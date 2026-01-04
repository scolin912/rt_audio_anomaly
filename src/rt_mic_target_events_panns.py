#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real-time mic (or WAV) target event detector using PANNs AudioTagging (Cnn14).
- Targets: scream / crying / dog_bark / shouting / gunshot
- Robust "sticky" prevention:
  * silence_frames: consecutive quiet frames force OFF
  * off_fast: fast-off threshold
  * max_on_s: hard timeout per event
- Prints EVERY line with:
  * per-target probability
  * per-target ON/OFF state
  * remaining timeout seconds for active ON events
  * suppression flags (speech/vehicle)
- Optional:
  * --debug_topk N : print top-N PANNs labels each frame (useful to debug "why")
  * --webhook_url : POST JSON on rising edges
  * --input_wav : simulate realtime from WAV (helps testing with your 20s clip)
"""

__VERSION__ = 'SAFETY_V14'

import argparse
import time
import json
import os
import sys
import math
import signal
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np



def _format_topk_debug(topk, k: int) -> str:
    """Human-readable top-k debug without dumping long vectors.

    - If topk is List[Tuple[label, score]]: prints "label=score".
    - If topk is a raw score vector (numpy array / list[float]): prints top-k indices with values.
    """
    if not k or k <= 0:
        return ""
    try:
        import numpy as np
    except Exception:
        np = None

    # Preferred: list of (label, score)
    if isinstance(topk, list) and (len(topk) == 0 or (isinstance(topk[0], tuple) and len(topk[0]) == 2)):
        if not topk:
            return "(none)"
        return ", ".join([f"{str(lbl)}={float(val):.3f}" for (lbl, val) in topk])

    # Fallback: raw score vector -> show only top indices
    if np is not None:
        arr = np.asarray(topk).reshape(-1)
        if arr.size == 0:
            return "(none)"
        kk = min(int(k), int(arr.size))
        idx = arr.argsort()[::-1][:kk]
        return ", ".join([f"{int(i)}:{float(arr[int(i)]):.3f}" for i in idx])

    # No numpy available
    if isinstance(topk, list) and topk and isinstance(topk[0], (int, float)):
        kk = min(int(k), len(topk))
        pairs = sorted(enumerate(topk), key=lambda t: float(t[1]), reverse=True)[:kk]
        return ", ".join([f"{i}:{float(v):.3f}" for i, v in pairs])

    return f"(unsupported topk type: {type(topk).__name__})"
try:
    import sounddevice as sd
except Exception as e:
    print("ERROR: sounddevice import failed. Install with: pip install sounddevice", file=sys.stderr)
    raise

try:
    import soundfile as sf
except Exception:
    sf = None  # only required for --input_wav

try:
    import torch
except Exception as e:
    print("ERROR: torch import failed. Install PyTorch first.", file=sys.stderr)
    raise

# PANNs inference wrapper (commonly used package name: panns_inference)
try:
    from panns_inference import AudioTagging
except Exception as e:
    print("ERROR: panns_inference import failed. Install with: pip install panns-inference", file=sys.stderr)
    raise

# Optional webhook
try:
    import requests
except Exception:
    requests = None


PANN_SR = 32000  # model expects 32kHz


def _label_to_str(x):
    """Normalize a label into a clean hashable string.

    Handles: str / bytes / bytearray / list / tuple / numpy scalar / numpy.ndarray.
    """
    try:
        import numpy as _np
    except Exception:
        _np = None

    if _np is not None:
        if isinstance(x, _np.ndarray):
            if x.size == 1:
                x = x.reshape(-1)[0]
            else:
                try:
                    flat = x.reshape(-1).tolist()
                    parts = []
                    for t in flat:
                        if isinstance(t, (bytes, bytearray)):
                            try:
                                parts.append(bytes(t).decode("utf-8", "ignore"))
                            except Exception:
                                parts.append(bytes(t).decode("latin-1", "ignore"))
                        else:
                            parts.append(str(t))
                    return " ".join(parts).strip()
                except Exception:
                    return str(x).strip()
        if isinstance(x, _np.generic):
            x = x.item()

    if isinstance(x, (list, tuple)):
        if len(x) == 1:
            x = x[0]
        else:
            return " ".join(_label_to_str(t) for t in x).strip()

    if isinstance(x, (bytes, bytearray)):
        try:
            return bytes(x).decode("utf-8").strip()
        except Exception:
            return bytes(x).decode("latin-1", "ignore").strip()

    if isinstance(x, str):
        return x.strip()

    return str(x).strip()



def list_devices() -> None:
    print("\n=== sounddevice devices ===")
    devices = sd.query_devices()
    default_hostapi = sd.query_hostapis()
    for i, d in enumerate(devices):
        api = default_hostapi[d["hostapi"]]["name"] if "hostapi" in d else "?"
        ins = d.get("max_input_channels", 0)
        outs = d.get("max_output_channels", 0)
        sr = d.get("default_samplerate", 0.0)
        print(f"[{i:2d}] {d['name']}  (in:{ins},out:{outs})  sr={sr}  api={api}")
    print("==========================\n")


def rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float32)
    return float(np.sqrt(np.mean(np.square(x)) + 1e-12))


def ascii_bar(v: float, width: int = 22) -> str:
    v = float(np.clip(v, 0.0, 1.0))
    n = int(round(v * width))
    return "█" * n + "·" * (width - n)


def now_s() -> float:
    return time.time()


def resample_linear(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Fast-ish linear resampler, good enough for real-time demo."""
    if sr_in == sr_out:
        return x.astype(np.float32, copy=False)
    x = x.astype(np.float32, copy=False)
    n_in = len(x)
    n_out = int(round(n_in * sr_out / sr_in))
    if n_out <= 1 or n_in <= 1:
        return np.zeros((max(n_out, 1),), dtype=np.float32)
    t_in = np.linspace(0.0, 1.0, n_in, endpoint=False, dtype=np.float32)
    t_out = np.linspace(0.0, 1.0, n_out, endpoint=False, dtype=np.float32)
    return np.interp(t_out, t_in, x).astype(np.float32)


@dataclass
class EventState:
    on: bool = False
    on_since: float = 0.0
    last_score: float = 0.0
    quiet_count: int = 0
    off_count: int = 0

    def remaining_timeout(self, max_on_s: float, t: float) -> Optional[float]:
        if (not self.on) or max_on_s <= 0:
            return None
        return max(0.0, (self.on_since + max_on_s) - t)


class MultiEventGate:
    """
    "Safety alarm" gate with:
      - per-event consecutive-frame confirmation
      - margin vs best competing class (approx. from top-K)
      - cooldown after an alarm (global)
      - (special) gunshot requires: score >= gun_on, in top-N, AND consecutive gun_frames

    This gate is intentionally conservative to reduce false alarms in a "safety" product.
    """

    def __init__(self, args):
        self.args = args

        # Generic thresholds
        self.on_thr = float(args.on)
        self.off_fast = float(args.off_fast)
        self.silence_frames = int(args.silence_frames)
        self.max_on_s = float(args.max_on_s)

        # Anti-false-positive controls
        self.margin = float(args.margin)
        self.cooldown_s = float(args.cooldown_s)
        self.min_on_frames = int(args.min_on_frames)

        # Gunshot-specific
        self.gun_on = float(getattr(args, "gun_on", 0.55))
        self.gun_topn = int(getattr(args, "gun_topn", 3))
        self.gun_frames = int(getattr(args, "gun_frames", 3))

        # Runtime state
        self.state_on = False
        self.active_event: Optional[str] = None
        self.ttl_end_t = 0.0
        self.cooldown_end_t = 0.0
        self.off_cnt = 0

        self.on_cnt = {"gunshot": 0, "scream": 0, "dog_bark": 0}

        # Label aliases used for margin/topN checks
        self.alias = {
            "gunshot": {"Gunshot, gunfire", "Gunshot"},
            "scream": {"Screaming", "Scream"},
            "dog_bark": {"Dog", "Dog bark", "Bark", "Bow-wow"},
        }

        # Priority: most safety-critical first
        self.priority = ["gunshot", "scream", "dog_bark"]

    def _best_other_from_topk(self, topk_pairs: List[Tuple[str, float]], event: str) -> float:
        """Approximate 'best competing class' using top-K list."""
        banned = self.alias.get(event, set())
        best = 0.0
        for lbl, sc in topk_pairs:
            if lbl in banned:
                continue
            if sc > best:
                best = sc
        return best

    def _gun_in_topn(self, topk_pairs: List[Tuple[str, float]]) -> bool:
        topn = topk_pairs[: max(self.gun_topn, 1)]
        for lbl, _sc in topn:
            if lbl in self.alias["gunshot"]:
                return True
        return False

    def step(
        self,
        t: float,
        target_scores: Dict[str, float],
        topk_pairs: List[Tuple[str, float]],
    ) -> Dict[str, object]:
        """
        Advances the gate by one frame.

        Returns a dict with keys:
          - state_on (bool)
          - active_event (str|None)
          - ttl_remaining_s (float)
          - cooldown_remaining_s (float)
          - changed (bool): whether ON/OFF changed this step
        """
        changed = False

        # Update OFF logic if currently ON
        if self.state_on:
            # TTL expiry
            if t >= self.ttl_end_t:
                self.state_on = False
                self.active_event = None
                self.cooldown_end_t = t + self.cooldown_s
                self.off_cnt = 0
                for k in self.on_cnt:
                    self.on_cnt[k] = 0
                changed = True
            else:
                # Early off if active event has fallen below off_fast for N frames
                if self.active_event:
                    cur = float(target_scores.get(self.active_event, 0.0))
                    if cur <= self.off_fast:
                        self.off_cnt += 1
                    else:
                        self.off_cnt = 0
                    if self.off_cnt >= self.silence_frames:
                        self.state_on = False
                        self.active_event = None
                        self.cooldown_end_t = t + self.cooldown_s
                        self.off_cnt = 0
                        for k in self.on_cnt:
                            self.on_cnt[k] = 0
                        changed = True

        # If OFF, possibly trigger
        if (not self.state_on) and (t >= self.cooldown_end_t):
            # Update per-event consecutive counters
            for ev in self.priority:
                score = float(target_scores.get(ev, 0.0))
                best_other = self._best_other_from_topk(topk_pairs, ev)
                thr = self.gun_on if ev == "gunshot" else self.on_thr

                ok = (score >= thr) and (score >= best_other + self.margin)

                if ev == "gunshot":
                    ok = ok and self._gun_in_topn(topk_pairs)

                if ok:
                    self.on_cnt[ev] += 1
                else:
                    self.on_cnt[ev] = 0

            # Trigger if any event meets its frame requirement (priority order)
            for ev in self.priority:
                need = self.gun_frames if ev == "gunshot" else self.min_on_frames
                if self.on_cnt[ev] >= need:
                    self.state_on = True
                    self.active_event = ev
                    self.ttl_end_t = t + self.max_on_s
                    self.off_cnt = 0
                    # Reset other counters to avoid immediate re-trigger noise
                    for k in self.on_cnt:
                        self.on_cnt[k] = 0
                    changed = True
                    break

        ttl_rem = max(self.ttl_end_t - t, 0.0) if self.state_on else 0.0
        cd_rem = max(self.cooldown_end_t - t, 0.0) if (not self.state_on) else 0.0

        return {
            "state_on": self.state_on,
            "active_event": self.active_event,
            "ttl_remaining_s": ttl_rem,
            "cooldown_remaining_s": cd_rem,
            "changed": changed,
        }


def send_webhook(url: str, payload: dict, timeout_s: float = 2.0) -> None:
    if not url:
        return
    if requests is None:
        print("WARN: requests not installed, webhook disabled. Install: pip install requests", file=sys.stderr)
        return
    try:
        requests.post(url, json=payload, timeout=timeout_s)
    except Exception as e:
        print(f"WARN: webhook post failed: {e}", file=sys.stderr)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--list_devices", action="store_true", help="List audio devices and exit")

    # input modes
    p.add_argument("--device", type=int, default=None, help="sounddevice input device index")
    p.add_argument("--sr_in", type=int, default=16000, help="Mic capture sample rate")
    p.add_argument("--input_wav", type=str, default="", help="Process a WAV file in pseudo-realtime (requires soundfile)")
    p.add_argument("--realtime_factor", type=float, default=1.0, help="WAV mode speed (1.0 = realtime)")

    # windowing
    p.add_argument("--window_ms", type=int, default=1500, help="Model window length")
    p.add_argument("--step_ms", type=int, default=200, help="Update hop")

    # gating
    p.add_argument("--rms_gate", type=float, default=0.008, help="Skip inference if too quiet (0 disables)")
    p.add_argument("--on", type=float, default=0.65, help="ON threshold for targets")
    p.add_argument("--off_fast", type=float, default=0.12, help="Fast OFF threshold")
    p.add_argument("--silence_frames", type=int, default=2, help="Force OFF after N consecutive quiet frames")
    p.add_argument("--max_on_s", type=float, default=3.0, help="Force OFF after N seconds per event (0 disables)")
    # --- safety gating controls ---
    p.add_argument("--print_mode", choices=["changes", "stream"], default="changes",
                   help="Print mode: 'changes' prints only ON/OFF/active changes; 'stream' prints every frame.")
    p.add_argument("--cooldown_s", type=float, default=3.0, help="Cooldown after an alarm OFF before re-arming.")
    p.add_argument("--margin", type=float, default=0.10,
                   help="Require target score >= best competing class + margin (approx via top-K).")
    p.add_argument("--min_on_frames", type=int, default=2,
                   help="Consecutive frames required to trigger for scream/dog_bark.")
    # Gunshot: very conservative defaults
    p.add_argument("--gun_on", type=float, default=0.55, help="Gunshot score threshold.")
    p.add_argument("--gun_topn", type=int, default=3, help="Gunshot must appear in top-N labels.")
    p.add_argument("--gun_frames", type=int, default=3, help="Consecutive frames required for gunshot trigger.")

    # suppression
    p.add_argument("--suppress_th", type=float, default=0.35, help="If speech/vehicle above this, suppress events (0 disables)")

    # model
    p.add_argument("--ckpt", type=str, default=os.path.expanduser("~/panns_data/Cnn14_mAP=0.431.pth"))

    # debug / output
    p.add_argument("--debug_topk", type=int, default=0, help="Print top-k labels each frame")
    p.add_argument("--print_json", action="store_true", help="Also print per-frame JSON (for piping/logging)")

    # notifications
    p.add_argument("--webhook_url", type=str, default="", help="POST JSON on rising edges")

    return p


def load_model(ckpt: str) -> AudioTagging:
    print("🧠 Initializing PANNs AudioTagging...")
    print(f"Checkpoint path: {ckpt}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device.upper()}.")
    model = AudioTagging(checkpoint_path=ckpt, device=device)
    return model






def infer_panns(
    model: AudioTagging,
    audio_32k: np.ndarray,
    *,
    debug_topk: int = 0,
    required_topn: int = 3,
) -> Tuple[Dict[str, float], Dict[str, float], List[Tuple[str, float]]]:
    """Safe PANNs inference across panns_inference versions (self-contained).

    Handles:
      - input shape (batch, samples)
      - output format dict vs tuple/list
      - label normalization without relying on external helpers
    """

    def _clean(s: str) -> str:
        s = str(s)
        s = s.lower().strip()
        s = " ".join(s.split())
        return s

    # Ensure float32
    if audio_32k.dtype != np.float32:
        audio_32k = audio_32k.astype(np.float32, copy=False)

    # Ensure (batch, samples)
    if audio_32k.ndim == 1:
        audio_32k = np.expand_dims(audio_32k, axis=0)

    with torch.no_grad():
        out = model.inference(audio_32k)

        clip = None
        if isinstance(out, dict):
            clip = out.get("clipwise_output", None)
        elif isinstance(out, (list, tuple)) and len(out) > 0:
            clip = out[0]
        else:
            clip = None

        if clip is None:
            raise RuntimeError(f"Unexpected model.inference() output type: {type(out).__name__}")

        if hasattr(clip, "detach"):
            scores = clip[0].detach().cpu().numpy().astype(np.float32, copy=False)
        else:
            scores = np.asarray(clip)[0].astype(np.float32, copy=False)

    labels = [str(x) for x in model.labels]

    k = int(max(int(debug_topk or 0), int(required_topn or 0), 1))
    top_idx = np.argsort(scores)[::-1][:k]
    topk_pairs = [(labels[i], float(scores[i])) for i in top_idx]

    norm_map = {_clean(labels[i]): i for i in range(len(labels))}

    def _get(label: str) -> float:
        i = norm_map.get(_clean(label))
        return float(scores[i]) if i is not None else 0.0

    dog = max(_get("Dog"), _get("Dog bark"), _get("Bark"), _get("Bow-wow"))
    scream = max(_get("Screaming"), _get("Scream"))
    gunshot = max(_get("Gunshot, gunfire"), _get("Gunshot"))

    target_scores = {"dog_bark": dog, "scream": scream, "gunshot": gunshot}
    suppress_scores = {"speech": _get("Speech"), "vehicle": _get("Vehicle")}

    return target_scores, suppress_scores, topk_pairs


def run_stream(args: argparse.Namespace) -> None:
    """Live microphone mode (sounddevice InputStream)."""
    if sd is None:
        raise RuntimeError("sounddevice is required for live mic mode. Please `pip install sounddevice`.")

    model = load_model(args.ckpt)
    gate = MultiEventGate(args)

    sr_in = int(args.sr_in)
    window_n = int(round(args.window_ms * sr_in / 1000.0))
    hop_n = int(round(args.step_ms * sr_in / 1000.0))
    window_n = max(window_n, 1)
    hop_n = max(hop_n, 1)

    required_topn = max(int(args.debug_topk or 0), int(getattr(args, "gun_topn", 3)))

    print("🎙️  Live mic mode")
    print(f"Input SR={sr_in}, device={args.device if args.device is not None else 'default'}")
    print(f"Window={args.window_ms}ms  Hop={args.step_ms}ms")
    print("Ctrl+C to stop.\n")

    import queue
    q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=50)

    def callback(indata, frames, time_info, status):
        if status:
            # drop status warnings silently (can be noisy)
            pass
        x = indata[:, 0].copy()  # float32
        try:
            q.put_nowait(x)
        except queue.Full:
            # drop if overloaded
            pass

    last_print_key = None
    last_active = None
    t_start = time.time()
    samples_seen = 0

    buf = np.zeros(0, dtype=np.float32)

    try:
        with sd.InputStream(
            device=args.device,
            channels=1,
            samplerate=sr_in,
            dtype="float32",
            blocksize=hop_n,
            callback=callback,
        ):
            while True:
                x = q.get()
                buf = np.concatenate([buf, x], axis=0)

                # Process as many hops as possible
                while len(buf) >= window_n:
                    frame = buf[:window_n]
                    buf = buf[hop_n:]  # advance by hop

                    rms_val = rms(frame)

                    if rms_val < args.rms_gate:
                        target_scores = {"dog_bark": 0.0, "scream": 0.0, "gunshot": 0.0}
                        suppress_scores = {"speech": 0.0, "vehicle": 0.0}
                        topk_pairs: List[Tuple[str, float]] = []
                    else:
                        audio_32k = resample_linear(frame, sr_in, 32000)
                        target_scores, suppress_scores, topk_pairs = infer_panns(
                            model,
                            audio_32k,
                            debug_topk=int(args.debug_topk or 0),
                            required_topn=required_topn,
                        )

                    now_t = samples_seen / float(sr_in)
                    info = gate.step(now_t, target_scores, topk_pairs)
                    samples_seen += hop_n

                    dog = float(target_scores.get("dog_bark", 0.0))
                    scr = float(target_scores.get("scream", 0.0))
                    gun = float(target_scores.get("gunshot", 0.0))
                    speech = float(suppress_scores.get("speech", 0.0))
                    veh = float(suppress_scores.get("vehicle", 0.0))

                    primary = max([("scream", scr), ("dog_bark", dog), ("gunshot", gun)], key=lambda x: x[1])
                    bar = ascii_bar(primary[1], width=20)

                    state = "ON " if info["state_on"] else "OFF"
                    active = (info["active_event"] or "-").ljust(8)
                    ttl = f'{info["ttl_remaining_s"]:.1f}s' if info["state_on"] else "--"
                    cd = info["cooldown_remaining_s"]
                    cd_txt = f"  CD {cd:.1f}s" if (not info["state_on"] and cd > 0.0) else ""

                    status_line = (
                        f"{primary[0]:8s} {primary[1]:.2f} |{bar}|  "
                        f"scr {scr:.2f}  dog {dog:.2f}  gun {gun:.2f}  "
                        f"(speech {speech:.2f}, veh {veh:.2f})  "
                        f"STATE {state} TL {ttl}  ACTIVE {active}  "
                        f"RMS {rms_val:.4f} gate {args.rms_gate:.3f}{cd_txt}"
                    )

                    print_this = False
                    if args.print_mode == "stream":
                        print_this = True
                    else:
                        if info["changed"]:
                            print_this = True
                        elif (info["state_on"] and info["active_event"] != last_active):
                            print_this = True

                    if print_this:
                        print(status_line)
                        if int(args.debug_topk or 0) > 0 and topk_pairs:
                            topk_show = topk_pairs[: int(args.debug_topk)]
                            pretty = ", ".join([f"{lbl}={sc:.3f}" for lbl, sc in topk_show])
                            print(f"🔎 top{len(topk_show)}: {pretty}")
                        last_print_key = status_line
                        last_active = info["active_event"]

    except KeyboardInterrupt:
        pass

    print("\nDone.")




def run_wav(args: argparse.Namespace) -> None:
    """Simulate realtime processing over a WAV file (sliding window)."""
    if sf is None:
        raise RuntimeError("soundfile is required for --input_wav mode. Please `pip install soundfile`.")

    # Init model
    model = load_model(args.ckpt)
    gate = MultiEventGate(args)

    # Load WAV
    wav, sr_in = sf.read(args.input_wav, dtype="float32", always_2d=False)
    if wav.ndim == 2:
        wav = np.mean(wav, axis=1).astype(np.float32, copy=False)
    wav = np.asarray(wav, dtype=np.float32)

    window_n = int(round(args.window_ms * sr_in / 1000.0))
    hop_n = int(round(args.step_ms * sr_in / 1000.0))
    window_n = max(window_n, 1)
    hop_n = max(hop_n, 1)

    print("🎧 WAV simulate mode")
    print(f"File={args.input_wav}  SR={sr_in}")
    print(f"Window={args.window_ms}ms  Hop={args.step_ms}ms  realtime_factor={args.realtime_factor}")
    print("Ctrl+C to stop.\n")

    # Needed top-N for gate constraints (gunshot top3) even if debug_topk=0
    required_topn = max(int(args.debug_topk or 0), int(getattr(args, "gun_topn", 3)))

    # Printing policy
    last_print_key = None
    last_active = None

    t0 = time.time()
    frame_idx = 0

    try:
        for start in range(0, max(len(wav) - window_n + 1, 1), hop_n):
            frame = wav[start : start + window_n]
            if len(frame) < window_n:
                # pad with zeros for last frame
                pad = np.zeros(window_n - len(frame), dtype=np.float32)
                frame = np.concatenate([frame, pad], axis=0)

            rms_val = rms(frame)
            # Gate on RMS (skip inference for very quiet frames)
            if rms_val < args.rms_gate:
                target_scores = {"dog_bark": 0.0, "scream": 0.0, "gunshot": 0.0}
                suppress_scores = {"speech": 0.0, "vehicle": 0.0}
                topk_pairs: List[Tuple[str, float]] = []
            else:
                # PANNs expects 32kHz
                audio_32k = resample_linear(frame, sr_in, 32000)
                target_scores, suppress_scores, topk_pairs = infer_panns(
                    model,
                    audio_32k,
                    debug_topk=int(args.debug_topk or 0),
                    required_topn=required_topn,
                )

            now_t = frame_idx * (hop_n / float(sr_in))
            info = gate.step(now_t, target_scores, topk_pairs)

            # Build a readable status line
            dog = float(target_scores.get("dog_bark", 0.0))
            scr = float(target_scores.get("scream", 0.0))
            gun = float(target_scores.get("gunshot", 0.0))
            speech = float(suppress_scores.get("speech", 0.0))
            veh = float(suppress_scores.get("vehicle", 0.0))

            # Primary label for the bar
            primary = max([("scream", scr), ("dog_bark", dog), ("gunshot", gun)], key=lambda x: x[1])

            # If truly no target evidence (quiet or all zeros), show "no event"
            if (rms_val < args.rms_gate) or (primary[1] <= 1e-6):
                primary = ("no event", 0.0)

            bar = ascii_bar(primary[1], width=20)


            state = "ON " if info["state_on"] else "OFF"
            active = (info["active_event"] or "-").ljust(8)
            ttl = f'{info["ttl_remaining_s"]:.1f}s' if info["state_on"] else "--"
            cd = info["cooldown_remaining_s"]
            cd_txt = f"  CD {cd:.1f}s" if (not info["state_on"] and cd > 0.0) else ""

            status_line = (
                f"{primary[0]:8s} {primary[1]:.2f} |{bar}|  "
                f"scr {scr:.2f}  dog {dog:.2f}  gun {gun:.2f}  "
                f"(speech {speech:.2f}, veh {veh:.2f})  "
                f"STATE {state} TL {ttl}  ACTIVE {active}  "
                f"RMS {rms_val:.4f} gate {args.rms_gate:.3f}{cd_txt}"
            )

            # Print decision
            print_this = False
            if args.print_mode == "stream":
                print_this = True
            else:
                # Only print when something meaningful changes
                if info["changed"]:
                    print_this = True
                elif (info["state_on"] and info["active_event"] != last_active):
                    print_this = True
                elif (status_line != last_print_key and info["state_on"]):
                    # optional: during ON, if numbers change, print (kept conservative)
                    print_this = False

            if print_this:
                print(status_line)
                if int(args.debug_topk or 0) > 0 and topk_pairs:
                    topk_show = topk_pairs[: int(args.debug_topk)]
                    pretty = ", ".join([f"{lbl}={sc:.3f}" for lbl, sc in topk_show])
                    print(f"🔎 top{len(topk_show)}: {pretty}")
                last_print_key = status_line
                last_active = info["active_event"]

            frame_idx += 1

            # Realtime pacing (optional)
            if args.realtime_factor > 0:
                target_dt = frame_idx * (hop_n / float(sr_in)) / float(args.realtime_factor)
                elapsed = time.time() - t0
                if target_dt > elapsed:
                    time.sleep(min(target_dt - elapsed, 0.25))

    except KeyboardInterrupt:
        pass

    print("\nDone.")




def main():
    args = build_argparser().parse_args()
    print("🔖 Script version:", __VERSION__)

    # Backward/forward compatible aliases between gate_rms and rms_gate
    if (not hasattr(args, "rms_gate")) and hasattr(args, "gate_rms"):
        args.rms_gate = args.gate_rms
    if (not hasattr(args, "gate_rms")) and hasattr(args, "rms_gate"):
        args.gate_rms = args.rms_gate

    if args.list_devices:
        list_devices()
        return

    if args.input_wav:
        run_wav(args)
        return

    # default mic mode
    run_stream(args)


if __name__ == "__main__":
    main()
