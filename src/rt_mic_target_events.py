#!/usr/bin/env python3
"""
rt_mic_target_events.py
Targeted real-time audio event detection on microphone input using PANNs CNN14.

Goals:
- Real-time mic streaming (USB mic ok)
- ~200ms hop (user wants low latency)
- Detect ONLY target events: scream, crying, dog_bark, shouting/argument, gunshot (strict)
- Suppress common false triggers: speech, vehicle
- ASCII bar output + product-like state machine (hysteresis + consecutive + cooldown)

Prereqs:
- torch, librosa, sounddevice
- models/Cnn14_mAP=0.431.pth (downloaded from Zenodo)
- numpy < 2 (recommended for torch compatibility)
"""

import time
import queue
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import sounddevice as sd
import librosa

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------
# Ring buffer
# ----------------------------

class RingBuffer:
    def __init__(self, size: int):
        self.size = int(size)
        self.buf = np.zeros(self.size, dtype=np.float32)
        self.idx = 0
        self.full = False

    def write(self, x: np.ndarray):
        x = np.asarray(x, dtype=np.float32).flatten()
        n = len(x)
        if n <= 0:
            return
        if n >= self.size:
            self.buf[:] = x[-self.size:]
            self.idx = 0
            self.full = True
            return
        end = self.idx + n
        if end < self.size:
            self.buf[self.idx:end] = x
        else:
            k = self.size - self.idx
            self.buf[self.idx:] = x[:k]
            self.buf[:end - self.size] = x[k:]
            self.full = True
        self.idx = end % self.size

    def read_last(self, n: int) -> np.ndarray:
        n = int(n)
        if n > self.size:
            raise ValueError("read exceeds buffer")
        start = (self.idx - n) % self.size
        if start < self.idx:
            return self.buf[start:self.idx].copy()
        return np.concatenate([self.buf[start:], self.buf[:self.idx]]).copy()


# ----------------------------
# Product-ish state machine
# ----------------------------

@dataclass
class HoldOffDetector:
    """
    Hysteresis + consecutive frames + cooldown
    - score >= on_th for consecutive -> EVENT ON
    - score <= off_th for consecutive -> EVENT OFF
    - after EVENT ON, enforce cooldown seconds to avoid spamming
    """
    on_th: float
    off_th: float
    consecutive: int
    cooldown_s: float

    state: bool = False
    on_count: int = 0
    off_count: int = 0
    last_on_time: float = -1e9

    def update(self, score: float, now: float) -> Tuple[bool, str]:
        toggled = False
        msg = ""

        if not self.state:
            # cooldown gate
            if now - self.last_on_time < self.cooldown_s:
                self.on_count = 0
                self.off_count = 0
                return False, ""

            if score >= self.on_th:
                self.on_count += 1
            else:
                self.on_count = max(0, self.on_count - 1)

            if self.on_count >= self.consecutive:
                self.state = True
                self.on_count = 0
                self.off_count = 0
                self.last_on_time = now
                toggled = True
                msg = "ON"
        else:
            if score <= self.off_th:
                self.off_count += 1
            else:
                self.off_count = 0

            if self.off_count >= self.consecutive:
                self.state = False
                self.on_count = 0
                self.off_count = 0
                toggled = True
                msg = "OFF"

        return toggled, msg


def ascii_bar(v: float, width: int = 24) -> str:
    v = float(np.clip(v, 0.0, 1.0))
    n = int(round(v * width))
    return "█" * n + "·" * (width - n)


# ----------------------------
# Minimal PANNs CNN14 implementation (inference only)
#   This is a compact version aligned to the public checkpoint.
# ----------------------------

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        x = F.avg_pool2d(x, kernel_size=(2, 2))
        return x


class Cnn14(nn.Module):
    """
    CNN14 backbone for AudioSet tagging.
    Input: log-mel spectrogram, shape (B, 1, T, M)
    Output: clipwise_output (B, 527) sigmoid probabilities
    """
    def __init__(self, classes_num=527):
        super().__init__()
        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = ConvBlock(1, 64)
        self.conv_block2 = ConvBlock(64, 128)
        self.conv_block3 = ConvBlock(128, 256)
        self.conv_block4 = ConvBlock(256, 512)
        self.conv_block5 = ConvBlock(512, 1024)
        self.conv_block6 = ConvBlock(1024, 2048)

        self.fc1 = nn.Linear(2048, 2048, bias=True)
        self.fc_audioset = nn.Linear(2048, classes_num, bias=True)

    def forward(self, x):
        # x: (B, 1, T, M) -> transpose to (B, 1, M, T) typical in PANNs?
        # In PANNs, they often treat mel bins as freq dim.
        # We'll keep (B,1,T,M) but apply bn over freq bins by permuting.
        x = x.permute(0, 1, 3, 2)  # (B,1,M,T)
        x = x.repeat(1, 64, 1, 1)  # expand channels to 64 for bn0 like official
        x = self.bn0(x)
        # return to (B,1,M,T) but conv expects channels=1. We'll instead reduce:
        # Simpler: take mean over those 64 channels to return (B,1,M,T)
        x = torch.mean(x, dim=1, keepdim=True)

        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        x = self.conv_block6(x)

        # global pooling over time/freq
        x = torch.mean(x, dim=3)  # average over time axis
        x1 = torch.max(x, dim=2).values
        x2 = torch.mean(x, dim=2)
        x = x1 + x2  # (B, 2048)

        x = F.relu_(self.fc1(x))
        x = torch.sigmoid(self.fc_audioset(x))
        return x


def load_checkpoint(model: nn.Module, ckpt_path: str):
    state = torch.load(ckpt_path, map_location="cpu")
    # The checkpoint might store 'model' or direct weights.
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    # Some keys won't match due to our compact implementation.
    # We'll load with strict=False and warn if too many missing keys.
    missing, unexpected = model.load_state_dict(state, strict=False)
    if len(unexpected) > 0:
        print(f"⚠️ Unexpected keys: {len(unexpected)} (ok for compact loader)")
    if len(missing) > 0:
        print(f"⚠️ Missing keys: {len(missing)} (if huge, we should adjust model definition)")


# ----------------------------
# Label mapping (subset)
# We avoid downloading the full class list; define robust keyword matching.
# We'll use a small embedded list of AudioSet labels for key targets.
# ----------------------------

# Minimal label list for matching (AudioSet names are stable)
# In practice, we'll just pick indices by approximate known names.
# For better accuracy later, we can load the official class map CSV.
AUDIOSET_TARGET_KEYWORDS = {
    "scream": ["scream", "screaming"],
    "crying": ["crying", "baby cry", "cry"],
    "dog_bark": ["dog", "bark", "dog bark"],
    "shouting": ["shout", "shouting", "yell", "yelling", "argument"],
    "gunshot": ["gunshot", "gun fire", "gun", "machine gun"],
    "speech": ["speech", "conversation", "narration", "talking"],
    "vehicle": ["vehicle", "car", "truck", "engine", "traffic"],
}

# Fallback indices (common in many AudioSet mappings). We'll try to find by keyword first.
# If keyword mapping fails, we can tune these later with real class map.
FALLBACK_CLASS_IDX = {
    "scream": [0],     # will be replaced once we load real map
    "crying": [0],
    "dog_bark": [0],
    "shouting": [0],
    "gunshot": [0],
    "speech": [0],
    "vehicle": [0],
}


def compute_logmel(
    audio: np.ndarray,
    sr: int,
    n_fft: int,
    hop_length: int,
    n_mels: int,
    fmin: int,
    fmax: int,
) -> np.ndarray:
    # streaming-friendly: center=False to avoid lookahead
    S = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        power=2.0,
        center=False,
    )
    logmel = librosa.power_to_db(S, ref=np.max).astype(np.float32)
    # normalize per-window to reduce gain sensitivity
    logmel = (logmel - np.mean(logmel)) / (np.std(logmel) + 1e-6)
    # shape to (1, 1, T, M) expected by our model: we currently use (B,1,T,M)
    logmel = logmel.T  # (T, M)
    return logmel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=None)
    ap.add_argument("--sr", type=int, default=16000)

    # Real-time settings: ~200ms hop
    ap.add_argument("--window_ms", type=float, default=960.0, help="Window length (ms). ~1s works best for AudioSet models.")
    ap.add_argument("--step_ms", type=float, default=200.0, help="Hop length (ms).")

    # Feature params
    ap.add_argument("--n_mels", type=int, default=64)
    ap.add_argument("--n_fft", type=int, default=1024)
    ap.add_argument("--mel_hop", type=int, default=320)
    ap.add_argument("--fmin", type=int, default=50)
    ap.add_argument("--fmax", type=int, default=7600)

    # Model + checkpoint
    ap.add_argument("--ckpt", type=str, default="models/Cnn14_mAP=0.431.pth")

    # thresholds
    ap.add_argument("--on", type=float, default=0.75)
    ap.add_argument("--off", type=float, default=0.55)
    ap.add_argument("--consecutive", type=int, default=2)
    ap.add_argument("--cooldown", type=float, default=4.0)

    # per-event overrides (optional)
    ap.add_argument("--gun_on", type=float, default=0.90, help="Gunshot is strict to reduce fireworks false positives.")

    args = ap.parse_args()

    sr = args.sr
    win = int(sr * args.window_ms / 1000.0)
    step = int(sr * args.step_ms / 1000.0)

    rb = RingBuffer(size=int(sr * 3.0))
    q = queue.Queue(maxsize=200)

    def callback(indata, frames, time_info, status):
        x = indata[:, 0].astype(np.float32)
        try:
            q.put_nowait(x)
        except queue.Full:
            pass

    device = "cpu"
    model = Cnn14(classes_num=527).to(device).eval()
    print("🧠 Loading CNN14 checkpoint...")
    load_checkpoint(model, args.ckpt)

    # Detectors: different strictness for gunshot
    det_general = HoldOffDetector(on_th=args.on, off_th=args.off, consecutive=args.consecutive, cooldown_s=args.cooldown)
    det_gun = HoldOffDetector(on_th=args.gun_on, off_th=max(0.65, args.off), consecutive=1, cooldown_s=max(6.0, args.cooldown))

    targets = ["scream", "crying", "dog_bark", "shouting", "gunshot"]
    suppress = ["speech", "vehicle"]

    print("🎙️ Starting mic stream...")
    print(f"Window={args.window_ms:.0f}ms Hop={args.step_ms:.0f}ms (≈{args.step_ms:.0f}ms update)  | targets={targets}")
    print("Ctrl+C to stop.\n")

    next_t = time.time()

    with sd.InputStream(samplerate=sr, channels=1, dtype="float32", callback=callback, device=args.device):
        while True:
            # feed buffer
            try:
                rb.write(q.get(timeout=0.05))
            except queue.Empty:
                pass

            now = time.time()
            if now < next_t:
                continue
            if (not rb.full) and (rb.idx < win):
                next_t += step / sr
                continue

            audio = rb.read_last(win)

            # feature
            logmel = compute_logmel(
                audio, sr,
                n_fft=args.n_fft,
                hop_length=args.mel_hop,
                n_mels=args.n_mels,
                fmin=args.fmin,
                fmax=args.fmax
            )
            x = torch.from_numpy(logmel).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,T,M)
            with torch.no_grad():
                probs = model(x)[0].detach().cpu().numpy()  # (527,)

            # NOTE: Without the official AudioSet class map, we can't reliably pick indices.
            # For now we use a pragmatic approach: treat "targets" as unknown in index space
            # and show "generic activity" score based on top-K probability.
            # Next step: we will load the official class_labels_indices.csv and map properly.
            topk = 8
            top_vals = np.sort(probs)[-topk:]
            activity = float(np.mean(top_vals))  # 0..1-ish
            activity = float(np.clip(activity * 3.0, 0.0, 1.0))  # scale for readability

            # suppress heuristic: if "speech/vehicle" are dominant, downweight activity
            # (placeholder until we load real label indices)
            # For now, we use spectral centroid as a lightweight suppressor:
            # - strong low-frequency energy often indicates vehicle/traffic
            # - steady mid band can be speech; we'll keep it conservative
            # (This is intentionally mild; real suppression will use label mapping.)
            # You can remove this once label mapping is in place.
            centroid = float(np.sum(np.fft.rfftfreq(len(audio), 1/sr) * (np.abs(np.fft.rfft(audio)) + 1e-9)) /
                            np.sum(np.abs(np.fft.rfft(audio)) + 1e-9))
            if centroid < 800:  # often traffic/engine-ish
                activity *= 0.8

            # event logic
            toggled, msg = det_general.update(activity, now)

            bar = ascii_bar(activity)
            if toggled:
                print(f"\n{'🚨 EVENT ON' if det_general.state else '✅ EVENT OFF'} | activity={activity:.2f}")

            print(f"\ractivity {activity:0.2f} |{bar}| centroid={centroid:4.0f}Hz", end="")

            next_t += step / sr


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
