# Real-Time Audio Event Detection (POC)

This repository contains a **real-time audio event detection proof-of-concept**, built as a continuation of a previous audio detection side project.

The purpose of this project is **not to achieve the highest possible accuracy**, but to **validate the concept and observe system behavior** when running audio ML models in a real-time streaming environment.

---

## 🎯 What This Project Does

- Detects **dog barking**, **screaming**, and **gunshot-like** audio events
- Uses **live microphone input**
  - Audio can come from the surrounding environment
  - Or sound effects played from an external device (e.g. iPhone)
- Runs **streaming inference** using sliding audio windows
- Includes practical logic for:
  - Event onset detection
  - Event stability
  - False-trigger control
  - On/off state transitions

This POC focuses on the kinds of behaviors that matter when AI models interact with **real hardware and real-time signals**.

---

## 🧠 Model

- Uses **PANNs (Pretrained Audio Neural Networks)** for audio tagging
- Example model:
  - `Cnn14 (mAP = 0.431)`
- Inference runs on CPU in this proof-of-concept

> Model choice here is pragmatic.  
> The emphasis is on **real-time system behavior**, not offline benchmarking or leaderboard scores.

---

## 🏗 Project Structure

```text
rt_audio_anomaly/
├── README.md
├── audio effect.wav        # Sample audio / sound effects
├── models/                 # Pretrained model files (path configurable)
├── src/                    # Real-time inference scripts
├── experiments/            # Experiments and variations
└── notes/                  # Design notes and observations
```

---

## ▶️ How to Run (Example)

```bash
python3 src/rt_mic_target_events_panns.py \
  --device 2 \
  --sr_in 44100 \
  --window_ms 1000 \
  --step_ms 250 \
  --print_mode stream
```

- Audio device index depends on your local setup  
- Use `sounddevice.query_devices()` to list available input devices

---

## ⚠️ Notes

- This project is intended for **concept validation and system experimentation**
- Thresholds and heuristics are intentionally simple
- Observing **event timing, stability, and behavior** is often more important than raw confidence values in real-time systems

---

## 🚀 Why This Matters

Offline metrics alone rarely reflect real-world performance.

This POC explores the gap between:

- Offline ML evaluation
- Real-time, streaming, hardware-facing systems

Starting around **2026**, more AI-powered hardware systems are expected to move from concepts into real-world products.  
Even in **multimedia product categories**, AI is likely to become a visible and integral part of system design.

---

## 📌 Status

- ✔ Real-time microphone input
- ✔ Streaming inference
- ✔ Event gating and stability logic
- ⏳ Future work: Edge deployment (e.g. NVIDIA Orin Nano)

---

## 📄 License

This repository is for research, learning, and experimentation purposes.
