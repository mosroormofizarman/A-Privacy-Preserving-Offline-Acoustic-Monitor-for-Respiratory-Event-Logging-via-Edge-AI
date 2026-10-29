# A Privacy-Preserving Offline Acoustic Monitor for Respiratory Event Logging via Edge AI

An offline, embedded acoustic monitoring pipeline designed for the **Raspberry Pi Zero W** and **INMP441 I2S Digital MEMS Microphone**. This system detects and logs respiratory events (`cough`, `wheeze`, `respiratory_anomaly`) at the edge without recording raw audio or requiring cloud connectivity.

## Key Engineering Highlights
- **Privacy-by-Design (Zero Raw Audio Storage):** 16kHz PCM audio frames captured via Linux ALSA I2S drivers are processed strictly in ephemeral RAM. Raw audio buffers are explicitly zeroed immediately after feature extraction (`extract_ephemeral_acoustic_features`), ensuring speech and ambient conversations can never be reconstructed.
- **Lightweight Pure-NumPy DSP:** Extracts 40-band Log-Mel / MFCC spectrograms using vectorized NumPy operations tailored for single-core ARMv6 processors.
- **Quantized INT8 TFLite Inference:** Runs a compact 2D Convolutional Neural Network (CNN) quantized to INT8 for sub-100ms real-time execution on 512MB RAM hardware.
- **Offline Metadata Logging:** Records only non-sensitive event timestamps, classification labels, confidence scores, and latency metrics to a local CSV/SQLite log.

## Built & Maintained with OpenAI Codex (`AGENTS.md`)
This repository includes a dedicated [`AGENTS.md`](./AGENTS.md) specification that enforces architectural constraints during agentic coding sessions with **OpenAI Codex**:
- Automated verification of in-RAM buffer wiping to guarantee privacy compliance.
- Refactoring desktop audio preprocessing libraries into zero-bloat NumPy routines for Raspberry Pi Zero W compatibility.
- Continuous latency benchmarking via `python edge_monitor.py --self-test`.

## Quickstart & Self-Test
```bash
pip install numpy
python edge_monitor.py --self-test
