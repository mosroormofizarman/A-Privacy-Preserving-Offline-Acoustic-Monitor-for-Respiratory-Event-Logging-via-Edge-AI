# AGENTS.md — Privacy-Preserving Offline Acoustic Respiratory Monitor

## Project Context
An offline, edge-native respiratory event detector (cough, wheeze, throat-clear, background) engineered for the **Raspberry Pi Zero W (ARMv6)** with an **INMP441 I2S digital MEMS microphone**.

## Strict Privacy & Architectural Rules for Codex Agents
1. **Zero Raw Audio Persistence:** Never write `.wav` or raw PCM buffers to disk or SD card. Audio frames must be processed strictly inside ephemeral in-RAM ring buffers and discarded immediately after 16kHz MFCC extraction.
2. **Offline-Only Execution:** Do not introduce any network, cloud API, or telemetry dependencies (`requests`, `urllib`, `socket`). All inference and event logging must operate 100% offline.
3. **Edge Resource Constraints:** Keep memory footprint minimal for 512MB RAM ARMv6 targets. Prefer vectorized `numpy` operations and `tflite-runtime` INT8 quantized inference over full TensorFlow packages.
4. **ALSA / I2S Compatibility:** Maintain 16,000 Hz, 1-channel (mono), 16-bit/32-bit PCM capture assumptions compatible with Linux ALSA `arecord` / `pyalsaaudio` I2S overlays.

## Verification & Testing Workflow
After modifying any DSP, feature extraction, or inference code, always run:
```bash
python edge_monitor.py --self-test
