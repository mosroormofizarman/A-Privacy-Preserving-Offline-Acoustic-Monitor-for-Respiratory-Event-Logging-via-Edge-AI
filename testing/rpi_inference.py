import argparse
import time
import queue
import sys
import numpy as np

from audio_utils import TARGET_SR, CHUNK_SAMPLES, make_input_tensor

# ── Class definitions ─────────────────────────────────────────────────────────
LABELS = {
    0: "Normal Breathing",
    1: "Cough",
    2: "Abnormal Resp.",      # shortened to fit 128px display
}

LABEL_ICONS = {
    0: "\u25cf",   # ● filled circle  — calm / normal
    1: "!",        # !  exclamation   — cough
    2: "\u26a0",   # ⚠  warning sign  — abnormal
}

ALERT_CLASSES = {1, 2}


# =============================================================================
# Binary Cough Event Detector
# =============================================================================
# The 3-class model is unreliable distinguishing Normal vs Abnormal but
# strong on Cough (perfect recall in the confusion matrix). So we collapse
# to a binary problem at inference time: cough vs not-cough.
#
# To kill false alarms we add three guards:
#   1. SILENCE_GATE      — peak below this is treated as silence (skip model)
#   2. COUGH_THRESHOLD   — Cough probability must exceed this to count
#   3. CONSECUTIVE_HITS  — N positive frames in a row to fire a cough event
#
# Tune by watching live output: stay silent → no events; cough → event.
# =============================================================================

SILENCE_GATE     = 0.02   # peak amplitude below this = ambient silence
COUGH_THRESHOLD  = 0.80   # min P(cough) to count a frame as positive
CONSECUTIVE_HITS = 2      # frames in a row required to fire an event
EVENT_COOLDOWN   = 1.5    # seconds to suppress repeats after firing

# ─── Transient discriminator ─────────────────────────────────────────────────
# A cough is a SHORT burst (<700ms) inside a quiet 1-second window. Speech,
# music, fans, white noise — all have sustained energy across most of the
# window. We reject sustained signals BEFORE running the model, which is what
# was producing all the "Cough 100%" false positives on talking.
#
# Two checks (both must pass):
#   A. Active-time fraction: <55% of the window has energy above 20% of peak.
#      Cough ≈ 30-50%. Speech ≈ 80-95%.
#   B. Sharp attack: 100ms before the envelope peak, energy is <40% of peak.
#      Coughs ramp up in ~30ms. Speech ramps up over hundreds of ms.
# ─────────────────────────────────────────────────────────────────────────────

ACTIVE_FRAC_MAX  = 0.55   # max fraction of window with active energy
ATTACK_RATIO_MAX = 0.40   # energy 100ms pre-peak / peak (must be below this)


def looks_like_cough_envelope(chunk: np.ndarray, samplerate: int) -> tuple:
    """
    Test whether the 1-second envelope shape is consistent with a cough
    (short burst + silence + sharp attack), as opposed to sustained sound
    (speech, music, fan, hum).

    Returns (is_cough_shape: bool, active_fraction: float, attack_ratio: float).
    """
    if len(chunk) < samplerate // 4:
        return False, 0.0, 1.0

    # Smooth absolute envelope with a 20 ms moving average
    win = max(1, samplerate // 50)
    env = np.convolve(np.abs(chunk), np.ones(win, dtype=np.float32) / win, mode="same")

    peak_val = float(env.max())
    if peak_val < 1e-5:
        return False, 0.0, 1.0

    # (A) Active-time fraction — what % of window is loud?
    active_fraction = float((env > 0.20 * peak_val).mean())

    # (B) Attack sharpness — how loud was it 100 ms before the peak?
    peak_idx = int(np.argmax(env))
    pre_idx  = max(0, peak_idx - samplerate // 10)   # 100 ms earlier
    attack_ratio = float(env[pre_idx] / (peak_val + 1e-9))

    is_cough_shape = (active_fraction <= ACTIVE_FRAC_MAX and
                      attack_ratio   <= ATTACK_RATIO_MAX)
    return is_cough_shape, active_fraction, attack_ratio


class CoughEventDetector:
    """
    Stateful cough-event detector with debouncing.

    Treats each 1-second model output as a vote. An "event" fires when
    we see CONSECUTIVE_HITS positive votes back-to-back, then enters a
    cooldown so a single 2-second cough doesn't fire 5 times.
    """

    def __init__(self, threshold=COUGH_THRESHOLD,
                 hits=CONSECUTIVE_HITS, cooldown=EVENT_COOLDOWN):
        self.threshold = threshold
        self.hits      = hits
        self.cooldown  = cooldown
        self._streak   = 0
        self._last_event_t = 0.0
        self.event_count   = 0

    def update(self, p_cough: float, peak: float, now: float) -> dict:
        """
        Feed one frame's stats. Returns a status dict for display.
        Status types:
          'silent'      — peak below gate, model output ignored
          'listening'   — model ran, no cough detected
          'building'    — frame was positive but streak < hits
          'EVENT'       — cough event just fired
          'cooldown'    — within cooldown window after a recent event
        """
        # 1. Cooldown overrides everything
        if now - self._last_event_t < self.cooldown:
            return {"status": "cooldown", "streak": self._streak}

        # 2. Silence gate
        if peak < SILENCE_GATE:
            self._streak = 0
            return {"status": "silent", "streak": 0}

        # 3. Threshold check
        if p_cough >= self.threshold:
            self._streak += 1
            if self._streak >= self.hits:
                self.event_count += 1
                self._last_event_t = now
                self._streak = 0
                return {"status": "EVENT", "count": self.event_count}
            return {"status": "building", "streak": self._streak}
        else:
            self._streak = 0
            return {"status": "listening", "streak": 0}


# =============================================================================
# OLED Display Manager
# =============================================================================

class OLEDDisplay:
    """
    Manages the SSD1306 128×64 I2C OLED display.

    Physical colour zones (hardware characteristic of this display model):
      y =  0–15  →  YELLOW  (16 px tall)  — used for timestamp
      y = 16–63  →  BLUE    (48 px tall)  — used for classification result

    Drawing uses PIL (Pillow). Text is white on black; the hardware
    LEDs make the top 16 rows appear yellow automatically.
    """

    WIDTH         = 128
    HEIGHT        = 64
    YELLOW_HEIGHT = 16   # top 16 rows are physically yellow on this display

    # Row positions (top of each text line)
    ROW_TIMESTAMP  = 1    # inside yellow zone
    ROW_LABEL      = 17   # first blue row
    ROW_CONF_BAR   = 29   # confidence bar
    ROW_PROBS      = 41   # per-class probability text
    ROW_ALERT      = 53   # alert banner

    def __init__(self, i2c_address: int = 0x3C):
        from luma.core.interface.serial import i2c
        from luma.oled.device import ssd1306
        from PIL import ImageFont

        serial       = i2c(port=1, address=i2c_address)
        self.device  = ssd1306(serial, width=self.WIDTH, height=self.HEIGHT)
        self.address = i2c_address

        # ── Fonts ──────────────────────────────────────────────────────────
        # Try DejaVu (installed by setup_rpi.sh); fall back to PIL default.
        DEJAVU = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        DEJAVU_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

        def _font(path, size):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                return ImageFont.load_default()

        self.font_time   = _font(DEJAVU_B, 11)   # timestamp (yellow zone)
        self.font_label  = _font(DEJAVU_B, 10)   # classification label
        self.font_small  = _font(DEJAVU,    8)   # confidence % and probs
        self.font_alert  = _font(DEJAVU_B, 10)   # alert banner

        print(f"[OLED] Initialised SSD1306 at I2C address 0x{i2c_address:02X}")
        self._show_boot_screen()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _new_canvas(self):
        """Return a fresh black (128×64) PIL image and its draw context."""
        from PIL import Image, ImageDraw
        img  = Image.new("1", (self.WIDTH, self.HEIGHT), 0)
        draw = ImageDraw.Draw(img)
        return img, draw

    def _draw_conf_bar(self, draw, y: int, confidence: float):
        """
        Draw a filled progress bar showing confidence.

        Layout:  [██████░░░░] 82%
        Bar area: x=0..99 (100 px wide), label: x=102..127
        """
        bar_w    = 100
        bar_h    = 8
        filled   = int(confidence * bar_w)
        # outer frame
        draw.rectangle([0, y, bar_w - 1, y + bar_h - 1], outline=1, fill=0)
        # filled portion
        if filled > 2:
            draw.rectangle([1, y + 1, filled - 1, y + bar_h - 2], fill=1)
        # percentage label
        pct_str = f"{confidence:.0%}"
        draw.text((bar_w + 2, y), pct_str, font=self.font_small, fill=1)

    def _draw_separator(self, draw, y: int):
        """Horizontal separator line across full width."""
        draw.line([(0, y), (self.WIDTH - 1, y)], fill=1)

    # ── Public API ────────────────────────────────────────────────────────────

    def _show_boot_screen(self):
        """Splash screen shown once at startup."""
        img, draw = self._new_canvas()
        draw.text((2,  1), "Audio Monitor",  font=self.font_time,  fill=1)
        draw.text((2, 17), "Initialising...", font=self.font_label, fill=1)
        draw.text((2, 30), f"I2C: 0x{self.address:02X}",
                  font=self.font_small, fill=1)
        draw.text((2, 40), f"SR: {TARGET_SR} Hz",
                  font=self.font_small, fill=1)
        draw.text((2, 50), "Nyquist: 8 kHz",
                  font=self.font_small, fill=1)
        self.device.display(img)
        time.sleep(2)

    def update(self, result: dict, timestamp: str):
        """
        Render one classification result to the OLED.

        Parameters
        ----------
        result    : dict returned by predict()
        timestamp : pre-formatted time string "HH:MM:SS Www DDMon"
        """
        img, draw = self._new_canvas()

        class_id   = result["class_id"]
        label      = result["label"]
        confidence = result["confidence"]
        probs      = list(result["probabilities"].values())  # [p0, p1, p2]
        is_alert   = class_id in ALERT_CLASSES

        # ── YELLOW ZONE: timestamp ────────────────────────────────────────
        draw.text((2, self.ROW_TIMESTAMP), timestamp,
                  font=self.font_time, fill=1)

        # ── Separator between yellow and blue zone ───────────────────────
        self._draw_separator(draw, self.YELLOW_HEIGHT - 1)

        # ── BLUE ZONE row 1: icon + label ────────────────────────────────
        icon = LABEL_ICONS[class_id]
        draw.text((2, self.ROW_LABEL), f"{icon} {label}",
                  font=self.font_label, fill=1)

        # ── BLUE ZONE row 2: confidence bar ──────────────────────────────
        self._draw_conf_bar(draw, self.ROW_CONF_BAR, confidence)

        # ── BLUE ZONE row 3: per-class probabilities ─────────────────────
        # Short labels: N=Normal, C=Cough, A=Abnormal
        prob_str = (
            f"N\u25b8{probs[0]:.0%} "
            f"C\u25b8{probs[1]:.0%} "
            f"A\u25b8{probs[2]:.0%}"
        )
        draw.text((2, self.ROW_PROBS), prob_str,
                  font=self.font_small, fill=1)

        # ── BLUE ZONE row 4: alert banner ─────────────────────────────────
        if is_alert:
            # Filled black rectangle with inverted text for high visibility
            draw.rectangle(
                [0, self.ROW_ALERT, self.WIDTH - 1, self.HEIGHT - 1],
                fill=1
            )
            alert_text = f"!! {label.upper()} DETECTED !!"
            # Centre the alert text
            try:
                bbox = draw.textbbox((0, 0), alert_text, font=self.font_alert)
                tw   = bbox[2] - bbox[0]
            except AttributeError:
                tw = len(alert_text) * 6   # fallback estimate
            x = max(0, (self.WIDTH - tw) // 2)
            draw.text((x, self.ROW_ALERT + 1), alert_text,
                      font=self.font_alert, fill=0)   # black text on white bg

        self.device.display(img)

    def show_message(self, line1: str = "", line2: str = "",
                     line3: str = "", line4: str = ""):
        """Generic message screen — used for startup/shutdown messages."""
        img, draw = self._new_canvas()
        for i, (y, text) in enumerate(zip(
            [1, 17, 30, 45], [line1, line2, line3, line4]
        )):
            if text:
                draw.text((2, y), text, font=self.font_small, fill=1)
        self.device.display(img)

    def clear(self):
        """Blank the display."""
        self.device.cleanup()


# =============================================================================
# TFLite Interpreter
# =============================================================================

def load_interpreter(model_path: str):
    try:
        import tflite_runtime.interpreter as tflite
        interp = tflite.Interpreter(model_path=model_path)
        print(f"[INFO] Loaded via tflite_runtime: {model_path}")
    except ImportError:
        import tensorflow as tf
        interp = tf.lite.Interpreter(model_path=model_path)
        print(f"[INFO] Loaded via tensorflow: {model_path}")
    interp.allocate_tensors()
    return interp


# =============================================================================
# Inference
# =============================================================================

def predict(interpreter, audio_chunk: np.ndarray, src_sr: int = TARGET_SR) -> dict:
    """
    Run one TFLite inference pass on a 1-second audio chunk.
    audio_chunk may be at any sample rate — make_input_tensor resamples
    to TARGET_SR (with anti-aliasing) using audio_utils.preprocess_signal.
    """
    tensor = make_input_tensor(audio_chunk, src_sr=src_sr)

    inp = interpreter.get_input_details()
    out = interpreter.get_output_details()

    interpreter.set_tensor(inp[0]["index"], tensor)
    interpreter.invoke()

    probs    = interpreter.get_tensor(out[0]["index"])[0]
    class_id = int(np.argmax(probs))

    return {
        "label":         LABELS[class_id],
        "class_id":      class_id,
        "confidence":    float(probs[class_id]),
        "probabilities": {LABELS[i]: float(p) for i, p in enumerate(probs)},
    }


# =============================================================================
# Audio Capture
# =============================================================================

def list_input_devices():
    """Print all audio input devices visible to PortAudio."""
    import sounddevice as sd
    print("\nAvailable input devices:")
    print("─" * 72)
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            print(f"  [{i}] {d['name']}")
            print(f"      channels={d['max_input_channels']}  "
                  f"default_sr={int(d['default_samplerate'])} Hz  "
                  f"hostapi={sd.query_hostapis(d['hostapi'])['name']}")
    print("─" * 72)


def pick_input_device(device=None):
    """
    Choose an input device and return (device_id, native_samplerate).

    Strategy:
      • If `device` is given (int index or name substring), use it.
      • Otherwise use PortAudio's default input device.
      • Sample rate = device's reported default_samplerate, which is
        what ALSA will actually open without complaint.
    """
    import sounddevice as sd

    if device is not None:
        # Allow integer index or substring of device name
        try:
            dev_id = int(device)
        except ValueError:
            dev_id = None
            for i, d in enumerate(sd.query_devices()):
                if d["max_input_channels"] > 0 and device.lower() in d["name"].lower():
                    dev_id = i
                    break
            if dev_id is None:
                raise RuntimeError(f"No input device matches '{device}'. "
                                   f"Run with --list-devices to see options.")
    else:
        dev_id = sd.default.device[0]
        if dev_id is None or dev_id < 0:
            # Fall back to first device with input channels
            for i, d in enumerate(sd.query_devices()):
                if d["max_input_channels"] > 0:
                    dev_id = i
                    break

    info = sd.query_devices(dev_id, "input")
    native_sr = int(info["default_samplerate"])
    print(f"[INFO] Using input device [{dev_id}] '{info['name']}' "
          f"@ native {native_sr} Hz")
    return dev_id, native_sr


def record_blocking(duration_sec: float = 1.0, device=None) -> tuple:
    """
    Blocking single-shot mic capture.
    Returns (audio_float32_mono, samplerate). Sample rate may differ from
    TARGET_SR — preprocess_signal() handles the resampling.
    """
    import sounddevice as sd
    dev_id, native_sr = pick_input_device(device)
    info = sd.query_devices(dev_id, "input")
    n_ch = min(2, int(info["max_input_channels"]))

    # Try TARGET_SR first, fall back to native if hardware refuses
    for rate in (TARGET_SR, native_sr):
        try:
            audio = sd.rec(
                frames=int(duration_sec * rate),
                samplerate=rate,
                channels=n_ch,
                dtype="float32",
                device=dev_id,
            )
            sd.wait()
            # Pick the populated channel (INMP441 fills only one)
            if audio.ndim == 2 and audio.shape[1] >= 2:
                l_peak = float(np.abs(audio[:, 0]).max())
                r_peak = float(np.abs(audio[:, 1]).max())
                audio = audio[:, 0 if l_peak >= r_peak else 1]
            return audio.flatten(), rate
        except Exception as e:
            print(f"[WARN] sd.rec @ {rate} Hz failed: {e}")
    raise RuntimeError("Could not record from any sample rate.")


class StreamingCapture:
    """
    Non-blocking continuous mic capture.

    Opens the device at whatever rate it actually supports (tries TARGET_SR
    first, then the device's native rate). Delivers fixed-duration chunks;
    inference handles resampling via audio_utils.
    """

    def __init__(self, chunk_sec: float = 1.0, device=None):
        import sounddevice as sd
        self._sd        = sd
        self.chunk_sec  = chunk_sec
        self.device     = device
        self.q          = queue.Queue(maxsize=4)
        self._stream    = None
        self.samplerate = None    # set in start()

    def _callback(self, indata, frames, time_info, status):
        if status:
            print(f"[WARN] Audio status: {status}", file=sys.stderr)
        # INMP441 with L/R→GND populates only ONE channel of the I2S frame
        # (left for L/R=GND, right for L/R=VDD). The silent channel has
        # peak ≈ 0, so the louder channel is the real mic. Auto-pick it.
        if indata.shape[1] >= 2:
            l_peak = float(np.abs(indata[:, 0]).max())
            r_peak = float(np.abs(indata[:, 1]).max())
            chunk  = indata[:, 0 if l_peak >= r_peak else 1].copy().astype(np.float32)
        else:
            chunk = indata[:, 0].copy().astype(np.float32)
        try:
            self.q.put_nowait(chunk)
        except queue.Full:
            try:
                self.q.get_nowait()   # drop oldest, keep fresh
            except queue.Empty:
                pass
            self.q.put_nowait(chunk)

    def start(self):
        sd = self._sd
        dev_id, native_sr = pick_input_device(self.device)

        # Determine how many channels the device exposes. I2S devices like
        # the INMP441/googlevoicehat are stereo even when only one mic is
        # wired up — we capture both and let _callback pick the live channel.
        info = sd.query_devices(dev_id, "input")
        n_ch = min(2, int(info["max_input_channels"]))

        # Try preferred (TARGET_SR), then native, then common fallbacks
        candidates = []
        for r in (TARGET_SR, native_sr, 48000, 44100, 32000):
            if r and r not in candidates:
                candidates.append(r)

        last_err = None
        for rate in candidates:
            try:
                self._stream = sd.InputStream(
                    samplerate=rate,
                    channels=n_ch,
                    dtype="float32",
                    blocksize=int(self.chunk_sec * rate),
                    device=dev_id,
                    callback=self._callback,
                )
                self._stream.start()
                self.samplerate = rate
                print(f"[INFO] Audio stream started at {rate} Hz, {n_ch} channels "
                      f"(target = {TARGET_SR} Hz, "
                      f"resampling {'in Python' if rate != TARGET_SR else 'not needed'})")
                return
            except Exception as e:
                last_err = e
                print(f"[WARN] InputStream @ {rate} Hz / {n_ch}ch failed: {e}")

        raise RuntimeError(
            f"Could not open input stream at any of {candidates} Hz.\n"
            f"Last error: {last_err}\n"
            f"Try: arecord -l   (to list devices)\n"
            f"     python3 rpi_inference.py --list-devices\n"
            f"     python3 rpi_inference.py --device <index>"
        )

    def get_chunk(self, timeout: float = 2.0) -> np.ndarray:
        return self.q.get(timeout=timeout)

    def stop(self):
        if self._stream:
            self._stream.stop()
            self._stream.close()


# =============================================================================
# Terminal printing helpers
# =============================================================================

def _print_result(result: dict, ts: str):
    """Print one result row to stdout."""
    probs = result["probabilities"]
    prob_str = "  ".join(
        f"{lbl[0]}:{p:.0%}" for lbl, p in probs.items()
    )
    alert = "  ⚠ ALERT" if result["class_id"] in ALERT_CLASSES else ""
    print(
        f"{ts}  {result['label']:<22}  {result['confidence']:>5.1%}"
        f"  [{prob_str}]{alert}"
    )


def _print_event(ts: str, status: dict, result: dict, peak: float, extra: str = ""):
    """
    Print one row in event-driven mode.
    Shows the full 3-class breakdown (Normal/Cough/Abnormal) plus the
    binary cough-detector state and any envelope-shape diagnostics.
    """
    s = status["status"]
    if s == "EVENT":
        note = f"🚨 COUGH EVENT #{status['count']}"
    elif s == "building":
        note = f"(streak {status['streak']})  {extra}"
    elif s == "cooldown":
        note = f"cooldown  {extra}"
    elif s == "sustained":
        note = extra or "sustained"
    elif s == "silent":
        note = extra
    else:
        note = extra

    if result is None:
        probs_str = "  (model skipped)    "
    else:
        p = result["probabilities"]
        probs_str = (f"N:{p['Normal Breathing']:>4.0%} "
                     f"C:{p['Cough']:>4.0%} "
                     f"A:{p['Abnormal Resp.']:>4.0%}")

    print(f"{ts:<10}  {s:<10}  peak={peak:.3f}  [{probs_str}]  {note}")


# =============================================================================
# Run modes
# =============================================================================

def run_single(interpreter, oled, args):
    """Record once → infer → display → exit."""
    print(f"[INFO] Recording {args.duration}s ...")
    audio, sr = record_blocking(args.duration, device=args.device)
    result = predict(interpreter, audio, src_sr=sr)
    ts     = time.strftime("%H:%M:%S %a %d%b")

    _print_result(result, ts)

    if oled:
        oled.update(result, ts)
        time.sleep(5)   # hold display 5 s before clearing
        oled.clear()


def run_continuous(interpreter, oled, args):
    """Stream audio indefinitely, classify every second."""
    capture = StreamingCapture(chunk_sec=1.0, device=args.device)
    capture.start()

    detector = CoughEventDetector(
        threshold=args.threshold,
        hits=args.hits,
        cooldown=args.cooldown,
    )

    if oled:
        oled.show_message(
            "Cough Monitor",
            "Listening...",
            f"thr={args.threshold:.2f} n={args.hits}",
        )
        time.sleep(1)

    print(f"\n[INFO] Monitoring — Ctrl-C to stop")
    print(f"[INFO] Classes: Normal (0), Cough (1), Abnormal (2)")
    print(f"[INFO] Silence gate    : peak < {SILENCE_GATE:.3f}")
    print(f"[INFO] Sustained gate  : active fraction > {ACTIVE_FRAC_MAX:.2f}")
    print(f"                          OR attack ratio > {ATTACK_RATIO_MAX:.2f}  (not cough-shaped)")
    print(f"[INFO] Cough event     : P(Cough) ≥ {args.threshold:.2f} for "
          f"{args.hits} consecutive frames, cooldown {args.cooldown:.1f}s\n")
    print(f"{'Time':<10}  {'State':<10}  {'Audio':<11}  {'[N / C / A probs]':<24}  Note")
    print("─" * 84)

    try:
        while True:
            t_start = time.time()
            chunk   = capture.get_chunk()
            peak    = float(np.abs(chunk).max())
            ts      = time.strftime("%H:%M:%S")

            # ── Gate 1: silence ──────────────────────────────────────────────
            if peak < SILENCE_GATE:
                status = detector.update(p_cough=0.0, peak=peak, now=t_start)
                _print_event(ts, status, result=None, peak=peak,
                             extra="(silence)")
                if oled:
                    oled.show_message(
                        "Listening...",
                        f"events: {detector.event_count}",
                        f"peak: {peak:.3f}",
                    )
                continue

            # ── Gate 2: envelope shape (reject sustained sounds) ─────────────
            is_cough_shape, active_frac, attack = looks_like_cough_envelope(
                chunk, capture.samplerate
            )
            if not is_cough_shape:
                # Sustained sound — talking, music, fan, etc. Don't run model;
                # don't fire events. Reset streak so partial buildups don't carry over.
                status = detector.update(p_cough=0.0, peak=peak, now=t_start)
                shape_note = f"af={active_frac:.2f} atk={attack:.2f}"
                _print_event(ts, {"status": "sustained", "streak": 0},
                             result=None, peak=peak,
                             extra=f"non-cough shape ({shape_note})")
                if oled:
                    oled.show_message(
                        "Sustained sound",
                        f"events: {detector.event_count}",
                        shape_note,
                    )
                continue

            # ── Gate 3: model classification ─────────────────────────────────
            result  = predict(interpreter, chunk, src_sr=capture.samplerate)
            p_cough = result["probabilities"]["Cough"]

            status = detector.update(p_cough=p_cough, peak=peak, now=t_start)
            _print_event(ts, status, result=result, peak=peak,
                         extra=f"af={active_frac:.2f} atk={attack:.2f}")

            # OLED: show live 3-class label, with event banner overlaid on cough
            if oled:
                if status["status"] == "EVENT":
                    oled.show_message(
                        "*** COUGH ***",
                        f"event #{detector.event_count}",
                        ts,
                    )
                else:
                    # Show what the 3-class model is currently predicting
                    oled.update(result, ts)

            # RPi Zero W performance warning
            elapsed = time.time() - t_start
            if elapsed > 1.1:
                print(f"  [WARN] Inference took {elapsed:.2f}s "
                      f"(>1s chunk — may drift)")

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")
    finally:
        capture.stop()
        if oled:
            oled.show_message(
                "Audio Monitor",
                "Stopped.",
                "",
                "Goodbye!",
            )
            time.sleep(2)
            oled.clear()


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Real-time audio monitor — RPi Zero W + INMP441 + SSD1306 OLED"
    )
    p.add_argument(
        "--model", default="model.tflite",
        help="Path to TFLite model (default: model.tflite)"
    )
    p.add_argument(
        "--mode", choices=["continuous", "single"], default="continuous",
        help="continuous = stream forever | single = one shot"
    )
    p.add_argument(
        "--duration", type=float, default=1.0,
        help="Recording length in seconds for single mode (default: 1.0)"
    )
    p.add_argument(
        "--i2c-address", type=lambda x: int(x, 16), default=0x3C,
        dest="i2c_address",
        help="OLED I2C address in hex (default: 0x3C — try 0x3D if 0x3C fails)"
    )
    p.add_argument(
        "--no-oled", action="store_true",
        help="Disable OLED output (terminal only)"
    )
    p.add_argument(
        "--device", default=None,
        help="Audio input device: integer index or name substring "
             "(default: PortAudio default). Use --list-devices to see options."
    )
    p.add_argument(
        "--list-devices", action="store_true", dest="list_devices",
        help="List available audio input devices and exit."
    )
    # ── Binary cough-event detector tunables ─────────────────────────────────
    p.add_argument(
        "--threshold", type=float, default=COUGH_THRESHOLD,
        help=f"P(cough) threshold for a positive frame "
             f"(default: {COUGH_THRESHOLD}, range 0–1, raise to reduce false alarms)"
    )
    p.add_argument(
        "--hits", type=int, default=CONSECUTIVE_HITS,
        help=f"Consecutive positive frames required to fire an event "
             f"(default: {CONSECUTIVE_HITS}, raise to reduce false alarms)"
    )
    p.add_argument(
        "--cooldown", type=float, default=EVENT_COOLDOWN,
        help=f"Seconds to suppress repeats after a fired event "
             f"(default: {EVENT_COOLDOWN})"
    )
    # ── Envelope-shape gate tunables ─────────────────────────────────────────
    p.add_argument(
        "--active-frac-max", type=float, default=ACTIVE_FRAC_MAX, dest="active_frac_max",
        help=f"Max fraction of window with active energy. Higher = more permissive. "
             f"(default: {ACTIVE_FRAC_MAX}; lower to reject more talking)"
    )
    p.add_argument(
        "--attack-ratio-max", type=float, default=ATTACK_RATIO_MAX, dest="attack_ratio_max",
        help=f"Max pre-peak/peak energy ratio. Higher = more permissive. "
             f"(default: {ATTACK_RATIO_MAX}; lower to require sharper attacks)"
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # ── List devices and exit ────────────────────────────────────────────────
    if args.list_devices:
        list_input_devices()
        sys.exit(0)

    # ── Model ────────────────────────────────────────────────────────────────
    interpreter = load_interpreter(args.model)
    inp_shape   = interpreter.get_input_details()[0]["shape"]
    out_shape   = interpreter.get_output_details()[0]["shape"]

    # Apply CLI overrides for envelope-gate thresholds
    ACTIVE_FRAC_MAX  = args.active_frac_max
    ATTACK_RATIO_MAX = args.attack_ratio_max
    # Push overrides into the module so looks_like_cough_envelope() picks them up
    sys.modules[__name__].ACTIVE_FRAC_MAX  = ACTIVE_FRAC_MAX
    sys.modules[__name__].ATTACK_RATIO_MAX = ATTACK_RATIO_MAX

    print(f"[INFO] Model input  : {inp_shape}  (expect [1, 60, 32, 1])")
    print(f"[INFO] Model output : {out_shape} (expect [1, 3])")
    print(f"[INFO] Target SR    : {TARGET_SR} Hz | Nyquist ceil: {TARGET_SR//2} Hz")
    print(f"[INFO] Chunk        : {CHUNK_SAMPLES} samples = 1 s @ target SR")

    # ── OLED ─────────────────────────────────────────────────────────────────
    oled = None
    if not args.no_oled:
        try:
            oled = OLEDDisplay(i2c_address=args.i2c_address)
        except Exception as e:
            print(f"[WARN] OLED init failed: {e}")
            print("[WARN] Continuing in terminal-only mode.")
            print("[WARN] Verify wiring and run: i2cdetect -y 1")

    # ── Run ───────────────────────────────────────────────────────────────────
    if args.mode == "single":
        run_single(interpreter, oled, args)
    else:
        run_continuous(interpreter, oled, args)
