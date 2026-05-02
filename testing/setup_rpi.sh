#!/bin/bash
# =============================================================================
# Raspberry Pi Zero W — Full Setup
# Devices: INMP441 I2S microphone  +  SSD1306 0.96" I2C OLED
# Run once as: sudo bash setup_rpi.sh
# =============================================================================

set -e

# ─────────────────────────────────────────────────────────────────────────────
# Wiring Reference
# ─────────────────────────────────────────────────────────────────────────────
#
#  INMP441 Microphone → RPi Zero W GPIO
#  ─────────────────────────────────────
#  INMP441 VDD  →  Pin 1   (3.3 V)
#  INMP441 GND  →  Pin 6   (GND)
#  INMP441 SCK  →  Pin 12  (GPIO 18, PCM_CLK / I2S BCLK)
#  INMP441 WS   →  Pin 35  (GPIO 19, PCM_FS  / I2S LRCLK)
#  INMP441 SD   →  Pin 38  (GPIO 20, PCM_DIN / I2S DATA)
#  INMP441 L/R  →  Pin 6   (GND → selects LEFT channel)
#
#  SSD1306 OLED (I2C) → RPi Zero W GPIO
#  ──────────────────────────────────────
#  OLED VCC  →  Pin 1   (3.3 V)   ← supports 3.3V–5V; use 3.3V on RPi
#  OLED GND  →  Pin 6   (GND)
#  OLED SDA  →  Pin 3   (GPIO 2,  I2C1 SDA)
#  OLED SCL  →  Pin 5   (GPIO 3,  I2C1 SCL)
#
#  NOTE: I2C and I2S use completely different GPIO pins — no conflict.
#  Default I2C address for SSD1306 is 0x3C (some boards use 0x3D).
# ─────────────────────────────────────────────────────────────────────────────

echo "========================================================"
echo "  RPi Zero W — Audio Monitor + OLED Setup"
echo "========================================================"

# ─── Step 1: Find the right config.txt ───────────────────────────────────────
# Raspberry Pi OS Bookworm (2023+) moved this file to /boot/firmware/config.txt.
# Older releases keep it at /boot/config.txt. We must write to whichever one
# the firmware is actually reading, otherwise our changes are silently ignored.
echo ""
echo "=== Step 1: Locate config.txt ==="

if   [ -f /boot/firmware/config.txt ]; then
    CONFIG_TXT=/boot/firmware/config.txt
elif [ -f /boot/config.txt ]; then
    CONFIG_TXT=/boot/config.txt
else
    echo "  ERROR: Cannot find config.txt at /boot/firmware/config.txt or /boot/config.txt."
    exit 1
fi
echo "  Using $CONFIG_TXT"

# ─── Step 2: /boot[/firmware]/config.txt ─────────────────────────────────────
echo ""
echo "=== Step 2: Configure $CONFIG_TXT ==="

add_if_missing() {
    grep -qF "$1" "$CONFIG_TXT" || echo "$1" >> "$CONFIG_TXT"
}

# I2S for INMP441
add_if_missing "dtparam=i2s=on"
add_if_missing "dtoverlay=i2s-mmap"
add_if_missing "dtoverlay=googlevoicehat-soundcard"

# I2C for SSD1306 OLED
add_if_missing "dtparam=i2c_arm=on"
add_if_missing "dtparam=i2c_arm_baudrate=400000"   # 400 kHz fast-mode for smooth OLED refresh

echo "  $CONFIG_TXT updated."

# Also enable I2C the official way so /dev/i2c-1 is created on boot
if command -v raspi-config >/dev/null 2>&1; then
    raspi-config nonint do_i2c 0 || true   # 0 = enable
    echo "  I2C enabled via raspi-config."
fi

# ─── Step 3: Kernel modules ──────────────────────────────────────────────────
echo ""
echo "=== Step 3: Kernel modules ==="

add_module() {
    grep -qF "$1" /etc/modules || echo "$1" >> /etc/modules
}

add_module "snd-bcm2835"
add_module "i2c-dev"
# NOTE: i2c-bcm2708 was removed in modern kernels — replaced by i2c-bcm2835
# which loads automatically via the dtparam=i2c_arm=on overlay. Don't add it.

modprobe snd-bcm2835   2>/dev/null || true
modprobe i2c-dev       2>/dev/null || true

echo "  Modules loaded."

# ─── Step 4: ALSA config ─────────────────────────────────────────────────────
echo ""
echo "=== Step 4: ALSA config (mic capture) ==="

cat > /etc/asound.conf << 'EOF'
# Raw INMP441 hardware interface
pcm.inmp441 {
    type hw
    card 0
    device 0
}

# Plug layer: converts stereo→mono, resamples to 16000 Hz at driver level.
# Note: PortAudio (used by sounddevice) bypasses this for raw hw access,
# so the Python script also handles resampling itself as a fallback.
# Useful for arecord/aplay testing.
pcm.mic_16k {
    type plug
    slave {
        pcm      "inmp441"
        rate     16000
        channels 1
        format   S32_LE    # INMP441 outputs 24-bit in 32-bit frames
    }
}

pcm.!default {
    type asym
    capture.pcm "mic_16k"
}
EOF

echo "  /etc/asound.conf written."

# ─── Step 5: System packages ─────────────────────────────────────────────────
echo ""
echo "=== Step 5: System packages ==="

apt-get update -q
apt-get install -y -q \
    python3-pip \
    python3-numpy \
    python3-scipy \
    python3-smbus \
    portaudio19-dev \
    libatlas-base-dev \
    libjpeg-dev \
    libfreetype6-dev \
    i2c-tools \
    fonts-dejavu-core    # DejaVu fonts for OLED text rendering

echo "  System packages installed."

# ─── Step 6: Python packages ─────────────────────────────────────────────────
echo ""
echo "=== Step 6: Python packages ==="

# audio_utils.py imports: numpy, scipy.signal, python_speech_features
# rpi_inference.py imports: sounddevice, tflite_runtime, luma.oled, PIL
# librosa is NOT required and was removed (heavy + needs numba on Pi Zero W).
pip3 install --upgrade pip --quiet
pip3 install \
    sounddevice \
    python_speech_features \
    tflite-runtime \
    luma.oled \
    Pillow

echo "  Python packages installed."

# ─── Step 7: Add user to i2c group ──────────────────────────────────────────
echo ""
echo "=== Step 7: I2C permissions ==="
# Use $SUDO_USER if available (real user who ran sudo), else fall back
TARGET_USER="${SUDO_USER:-pi}"
usermod -aG i2c "$TARGET_USER" 2>/dev/null || true
echo "  User '$TARGET_USER' added to i2c group (takes effect after reboot)."

# ─── Step 8: Verify devices ──────────────────────────────────────────────────
echo ""
echo "=== Step 8: Pre-reboot device check ==="

echo "  Microphone devices:"
arecord -l 2>/dev/null || echo "  WARNING: No capture device found — normal before reboot."

echo ""
echo "  I2C bus scan (may be empty before reboot):"
i2cdetect -y 1 2>/dev/null || echo "  I2C not ready yet — reboot required."

# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "========================================================"
echo "  Setup complete — REBOOT NOW:"
echo "    sudo reboot"
echo ""
echo "  After reboot, verify with:"
echo "    # Microphone"
echo "    arecord -D mic_16k -r 16000 -c 1 -f S32_LE -d 3 test.wav"
echo "    aplay test.wav"
echo ""
echo "    # OLED (should show 0x3C or 0x3D)"
echo "    i2cdetect -y 1"
echo ""
echo "  Then run:"
echo "    python3 rpi_inference.py --model model.tflite --mode continuous"
echo "    # If audio fails: python3 rpi_inference.py --list-devices"
echo "========================================================"