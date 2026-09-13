from __future__ import annotations

import shutil
import subprocess
import tempfile
import threading
import time
import msvcrt
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import serial
import serial.tools.list_ports
from pyo import (
    Biquadx,
    Chorus,
    Delay,
    Disto,
    Freeverb,
    Interp,
    Mix,
    Pan,
    Port,
    Server,
    SfPlayer,
    Sig,
)

# =============================================================================
# USER SETTINGS
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = str(BASE_DIR / "driver_state_model_driver_centric.joblib")
SONG_FILE = str(BASE_DIR / "song2.mp3")
BAUD_RATE = 115200

WINDOW_SIZE = 20
PREDICT_EVERY = 5

INITIAL_STATE = "focused"
INITIAL_CONFIDENCE = 0.50
STATE_CONFIRMATIONS = 2
MIN_STATE_CONFIDENCE = 0.40

# Audio
SAMPLE_RATE = 44100
AUDIO_BUFFER_SIZE = 1024
AUDIO_SERVER_AMPLITUDE = 0.90
LOOP_SONG = True

# Stronger demo mapping. This changes intrinsic sound character more clearly.
EXTREME_AUDIO_DEMO = False

FFMPEG_BIN = "ffmpeg"
KEEP_DECODED_WAV = False

# =============================================================================
# SENSOR-DRIVEN AUDIO EFFECTS
# =============================================================================

SPIKE_THRESHOLD = 300.0
GYRO_DECAY = 0.93
DRIVING_DUCK_DEPTH = 0.85

GYRO_FLOOR = 10.0
GYRO_CEIL = 150.0
TENSION_ALPHA = 0.30

LOUDNESS_CALIBRATION_SECONDS = 3.0
LOUDNESS_THRESHOLD_RATIO = 1.8
LOUDNESS_SUSTAIN_RISE_SECONDS = 0.4
LOUDNESS_SUSTAIN_FALL_SECONDS = 2.0
LOUDNESS_DUCK_DEPTH = 0.90

STEERING_CALIBRATION_SECONDS = 2.0
LEFT_EXTENT = 231.0
RIGHT_EXTENT = 530.0
TURN_THRESHOLD = 0.15
STEERING_HOLD_SECONDS = 3.5
STEERING_RELEASE_SECONDS = 1.5
PAN_BOOST = 0.60
PAN_CUT = 0.90
STEERING_SMOOTHING = 0.30

# =============================================================================
# RAW 110-COLUMN ARDUINO SCHEMA
# =============================================================================

RAW_COLUMNS = [
    "index", "timestamp_ms", "temperature", "humidity",
    "temperature_valid", "humidity_valid", "accel_x", "accel_y",
    "accel_z", "accel_magnitude", "gyro_x", "gyro_y", "gyro_z",
    "gyro_magnitude", "imu_valid", "touch_0", "touch_1", "touch_2",
    "touch_3", "touch_4", "touch_5", "touch_6", "touch_7", "touch_8",
    "touch_9", "touch_10", "touch_11", "touch_count", "touch_valid",
    "heart_rate", "heart_rate_valid",
]
RAW_COLUMNS += [f"thermal_{i}" for i in range(64)]
RAW_COLUMNS += [
    "thermal_min", "thermal_max", "thermal_avg", "thermal_range",
    "thermal_valid", "joystick_x", "joystick_y", "joystick_dx",
    "joystick_dy", "joystick_magnitude", "joystick_button", "light",
    "light_normalized", "loudness", "loudness_normalized",
]
assert len(RAW_COLUMNS) == 110

DROP_COLUMNS = [
    "index", "timestamp_ms", "temperature", "humidity",
    "temperature_valid", "humidity_valid", "light", "light_normalized",
    "loudness", "loudness_normalized", "imu_valid", "touch_valid",
    "heart_rate_valid", "thermal_valid",
]
MODEL_COLUMNS = [c for c in RAW_COLUMNS if c not in DROP_COLUMNS]
assert len(MODEL_COLUMNS) == 96

# =============================================================================
# DRIVER STATES
# =============================================================================

CALM = "calm"
FOCUSED = "focused"
DROWSY = "drowsy_fatigue"
STRESSED = "stressed"
HIGH_WORKLOAD = "high_workload"

VALID_STATES = {CALM, FOCUSED, DROWSY, STRESSED, HIGH_WORKLOAD}
NUMERIC_STATE_MAP = {
    0: CALM,
    1: FOCUSED,
    2: DROWSY,
    3: STRESSED,
    4: HIGH_WORKLOAD,
}

# Presentation keyboard shortcuts.
# 1 = Calm, 2 = Focused, 3 = Drowsy/Fatigue,
# 4 = Stressed, 5 = High workload.
# L = return to live AI. Q = quit.
KEY_STATE_MAP = {
    "1": CALM,
    "2": FOCUSED,
    "3": DROWSY,
    "4": STRESSED,
    "5": HIGH_WORKLOAD,
}

# =============================================================================
# MUSIC PROFILES
# =============================================================================

@dataclass(frozen=True)
class MusicProfile:
    speed: float
    reverb_mix: float
    reverb_size: float
    reverb_damp: float
    delay_mix: float
    delay_time: float
    delay_feedback: float
    chorus_mix: float
    chorus_depth_ms: float
    stereo_width: float
    tone_softness: float

    # Intrinsic timbre controls
    warmth: float
    brightness: float
    presence: float
    saturation: float


NORMAL_PROFILES: Dict[str, MusicProfile] = {
    CALM: MusicProfile(
        speed=1.00,
        reverb_mix=0.60, reverb_size=0.92, reverb_damp=0.28,
        delay_mix=0.18, delay_time=0.46, delay_feedback=0.30,
        chorus_mix=0.30, chorus_depth_ms=12.0, stereo_width=1.25,
        tone_softness=0.00,
        warmth=0.78, brightness=0.00, presence=0.04, saturation=0.04,
    ),
    FOCUSED: MusicProfile(
        speed=1.00,
        reverb_mix=0.025, reverb_size=0.35, reverb_damp=0.55,
        delay_mix=0.00, delay_time=0.20, delay_feedback=0.05,
        chorus_mix=0.00, chorus_depth_ms=0.0, stereo_width=0.92,
        tone_softness=0.00,
        warmth=0.02, brightness=0.42, presence=0.16, saturation=0.00,
    ),
    DROWSY: MusicProfile(
        speed=1.10,
        reverb_mix=0.18, reverb_size=0.62, reverb_damp=0.42,
        delay_mix=0.42, delay_time=0.20, delay_feedback=0.28,
        chorus_mix=0.22, chorus_depth_ms=9.0, stereo_width=1.18,
        tone_softness=0.00,
        warmth=0.68, brightness=0.00, presence=0.48, saturation=0.14,
    ),
    STRESSED: MusicProfile(
        speed=1.00,
        reverb_mix=0.42, reverb_size=0.68, reverb_damp=0.38,
        delay_mix=0.20, delay_time=0.11, delay_feedback=0.16,
        chorus_mix=0.07, chorus_depth_ms=4.0, stereo_width=0.72,
        tone_softness=0.22,
        warmth=0.00, brightness=0.82, presence=0.30, saturation=0.42,
    ),
    HIGH_WORKLOAD: MusicProfile(
        speed=1.00,
        reverb_mix=0.01, reverb_size=0.25, reverb_damp=0.62,
        delay_mix=0.11, delay_time=0.065, delay_feedback=0.06,
        chorus_mix=0.00, chorus_depth_ms=0.0, stereo_width=0.52,
        tone_softness=0.35,
        warmth=0.88, brightness=0.00, presence=0.02, saturation=0.30,
    ),
}

EXTREME_PROFILES: Dict[str, MusicProfile] = {
    CALM: MusicProfile(
        speed=1.00,
        reverb_mix=0.70, reverb_size=0.97, reverb_damp=0.22,
        delay_mix=0.26, delay_time=0.48, delay_feedback=0.34,
        chorus_mix=0.38, chorus_depth_ms=14.0, stereo_width=1.28,
        tone_softness=0.10,
        warmth=0.85, brightness=0.00, presence=0.05, saturation=0.04,
    ),
    FOCUSED: MusicProfile(
        speed=1.00,
        reverb_mix=0.01, reverb_size=0.20, reverb_damp=0.65,
        delay_mix=0.00, delay_time=0.20, delay_feedback=0.02,
        chorus_mix=0.00, chorus_depth_ms=0.0, stereo_width=0.86,
        tone_softness=0.00,
        warmth=0.00, brightness=0.40, presence=0.18, saturation=0.00,
    ),
    DROWSY: MusicProfile(
        speed=1.10,
        reverb_mix=0.22, reverb_size=0.68, reverb_damp=0.40,
        delay_mix=0.55, delay_time=0.17, delay_feedback=0.34,
        chorus_mix=0.28, chorus_depth_ms=10.0, stereo_width=1.30,
        tone_softness=0.00,
        warmth=0.70, brightness=0.00, presence=0.50, saturation=0.16,
    ),
    STRESSED: MusicProfile(
        speed=1.00,
        reverb_mix=0.46, reverb_size=0.72, reverb_damp=0.32,
        delay_mix=0.28, delay_time=0.10, delay_feedback=0.22,
        chorus_mix=0.05, chorus_depth_ms=3.0, stereo_width=0.62,
        tone_softness=0.32,
        warmth=0.00, brightness=0.80, presence=0.28, saturation=0.45,
    ),
    HIGH_WORKLOAD: MusicProfile(
        speed=1.00,
        reverb_mix=0.00, reverb_size=0.18, reverb_damp=0.70,
        delay_mix=0.14, delay_time=0.06, delay_feedback=0.08,
        chorus_mix=0.00, chorus_depth_ms=0.0, stereo_width=0.48,
        tone_softness=0.48,
        warmth=0.90, brightness=0.00, presence=0.00, saturation=0.28,
    ),
}

PROFILES = EXTREME_PROFILES if EXTREME_AUDIO_DEMO else NORMAL_PROFILES


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


# =============================================================================
# PYO CONTROL BUS
# =============================================================================

class PyoControls:
    def __init__(self):
        initial = PROFILES[INITIAL_STATE]

        self.speed = Sig(initial.speed)
        self.reverb_mix = Sig(initial.reverb_mix)
        self.reverb_size = Sig(initial.reverb_size)
        self.reverb_damp = Sig(initial.reverb_damp)
        self.delay_mix = Sig(initial.delay_mix)
        self.delay_time = Sig(initial.delay_time)
        self.delay_feedback = Sig(initial.delay_feedback)
        self.chorus_mix = Sig(initial.chorus_mix)
        self.chorus_depth_ms = Sig(initial.chorus_depth_ms)
        self.stereo_width = Sig(initial.stereo_width)
        self.tone_softness = Sig(initial.tone_softness)

        # Intrinsic timbre
        self.warmth = Sig(initial.warmth)
        self.brightness = Sig(initial.brightness)
        self.presence = Sig(initial.presence)
        self.saturation = Sig(initial.saturation)

        # Sensor driven
        self.driving_duck = Sig(0.0)
        self.loudness_duck = Sig(0.0)
        self.left_gain = Sig(1.0)
        self.right_gain = Sig(1.0)
        self.master_gain = Sig(0.80)
        self.pan_position = Sig(0.50)
        self.pan_spread = Sig(0.80)

        self.speed_s = Port(self.speed, risetime=1.50, falltime=0.80)
        self.reverb_mix_s = Port(self.reverb_mix, risetime=1.25, falltime=1.25)
        self.reverb_size_s = Port(self.reverb_size, risetime=1.50, falltime=1.50)
        self.reverb_damp_s = Port(self.reverb_damp, risetime=1.50, falltime=1.50)
        self.delay_mix_s = Port(self.delay_mix, risetime=1.25, falltime=1.25)
        self.delay_time_s = Port(self.delay_time, risetime=1.25, falltime=1.25)
        self.delay_feedback_s = Port(self.delay_feedback, risetime=1.25, falltime=1.25)
        self.chorus_mix_s = Port(self.chorus_mix, risetime=1.25, falltime=1.25)
        self.chorus_depth_ms_s = Port(self.chorus_depth_ms, risetime=1.25, falltime=1.25)
        self.stereo_width_s = Port(self.stereo_width, risetime=1.00, falltime=1.00)
        self.tone_softness_s = Port(self.tone_softness, risetime=0.75, falltime=0.75)

        self.warmth_s = Port(self.warmth, risetime=1.0, falltime=1.0)
        self.brightness_s = Port(self.brightness, risetime=1.0, falltime=1.0)
        self.presence_s = Port(self.presence, risetime=1.0, falltime=1.0)
        self.saturation_s = Port(self.saturation, risetime=0.8, falltime=0.8)

        self.driving_duck_s = Port(self.driving_duck, risetime=0.05, falltime=0.25)
        self.loudness_duck_s = Port(self.loudness_duck, risetime=0.10, falltime=1.50)
        self.left_gain_s = Port(self.left_gain, risetime=0.20, falltime=1.20)
        self.right_gain_s = Port(self.right_gain, risetime=0.20, falltime=1.20)
        self.master_gain_s = Port(self.master_gain, risetime=0.10, falltime=1.20)
        self.pan_position_s = Port(self.pan_position, risetime=0.20, falltime=1.20)
        self.pan_spread_s = Port(self.pan_spread, risetime=0.80, falltime=0.80)

    def set_profile(self, profile: MusicProfile) -> None:
        self.speed.value = profile.speed
        self.reverb_mix.value = profile.reverb_mix
        self.reverb_size.value = profile.reverb_size
        self.reverb_damp.value = profile.reverb_damp
        self.delay_mix.value = profile.delay_mix
        self.delay_time.value = profile.delay_time
        self.delay_feedback.value = profile.delay_feedback
        self.chorus_mix.value = profile.chorus_mix
        self.chorus_depth_ms.value = profile.chorus_depth_ms
        self.stereo_width.value = profile.stereo_width
        self.tone_softness.value = profile.tone_softness

        self.warmth.value = profile.warmth
        self.brightness.value = profile.brightness
        self.presence.value = profile.presence
        self.saturation.value = profile.saturation

    def set_sensor_effects(
        self,
        driving_event: float,
        loudness_sustain: float,
        left_gain: float,
        right_gain: float,
        pan_position: float,
        pan_spread: float,
    ) -> None:
        driving_event = clamp(driving_event, 0.0, 1.0)
        loudness_sustain = clamp(loudness_sustain, 0.0, 1.0)
        left_gain = clamp(left_gain, 0.0, 1.6)
        right_gain = clamp(right_gain, 0.0, 1.6)

        master = (
            1.0 - DRIVING_DUCK_DEPTH * driving_event
        ) * (
            1.0 - LOUDNESS_DUCK_DEPTH * loudness_sustain
        )

        self.driving_duck.value = driving_event
        self.loudness_duck.value = loudness_sustain
        self.left_gain.value = left_gain
        self.right_gain.value = right_gain
        self.master_gain.value = clamp(master * 0.80, 0.0, 0.80)
        self.pan_position.value = clamp(pan_position, 0.0, 1.0)
        self.pan_spread.value = clamp(pan_spread, 0.0, 1.0)


# =============================================================================
# REAL MP3 -> TEMPORARY WAV + PYO DSP
# =============================================================================

class PyoSongEngine:
    def __init__(self, song_path: str, server: Server):
        self.song_path = Path(song_path)
        self.server = server
        self.controls = PyoControls()
        self._temp_wav: Optional[Path] = None
        self._build_graph()

    def _decode_mp3_to_wav(self) -> Path:
        if not self.song_path.exists():
            raise FileNotFoundError(
                f"Song not found: {self.song_path.resolve()}"
            )

        if shutil.which(FFMPEG_BIN) is None:
            raise RuntimeError(
                "FFmpeg is required for reliable MP3 -> WAV conversion but was "
                "not found in PATH. Install FFmpeg and verify `ffmpeg -version`."
            )

        temp_dir = Path(tempfile.mkdtemp(prefix="musiai_pyo_"))
        output = temp_dir / "decoded.wav"

        cmd = [
            FFMPEG_BIN, "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(self.song_path),
            "-vn", "-ac", "2", "-ar", str(SAMPLE_RATE),
            "-acodec", "pcm_s16le", str(output),
        ]

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

        if result.returncode != 0:
            error = result.stderr.decode("utf-8", errors="replace")
            raise RuntimeError(f"FFmpeg failed to decode MP3:\n{error}")

        self._temp_wav = output
        print(f"Decoded MP3 for Pyo: {output}")
        return output

    def _build_graph(self) -> None:
        wav_path = self._decode_mp3_to_wav()

        print("Building Pyo audio graph...")

        self.player = SfPlayer(
            str(wav_path),
            speed=self.controls.speed_s,
            loop=LOOP_SONG,
            mul=0.82,
        )

        src_l = self.player[0]
        src_r = self.player[1]

        # ------------------------------------------------------------------
        # 1. INTRINSIC TIMBRE
        # ------------------------------------------------------------------
        # Warm branch: low-pass-like coloration
        warm_l = Biquadx(src_l, freq=5200.0, q=0.8, type=0)
        warm_r = Biquadx(src_r, freq=5200.0, q=0.8, type=0)

        # Bright branch: brighter spectral character
        bright_l = Biquadx(src_l, freq=2500.0, q=0.8, type=1)
        bright_r = Biquadx(src_r, freq=2500.0, q=0.8, type=1)

        # Presence branch: strong mid character
        presence_l = Biquadx(src_l, freq=1800.0, q=0.65, type=2)
        presence_r = Biquadx(src_r, freq=1800.0, q=0.65, type=2)

        timbre_l = Interp(
            src_l, warm_l, interp=self.controls.warmth_s
        )
        timbre_r = Interp(
            src_r, warm_r, interp=self.controls.warmth_s
        )

        timbre_l = Interp(
            timbre_l, bright_l, interp=self.controls.brightness_s
        )
        timbre_r = Interp(
            timbre_r, bright_r, interp=self.controls.brightness_s
        )

        timbre_l = Interp(
            timbre_l, presence_l, interp=self.controls.presence_s
        )
        timbre_r = Interp(
            timbre_r, presence_r, interp=self.controls.presence_s
        )

        # Harmonic saturation makes stressed/workload audibly different.
        sat_l = Disto(
            timbre_l,
            drive=0.60,
            slope=0.65,
        )
        sat_r = Disto(
            timbre_r,
            drive=0.60,
            slope=0.65,
        )

        tone_l = Interp(
            timbre_l, sat_l, interp=self.controls.saturation_s
        )
        tone_r = Interp(
            timbre_r, sat_r, interp=self.controls.saturation_s
        )

        # Additional tone-softening branch from original system.
        low2_l = Biquadx(tone_l, freq=7000.0, q=0.8, type=0)
        low2_r = Biquadx(tone_r, freq=7000.0, q=0.8, type=0)
        tone_l = Interp(
            tone_l, low2_l, interp=self.controls.tone_softness_s
        )
        tone_r = Interp(
            tone_r, low2_r, interp=self.controls.tone_softness_s
        )

        # ------------------------------------------------------------------
        # 2. DELAY / ECHO
        # ------------------------------------------------------------------
        delay_l = Delay(
            tone_l,
            delay=0.20,
            feedback=0.20,
        )
        delay_r = Delay(
            tone_r,
            delay=0.20,
            feedback=0.20,
        )

        delay_mix_l = Interp(
            tone_l, delay_l, interp=self.controls.delay_mix_s
        )
        delay_mix_r = Interp(
            tone_r, delay_r, interp=self.controls.delay_mix_s
        )

        # ------------------------------------------------------------------
        # 3. CHORUS
        # ------------------------------------------------------------------
        chorus_l = Chorus(
            delay_mix_l,
            depth=10.0,
            feedback=0.20,
            bal=1.0,
        )
        chorus_r = Chorus(
            delay_mix_r,
            depth=10.0,
            feedback=0.20,
            bal=1.0,
        )

        chorus_mix_l = Interp(
            delay_mix_l, chorus_l, interp=self.controls.chorus_mix_s
        )
        chorus_mix_r = Interp(
            delay_mix_r, chorus_r, interp=self.controls.chorus_mix_s
        )

        # ------------------------------------------------------------------
        # 4. REVERB
        # ------------------------------------------------------------------
        rev_l = Freeverb(
            chorus_mix_l,
            size=0.80,
            damp=0.40,
            bal=1.0,
        )
        rev_r = Freeverb(
            chorus_mix_r,
            size=0.80,
            damp=0.40,
            bal=1.0,
        )

        final_l = Interp(
            chorus_mix_l, rev_l, interp=self.controls.reverb_mix_s
        )
        final_r = Interp(
            chorus_mix_r, rev_r, interp=self.controls.reverb_mix_s
        )

        # ------------------------------------------------------------------
        # 5. SPATIAL OUTPUT + MASTER GAIN
        # ------------------------------------------------------------------
        mono = Mix([final_l, final_r], voices=1)

        panned = Pan(
            mono,
            outs=2,
            pan=self.controls.pan_position_s,
            spread=self.controls.pan_spread_s,
            mul=self.controls.master_gain_s,
        )

        self.output = panned
        print("Pyo audio graph ready.")

    def start(self) -> None:
        self.output.out()

    def stop(self) -> None:
        try:
            self.output.stop()
        except Exception:
            pass

    def cleanup(self) -> None:
        if self._temp_wav is not None and not KEEP_DECODED_WAV:
            shutil.rmtree(self._temp_wav.parent, ignore_errors=True)


# =============================================================================
# MODEL LOADING
# =============================================================================

def find_predictor(obj, path="model"):
    if hasattr(obj, "predict"):
        return obj, path

    if isinstance(obj, dict):
        preferred = [
            "model", "pipeline", "classifier", "estimator", "best_model", "clf"
        ]
        for key in preferred:
            if key in obj:
                result = obj[key]
                if hasattr(result, "predict"):
                    return result, f"{path}[{key!r}]"

        for key, value in obj.items():
            result = find_predictor(value, f"{path}[{key!r}]")
            if result is not None:
                return result

    elif isinstance(obj, (list, tuple)):
        for i, value in enumerate(obj):
            result = find_predictor(value, f"{path}[{i}]")
            if result is not None:
                return result

    return None


def load_model():
    path = Path(MODEL_FILE)
    if not path.exists():
        raise FileNotFoundError(f"Model not found:\n{path.resolve()}")

    loaded = joblib.load(path)
    result = find_predictor(loaded)

    if result is None:
        raise TypeError("No .predict() estimator found inside joblib.")

    model, location = result

    print(f"Loaded model file: {path.resolve()}")
    print(f"Using predictor: {location}")
    print(f"Predictor type: {type(model).__name__}")

    return model


# =============================================================================
# TEMPORAL FEATURES / PREDICTION
# =============================================================================

def make_temporal_features(window: pd.DataFrame) -> np.ndarray:
    values = (
        window[MODEL_COLUMNS]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=np.float64)
    )

    if len(window) != WINDOW_SIZE:
        raise ValueError(
            f"Expected {WINDOW_SIZE} rows, got {len(window)}."
        )

    if not np.isfinite(values).all():
        raise ValueError("Window contains NaN/inf.")

    features: List[float] = []

    for i in range(values.shape[1]):
        x = values[:, i]
        features.extend([
            float(np.mean(x)),
            float(np.std(x)),
            float(np.min(x)),
            float(np.max(x)),
            float(x[-1] - x[0]),
        ])

    result = np.asarray(features, dtype=np.float64)

    if len(result) != 480:
        raise RuntimeError(
            f"Expected 480 features, got {len(result)}."
        )

    return result


def normalize_state(raw_prediction) -> str:
    if isinstance(raw_prediction, str):
        text = raw_prediction.strip()

        if text in VALID_STATES:
            return text

        try:
            return NUMERIC_STATE_MAP[int(float(text))]
        except (ValueError, KeyError):
            pass

    try:
        return NUMERIC_STATE_MAP[int(raw_prediction)]
    except (TypeError, ValueError, KeyError) as exc:
        raise ValueError(
            f"Unknown model prediction: {raw_prediction!r}"
        ) from exc


def predict_state(
    model,
    feature_vector: np.ndarray,
) -> Tuple[str, float]:
    x = feature_vector.reshape(1, -1)

    raw_prediction = model.predict(x)[0]
    state = normalize_state(raw_prediction)

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(x)[0]
        confidence = float(np.max(probs))
    else:
        confidence = 1.0

    return state, confidence


# =============================================================================
# STATE FILTER
# =============================================================================

class StateFilter:
    def __init__(self):
        self.current_state = INITIAL_STATE
        self.pending_state = None
        self.pending_count = 0

    def update(self, state: str, confidence: float) -> str:
        if confidence < MIN_STATE_CONFIDENCE:
            return self.current_state

        if state == self.current_state:
            self.pending_state = None
            self.pending_count = 0
            return self.current_state

        if state != self.pending_state:
            self.pending_state = state
            self.pending_count = 1
            return self.current_state

        self.pending_count += 1

        if self.pending_count >= STATE_CONFIRMATIONS:
            self.current_state = state
            self.pending_state = None
            self.pending_count = 0

        return self.current_state


# =============================================================================
# PRESENTATION KEYBOARD OVERRIDE
# =============================================================================

def keyboard_worker(
    stop_event: threading.Event,
    state_lock: threading.Lock,
    shared_state: Dict[str, object],
    player: "PyoSongEngine",
    state_filter: "StateFilter",
    sensor_runtime: Dict[str, object],
) -> None:
    """
    Windows presentation shortcuts.

    1 = CALM
    2 = FOCUSED
    3 = DROWSY / FATIGUE
    4 = STRESSED
    5 = HIGH WORKLOAD
    L = return to live AI
    Q = quit

    Sensor acquisition and live Model E predictions continue in the background.
    A forced state only takes priority for music playback until L is pressed.
    """
    print()
    print("[Keyboard] PRESENTATION SHORTCUTS:")
    print("[Keyboard] 1=CALM | 2=FOCUSED | 3=DROWSY | 4=STRESSED | 5=HIGH_WORKLOAD")
    print("[Keyboard] L=LIVE AI | Q=QUIT")

    while not stop_event.is_set():
        try:
            if msvcrt.kbhit():
                key = msvcrt.getwch().lower()

                if key == "q":
                    print("\n[Keyboard] Quit requested.")
                    stop_event.set()
                    break

                if key == "l":
                    with state_lock:
                        shared_state["manual_override"] = None
                        live_state = shared_state["current_state"]
                        live_confidence = shared_state["current_confidence"]

                    sensor_runtime["music_state"] = live_state
                    player.controls.set_profile(PROFILES[live_state])

                    print(
                        f"\n[Keyboard] LIVE AI CONTROL -> "
                        f"{live_state} (confidence={live_confidence:.3f})"
                    )
                    continue

                if key in KEY_STATE_MAP:
                    forced_state = KEY_STATE_MAP[key]
                    profile = PROFILES[forced_state]

                    with state_lock:
                        shared_state["manual_override"] = forced_state
                        shared_state["current_state"] = forced_state
                        shared_state["current_confidence"] = 1.0

                    state_filter.current_state = forced_state
                    state_filter.pending_state = None
                    state_filter.pending_count = 0

                    sensor_runtime["music_state"] = forced_state
                    player.controls.set_profile(profile)

                    print()
                    print("=" * 78)
                    print(f"[KEYBOARD DEMO] FORCED STATE = {forced_state.upper()}")
                    print("=" * 78)
                    print(
                        f"speed={profile.speed:.2f}x | "
                        f"reverb={profile.reverb_mix:.2f} | "
                        f"echo={profile.delay_mix:.2f} | "
                        f"chorus={profile.chorus_mix:.2f}"
                    )
                    print(
                        f"warmth={profile.warmth:.2f} | "
                        f"brightness={profile.brightness:.2f} | "
                        f"presence={profile.presence:.2f} | "
                        f"saturation={profile.saturation:.2f}"
                    )
                    print("Press L to return to live AI.")
                    print("=" * 78)

            time.sleep(0.03)

        except Exception as exc:
            print(f"\n[Keyboard error] {exc!r}")
            time.sleep(0.10)

    print("[Keyboard] Controller stopped.")


# =============================================================================
# SENSOR EFFECT LOGIC
# =============================================================================

def update_sensor_effects(
    controls: PyoControls,
    row: List[float],
    runtime: Dict[str, object],
    now: float,
) -> None:
    row_dict = dict(zip(RAW_COLUMNS, row))

    try:
        gyro = float(row_dict["gyro_magnitude"])
        loudness = float(row_dict["loudness"])
        joystick_x = float(row_dict["joystick_x"])
    except (KeyError, TypeError, ValueError):
        return

    if gyro > SPIKE_THRESHOLD:
        runtime["driving_event"] = 1.0
    else:
        runtime["driving_event"] *= GYRO_DECAY

    norm = np.clip(
        (gyro - GYRO_FLOOR) / max(1e-6, GYRO_CEIL - GYRO_FLOOR),
        0.0,
        1.0,
    )

    runtime["tension"] = (
        TENSION_ALPHA * norm
        + (1.0 - TENSION_ALPHA) * runtime["tension"]
    )

    # Loudness calibration
    if runtime["loudness_baseline"] is None:
        if runtime["loudness_calibration_deadline"] is None:
            runtime["loudness_calibration_deadline"] = (
                now + LOUDNESS_CALIBRATION_SECONDS
            )

            if not runtime["loudness_calibration_printed"]:
                print(
                    f"\n[Sensor] Calibrating loudness for "
                    f"{LOUDNESS_CALIBRATION_SECONDS:.0f}s — stay quiet..."
                )
                runtime["loudness_calibration_printed"] = True

        runtime["loudness_samples"].append(loudness)

        if now >= runtime["loudness_calibration_deadline"]:
            samples = runtime["loudness_samples"]

            runtime["loudness_baseline"] = (
                float(sum(samples) / len(samples))
                if samples
                else loudness
            )

            print(
                f"[Sensor] Loudness baseline: "
                f"{runtime['loudness_baseline']:.1f}"
            )

    else:
        dt = max(now - runtime["prev_sensor_time"], 1e-3)
        threshold = (
            runtime["loudness_baseline"]
            * LOUDNESS_THRESHOLD_RATIO
        )

        if loudness > threshold:
            runtime["loudness_sustain"] = min(
                1.0,
                runtime["loudness_sustain"]
                + dt / LOUDNESS_SUSTAIN_RISE_SECONDS,
            )
        else:
            runtime["loudness_sustain"] = max(
                0.0,
                runtime["loudness_sustain"]
                - dt / LOUDNESS_SUSTAIN_FALL_SECONDS,
            )

    # Joystick calibration
    if runtime["joystick_baseline_x"] is None:
        if runtime["joystick_calibration_deadline"] is None:
            runtime["joystick_calibration_deadline"] = (
                now + STEERING_CALIBRATION_SECONDS
            )

            if not runtime["joystick_calibration_printed"]:
                print(
                    f"[Sensor] Calibrating joystick center for "
                    f"{STEERING_CALIBRATION_SECONDS:.0f}s — "
                    f"leave centered..."
                )
                runtime["joystick_calibration_printed"] = True

        runtime["joystick_samples"].append(joystick_x)

        if now >= runtime["joystick_calibration_deadline"]:
            samples = runtime["joystick_samples"]

            runtime["joystick_baseline_x"] = (
                float(sum(samples) / len(samples))
                if samples
                else joystick_x
            )

            print(
                f"[Sensor] Joystick center: "
                f"{runtime['joystick_baseline_x']:.1f}"
            )

    else:
        dt = max(now - runtime["prev_sensor_time"], 1e-3)
        dx = joystick_x - runtime["joystick_baseline_x"]

        if dx > 0:
            raw_steering = dx / max(1.0, RIGHT_EXTENT)
        else:
            raw_steering = dx / max(1.0, LEFT_EXTENT)

        raw_steering = clamp(raw_steering, -1.0, 1.0)

        runtime["steering"] += (
            raw_steering - runtime["steering"]
        ) * STEERING_SMOOTHING

        steering = runtime["steering"]

        if steering > TURN_THRESHOLD:
            runtime["steering_direction"] = 1
            runtime["steering_hold"] = STEERING_HOLD_SECONDS

        elif steering < -TURN_THRESHOLD:
            runtime["steering_direction"] = -1
            runtime["steering_hold"] = STEERING_HOLD_SECONDS

        else:
            runtime["steering_hold"] = max(
                0.0,
                runtime["steering_hold"] - dt,
            )

        if runtime["steering_hold"] > 0:
            runtime["steering_intensity"] = 1.0
        else:
            runtime["steering_intensity"] = max(
                0.0,
                runtime["steering_intensity"]
                - dt / STEERING_RELEASE_SECONDS,
            )

        intensity = runtime["steering_intensity"]
        direction = runtime["steering_direction"]

        right_active = 1.0 if direction == 1 else 0.0
        left_active = 1.0 if direction == -1 else 0.0

        runtime["right_gain"] = (
            1.0
            + intensity * right_active * PAN_BOOST
            - intensity * left_active * PAN_CUT
        )

        runtime["left_gain"] = (
            1.0
            + intensity * left_active * PAN_BOOST
            - intensity * right_active * PAN_CUT
        )

    runtime["gyro_magnitude"] = gyro
    runtime["loudness"] = loudness
    runtime["joystick_x"] = joystick_x
    runtime["prev_sensor_time"] = now

    profile = PROFILES.get(
        runtime.get("music_state", INITIAL_STATE),
        PROFILES[INITIAL_STATE],
    )

    direction = int(runtime["steering_direction"])
    intensity = float(runtime["steering_intensity"])

    if direction > 0:
        pan_position = 0.50 + 0.42 * intensity
    elif direction < 0:
        pan_position = 0.50 - 0.42 * intensity
    else:
        pan_position = 0.50

    pan_spread = clamp(
        0.25
        + 0.58 * ((profile.stereo_width - 0.45) / 0.85),
        0.0,
        1.0,
    )

    controls.set_sensor_effects(
        driving_event=runtime["driving_event"],
        loudness_sustain=runtime["loudness_sustain"],
        left_gain=runtime["left_gain"],
        right_gain=runtime["right_gain"],
        pan_position=pan_position,
        pan_spread=pan_spread,
    )


def print_sensor_status(runtime: Dict[str, object]) -> None:
    print(
        f"[SENS] gyro={runtime['gyro_magnitude']:.1f} "
        f"event={runtime['driving_event']:.2f} "
        f"tension={runtime['tension']:.2f} | "
        f"loud={runtime['loudness']:.0f} "
        f"sustain={runtime['loudness_sustain']:.2f} | "
        f"steer={runtime['steering']:.2f} "
        f"L={runtime['left_gain']:.2f} "
        f"R={runtime['right_gain']:.2f}"
    )


# =============================================================================
# SERIAL HELPERS
# =============================================================================

def choose_serial_port() -> str:
    ports = list(serial.tools.list_ports.comports())

    if not ports:
        raise RuntimeError(
            "No serial ports found. Connect the Arduino and try again."
        )

    print()
    print("=" * 78)
    print("AVAILABLE SERIAL / ARDUINO PORTS")
    print("=" * 78)

    for i, port in enumerate(ports, start=1):
        print(f"[{i}] {port.device:<14} | {port.description}")

        if port.manufacturer:
            print(f"     Manufacturer: {port.manufacturer}")

        if port.hwid:
            print(f"     HWID:         {port.hwid}")

    print("=" * 78)

    while True:
        answer = input(
            "Select Arduino port by number or device name "
            "(e.g. 1 / COM5): "
        ).strip()

        if answer.isdigit():
            idx = int(answer) - 1

            if 0 <= idx < len(ports):
                selected = ports[idx].device
                print(f"Selected: {selected}")
                return selected

        else:
            for port in ports:
                if port.device.upper() == answer.upper():
                    print(f"Selected: {port.device}")
                    return port.device

        print("Invalid selection. Please choose one of the listed ports.")


def parse_serial_row(line: str) -> Optional[List[float]]:
    line = line.strip()

    if not line:
        return None

    if line.lower().startswith("index,"):
        return None

    parts = line.split(",")

    if len(parts) != 110:
        return None

    values: List[float] = []

    for item in parts:
        try:
            value = float(item)
        except ValueError:
            return None

        if not np.isfinite(value):
            return None

        values.append(value)

    return values


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    print()
    print("=" * 78)
    print("MusiAI — LIVE DRIVER AI -> REAL MP3 -> PYO ADAPTIVE MUSIC")
    print("=" * 78)
    print()
    print(
        "Arduino -> Model E -> driver state -> intrinsic timbre DSP -> speaker"
    )
    print(f"Song: {SONG_FILE}")
    print(f"Extreme audio demo: {EXTREME_AUDIO_DEMO}")
    print()

    model = load_model()
    port = choose_serial_port()

    print()
    print(f"Opening {port} at {BAUD_RATE} baud...")

    ser = serial.Serial(
        port,
        BAUD_RATE,
        timeout=0.20,
    )

    print("Starting Pyo server...")

    pyo_server = Server(
        sr=SAMPLE_RATE,
        buffersize=AUDIO_BUFFER_SIZE,
        duplex=0,
    ).boot()

    pyo_server.amp = AUDIO_SERVER_AMPLITUDE

    player = PyoSongEngine(
        SONG_FILE,
        pyo_server,
    )

    state_filter = StateFilter()
    player.controls.set_profile(
        PROFILES[INITIAL_STATE]
    )

    stop_event = threading.Event()
    state_lock = threading.Lock()

    shared_state = {
        "current_state": INITIAL_STATE,
        "current_confidence": INITIAL_CONFIDENCE,
        "raw_rows": 0,
        "prediction_number": 0,
        "last_message_time": time.monotonic(),
    }

    sensor_runtime: Dict[str, object] = {
        "gyro_magnitude": 0.0,
        "driving_event": 0.0,
        "tension": 0.0,
        "loudness": 0.0,
        "loudness_baseline": None,
        "loudness_samples": [],
        "loudness_calibration_deadline": None,
        "loudness_calibration_printed": False,
        "loudness_sustain": 0.0,
        "joystick_x": None,
        "joystick_baseline_x": None,
        "joystick_samples": [],
        "joystick_calibration_deadline": None,
        "joystick_calibration_printed": False,
        "steering": 0.0,
        "steering_direction": 0,
        "steering_hold": 0.0,
        "steering_intensity": 0.0,
        "left_gain": 1.0,
        "right_gain": 1.0,
        "music_state": INITIAL_STATE,
        "prev_sensor_time": time.monotonic(),
    }

    window_rows: List[List[float]] = []

    def serial_worker() -> None:
        nonlocal window_rows

        print("[Serial] Arduino worker started.")

        while not stop_event.is_set():
            try:
                raw_line = ser.readline()

                if not raw_line:
                    continue

                row = parse_serial_row(
                    raw_line.decode(
                        "utf-8",
                        errors="ignore",
                    )
                )

                if row is None:
                    continue

                now = time.monotonic()

                update_sensor_effects(
                    player.controls,
                    row,
                    sensor_runtime,
                    now,
                )

                with state_lock:
                    shared_state["raw_rows"] += 1
                    shared_state["last_message_time"] = now
                    raw_count = shared_state["raw_rows"]

                window_rows.append(row)

                if len(window_rows) > WINDOW_SIZE:
                    window_rows = window_rows[-WINDOW_SIZE:]

                rows_since_full = raw_count - WINDOW_SIZE

                should_predict = (
                    len(window_rows) == WINDOW_SIZE
                    and rows_since_full >= 0
                    and rows_since_full % PREDICT_EVERY == 0
                )

                if not should_predict:
                    continue

                try:
                    frame = pd.DataFrame(
                        window_rows,
                        columns=RAW_COLUMNS,
                    )

                    features = make_temporal_features(frame)

                    predicted_state, confidence = predict_state(
                        model,
                        features,
                    )

                    applied_state = state_filter.update(
                        predicted_state,
                        confidence,
                    )

                    with state_lock:
                        shared_state["current_state"] = applied_state
                        shared_state["current_confidence"] = confidence
                        shared_state["prediction_number"] += 1
                        pred_no = shared_state["prediction_number"]
                        manual_override = shared_state.get("manual_override")

                    if manual_override is None:
                        sensor_runtime["music_state"] = applied_state
                        player.controls.set_profile(PROFILES[applied_state])
                        playback_state = applied_state
                    else:
                        playback_state = manual_override

                    profile = PROFILES[playback_state]

                    print()
                    print(
                        f"[AI #{pred_no:04d}] "
                        f"predicted={predicted_state:<18} "
                        f"confidence={confidence:.3f}"
                    )

                    print(
                        f"           music_state={playback_state:<18} "
                        f"speed={profile.speed:.2f}x "
                        f"reverb={profile.reverb_mix:.2f} "
                        f"echo={profile.delay_mix:.2f} "
                        f"chorus={profile.chorus_mix:.2f}"
                    )

                    print(
                        f"           warmth={profile.warmth:.2f} "
                        f"brightness={profile.brightness:.2f} "
                        f"presence={profile.presence:.2f} "
                        f"saturation={profile.saturation:.2f}"
                    )

                    print(
                        f"           stereo={profile.stereo_width:.2f} "
                        f"soften={profile.tone_softness:.2f} "
                        f"rows={raw_count}"
                    )

                except Exception as exc:
                    print(
                        f"\n[Prediction error] {exc!r}"
                    )

            except serial.SerialException as exc:
                print(
                    f"\n[Serial error] {exc}"
                )
                stop_event.set()
                break

            except Exception as exc:
                print(
                    f"\n[Serial worker error] {exc!r}"
                )
                time.sleep(0.10)

        print(
            "[Serial] Arduino worker stopped."
        )

    worker = threading.Thread(
        target=serial_worker,
        name="arduino-reader",
        daemon=True,
    )

    keyboard_thread = threading.Thread(
        target=keyboard_worker,
        args=(
            stop_event,
            state_lock,
            shared_state,
            player,
            state_filter,
            sensor_runtime,
        ),
        name="presentation-keyboard",
        daemon=True,
    )

    try:
        print()
        print("Starting Pyo audio graph...")

        pyo_server.start()
        player.start()
        worker.start()
        keyboard_thread.start()

        print()
        print(
            "REAL MP3 is now playing through Pyo."
        )
        print(
            "AI changes intrinsic timbre: warmth, brightness, presence,"
            " saturation, reverb, delay, chorus and stereo."
        )
        print(
            "Sensor effects: gyro duck + loudness duck + steering stereo pan."
        )
        print()
        print("PRESENTATION SHORTCUTS:")
        print("  1 = CALM")
        print("  2 = FOCUSED")
        print("  3 = DROWSY / FATIGUE")
        print("  4 = STRESSED")
        print("  5 = HIGH WORKLOAD")
        print("  L = RETURN TO LIVE AI TRACKING")
        print("  Q = QUIT")
        print()
        print(
            f"Waiting for {WINDOW_SIZE} valid Arduino rows before first AI prediction..."
        )
        print("Live AI continues predicting while a manual demo key is active.")
        print("Press Ctrl+C to stop.")

        last_status_print = time.monotonic()

        while not stop_event.is_set():
            now = time.monotonic()

            if now - last_status_print >= 5.0:
                with state_lock:
                    raw_count = shared_state["raw_rows"]
                    pred_count = shared_state["prediction_number"]
                    state = shared_state["current_state"]
                    confidence = shared_state["current_confidence"]
                    last_message = shared_state["last_message_time"]

                age = now - last_message

                with state_lock:
                    manual_override = shared_state.get("manual_override")

                control_mode = (
                    f"MANUAL:{manual_override}"
                    if manual_override is not None
                    else "LIVE_AI"
                )

                print(
                    f"\n[STATUS] rows={raw_count} "
                    f"predictions={pred_count} "
                    f"music_state={state} "
                    f"confidence={confidence:.3f} "
                    f"control={control_mode} "
                    f"last_valid_row={age:.1f}s ago"
                )

                print_sensor_status(
                    sensor_runtime
                )

                if raw_count == 0:
                    print(
                        "[STATUS] No valid 110-column Arduino rows yet. "
                        "Close PlatformIO Serial Monitor if it is using the port."
                    )

                elif raw_count < WINDOW_SIZE:
                    print(
                        f"[STATUS] Need {WINDOW_SIZE - raw_count} more "
                        "valid rows before the first prediction."
                    )

                last_status_print = now

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\nStopping...")

    finally:
        stop_event.set()

        try:
            worker.join(timeout=2.0)
        except Exception:
            pass

        try:
            player.stop()
        except Exception:
            pass

        try:
            pyo_server.stop()
        except Exception:
            pass

        try:
            pyo_server.shutdown()
        except Exception:
            pass

        try:
            ser.close()
        except Exception:
            pass

        try:
            player.cleanup()
        except Exception:
            pass

        print(
            "MusiAI Pyo real-song player stopped."
        )


if __name__ == "__main__":
    main()
