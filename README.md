# MusiAI — Adaptive Driver-State Music System

## 1. Project overview

MusiAI is a real-time, driver-aware adaptive audio prototype.

The system continuously reads multiple driver/cabin sensors from an Arduino UNO R4 Minima, sends the sensor data to a Python application, estimates the current driver state with a trained machine-learning model, and changes the sonic character of one real MP3 track accordingly.

The system is designed around the following loop:

```text
Physical sensors
      ↓
Arduino UNO R4 Minima
      ↓  USB Serial @ 115200
110-column sensor row
      ↓
Python preprocessing
      ↓
20-sample temporal window
      ↓
480 temporal features
      ↓
Trained driver-state model
      ↓
Calm / Focused / Drowsy / Stressed / High Workload
      ↓
Pyo real-time DSP
      ↓
One continuously playing MP3
      ↓
Laptop / speaker
```

The final prototype also contains a presentation mode. The live AI continues running, while keys `1–5` can temporarily force one of the five music states so the jury can immediately hear the different sound profiles. Press `L` to return control to live AI.

---

# 2. Main goals

The project demonstrates four main ideas:

1. **Multimodal driver sensing**
   - Motion
   - Touch interaction
   - Heart rate
   - Thermal information
   - Joystick / steering interaction
   - Cabin sound and light context

2. **Temporal driver-state estimation**
   - A driver state is not inferred from one sensor sample.
   - The Python application uses a short sequence of sensor measurements.

3. **Adaptive music**
   - The prototype uses one real MP3 song.
   - The song continuously changes its sonic character rather than switching between separate prerecorded tracks.

4. **Immediate demonstration / human interaction**
   - The autonomous system is continuously running.
   - Keyboard shortcuts allow the five states to be demonstrated deterministically during a short presentation.

---

# 3. Final submitted files

## `main.cpp`

Arduino firmware.

Responsibilities:

- Initialize the I2C sensors.
- Read the analog sensors.
- Calculate derived values such as magnitudes and normalized values.
- Produce a stable 110-column CSV row.
- Print sensor validity information.
- Record up to 500 samples at 500 ms intervals.

The firmware uses the following I2C addresses:

| Device | Address |
|---|---:|
| SHT31 | `0x44` |
| Heart-rate sensor | `0x50` |
| MPR121 | `0x5B` |
| AMG8833 | `0x68` |
| LSM6DS3 | `0x6A` |

Analog inputs:

| Pin | Signal |
|---|---|
| A0 | Joystick X |
| A1 | Joystick Y |
| A2 | Light |
| A3 | Loudness |

The firmware configures the LSM6DS3 for a 104 Hz accelerometer and 104 Hz gyroscope configuration, with ±4 g acceleration and ±2000 dps gyro ranges.

The AMG8833 reads all 64 thermal pixels.

## `live_ai_music_modelE_keyboard_intrinsic_demo(1).py`

Python live application.

Responsibilities:

- Load the trained driver-state model.
- Find the predictor inside the `.joblib` file.
- Connect to the Arduino serial stream.
- Validate 110-column rows.
- Maintain a 20-sample temporal window.
- Generate 480 temporal features.
- Predict driver state.
- Apply confidence and state-confirmation filtering.
- Control Pyo playback and DSP.
- Apply real-time sensor-driven effects.
- Provide keyboard demo overrides.

---

# 4. Required runtime files

The submitted Python file expects these files in the same project directory:

```text
MusicAI_Project/
│
├── main.cpp
├── live_ai_music_modelE_keyboard_intrinsic_demo(1).py
├── driver_state_model_driver_centric.joblib
├── song2.mp3
└── README.md
```

The `.joblib` model and `song2.mp3` are runtime dependencies of the Python application.

The Python application looks for:

```python
MODEL_FILE = "driver_state_model_driver_centric.joblib"
SONG_FILE = "song2.mp3"
```

---

# 5. Hardware

## Core controller

- Arduino UNO R4 Minima
- Grove Base Shield
- Grove I2C hub

## Sensors

### SHT31
Measures:

- Temperature
- Humidity

### LSM6DS3
Measures:

- Acceleration X/Y/Z
- Gyroscope X/Y/Z
- Derived acceleration magnitude
- Derived gyro magnitude

### MPR121
Provides:

- 12 capacitive touch channels
- Touch count
- Touch validity flag

### Grove finger-clip heart-rate sensor

Provides:

- Heart-rate value in BPM
- Heart-rate validity

The firmware treats readings below 30 BPM as invalid because the sensor can produce zero/startup values before a valid reading is available.

### AMG8833

Provides:

- 64 individual thermal pixels
- Minimum thermal value
- Maximum thermal value
- Average thermal value
- Thermal range
- Thermal validity

### Joystick

Provides:

- X position
- Y position
- X displacement
- Y displacement
- Magnitude
- Button state

### Analog cabin sensors

- Light: A2
- Loudness: A3

---

# 6. Raw Arduino data format

The Arduino produces exactly **110 columns** in a fixed order.

The schema is:

```text
index
timestamp_ms
temperature
humidity
temperature_valid
humidity_valid

accel_x
accel_y
accel_z
accel_magnitude

gyro_x
gyro_y
gyro_z
gyro_magnitude
imu_valid

touch_0 ... touch_11
touch_count
touch_valid

heart_rate
heart_rate_valid

thermal_0 ... thermal_63
thermal_min
thermal_max
thermal_avg
thermal_range
thermal_valid

joystick_x
joystick_y
joystick_dx
joystick_dy
joystick_magnitude
joystick_button

light
light_normalized
loudness
loudness_normalized
```

The firmware explicitly checks/maintains this 110-column output structure.

The Python application rejects a row unless it contains exactly 110 numeric, finite values.

---

# 7. Sampling and Arduino behavior

The Arduino firmware uses:

```text
TOTAL_SAMPLES = 500
SAMPLE_INTERVAL_MS = 500
```

Therefore the dataset/recording mode produces up to:

```text
500 samples
×
0.5 seconds
=
250 seconds
≈ 4 minutes 10 seconds
```

When the requested number of samples is reached, the firmware prints:

```text
DATASET_END
```

and remains stopped.

---

# 8. Driver-state model pipeline

The live Python application uses the trained **Model E driver-centric model**.

The raw Arduino stream contains 110 columns, but not every field is used as a driver-state model input.

The live code removes 14 metadata/environment/validity fields:

```text
index
timestamp_ms
temperature
humidity
temperature_valid
humidity_valid
light
light_normalized
loudness
loudness_normalized
imu_valid
touch_valid
heart_rate_valid
thermal_valid
```

That leaves:

```text
110 raw columns
       ↓
14 removed
       ↓
96 driver-centric features
```

The idea is to reduce dependence on identifiers, timestamps, validity flags and direct environmental shortcuts.

---

# 9. Temporal feature extraction

The system does not classify a driver from a single sensor row.

The Python application waits for:

```text
20 valid rows
```

before the first prediction.

It then maintains a sliding window of 20 rows and generates five statistics for each of the 96 model features:

```text
mean
standard deviation
minimum
maximum
last value - first value
```

Therefore:

```text
96 features
×
5 statistics
=
480 model features
```

The live predictor checks that the final feature vector contains exactly 480 values.

---

# 10. Prediction cadence

The Python application uses:

```python
WINDOW_SIZE = 20
PREDICT_EVERY = 5
```

Meaning:

- 20 samples are needed initially.
- Once the window is full, prediction is refreshed every 5 new valid sensor rows.
- The window continues sliding over the incoming data.

This creates a balance between:

- temporal context
- responsiveness
- reduced prediction noise

---

# 11. Driver states

The model maps predictions to five states:

```text
0 → calm
1 → focused
2 → drowsy_fatigue
3 → stressed
4 → high_workload
```

The application normalizes either numeric or string predictions into these five state names.

---

# 12. Prediction confidence and state filtering

The live application does not immediately switch the music for every small prediction change.

Two protections are used:

```python
MIN_STATE_CONFIDENCE = 0.40
STATE_CONFIRMATIONS = 2
```

A new state must:

1. Reach at least the minimum confidence.
2. Be predicted consistently for two consecutive confirmations.

This reduces rapid state flickering caused by sensor noise.

---

# 13. Audio architecture

The Python application uses Pyo for continuous playback and real-time DSP.

The MP3 is first converted by FFmpeg to:

```text
Stereo
44.1 kHz
16-bit PCM WAV
```

The temporary WAV is then played through Pyo.

Important architecture:

```text
song2.mp3
   ↓
FFmpeg
   ↓
temporary WAV
   ↓
Pyo SfPlayer
   ↓
Timbre processing
   ↓
Delay
   ↓
Chorus
   ↓
Reverb
   ↓
Spatial output
   ↓
Speaker
```

The source song therefore remains continuous while its sonic character changes.

---

# 14. Intrinsic music adaptation

A major feature of the final version is that state changes are not represented primarily by volume.

The `MusicProfile` contains intrinsic timbre parameters:

```text
warmth
brightness
presence
saturation
```

These are combined with:

```text
reverb
delay
chorus
stereo width
tone softening
playback speed
```

The intrinsic timbre controls are smoothed using Pyo `Port` objects so state changes do not produce abrupt clicks.

---

# 15. Five music profiles

## Calm

Characteristics:

- Warm
- Spacious
- Soft
- Large reverb contribution
- Moderate delay
- Noticeable chorus
- Wide stereo field
- Very low saturation
- Normal speed

Interpretation:

> A relaxed and immersive sonic environment.

---

## Focused

Characteristics:

- Clean
- Relatively bright
- More present
- Almost dry
- No noticeable chorus
- Very little delay
- Moderate stereo
- No saturation
- Normal speed

Interpretation:

> Clear, controlled and less distracting audio.

---

## Drowsy / fatigue

Characteristics:

- 1.10× playback speed
- Warm
- Stronger mid/presence character
- Delay
- Chorus
- More spatial activity
- Mild saturation

Interpretation:

> A more active sound profile designed to make the audio feel more engaging.

The prototype deliberately limits speed adaptation to the drowsy/fatigue state.

---

## Stressed

Characteristics:

- Bright
- More presence
- Noticeable harmonic saturation
- Reverb
- Some delay
- Some chorus
- Narrower stereo field
- Normal speed

Interpretation:

> More energetic and sonically intense.

---

## High workload

Characteristics:

- Dark / warm
- Strong tone softening
- Harmonic saturation
- Very little reverb
- Small delay
- Narrower stereo field
- Normal speed

Interpretation:

> A denser and more controlled sonic character.

These mappings are prototype design choices intended to make the five states audibly distinguishable. They are not intended to claim that a particular frequency response is universally optimal for every driver.

---

# 16. Real-time intrinsic DSP implementation

The final Python application creates separate DSP branches for:

### Warmth

Uses a filtered branch around:

```text
5200 Hz
```

and blends it according to the warmth parameter.

### Brightness

Uses another spectral branch around:

```text
2500 Hz
```

and blends it according to brightness.

### Presence

Uses a stronger mid-frequency branch around:

```text
1800 Hz
```

and blends it according to presence.

### Saturation

Uses Pyo `Disto` to generate additional harmonics.

The amount of saturation is controlled per state.

This creates audible differences in the **tone and texture** of the same song.

---

# 17. Additional audio effects

After the intrinsic timbre stage, the signal can receive:

## Delay

Used to create echo/spatial repetition.

## Chorus

Adds modulation and width.

## Reverb

Uses Freeverb for ambience and space.

## Stereo positioning

Pyo `Pan` controls:

- stereo position
- stereo spread

The joystick can dynamically influence the stereo position.

---

# 18. Immediate sensor-driven audio reactions

The AI state adaptation is relatively deliberate and filtered.

Separate from this, the prototype has fast sensor reactions.

## Gyroscope reaction

A strong gyro magnitude above:

```text
SPIKE_THRESHOLD = 300
```

creates a temporary audio duck.

The driving event then decays over time.

This gives the system a fast reaction to sudden motion.

## Loudness reaction

At startup, loudness is calibrated for:

```text
3 seconds
```

The measured average becomes the baseline.

A sustained sound level significantly above the baseline can trigger audio attenuation.

This allows the system to adapt to the cabin's own sound conditions instead of relying only on one universal threshold.

## Joystick reaction

The joystick is calibrated for:

```text
2 seconds
```

The calibrated center is then used to estimate left/right steering input.

The music's stereo position is moved toward the corresponding side and smoothly released.

---

# 19. Keyboard presentation mode

The final Python program contains a Windows keyboard controller using `msvcrt`.

The shortcut mapping is:

```text
1 → CALM
2 → FOCUSED
3 → DROWSY / FATIGUE
4 → STRESSED
5 → HIGH WORKLOAD
L → LIVE AI CONTROL
Q → QUIT
```

This is only a presentation convenience.

The live AI and Arduino sensor processing continue in the background while a manual demo state is active.

Therefore:

```text
Normal operation
    ↓
Live AI controls music

Press 1–5
    ↓
Selected state controls music
    ↓
Sensor collection + AI prediction continue

Press L
    ↓
Live AI gets control again
```

This makes it possible to demonstrate all five states reliably during a short jury presentation without having to physically reproduce five different driver conditions.

---

# 20. Difference between live AI and keyboard mode

### Live AI mode

The model decides the current state.

```text
Sensors
  ↓
20-sample window
  ↓
480 features
  ↓
Model prediction
  ↓
Confidence filter
  ↓
Music profile
```

### Keyboard demo mode

The keyboard selects the state.

```text
Keyboard
  ↓
Forced state
  ↓
Music profile
```

At the same time, the live sensor/model pipeline keeps operating in the background.

Pressing `L` returns music control to the live AI state.

---

# 21. Why use one MP3 instead of separate songs?

The prototype intentionally uses one continuous source track.

This allows the demonstration to show:

> “The system changes the music experience itself.”

rather than:

> “The system selects a different song.”

This makes the adaptive-audio concept more direct.

---

# 22. Serial communication

The Arduino communicates to the Python application through USB serial:

```text
Baud rate: 115200
```

The Python application:

1. Enumerates available serial ports.
2. Lets the user choose the Arduino.
3. Opens the selected port.
4. Reads lines continuously.
5. Rejects empty/malformed rows.
6. Requires exactly 110 numeric finite values.

This prevents malformed serial lines from entering the model pipeline.

---

# 23. Startup sequence

A typical run follows this sequence:

```text
1. Start Python application
2. Load driver-state model
3. Select Arduino COM port
4. Open serial @ 115200
5. Start Pyo server
6. Decode song2.mp3 through FFmpeg
7. Build the Pyo DSP graph
8. Start continuous MP3 playback
9. Start Arduino reader thread
10. Start keyboard presentation thread
11. Collect initial 20 valid sensor rows
12. Start AI predictions
13. Adapt music continuously
```

At startup, the system also calibrates:

- loudness baseline
- joystick center

---

# 24. Software architecture

The Python application is divided into several functional parts:

```text
Configuration
    ↓
Raw Arduino schema
    ↓
Driver-state definitions
    ↓
Music profiles
    ↓
Pyo control bus
    ↓
MP3/Pyo engine
    ↓
Model loading
    ↓
Temporal feature extraction
    ↓
Prediction
    ↓
State filtering
    ↓
Keyboard override
    ↓
Sensor-effect logic
    ↓
Serial worker
    ↓
Main runtime
```

The serial/model work runs in a separate thread so that Arduino reading and prediction do not become the Pyo audio clock.

Pyo owns the continuous playback stream.

---

# 25. Threading model

The main Python application uses separate responsibilities:

### Main thread

- Starts the application.
- Runs status output.
- Handles shutdown.

### Arduino reader thread

- Reads serial data.
- Updates sensor effects.
- Builds the 20-sample window.
- Runs model inference.
- Applies the resulting music profile.

### Keyboard thread

- Reads `1–5`, `L`, and `Q`.
- Provides deterministic presentation control.

### Pyo audio engine

- Continuously plays the MP3.
- Applies DSP and smoothing in the audio engine.

---

# 26. Model artifact dependency

The Python file dynamically searches inside:

```text
driver_state_model_driver_centric.joblib
```

for an object exposing:

```python
.predict()
```

and, when available:

```python
.predict_proba()
```

This means the estimator itself is stored separately from the live Python source code.

For a complete deployment, the model artifact must therefore be copied beside the Python application.

---

# 27. Required Python packages

The final Python application imports:

```text
joblib
numpy
pandas
pyserial
pyo
```

FFmpeg is also required as an executable available in the system `PATH`.

A typical environment can be prepared with:

```powershell
pip install joblib numpy pandas pyserial pyo
```

Then verify FFmpeg:

```powershell
ffmpeg -version
```

The exact Pyo installation method can depend on the Windows/Python environment.

---

# 28. Arduino build requirements

The firmware is written for Arduino and includes:

```cpp
#include <Arduino.h>
#include <Wire.h>
#include <math.h>
```

Build it with the Arduino/PlatformIO environment configured for the Arduino UNO R4 Minima.

---

# 29. Running the complete system

## Step 1 — Upload Arduino firmware

Open the Arduino project containing `main.cpp`.

Upload it to the UNO R4 Minima.

Do not leave another serial monitor connected to the same COM port when the Python application needs exclusive access.

## Step 2 — Put required runtime files together

The Python project directory should contain:

```text
main.cpp
live_ai_music_modelE_keyboard_intrinsic_demo(1).py
driver_state_model_driver_centric.joblib
song2.mp3
README.md
```

## Step 3 — Start Python

Example on Windows:

```powershell
& C:\Users\<YOUR_USER>\AppData\Local\Microsoft\WindowsApps\python3.11.exe .\live_ai_music_modelE_keyboard_intrinsic_demo(1).py
```

## Step 4 — Select Arduino port

The application lists detected serial ports.

Select the UNO R4 Minima COM port.

## Step 5 — Wait for calibration

The application performs:

```text
Loudness calibration: 3 seconds
Joystick calibration: 2 seconds
```

Leave the joystick centered during joystick calibration and keep the cabin relatively quiet during loudness calibration.

## Step 6 — Wait for the first prediction

The system needs:

```text
20 valid Arduino rows
```

before its first model prediction.

## Step 7 — Demonstrate

Use:

```text
1 = Calm
2 = Focused
3 = Drowsy
4 = Stressed
5 = High workload
L = Live AI
Q = Quit
```

---

# 30. Recommended 2-minute demonstration

A simple presentation sequence:

### 0:00–0:20 — Explain the concept

Say:

> “We sense the driver using multiple physical sensors, estimate a driver state over a temporal window, and adapt one real song in real time.”

### 0:20–0:40 — Show the live system

Start the program and point out:

```text
Arduino rows
AI prediction
confidence
current music state
```

### 0:40–1:30 — Use keyboard demonstration

Press:

```text
1 → Calm
2 → Focused
3 → Drowsy
4 → Stressed
5 → High workload
```

Let the jury hear the differences.

Do not spend time physically triggering eight sensors one by one.

### 1:30–1:50 — Show sensor interaction

Move the joystick and/or create a gyro event.

Explain that those sensor streams can also alter stereo positioning or temporarily attenuate the music.

### 1:50–2:00 — Return to autonomous mode

Press:

```text
L
```

Then say:

> “The keyboard was only for deterministic demonstration. In normal operation, control returns to the live AI.”

---

# 31. What makes the audio adaptation different

The prototype does not rely only on volume.

The state profiles change:

```text
Warmth
Brightness
Mid presence
Harmonic saturation
Reverb
Delay
Chorus
Stereo width
Tone softness
Playback speed
```

The strongest conceptual distinction is:

```text
Same song
     +
different driver state
     =
different sonic character
```

---

# 32. Safety / prototype scope

This is a research/demo prototype and should not be treated as a certified automotive safety system.

The driver-state classes and audio mappings are experimental.

The model should not be interpreted as a medical diagnosis or a definitive measurement of a person's mental state.

Before a production automotive deployment, the system would require substantially more validation, including:

- more participants
- multiple recording sessions per participant
- broader driving conditions
- more rigorous validation against independent test subjects
- latency and fail-safe testing
- automotive-grade hardware and software integration
- human-factors evaluation
- safety and regulatory assessment

---

# 33. Known prototype limitations

## Limited training data

The trained Model E was developed from a small prototype dataset with separate recordings for the five state labels.

High internal validation performance should therefore not be interpreted as guaranteed real-world generalization to every driver.

## Hand-designed audio mapping

The mapping between driver state and musical parameters is an engineered prototype choice.

It demonstrates the concept but is not claimed to be universally optimal.

## Keyboard override

The keyboard shortcuts intentionally bypass autonomous state selection for presentation purposes.

They are not part of an intended production control interface.

## Fixed DSP branches

Some profile parameters are stored for the music profile structure even when the underlying Pyo branch uses fixed DSP settings.

The main audible state differences come from the profile controls that are connected into the actual DSP graph.

---

# 34. Future development

Potential next steps include:

### Better model generalization

Collect:

```text
many drivers
×
many sessions
×
many driving conditions
```

and use subject-independent validation.

### Sensor fusion improvements

Explore:

- temporal deep learning
- sequence models
- calibrated probabilities
- per-sensor reliability weighting
- missing-sensor robustness

### Personalized music mapping

Allow the driver to choose:

```text
more calming
more energetic
more neutral
```

and learn their preferred mappings over time.

### Automotive integration

Move from laptop audio to:

- vehicle infotainment
- automotive DSP
- embedded compute
- production-grade cabin microphones/sensors

### Closed-loop evaluation

Measure whether adaptive audio actually improves:

- driver engagement
- workload
- perceived comfort
- fatigue-related behavior
- distraction

rather than evaluating only the classifier.

---

# 35. Project summary

MusiAI demonstrates a complete real-time multimodal pipeline:

```text
SENSE
  ↓
Arduino sensors
  ↓
110-column real-time stream

UNDERSTAND
  ↓
96 driver-centric inputs
  ↓
20-sample temporal window
  ↓
480 temporal features
  ↓
trained Model E classifier

ADAPT
  ↓
five driver states
  ↓
intrinsic audio profile

OUTPUT
  ↓
continuous real MP3
  ↓
timbre + ambience + spatial adaptation
  ↓
speaker
```

The core concept is:

> **Sense the driver → understand the state → adapt the audio experience.**

The keyboard control is a presentation aid; the intended operating concept is autonomous live sensing and adaptation.
