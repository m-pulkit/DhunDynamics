#include <Arduino.h>
#include <Wire.h>
#include <math.h>

// ============================================================
// I2C ADDRESSES
// ============================================================

constexpr uint8_t ADDR_SHT31    = 0x44;
constexpr uint8_t ADDR_HR       = 0x50;
constexpr uint8_t ADDR_MPR121   = 0x5B;
constexpr uint8_t ADDR_AMG8833  = 0x68;
constexpr uint8_t ADDR_LSM6DS3  = 0x6A;

// ============================================================
// ANALOG INPUTS
// ============================================================

constexpr int PIN_JOYSTICK_X = A0;
constexpr int PIN_JOYSTICK_Y = A1;

constexpr int PIN_LIGHT      = A2;
constexpr int PIN_LOUDNESS   = A3;

// ============================================================
// DATASET SETTINGS
// ============================================================

constexpr int TOTAL_SAMPLES = 500;
constexpr unsigned long SAMPLE_INTERVAL_MS = 500;

int datasetCount = 0;

// ============================================================
// JOYSTICK REFERENCE VALUES
// Seeed documentation values
// ============================================================

constexpr float JOY_X_CENTER = 516.0f;
constexpr float JOY_Y_CENTER = 507.0f;

constexpr float JOY_X_MIN = 206.0f;
constexpr float JOY_X_MAX = 798.0f;

constexpr float JOY_Y_MIN = 203.0f;
constexpr float JOY_Y_MAX = 797.0f;

// ============================================================
// HELPERS
// ============================================================

bool i2cPresent(uint8_t address)
{
    Wire.beginTransmission(address);
    return Wire.endTransmission() == 0;
}

bool writeRegister8(
    uint8_t address,
    uint8_t reg,
    uint8_t value
)
{
    Wire.beginTransmission(address);
    Wire.write(reg);
    Wire.write(value);

    return Wire.endTransmission() == 0;
}

bool readRegister8(
    uint8_t address,
    uint8_t reg,
    uint8_t &value
)
{
    Wire.beginTransmission(address);
    Wire.write(reg);

    if (Wire.endTransmission(false) != 0)
        return false;

    if (Wire.requestFrom((int)address, 1) != 1)
        return false;

    value = Wire.read();

    return true;
}

float clamp01(float x)
{
    if (x < 0.0f)
        return 0.0f;

    if (x > 1.0f)
        return 1.0f;

    return x;
}

float magnitude3(
    float x,
    float y,
    float z
)
{
    return sqrtf(
        x * x +
        y * y +
        z * z
    );
}

// ============================================================
// SHT31
// ============================================================

bool readSHT31(
    float &temperatureC,
    float &humidity
)
{
    Wire.beginTransmission(ADDR_SHT31);

    Wire.write(0x24);
    Wire.write(0x00);

    if (Wire.endTransmission() != 0)
        return false;

    delay(20);

    if (Wire.requestFrom((int)ADDR_SHT31, 6) != 6)
        return false;

    uint16_t rawTemp =
        ((uint16_t)Wire.read() << 8) |
        Wire.read();

    uint8_t tempCRC = Wire.read();

    uint16_t rawHum =
        ((uint16_t)Wire.read() << 8) |
        Wire.read();

    uint8_t humCRC = Wire.read();

    (void)tempCRC;
    (void)humCRC;

    temperatureC =
        -45.0f +
        175.0f *
        ((float)rawTemp / 65535.0f);

    humidity =
        100.0f *
        ((float)rawHum / 65535.0f);

    return true;
}

// ============================================================
// LSM6DS3
// ============================================================

bool initLSM6DS3()
{
    uint8_t whoAmI = 0;

    if (!readRegister8(
            ADDR_LSM6DS3,
            0x0F,
            whoAmI))
    {
        return false;
    }

    if (whoAmI != 0x6A &&
        whoAmI != 0x69)
    {
        return false;
    }

    // Accelerometer: 104 Hz, +/-4g
    // Gyroscope:     104 Hz, +/-2000 dps

    writeRegister8(
        ADDR_LSM6DS3,
        0x11,
        0x4C
    );

    writeRegister8(
        ADDR_LSM6DS3,
        0x10,
        0x4A
    );

    writeRegister8(
        ADDR_LSM6DS3,
        0x16,
        0x00
    );

    writeRegister8(
        ADDR_LSM6DS3,
        0x17,
        0x09
    );

    return true;
}

bool readLSM6DS3(
    float &ax,
    float &ay,
    float &az,
    float &gx,
    float &gy,
    float &gz
)
{
    // -------------------------
    // Accelerometer
    // -------------------------

    Wire.beginTransmission(ADDR_LSM6DS3);
    Wire.write(0x28);

    if (Wire.endTransmission(false) != 0)
        return false;

    if (Wire.requestFrom(
            (int)ADDR_LSM6DS3,
            6
        ) != 6)
    {
        return false;
    }

    int16_t rawAX =
        (int16_t)(
            Wire.read() |
            (Wire.read() << 8)
        );

    int16_t rawAY =
        (int16_t)(
            Wire.read() |
            (Wire.read() << 8)
        );

    int16_t rawAZ =
        (int16_t)(
            Wire.read() |
            (Wire.read() << 8)
        );

    ax = rawAX * 4.0f / 32768.0f;
    ay = rawAY * 4.0f / 32768.0f;
    az = rawAZ * 4.0f / 32768.0f;

    // -------------------------
    // Gyroscope
    // -------------------------

    Wire.beginTransmission(ADDR_LSM6DS3);
    Wire.write(0x22);

    if (Wire.endTransmission(false) != 0)
        return false;

    if (Wire.requestFrom(
            (int)ADDR_LSM6DS3,
            6
        ) != 6)
    {
        return false;
    }

    int16_t rawGX =
        (int16_t)(
            Wire.read() |
            (Wire.read() << 8)
        );

    int16_t rawGY =
        (int16_t)(
            Wire.read() |
            (Wire.read() << 8)
        );

    int16_t rawGZ =
        (int16_t)(
            Wire.read() |
            (Wire.read() << 8)
        );

    gx = rawGX * 2000.0f / 32768.0f;
    gy = rawGY * 2000.0f / 32768.0f;
    gz = rawGZ * 2000.0f / 32768.0f;

    return true;
}

// ============================================================
// MPR121
// ============================================================

bool initMPR121()
{
    if (!i2cPresent(ADDR_MPR121))
        return false;

    // Stop sensing
    if (!writeRegister8(
            ADDR_MPR121,
            0x5E,
            0x00))
    {
        return false;
    }

    delay(50);

    writeRegister8(
        ADDR_MPR121,
        0x5C,
        0x10
    );

    writeRegister8(
        ADDR_MPR121,
        0x5D,
        0x23
    );

    writeRegister8(
        ADDR_MPR121,
        0x5B,
        0x22
    );

    // Enable electrodes 0-11
    writeRegister8(
        ADDR_MPR121,
        0x5E,
        0x3C
    );

    delay(100);

    return true;
}

bool readMPR121(
    uint16_t &touchMask
)
{
    Wire.beginTransmission(ADDR_MPR121);
    Wire.write(0x00);

    if (Wire.endTransmission(false) != 0)
        return false;

    if (Wire.requestFrom(
            (int)ADDR_MPR121,
            2
        ) != 2)
    {
        return false;
    }

    uint8_t low = Wire.read();
    uint8_t high = Wire.read();

    touchMask =
        ((uint16_t)high << 8) |
        low;

    touchMask &= 0x0FFF;

    return true;
}

// ============================================================
// HEART RATE
// ============================================================

bool readHeartRate(
    int &bpm
)
{
    Wire.requestFrom(
        (int)ADDR_HR,
        1
    );

    if (Wire.available() != 1)
        return false;

    bpm = Wire.read();

    return true;
}

// ============================================================
// AMG8833
// Read ALL 64 pixels
// ============================================================

bool initAMG8833()
{
    if (!i2cPresent(ADDR_AMG8833))
        return false;

    // Normal operating mode
    writeRegister8(
        ADDR_AMG8833,
        0x00,
        0x00
    );

    // Software reset
    writeRegister8(
        ADDR_AMG8833,
        0x01,
        0x3F
    );

    delay(100);

    // 10 FPS
    writeRegister8(
        ADDR_AMG8833,
        0x02,
        0x00
    );

    return true;
}

bool readAMG8833(
    float thermal[64],
    float &minTemp,
    float &maxTemp,
    float &averageTemp
)
{
    float sum = 0.0f;

    minTemp = 1000.0f;
    maxTemp = -1000.0f;

    for (
        int startPixel = 0;
        startPixel < 64;
        startPixel += 8
    )
    {
        uint8_t reg =
            0x80 +
            (startPixel * 2);

        Wire.beginTransmission(
            ADDR_AMG8833
        );

        Wire.write(reg);

        if (Wire.endTransmission(false) != 0)
            return false;

        if (Wire.requestFrom(
                (int)ADDR_AMG8833,
                16
            ) != 16)
        {
            return false;
        }

        for (int i = 0; i < 8; i++)
        {
            uint8_t lowByte = Wire.read();
            uint8_t highByte = Wire.read();

            int16_t raw =
                (int16_t)(
                    lowByte |
                    ((highByte & 0x0F) << 8)
                );

            if (raw & 0x0800)
                raw |= 0xF000;

            float tempC =
                raw * 0.25f;

            int index =
                startPixel + i;

            thermal[index] = tempC;

            sum += tempC;

            if (tempC < minTemp)
                minTemp = tempC;

            if (tempC > maxTemp)
                maxTemp = tempC;
        }
    }

    averageTemp =
        sum / 64.0f;

    return true;
}

// ============================================================
// JOYSTICK
// ============================================================

void readJoystick(
    int &x,
    int &y,
    bool &buttonPressed
)
{
    x = analogRead(PIN_JOYSTICK_X);
    y = analogRead(PIN_JOYSTICK_Y);

    // Button is multiplexed through X
    buttonPressed = (x > 950);
}

// ============================================================
// CSV HEADER
// ============================================================

void printCSVHeader()
{
    Serial.print("index,timestamp_ms,");

    // SHT31
    Serial.print("temperature,");
    Serial.print("humidity,");
    Serial.print("temperature_valid,");
    Serial.print("humidity_valid,");

    // IMU
    Serial.print("accel_x,");
    Serial.print("accel_y,");
    Serial.print("accel_z,");
    Serial.print("accel_magnitude,");

    Serial.print("gyro_x,");
    Serial.print("gyro_y,");
    Serial.print("gyro_z,");
    Serial.print("gyro_magnitude,");
    Serial.print("imu_valid,");

    // MPR121
    for (int i = 0; i < 12; i++)
    {
        Serial.print("touch_");
        Serial.print(i);
        Serial.print(",");
    }

    Serial.print("touch_count,");
    Serial.print("touch_valid,");

    // Heart rate
    Serial.print("heart_rate,");
    Serial.print("heart_rate_valid,");

    // AMG8833
    for (int i = 0; i < 64; i++)
    {
        Serial.print("thermal_");
        Serial.print(i);
        Serial.print(",");
    }

    Serial.print("thermal_min,");
    Serial.print("thermal_max,");
    Serial.print("thermal_avg,");
    Serial.print("thermal_range,");
    Serial.print("thermal_valid,");

    // Joystick
    Serial.print("joystick_x,");
    Serial.print("joystick_y,");
    Serial.print("joystick_dx,");
    Serial.print("joystick_dy,");
    Serial.print("joystick_magnitude,");
    Serial.print("joystick_button,");

    // Analog
    Serial.print("light,");
    Serial.print("light_normalized,");
    Serial.print("loudness,");
    Serial.println("loudness_normalized");
}

// ============================================================
// SETUP
// ============================================================

void setup()
{
    Serial.begin(115200);

    unsigned long start = millis();

    while (
        !Serial &&
        millis() - start < 3000
    )
    {
        delay(10);
    }

    analogReadResolution(10);

    Wire.begin();
    Wire.setClock(100000);

    // ========================================================
    // DEVICE CHECK
    // ========================================================

    Serial.println(
        "DATASET_START"
    );

    Serial.println(
        "# I2C DEVICE CHECK"
    );

    Serial.print("# SHT31=");
    Serial.println(
        i2cPresent(ADDR_SHT31)
        ? "FOUND"
        : "NOT_FOUND"
    );

    Serial.print("# HEART_RATE=");
    Serial.println(
        i2cPresent(ADDR_HR)
        ? "FOUND"
        : "NOT_FOUND"
    );

    Serial.print("# MPR121=");
    Serial.println(
        i2cPresent(ADDR_MPR121)
        ? "FOUND"
        : "NOT_FOUND"
    );

    Serial.print("# AMG8833=");
    Serial.println(
        i2cPresent(ADDR_AMG8833)
        ? "FOUND"
        : "NOT_FOUND"
    );

    Serial.print("# LSM6DS3=");
    Serial.println(
        i2cPresent(ADDR_LSM6DS3)
        ? "FOUND"
        : "NOT_FOUND"
    );

    // ========================================================
    // SENSOR INITIALIZATION
    // ========================================================

    Serial.print("# LSM6DS3_INIT=");
    Serial.println(
        initLSM6DS3()
        ? "OK"
        : "FAILED"
    );

    Serial.print("# MPR121_INIT=");
    Serial.println(
        initMPR121()
        ? "OK"
        : "FAILED"
    );

    Serial.print("# AMG8833_INIT=");
    Serial.println(
        initAMG8833()
        ? "OK"
        : "FAILED"
    );

    Serial.println(
        "# CONFIG=A0_JOY_X,A1_JOY_Y,A2_LIGHT,A3_LOUDNESS"
    );

    Serial.println(
        "# I2C=SHT31,LSM6DS3,MPR121,HEART_RATE,AMG8833"
    );

    // ========================================================
    // CSV HEADER
    // ========================================================

    printCSVHeader();
}

// ============================================================
// LOOP
// ============================================================

void loop()
{
    // --------------------------------------------------------
    // Stop after requested amount
    // --------------------------------------------------------

    if (datasetCount >= TOTAL_SAMPLES)
    {
        Serial.println(
            "DATASET_END"
        );

        while (true)
        {
            delay(1000);
        }
    }

    // ========================================================
    // VARIABLES
    // ========================================================

    float temperature = NAN;
    float humidity = NAN;

    float ax = NAN;
    float ay = NAN;
    float az = NAN;

    float gx = NAN;
    float gy = NAN;
    float gz = NAN;

    uint16_t touchMask = 0;

    int heartRate = 0;

    float thermal[64];

    for (int i = 0; i < 64; i++)
        thermal[i] = NAN;

    float thermalMin = NAN;
    float thermalMax = NAN;
    float thermalAverage = NAN;

    int joystickX = 0;
    int joystickY = 0;

    bool joystickButton = false;

    int lightValue = 0;
    int loudnessValue = 0;

    // ========================================================
    // SENSOR VALID FLAGS
    // ========================================================

    bool shtOK =
        readSHT31(
            temperature,
            humidity
        );

    bool imuOK =
        readLSM6DS3(
            ax,
            ay,
            az,
            gx,
            gy,
            gz
        );

    bool touchOK =
        readMPR121(
            touchMask
        );

    bool heartRateOK =
        readHeartRate(
            heartRate
        );

    bool thermalOK =
        readAMG8833(
            thermal,
            thermalMin,
            thermalMax,
            thermalAverage
        );

    // ========================================================
    // JOYSTICK
    // ========================================================

    readJoystick(
        joystickX,
        joystickY,
        joystickButton
    );

    // ========================================================
    // ANALOG SENSORS
    // ========================================================

    lightValue =
        analogRead(PIN_LIGHT);

    loudnessValue =
        analogRead(PIN_LOUDNESS);

    // ========================================================
    // DERIVED FEATURES
    // ========================================================

    float accelMagnitude =
        magnitude3(
            ax,
            ay,
            az
        );

    float gyroMagnitude =
        magnitude3(
            gx,
            gy,
            gz
        );

    float joystickDX =
        (float)joystickX -
        JOY_X_CENTER;

    float joystickDY =
        (float)joystickY -
        JOY_Y_CENTER;

    float joystickMagnitude =
        sqrtf(
            joystickDX * joystickDX +
            joystickDY * joystickDY
        );

    float lightNormalized =
        ((float)lightValue / 1023.0f);

    float loudnessNormalized =
        ((float)loudnessValue / 1023.0f);

    // ========================================================
    // MPR121 -> individual electrode states
    // ========================================================

    int touchCount = 0;

    bool touchStates[12];

    for (int i = 0; i < 12; i++)
    {
        touchStates[i] =
            (touchMask & (1 << i)) != 0;

        if (touchStates[i])
            touchCount++;
    }

    // ========================================================
    // HEART RATE VALIDITY
    //
    // The sensor may output 0 during startup.
    // The supplied kit documentation notes that valid readings
    // can take about a minute.
    // ========================================================

    bool heartRateValid =
        heartRateOK &&
        heartRate >= 30;

    // ========================================================
    // SAMPLE NUMBER
    // ========================================================

    datasetCount++;

    unsigned long timestamp =
        millis();

    // ========================================================
    // CSV
    // ========================================================

    // Basic
    Serial.print(datasetCount);
    Serial.print(",");

    Serial.print(timestamp);
    Serial.print(",");

    // --------------------------------------------------------
    // SHT31
    // --------------------------------------------------------

    Serial.print(temperature, 2);
    Serial.print(",");

    Serial.print(humidity, 2);
    Serial.print(",");

    Serial.print(shtOK ? 1 : 0);
    Serial.print(",");

    Serial.print(shtOK ? 1 : 0);
    Serial.print(",");

    // --------------------------------------------------------
    // ACCEL
    // --------------------------------------------------------

    Serial.print(ax, 4);
    Serial.print(",");

    Serial.print(ay, 4);
    Serial.print(",");

    Serial.print(az, 4);
    Serial.print(",");

    Serial.print(accelMagnitude, 4);
    Serial.print(",");

    // --------------------------------------------------------
    // GYRO
    // --------------------------------------------------------

    Serial.print(gx, 2);
    Serial.print(",");

    Serial.print(gy, 2);
    Serial.print(",");

    Serial.print(gz, 2);
    Serial.print(",");

    Serial.print(gyroMagnitude, 2);
    Serial.print(",");

    Serial.print(imuOK ? 1 : 0);
    Serial.print(",");

    // --------------------------------------------------------
    // MPR121 - 12 separate columns
    // --------------------------------------------------------

    for (int i = 0; i < 12; i++)
    {
        Serial.print(
            touchStates[i]
            ? 1
            : 0
        );

        Serial.print(",");
    }

    Serial.print(touchCount);
    Serial.print(",");

    Serial.print(touchOK ? 1 : 0);
    Serial.print(",");

    // --------------------------------------------------------
    // HEART RATE
    // --------------------------------------------------------

    Serial.print(heartRate);
    Serial.print(",");

    Serial.print(
        heartRateValid
        ? 1
        : 0
    );

    Serial.print(",");

    // --------------------------------------------------------
    // AMG8833 - 64 individual pixels
    // --------------------------------------------------------

    for (int i = 0; i < 64; i++)
    {
        Serial.print(
            thermal[i],
            2
        );

        Serial.print(",");
    }

    Serial.print(
        thermalMin,
        2
    );

    Serial.print(",");

    Serial.print(
        thermalMax,
        2
    );

    Serial.print(",");

    Serial.print(
        thermalAverage,
        2
    );

    Serial.print(",");

    float thermalRange =
        thermalMax -
        thermalMin;

    Serial.print(
        thermalRange,
        2
    );

    Serial.print(",");

    Serial.print(
        thermalOK
        ? 1
        : 0
    );

    Serial.print(",");

    // --------------------------------------------------------
    // JOYSTICK
    // --------------------------------------------------------

    Serial.print(
        joystickX
    );

    Serial.print(",");

    Serial.print(
        joystickY
    );

    Serial.print(",");

    Serial.print(
        joystickDX,
        2
    );

    Serial.print(",");

    Serial.print(
        joystickDY,
        2
    );

    Serial.print(",");

    Serial.print(
        joystickMagnitude,
        2
    );

    Serial.print(",");

    Serial.print(
        joystickButton
        ? 1
        : 0
    );

    Serial.print(",");

    // --------------------------------------------------------
    // LIGHT
    // --------------------------------------------------------

    Serial.print(
        lightValue
    );

    Serial.print(",");

    Serial.print(
        lightNormalized,
        4
    );

    Serial.print(",");

    // --------------------------------------------------------
    // LOUDNESS
    // --------------------------------------------------------

    Serial.print(
        loudnessValue
    );

    Serial.print(",");

    Serial.println(
        loudnessNormalized,
        4
    );

    // ========================================================
    // NEXT SAMPLE
    // ========================================================

    delay(
        SAMPLE_INTERVAL_MS
    );
}