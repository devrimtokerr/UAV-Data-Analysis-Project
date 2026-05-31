import numpy as np
import os

# ============================================================
# CONFIGURATION
# ============================================================

NUM_POINTS            = 200
TARGET_WAYPOINT_COUNT = 20
RANDOM_SEED           = 42
OUTPUT_DIR = r"C:\Users\devri\OneDrive\Masaüstü\proje 2\mission_planner_output"

# ============================================================
# CATEGORY-SPECIFIC NOISE PROFILES (Q and R parameters)
# ============================================================

NOISE_PROFILES = {
    "IMU"   : {"Q": 0.10,  "R": 1.50},
    "GPS"   : {"Q": 0.01,  "R": 0.50},
    "LiDAR" : {"Q": 0.005, "R": 0.30},
}

# ============================================================
# 13 SENSOR CHANNELS
# IMU(6) + GPS(3) + LiDAR(4)
# ============================================================

CHANNELS = {
    "accel_x"    : ("IMU",    lambda n: np.zeros(n),                        0.50),
    "accel_y"    : ("IMU",    lambda n: np.zeros(n),                        0.50),
    "accel_z"    : ("IMU",    lambda n: np.full(n, -9.81),                  0.80),
    "gyro_x"     : ("IMU",    lambda n: np.zeros(n),                        0.05),
    "gyro_y"     : ("IMU",    lambda n: np.zeros(n),                        0.05),
    "gyro_z"     : ("IMU",    lambda n: np.zeros(n),                        0.05),
    "latitude"   : ("GPS",    lambda n: np.linspace(39.9000, 39.9200, n),   0.0005),
    "longitude"  : ("GPS",    lambda n: np.linspace(32.8500, 32.8700, n),   0.0005),
    "altitude"   : ("GPS",    lambda n: np.linspace(50.0, 100.0, n),        2.00),
    "lidar_x"    : ("LiDAR",  lambda n: np.linspace(0, 50, n),              0.10),
    "lidar_y"    : ("LiDAR",  lambda n: np.linspace(0, 50, n),              0.10),
    "lidar_z"    : ("LiDAR",  lambda n: np.linspace(0, 20, n),              0.05),
    "lidar_range": ("LiDAR",  lambda n: np.linspace(5, 30, n),              0.20),
}

# ============================================================
# DISCRETE-TIME SCALAR LINEAR KALMAN FILTER
# Recursive predict-correct cycle
# ============================================================

def kalman_filter(measurements, Q, R):
    n        = len(measurements)
    filtered = np.zeros(n)
    x = measurements[0]
    P = 1.0

    for i in range(n):
        # PREDICT
        x_pred = x
        P_pred = P + Q
        # CORRECT
        K = P_pred / (P_pred + R)
        x = x_pred + K * (measurements[i] - x_pred)
        P = (1 - K) * P_pred
        filtered[i] = x

    return filtered

# ============================================================
# PERFORMANCE METRICS
# ============================================================

def noise_reduction(raw, filtered):
    """Noise Reduction (%) = (std_raw - std_filtered) / std_raw * 100"""
    std_raw      = np.std(raw)
    std_filtered = np.std(filtered)
    if std_raw == 0:
        return 0.0
    return (std_raw - std_filtered) / std_raw * 100.0

def snr_db(true_signal, noisy_signal):
    """SNR (dB) = 10 * log10(Signal Power / Noise Power)"""
    noise        = noisy_signal - true_signal
    signal_power = np.mean(true_signal ** 2)
    noise_power  = np.mean(noise ** 2)
    if noise_power == 0:
        return float('inf')
    return 10 * np.log10(signal_power / noise_power)

def snr_improvement(true_signal, raw, filtered):
    """SNR Improvement (dB) = SNR_filtered - SNR_raw"""
    return snr_db(true_signal, filtered) - snr_db(true_signal, raw)

# ============================================================
# DOWNSAMPLE
# ============================================================

def downsample(arr, target_count):
    total = len(arr)
    if target_count >= total:
        return arr
    indices = np.round(np.linspace(0, total - 1, target_count)).astype(int)
    return arr[indices]

# ============================================================
# GENERATE RAW DATA FOR ALL 13 CHANNELS
# ============================================================

np.random.seed(RANDOM_SEED)

true_signals     = {}
raw_signals      = {}
filtered_signals = {}

for ch_name, (sensor_type, signal_fn, noise_std) in CHANNELS.items():
    true_sig              = signal_fn(NUM_POINTS)
    true_signals[ch_name] = true_sig
    raw_signals[ch_name]  = true_sig + np.random.normal(0, noise_std, NUM_POINTS)

# ============================================================
# APPLY KALMAN FILTER TO ALL 13 CHANNELS
# ============================================================

for ch_name, (sensor_type, _, _) in CHANNELS.items():
    Q = NOISE_PROFILES[sensor_type]["Q"]
    R = NOISE_PROFILES[sensor_type]["R"]
    filtered_signals[ch_name] = kalman_filter(raw_signals[ch_name], Q, R)

# ============================================================
# RAW vs KALMAN FILTERED DATA TABLE (Sample: first 5 points)
# Shows Raw Data | Kalman Filtered | Noise Reduction per channel
# ============================================================

SAMPLE_POINTS = 5   # how many rows to display per channel

print("\n" + "=" * 95)
print("RAW DATA vs KALMAN FILTERED DATA  —  Sample Points per Channel")
print("=" * 95)

for ch_name, (sensor_type, _, _) in CHANNELS.items():
    raw  = raw_signals[ch_name]
    filt = filtered_signals[ch_name]
    nr   = noise_reduction(raw, filt)

    print(f"\n  Channel: {ch_name:<14} | Type: {sensor_type:<6} | Noise Reduction: {nr:.2f}%")
    print(f"  {'Index':<8} {'Raw Data':>14} {'Kalman Filtered':>16} {'Difference':>12}")
    print(f"  {'-'*8} {'-'*14} {'-'*16} {'-'*12}")

    for i in range(SAMPLE_POINTS):
        diff = filt[i] - raw[i]
        print(f"  {i:<8} {raw[i]:>14.6f} {filt[i]:>16.6f} {diff:>+12.6f}")

print("\n" + "=" * 95)

# ============================================================
# PERFORMANCE METRICS REPORT
# ============================================================

print("\n" + "=" * 80)
print("PERFORMANCE METRICS REPORT")
print("Noise Reduction & SNR Improvement per Channel")
print("=" * 80)
print(f"  {'Channel':<14} {'Type':<8} {'NR (%)':>8} {'Raw SNR':>10} {'Filt SNR':>10} {'SNR Impr':>10}")
print(f"  {'-'*14} {'-'*8} {'-'*8} {'-'*10} {'-'*10} {'-'*10}")

for ch_name, (sensor_type, _, _) in CHANNELS.items():
    true = true_signals[ch_name]
    raw  = raw_signals[ch_name]
    filt = filtered_signals[ch_name]

    nr    = noise_reduction(raw, filt)
    snr_r = snr_db(true, raw)
    snr_f = snr_db(true, filt)
    snr_i = snr_improvement(true, raw, filt)

    print(f"  {ch_name:<14} {sensor_type:<8} {nr:>7.2f}% {snr_r:>9.2f}dB {snr_f:>9.2f}dB {snr_i:>+9.2f}dB")

print("=" * 80)

# ============================================================
# WAYPOINT GENERATION (GPS channels only)
# ============================================================

raw_lat = raw_signals["latitude"]
raw_lon = raw_signals["longitude"]
raw_alt = raw_signals["altitude"]

fil_lat = filtered_signals["latitude"]
fil_lon = filtered_signals["longitude"]
fil_alt = filtered_signals["altitude"]

ds_raw_lat = downsample(raw_lat, TARGET_WAYPOINT_COUNT)
ds_raw_lon = downsample(raw_lon, TARGET_WAYPOINT_COUNT)
ds_raw_alt = downsample(raw_alt, TARGET_WAYPOINT_COUNT)

ds_fil_lat = downsample(fil_lat, TARGET_WAYPOINT_COUNT)
ds_fil_lon = downsample(fil_lon, TARGET_WAYPOINT_COUNT)
ds_fil_alt = downsample(fil_alt, TARGET_WAYPOINT_COUNT)

# ============================================================
# CREATE WAYPOINTS FILE
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_waypoints_file(filename, latitudes, longitudes, altitudes):
    full_path = os.path.join(OUTPUT_DIR, filename)
    with open(full_path, 'w') as f:
        f.write("QGC WPL 110\n")
        # HOME
        f.write(
            f"0\t1\t0\t16\t0\t0\t0\t0\t"
            f"{latitudes[0]:.7f}\t{longitudes[0]:.7f}\t0.00\t1\n"
        )
        # TAKEOFF
        f.write(
            f"1\t0\t3\t22\t0\t0\t0\t0\t"
            f"{latitudes[0]:.7f}\t{longitudes[0]:.7f}\t{altitudes[0]:.2f}\t1\n"
        )
        # WAYPOINTS
        for i in range(len(latitudes)):
            f.write(
                f"{i+2}\t0\t3\t16\t0\t0\t0\t0\t"
                f"{latitudes[i]:.7f}\t{longitudes[i]:.7f}\t{altitudes[i]:.2f}\t1\n"
            )
        # LAND
        f.write(
            f"{len(latitudes)+2}\t0\t3\t21\t0\t0\t0\t0\t"
            f"{latitudes[-1]:.7f}\t{longitudes[-1]:.7f}\t0.00\t1\n"
        )
    print(f"[OK] '{full_path}' saved! ({len(latitudes)+3} waypoints)")

create_waypoints_file("RAW_DATA_waypoints.waypoints",
                      ds_raw_lat, ds_raw_lon, ds_raw_alt)

create_waypoints_file("KALMAN_FILTERED_waypoints.waypoints",
                      ds_fil_lat, ds_fil_lon, ds_fil_alt)

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 55)
print("SUMMARY")
print("=" * 55)
print(f"  Total channels processed : 13 (IMU:6, GPS:3, LiDAR:4)")
print(f"  Filter type              : Discrete-time scalar LKF")
print(f"  Raw data points          : {NUM_POINTS}")
print(f"  Waypoints (downsampled)  : {TARGET_WAYPOINT_COUNT}")
print(f"  Output folder            : {OUTPUT_DIR}")
print("=" * 55)
print("  Files created:")
print("    - RAW_DATA_waypoints.waypoints")
print("    - KALMAN_FILTERED_waypoints.waypoints")
print("=" * 55)
