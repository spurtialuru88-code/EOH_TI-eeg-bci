import time
import numpy as np

def collect_samples(reader, duration=5):
    values = []
    start = time.time()

    while time.time() - start < duration:
        v = reader.read_latest()
        if v is not None:
            values.append(v)
        time.sleep(0.02)

    return np.array(values)


def calibrate(reader):
    print("\n=== CALIBRATION START ===")

    input("Relax your brain (REST) and press Enter...")
    rest_vals = collect_samples(reader)

    input("Now FOCUS / FLEX and press Enter...")
    flex_vals = collect_samples(reader)

    rest_mean = np.mean(rest_vals)
    flex_mean = np.mean(flex_vals)

    threshold_rest = rest_mean
    threshold_flex = flex_mean

    print(f"\nCalibrated:")
    print(f"REST: {threshold_rest:.3f}")
    print(f"FLEX: {threshold_flex:.3f}")

    return threshold_rest, threshold_flex