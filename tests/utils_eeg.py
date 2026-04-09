import numpy as np

def generate_eeg_signal(freq=10, noise_level=0.5, samples=1000, fs=250):
    """
    Synthetic EEG-like signal (alpha rhythm + noise)
    """
    t = np.arange(samples) / fs
    signal = np.sin(2 * np.pi * freq * t)

    noise = noise_level * np.random.randn(samples)
    return signal + noise


def generate_dataset(n_samples=50):
    """
    Generates labeled synthetic EEG dataset
    Class 0: 10 Hz alpha
    Class 1: 20 Hz beta
    """
    X = []
    y = []

    for _ in range(n_samples // 2):
        X.append(generate_eeg_signal(freq=10))
        y.append(0)

        X.append(generate_eeg_signal(freq=20))
        y.append(1)

    return X, y