import numpy as np
from tests.utils_eeg import generate_eeg_signal
from preprocessing import bandpass_filter, notch_filter

def test_bandpass_preserves_signal_length():
    signal = generate_eeg_signal(freq=10)
    filtered = bandpass_filter(signal)

    assert len(signal) == len(filtered)

def test_bandpass_reduces_high_frequency_noise():
    # high-frequency noisy signal
    signal = generate_eeg_signal(freq=50, noise_level=1.0)
    filtered = bandpass_filter(signal)

    # bandpass should reduce variance outside band
    assert np.var(filtered) < np.var(signal)

def test_notch_removes_powerline_noise():
    signal = generate_eeg_signal(freq=60)  # simulate mains interference
    filtered = notch_filter(signal)

    assert np.var(filtered) < np.var(signal)