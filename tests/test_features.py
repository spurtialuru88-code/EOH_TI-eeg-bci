import numpy as np
from classifier import train_model
from EOH_BCI.eoh_bci.utils.utils_eeg import generate_eeg_signal
from features import extract_features
from preprocessing import bandpass_filter, notch_filter

def test_feature_output_shape():
    signal = generate_eeg_signal()
    features = extract_features(signal)

    assert isinstance(features, (list, np.ndarray))
    assert len(features) > 0

def test_features_are_finite():
    signal = generate_eeg_signal()
    features = extract_features(signal)

    assert not np.any(np.isnan(features))
    assert not np.any(np.isinf(features))

import numpy as np
from preprocessing import bandpass_filter, notch_filter
from EOH_BCI.eoh_bci.utils.utils_eeg import generate_eeg_signal

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

