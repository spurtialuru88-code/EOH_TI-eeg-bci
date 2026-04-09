import numpy as np

from preprocessing import bandpass_filter
from features import extract_features
from classifier import train_model
from tests.utils_eeg import generate_dataset
from preprocessing import bandpass_filter, notch_filter
from features import extract_features
from classifier import train_model, predict

def test_full_bci_pipeline():
    X_raw, y = generate_dataset()

    processed = []
    for signal in X_raw:
        filtered = bandpass_filter(signal)
        feats = extract_features(filtered)
        processed.append(feats)

    model = train_model(processed, y)
    preds = predict(model, processed)

    assert len(preds) == len(y)

    # sanity check: predictions are binary
    assert set(preds).issubset({0, 1})

from tests.utils_eeg import generate_dataset
from preprocessing import bandpass_filter
from features import extract_features
from classifier import train_model

def test_full_bci_pipeline():
    X_raw, y = generate_dataset()

    processed = []
    for signal in X_raw:
        filtered = bandpass_filter(signal)
        feats = extract_features(filtered)
        processed.append(feats)

    model = train_model(processed, y)
    preds = predict(model, processed)

    assert len(preds) == len(y)
    assert set(preds).issubset({0, 1})