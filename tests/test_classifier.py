import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from preprocessing import bandpass_filter
from tests.utils_eeg import generate_dataset
from classifier import train_model, predict
from features import extract_features

def test_classifier_accuracy():
    X_raw, y = generate_dataset()

    # Assume extract_features is used inside classifier pipeline
    from classifier import train_model
    X = [extract_features(x) for x in X_raw]

    model = train_model(X, y)
    preds = predict(model, X)

    acc = accuracy_score(y, preds)

    assert acc > 0.7  # realistic threshold for synthetic EEG

def test_confusion_matrix_shape():
    X_raw, y = generate_dataset()

    from classifier import train_model
    X = [extract_features(x) for x in X_raw]

    model = train_model(X, y)
    preds = predict(model, X)

    cm = confusion_matrix(y, preds)

    assert cm.shape == (2, 2)

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

from classifier import train_model
from tests.utils_eeg import generate_dataset

def test_classifier_accuracy_threshold():
    X_raw, y = generate_dataset()

    from classifier import train_model
    X = [extract_features(x) for x in X_raw]

    model = train_model(X, y)
    preds = predict(model, X)

    acc = accuracy_score(y, preds)

    assert acc > 0.7  # EEG synthetic baseline

def test_confusion_matrix_valid():
    X_raw, y = generate_dataset()

    from classifier import train_model
    X = [extract_features(x) for x in X_raw]

    model = train_model(X, y)
    preds = predict(model, X)

    cm = confusion_matrix(y, preds)

    assert cm.shape == (2, 2)
