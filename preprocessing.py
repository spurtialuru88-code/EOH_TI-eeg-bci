"""
preprocessing.py — EEG Signal Preprocessing Pipeline.

Raw EEG is MESSY. Before you can classify anything, you need to:
  1. Remove power line noise (60 Hz hum from wall outlets)
  2. Bandpass filter to isolate frequencies of interest (8-30 Hz)
  3. Reject artifacts (blinks, jaw clenches, movement)
  4. Segment into analysis windows

WHY THESE SPECIFIC FREQUENCIES?
  Motor imagery produces changes in two frequency bands:
  - Mu rhythm (8-13 Hz): suppressed when you imagine hand movement
  - Beta rhythm (13-30 Hz): also suppressed during motor imagery
  Together these are called "sensorimotor rhythms" or SMR.
  
  Below 8 Hz = slow drifts, eye movements, not useful for MI
  Above 30 Hz = muscle artifacts (EMG), also not useful

WHAT IS A BUTTERWORTH FILTER?
  A type of digital filter with a maximally flat passband.
  "Order 4" means it uses 4 poles — steeper rolloff = better
  frequency separation, but more phase distortion. Order 4 is
  the sweet spot for EEG.
"""

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch
import config


def bandpass_filter(data, low=None, high=None, fs=None, order=None):
    """
    Apply a bandpass Butterworth filter to EEG data.
    
    This keeps frequencies between `low` and `high` Hz,
    and attenuates everything else.
    
    Args:
        data: np.ndarray, shape (n_channels, n_samples) or (n_samples,)
        low: Lower cutoff frequency in Hz (default: config.BANDPASS_LOW)
        high: Upper cutoff frequency in Hz (default: config.BANDPASS_HIGH)
        fs: Sampling rate in Hz (default: config.SAMPLING_RATE)
        order: Filter order (default: config.FILTER_ORDER)
    
    Returns:
        Filtered data, same shape as input.
    
    HOW IT WORKS:
      1. butter() designs the filter coefficients (b, a)
      2. filtfilt() applies the filter FORWARD then BACKWARD
         → This gives zero phase distortion (important for BCI!)
         → Equivalent to doubling the filter order
    """
    low = low or config.BANDPASS_LOW
    high = high or config.BANDPASS_HIGH
    fs = fs or config.SAMPLING_RATE
    order = order or config.FILTER_ORDER
    
    # Nyquist frequency = half the sampling rate
    nyq = fs / 2.0
    
    # Normalize frequencies to [0, 1] where 1 = Nyquist
    b, a = butter(order, [low / nyq, high / nyq], btype='band')
    
    # Apply filter
    if data.ndim == 1:
        return filtfilt(b, a, data)
    else:
        # Filter each channel independently
        filtered = np.zeros_like(data)
        for ch in range(data.shape[0]):
            filtered[ch, :] = filtfilt(b, a, data[ch, :])
        return filtered


def notch_filter(data, freq=None, fs=None, quality=30.0):
    """
    Remove power line noise at a specific frequency.
    
    In the US, wall power is 60 Hz. This creates electromagnetic
    interference that your EEG electrodes pick up. A notch filter
    removes JUST that frequency while preserving everything else.
    
    Args:
        data: np.ndarray, shape (n_channels, n_samples)
        freq: Frequency to remove in Hz (default: 60 Hz)
        fs: Sampling rate
        quality: Q factor. Higher = narrower notch. 30 is standard.
    
    Returns:
        Filtered data.
    """
    freq = freq or config.NOTCH_FREQ
    fs = fs or config.SAMPLING_RATE
    
    b, a = iirnotch(freq, quality, fs)
    
    if data.ndim == 1:
        return filtfilt(b, a, data)
    else:
        filtered = np.zeros_like(data)
        for ch in range(data.shape[0]):
            filtered[ch, :] = filtfilt(b, a, data[ch, :])
        return filtered


def preprocess_pipeline(data, fs=None):
    """
    Full preprocessing pipeline: notch → bandpass.
    
    This is the function you'll call most often. It takes raw EEG
    and returns clean, filtered data ready for feature extraction.
    
    Args:
        data: np.ndarray, shape (n_channels, n_samples)
        fs: Sampling rate
    
    Returns:
        Preprocessed data, same shape.
    """
    fs = fs or config.SAMPLING_RATE
    
    # Step 1: Remove power line noise
    data = notch_filter(data, fs=fs)
    
    # Step 2: Bandpass filter to mu+beta band
    data = bandpass_filter(data, fs=fs)
    
    return data


def reject_bad_epochs(epochs, labels, threshold_uv=150.0):
    """
    Remove epochs that contain artifacts (blinks, movement, etc.)
    
    Simple amplitude-based rejection:
    If any channel in an epoch exceeds `threshold_uv` microvolts,
    that epoch is thrown out.
    
    WHY THIS WORKS:
      Normal EEG is typically < 100 µV. Blinks are ~200-300 µV.
      Jaw clenches can be ~500+ µV. So 150 µV is a reasonable
      threshold that keeps good data and rejects obvious artifacts.
    
    Args:
        epochs: np.ndarray, shape (n_epochs, n_channels, n_samples)
        labels: np.ndarray, shape (n_epochs,) — corresponding labels
        threshold_uv: Rejection threshold in microvolts
    
    Returns:
        clean_epochs, clean_labels (with bad epochs removed)
    """
    good_mask = np.ones(len(epochs), dtype=bool)
    
    for i in range(len(epochs)):
        # Check if any channel exceeds threshold
        if np.max(np.abs(epochs[i])) > threshold_uv:
            good_mask[i] = False
    
    n_rejected = np.sum(~good_mask)
    n_total = len(epochs)
    print(f"[Preprocessing] Artifact rejection: {n_rejected}/{n_total} "
          f"epochs rejected ({n_rejected/n_total*100:.1f}%)")
    
    if n_rejected > n_total * 0.5:
        print("  ⚠️  WARNING: >50% of epochs rejected!")
        print("  → Check electrode contact and ask subject to minimize movement")
    
    return epochs[good_mask], labels[good_mask]


def segment_into_epochs(data, epoch_length_samples, overlap_samples=0):
    """
    Chop continuous data into overlapping windows (epochs).
    
    This is how you go from a long continuous recording to
    discrete chunks that the classifier can process.
    
    Args:
        data: np.ndarray, shape (n_channels, n_samples)
        epoch_length_samples: Length of each epoch in samples
        overlap_samples: How many samples overlap between epochs
    
    Returns:
        np.ndarray, shape (n_epochs, n_channels, epoch_length_samples)
    
    Example:
        If data is 10 seconds at 250 Hz = 2500 samples,
        and epoch_length = 250 (1 sec), overlap = 125 (50%):
        → You get (2500 - 250) / 125 + 1 = 19 epochs
    """
    n_channels, n_total = data.shape
    step = epoch_length_samples - overlap_samples
    
    n_epochs = (n_total - epoch_length_samples) // step + 1
    
    epochs = np.zeros((n_epochs, n_channels, epoch_length_samples))
    
    for i in range(n_epochs):
        start = i * step
        end = start + epoch_length_samples
        epochs[i, :, :] = data[:, start:end]
    
    return epochs


def common_average_reference(data):
    """
    Apply Common Average Reference (CAR) to EEG data.
    
    For each time point, subtract the mean across all channels.
    This removes noise that's common to all electrodes (like
    the 60 Hz hum that doesn't fully get caught by the notch filter).
    
    This is a standard EEG preprocessing step used in research.
    
    Args:
        data: np.ndarray, shape (n_channels, n_samples)
    
    Returns:
        Re-referenced data, same shape.
    """
    mean_signal = np.mean(data, axis=0, keepdims=True)
    return data - mean_signal
