"""
features.py — Feature Extraction for Motor Imagery BCI.

This is the module that turns filtered EEG into numbers the ML model
can understand. Two main approaches:

1. CSP (Common Spatial Pattern) — THE gold standard for motor imagery
2. Band Power Features — simpler, good fallback

=== WHAT IS CSP? ===

Common Spatial Pattern finds spatial filters (linear combinations of
EEG channels) that MAXIMIZE the variance for one class and MINIMIZE
it for the other class.

Intuition: When you imagine moving your right hand, the LEFT motor
cortex (C3 area) shows mu suppression = LESS power. The RIGHT motor
cortex (C4 area) stays the same or increases. CSP finds the exact
channel combinations that best capture this asymmetry.

Mathematically:
  - You have covariance matrices for each class: Σ₁, Σ₂
  - CSP solves the generalized eigenvalue problem: Σ₁w = λΣ₂w
  - The eigenvectors w are your spatial filters
  - The eigenvalues tell you how discriminative each filter is

After CSP, you take the log-variance of each component as your feature.
These features go directly into the classifier (LDA).

=== WHAT IS BAND POWER? ===

Simpler approach: for each channel, compute the power in specific
frequency bands (mu: 8-13 Hz, beta: 13-30 Hz) using Welch's method.
Less powerful than CSP but useful as additional features.
"""

import numpy as np
from scipy.signal import welch
from sklearn.base import BaseEstimator, TransformerMixin
import config


class CSPFilter(BaseEstimator, TransformerMixin):
    """
    Common Spatial Pattern implementation.
    
    Compatible with scikit-learn Pipeline (has fit/transform methods).
    
    Usage:
        csp = CSPFilter(n_components=6)
        csp.fit(X_train, y_train)            # learn spatial filters
        features_train = csp.transform(X_train)  # extract features
        features_test = csp.transform(X_test)
    
    Input shape:  (n_epochs, n_channels, n_samples)
    Output shape: (n_epochs, n_components)
    """
    
    def __init__(self, n_components=None):
        """
        Args:
            n_components: Number of CSP components to keep.
                         Must be even. Default from config.
                         Uses the top n/2 and bottom n/2 components
                         (maximally discriminative for each class).
        """
        self.n_components = n_components or config.CSP_N_COMPONENTS
        self.filters_ = None
        self.mean_ = None
        self.std_ = None
    
    def _compute_covariance(self, X):
        """
        Compute the normalized covariance matrix for a set of epochs.
        
        For each epoch, compute Σ = XX^T / trace(XX^T)
        Then average across epochs.
        
        Args:
            X: np.ndarray, shape (n_epochs, n_channels, n_samples)
        
        Returns:
            Average covariance matrix, shape (n_channels, n_channels)
        """
        n_epochs, n_channels, n_samples = X.shape
        covs = np.zeros((n_epochs, n_channels, n_channels))
        
        for i in range(n_epochs):
            epoch = X[i]
            cov = np.dot(epoch, epoch.T)
            cov /= np.trace(cov)  # normalize by trace
            covs[i] = cov
        
        return np.mean(covs, axis=0)
    
    def fit(self, X, y):
        """
        Learn the CSP spatial filters from labeled training data.
        
        Args:
            X: np.ndarray, shape (n_epochs, n_channels, n_samples)
            y: np.ndarray, shape (n_epochs,) — class labels (0 or 1)
        
        Returns:
            self (for pipeline compatibility)
        
        WHAT'S HAPPENING:
          1. Compute average covariance for each class
          2. Compute composite covariance Σ_c = Σ₁ + Σ₂
          3. Whitening: P = eigendecomposition of Σ_c
          4. Solve generalized eigenvalue problem
          5. Sort by eigenvalue → first components maximize class 1,
             last components maximize class 2
          6. Keep top n/2 + bottom n/2 = n_components filters
        """
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError(f"CSP requires exactly 2 classes, got {len(classes)}")
        
        # Covariance for each class
        cov_1 = self._compute_covariance(X[y == classes[0]])
        cov_2 = self._compute_covariance(X[y == classes[1]])
        
        # Composite covariance
        cov_composite = cov_1 + cov_2
        
        # Eigendecomposition of composite (for whitening)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_composite)
        
        # Sort descending
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Whitening transform
        # P = D^(-1/2) * U^T  where D=eigenvalues, U=eigenvectors
        D_inv_sqrt = np.diag(1.0 / np.sqrt(eigenvalues + 1e-10))
        P = D_inv_sqrt @ eigenvectors.T
        
        # Transform class covariances into whitened space
        S1 = P @ cov_1 @ P.T
        
        # Eigendecomposition of whitened class-1 covariance
        eig_vals, W = np.linalg.eigh(S1)
        
        # Sort by eigenvalue (descending)
        idx = np.argsort(eig_vals)[::-1]
        W = W[:, idx]
        
        # Full spatial filters
        full_filters = (W.T @ P)  # shape: (n_channels, n_channels)
        
        # Select top and bottom components
        n = self.n_components // 2
        selected = np.concatenate([
            full_filters[:n],         # best for class 1
            full_filters[-n:]         # best for class 2
        ], axis=0)
        
        self.filters_ = selected
        
        # Compute normalization params on training data
        features = self._extract_features(X)
        self.mean_ = np.mean(features, axis=0)
        self.std_ = np.std(features, axis=0) + 1e-10
        
        print(f"[CSP] Fitted {self.n_components} spatial filters on "
              f"{len(X)} epochs ({len(classes)} classes)")
        
        return self
    
    def _extract_features(self, X):
        """
        Apply spatial filters and extract log-variance features.
        
        For each epoch:
          1. Project through spatial filters: Z = W @ X
          2. Compute variance of each component
          3. Take log (normalizes the distribution)
        
        Args:
            X: shape (n_epochs, n_channels, n_samples)
        
        Returns:
            features: shape (n_epochs, n_components)
        """
        n_epochs = X.shape[0]
        features = np.zeros((n_epochs, self.n_components))
        
        for i in range(n_epochs):
            # Apply spatial filters
            Z = self.filters_ @ X[i]  # shape: (n_components, n_samples)
            
            # Log-variance of each component
            var = np.var(Z, axis=1)
            features[i] = np.log(var + 1e-10)
        
        return features
    
    def transform(self, X):
        """
        Transform epochs into CSP feature vectors.
        
        Args:
            X: shape (n_epochs, n_channels, n_samples)
        
        Returns:
            features: shape (n_epochs, n_components), normalized
        """
        if self.filters_ is None:
            raise RuntimeError("Call fit() first!")
        
        features = self._extract_features(X)
        
        # Normalize using training stats
        if self.mean_ is not None:
            features = (features - self.mean_) / self.std_
        
        return features


def compute_band_powers(epoch, fs=None):
    """
    Compute power in specific frequency bands for each channel.
    
    Uses Welch's method (PSD estimation) then integrates power
    within each band.
    
    Args:
        epoch: np.ndarray, shape (n_channels, n_samples)
        fs: Sampling rate
    
    Returns:
        features: np.ndarray, shape (n_channels * n_bands,)
        
    Bands computed:
        mu:    8-13 Hz  (motor imagery specific)
        beta:  13-30 Hz (also modulated by MI)
        ratio: mu/beta  (discriminative feature)
    
    WHY WELCH'S METHOD?
      It splits the signal into overlapping segments, computes
      FFT on each, and averages. This reduces variance of the
      spectral estimate compared to a single FFT.
    """
    fs = fs or config.SAMPLING_RATE
    n_channels = epoch.shape[0]
    
    bands = {
        'mu': (8, 13),
        'beta': (13, 30),
    }
    
    features = []
    
    for ch in range(n_channels):
        # Welch PSD: returns frequencies and power spectral density
        freqs, psd = welch(epoch[ch], fs=fs, nperseg=min(fs, epoch.shape[1]))
        
        for band_name, (f_low, f_high) in bands.items():
            # Find frequency indices within the band
            idx = np.where((freqs >= f_low) & (freqs <= f_high))[0]
            
            # Integrate (sum) the power in this band
            band_power = np.sum(psd[idx])
            features.append(np.log(band_power + 1e-10))
        
        # Mu/Beta ratio (discriminative for MI)
        mu_idx = np.where((freqs >= 8) & (freqs <= 13))[0]
        beta_idx = np.where((freqs >= 13) & (freqs <= 30))[0]
        mu_power = np.sum(psd[mu_idx])
        beta_power = np.sum(psd[beta_idx])
        features.append(np.log((mu_power / (beta_power + 1e-10)) + 1e-10))
    
    return np.array(features)


def extract_combined_features(epochs, csp_filter=None, fs=None):
    """
    Extract both CSP and band power features for each epoch.
    
    This gives the classifier more information to work with.
    CSP captures spatial patterns; band power captures spectral patterns.
    
    Args:
        epochs: shape (n_epochs, n_channels, n_samples)
        csp_filter: Fitted CSPFilter object (optional)
        fs: Sampling rate
    
    Returns:
        features: shape (n_epochs, n_features)
    """
    fs = fs or config.SAMPLING_RATE
    all_features = []
    
    for i in range(len(epochs)):
        bp = compute_band_powers(epochs[i], fs)
        all_features.append(bp)
    
    bp_features = np.array(all_features)
    
    if csp_filter is not None and csp_filter.filters_ is not None:
        csp_features = csp_filter.transform(epochs)
        return np.hstack([csp_features, bp_features])
    
    return bp_features
