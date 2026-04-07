#!/usr/bin/env python3
"""
03_train_classifier.py — Train the CSP+LDA Motor Imagery Classifier.

Takes the data you collected in step 2 and:
  1. Loads the epochs and labels
  2. Segments into analysis windows
  3. Trains CSP spatial filters
  4. Trains LDA classifier
  5. Reports cross-validation accuracy
  6. Saves the trained model

Usage:
  python 03_train_classifier.py                           # uses most recent data
  python 03_train_classifier.py --data data/training_*.npz  # specific file
"""

import sys
import argparse
import numpy as np
import os
import glob
import matplotlib.pyplot as plt

sys.path.insert(0, '.')
import config
from bci.preprocessing import segment_into_epochs, common_average_reference
from bci.classifier import BCIClassifier
from bci.features import CSPFilter, compute_band_powers


def find_latest_data():
    """Find the most recently created training data file."""
    pattern = os.path.join(config.DATA_DIR, "training_*.npz")
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def plot_csp_patterns(csp, epochs, labels, save_path=None):
    """
    Visualize what the CSP filters learned.
    
    This shows the SPATIAL PATTERNS — i.e., which brain regions
    are most important for distinguishing rest vs imagery.
    
    You should see:
      - Strong weights over C3/C4 (motor cortex)
      - Opposite polarities for the two classes
    """
    # Get CSP features for visualization
    features = csp.transform(epochs)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: CSP feature distributions per class
    ax = axes[0]
    for c in np.unique(labels):
        class_feats = features[labels == c]
        ax.scatter(class_feats[:, 0], class_feats[:, 1],
                  label=config.CLASS_NAMES[c], alpha=0.6, s=30)
    ax.set_xlabel('CSP Component 1 (log-var)')
    ax.set_ylabel('CSP Component 2 (log-var)')
    ax.set_title('CSP Feature Space')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Feature importance (variance explained)
    ax = axes[1]
    n_comp = features.shape[1]
    
    # Compare mean features between classes
    rest_mean = np.mean(features[labels == 0], axis=0)
    close_mean = np.mean(features[labels == 1], axis=0)
    diff = close_mean - rest_mean
    
    colors = ['#FF5722' if d > 0 else '#2196F3' for d in diff]
    ax.barh(range(n_comp), diff, color=colors)
    ax.set_yticks(range(n_comp))
    ax.set_yticklabels([f'CSP {i+1}' for i in range(n_comp)])
    ax.set_xlabel('Feature Difference (CLOSE − REST)')
    ax.set_title('CSP Discriminative Power')
    ax.axvline(0, color='black', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.suptitle('CSP Analysis — Motor Imagery BCI', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  [Plot] Saved to {save_path}")
    
    plt.show()


def plot_training_results(epochs, labels, csp, save_path=None):
    """
    Plot average spectral power per class per channel.
    Shows the mu/beta desynchronization pattern.
    """
    from scipy.signal import welch
    
    n_channels = epochs.shape[1]
    n_cols = min(4, n_channels)
    n_rows = (n_channels + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
    axes = np.atleast_2d(axes)
    
    for ch in range(n_channels):
        ax = axes[ch // n_cols, ch % n_cols]
        
        for c in np.unique(labels):
            class_epochs = epochs[labels == c, ch, :]
            
            # Average PSD across epochs
            psds = []
            for ep in class_epochs:
                f, psd = welch(ep, fs=config.SAMPLING_RATE,
                              nperseg=min(config.SAMPLING_RATE, len(ep)))
                psds.append(psd)
            
            mean_psd = np.mean(psds, axis=0)
            ax.semilogy(f, mean_psd, label=config.CLASS_NAMES[c], alpha=0.8)
        
        ch_name = config.CHANNEL_NAMES[ch] if ch < len(config.CHANNEL_NAMES) else f"Ch{ch}"
        ax.set_title(ch_name)
        ax.set_xlim(1, 45)
        ax.axvspan(8, 13, alpha=0.1, color='green', label='Mu')
        ax.axvspan(13, 30, alpha=0.1, color='orange', label='Beta')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    
    # Remove empty subplots
    for ch in range(n_channels, n_rows * n_cols):
        axes[ch // n_cols, ch % n_cols].set_visible(False)
    
    fig.suptitle('Average PSD per Channel per Class\n'
                 '(Green = Mu band, Orange = Beta band)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Train motor imagery classifier')
    parser.add_argument('--data', type=str, default=None,
                       help='Path to training data (.npz)')
    args = parser.parse_args()
    
    # Find data file
    data_path = args.data or find_latest_data()
    if data_path is None:
        print("❌ No training data found! Run 02_collect_training.py first.")
        sys.exit(1)
    
    print("=" * 60)
    print("  TRAINING CSP+LDA CLASSIFIER")
    print("=" * 60)
    print(f"  Data: {data_path}")
    
    # Load data
    data = np.load(data_path, allow_pickle=True)
    epochs = data['epochs']    # shape: (n_epochs, n_channels, n_samples)
    labels = data['labels']    # shape: (n_epochs,)
    
    print(f"  Loaded: {len(epochs)} epochs, {epochs.shape[1]} channels, "
          f"{epochs.shape[2]} samples each")
    
    # Check class balance
    for c in np.unique(labels):
        n = np.sum(labels == c)
        name = config.CLASS_NAMES[c] if c < len(config.CLASS_NAMES) else f"Class {c}"
        print(f"  Class {c} ({name}): {n} epochs")
    
    # Apply common average reference
    print("\n  Applying common average reference...")
    for i in range(len(epochs)):
        epochs[i] = common_average_reference(epochs[i])
    
    # Optionally segment long epochs into shorter windows
    # (more training data from the same recording)
    window_samples = config.WINDOW_LENGTH
    epoch_samples = epochs.shape[2]
    
    if epoch_samples > window_samples * 2:
        print(f"\n  Segmenting {epoch_samples}-sample epochs into "
              f"{window_samples}-sample windows...")
        
        new_epochs = []
        new_labels = []
        
        for i in range(len(epochs)):
            segs = segment_into_epochs(
                epochs[i],
                window_samples,
                overlap_samples=window_samples // 2
            )
            # segment_into_epochs returns (n_segs, n_ch, n_samples)
            # but input is (n_ch, n_samples), so we need to handle this
            # Let's do it manually here
            step = window_samples // 2
            n_segs = (epoch_samples - window_samples) // step + 1
            for s in range(n_segs):
                start = s * step
                end = start + window_samples
                new_epochs.append(epochs[i, :, start:end])
                new_labels.append(labels[i])
        
        epochs = np.array(new_epochs)
        labels = np.array(new_labels)
        print(f"  → {len(epochs)} windows")
    
    # Train classifier
    clf = BCIClassifier()
    results = clf.train(epochs, labels)
    
    # Save model
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    clf.save()
    
    # Generate plots for your poster
    print("\n  Generating analysis plots...")
    
    os.makedirs("plots", exist_ok=True)
    
    plot_csp_patterns(clf.csp, epochs, labels,
                     save_path="plots/csp_analysis.png")
    
    plot_training_results(epochs, labels, clf.csp,
                         save_path="plots/spectral_analysis.png")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"  Cross-validation accuracy: {results['cv_accuracy']:.1%}")
    print(f"  Model saved to: {config.MODEL_FILE}")
    print(f"  Plots saved to: plots/")
    print(f"\n  Next: python 04_live_control.py")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
