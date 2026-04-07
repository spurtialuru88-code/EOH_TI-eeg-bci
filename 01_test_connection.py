#!/usr/bin/env python3
"""
01_test_connection.py — Verify your EEG board connects and streams data.

RUN THIS FIRST before anything else.

What it does:
  1. Connects to your OpenBCI board (or synthetic board for testing)
  2. Streams 5 seconds of data
  3. Checks signal quality per channel
  4. Plots the raw EEG so you can visually verify

Usage:
  python 01_test_connection.py              # real board
  python 01_test_connection.py --synthetic  # no hardware (test mode)
"""

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, '.')
import config
from bci.acquisition import EEGAcquisition
from bci.preprocessing import notch_filter, bandpass_filter


def main():
    parser = argparse.ArgumentParser(description='Test EEG board connection')
    parser.add_argument('--synthetic', action='store_true',
                       help='Use synthetic board (no hardware)')
    parser.add_argument('--duration', type=float, default=5.0,
                       help='Recording duration in seconds')
    args = parser.parse_args()
    
    print("=" * 60)
    print("  EEG BOARD CONNECTION TEST")
    print("=" * 60)
    
    # Create DAQ
    daq = EEGAcquisition(use_synthetic=args.synthetic)
    
    try:
        # Connect
        daq.connect()
        daq.start_stream()
        
        print(f"\nRecording {args.duration} seconds of data...")
        
        # Check signal quality
        import time
        time.sleep(2)
        daq.check_signal_quality()
        
        # Record data
        data = daq.get_data_chunk(args.duration)
        
        print(f"\nData shape: {data.shape}")
        print(f"  → {data.shape[0]} channels × {data.shape[1]} samples")
        print(f"  → {data.shape[1] / config.SAMPLING_RATE:.1f} seconds")
        
        # Basic stats
        print(f"\nSignal statistics (µV):")
        for i, name in enumerate(config.CHANNEL_NAMES[:data.shape[0]]):
            print(f"  {name}: mean={np.mean(data[i]):.1f}, "
                  f"std={np.std(data[i]):.1f}, "
                  f"range=[{np.min(data[i]):.1f}, {np.max(data[i]):.1f}]")
        
        # Plot raw EEG
        print("\nPlotting raw EEG (close window to continue)...")
        plot_eeg(data, "Raw EEG", config.SAMPLING_RATE)
        
        # Plot filtered EEG
        filtered = notch_filter(data)
        filtered = bandpass_filter(filtered)
        plot_eeg(filtered, "Filtered EEG (8-30 Hz)", config.SAMPLING_RATE)
        
        print("\n✅ Connection test PASSED!")
        print("   Your board is working. Proceed to 02_collect_training.py")
        
    except Exception as e:
        print(f"\n❌ Connection FAILED: {e}")
        print("\nTroubleshooting:")
        print("  1. Check USB dongle is plugged in")
        print("  2. Check serial port in config.py")
        print("  3. Make sure OpenBCI GUI is closed (can't share the port)")
        print("  4. Try: python 01_test_connection.py --synthetic")
        raise
    
    finally:
        daq.stop()


def plot_eeg(data, title, fs):
    """Plot multi-channel EEG data."""
    n_channels, n_samples = data.shape
    t = np.arange(n_samples) / fs
    
    fig, axes = plt.subplots(n_channels, 1, figsize=(14, 2 * n_channels),
                            sharex=True)
    if n_channels == 1:
        axes = [axes]
    
    for i, ax in enumerate(axes):
        ch_name = config.CHANNEL_NAMES[i] if i < len(config.CHANNEL_NAMES) else f"Ch{i}"
        ax.plot(t, data[i], linewidth=0.5, color='#2196F3')
        ax.set_ylabel(f'{ch_name}\n(µV)')
        ax.set_xlim(t[0], t[-1])
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Time (s)')
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
