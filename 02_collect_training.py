#!/usr/bin/env python3
"""
02_collect_training.py — Guided Motor Imagery Training Data Collection.

This script runs a cued paradigm where you:
  1. See "REST" → relax, think of nothing
  2. See "CLOSE" → imagine squeezing your right hand into a fist
  3. Repeat for ~5 minutes

The EEG data is saved with labels so you can train the classifier.

IMPORTANT TIPS FOR GOOD DATA:
  - Sit still. Minimize ALL movement (especially jaw, eyes, head).
  - IMAGINE the movement, don't actually do it.
  - Focus on the sensation of your hand closing, not the visual image.
  - Close your eyes during imagery if it helps.
  - Keep your real hands completely still on the desk.
  - The more vivid and consistent your imagery, the better the classifier.

Usage:
  python 02_collect_training.py              # real board
  python 02_collect_training.py --synthetic  # test mode
"""

import sys
import argparse
import numpy as np
import time
import os

sys.path.insert(0, '.')
import config
from bci.acquisition import EEGAcquisition
from bci.preprocessing import preprocess_pipeline, reject_bad_epochs


def print_cue(text, style="normal"):
    """Print a large, visible cue to the terminal."""
    if style == "rest":
        print(f"\n{'='*50}")
        print(f"       😌  REST  —  Relax, clear your mind")
        print(f"{'='*50}")
    elif style == "close":
        print(f"\n{'='*50}")
        print(f"       ✊  CLOSE  —  Imagine squeezing your fist!")
        print(f"{'='*50}")
    elif style == "ready":
        print(f"\n{'~'*50}")
        print(f"       ⏳  Get ready...")
        print(f"{'~'*50}")
    else:
        print(f"\n  {text}")


def countdown(seconds):
    """Visual countdown timer."""
    for i in range(seconds, 0, -1):
        print(f"  Starting in {i}...", end='\r')
        time.sleep(1)
    print(" " * 30, end='\r')


def main():
    parser = argparse.ArgumentParser(description='Collect motor imagery training data')
    parser.add_argument('--synthetic', action='store_true',
                       help='Use synthetic board')
    parser.add_argument('--trials', type=int, default=config.TRIALS_PER_CLASS,
                       help='Trials per class')
    parser.add_argument('--session', type=str, default=None,
                       help='Session name (for filename)')
    args = parser.parse_args()
    
    n_trials = args.trials
    n_classes = config.NUM_CLASSES
    total_trials = n_trials * n_classes
    
    trial_duration = config.TRIAL_DURATION_SEC
    rest_duration = config.REST_BETWEEN_SEC
    
    # Estimate total time
    total_time_min = (total_trials * (trial_duration + rest_duration + 1)) / 60
    
    print("=" * 60)
    print("  MOTOR IMAGERY TRAINING DATA COLLECTION")
    print("=" * 60)
    print(f"  Classes: {config.CLASS_NAMES[:n_classes]}")
    print(f"  Trials per class: {n_trials}")
    print(f"  Total trials: {total_trials}")
    print(f"  Trial duration: {trial_duration}s")
    print(f"  Estimated time: {total_time_min:.1f} minutes")
    print()
    print("  INSTRUCTIONS:")
    print("  • Sit comfortably with hands on desk")
    print("  • When you see REST: relax, clear your mind")
    print("  • When you see CLOSE: imagine squeezing your RIGHT fist")
    print("  • Do NOT actually move — only imagine!")
    print("  • Minimize eye blinks and jaw movement")
    print()
    
    input("  Press ENTER when you're ready to begin...")
    
    # Create trial sequence (randomized)
    # Each class appears n_trials times, in random order
    trial_labels = []
    for c in range(n_classes):
        trial_labels.extend([c] * n_trials)
    
    np.random.seed(config.RANDOM_SEED)
    np.random.shuffle(trial_labels)
    
    # Initialize DAQ
    daq = EEGAcquisition(use_synthetic=args.synthetic)
    
    try:
        daq.connect()
        daq.start_stream()
        
        # Wait for stream to stabilize
        print("\n  Stabilizing EEG stream...")
        time.sleep(3)
        
        # Check signal quality before starting
        daq.check_signal_quality()
        
        input("\n  Signal looks good? Press ENTER to start collection...\n")
        
        countdown(5)
        
        # Storage for all epochs
        all_epochs = []
        all_labels = []
        
        for trial_idx, label in enumerate(trial_labels):
            class_name = config.CLASS_NAMES[label]
            
            print(f"\n  ── Trial {trial_idx + 1}/{total_trials} ──")
            
            # Preparation phase
            print_cue("", "ready")
            time.sleep(1.0)
            
            # Cue phase — show the instruction
            if label == 0:
                print_cue("", "rest")
            else:
                print_cue("", "close")
            
            # Baseline: clear buffer, record brief baseline
            daq.board.get_board_data()  # flush
            time.sleep(config.BASELINE_SEC)
            
            # Imagery period: THIS is the data we keep
            daq.board.get_board_data()  # flush again
            
            # Progress bar during imagery
            steps = 20
            step_duration = trial_duration / steps
            for s in range(steps):
                bar = '█' * (s + 1) + '░' * (steps - s - 1)
                print(f"  [{bar}] {(s+1)*step_duration:.1f}s / {trial_duration}s",
                      end='\r')
                time.sleep(step_duration)
            print()
            
            # Retrieve the data from the imagery period
            raw_data = daq.board.get_board_data()
            eeg_channels = daq.eeg_channel_indices
            eeg_data = raw_data[eeg_channels, :]
            
            if len(config.EEG_CHANNELS) < len(eeg_channels):
                eeg_data = eeg_data[config.EEG_CHANNELS, :]
            
            # We need exactly trial_duration * fs samples
            expected_samples = int(trial_duration * config.SAMPLING_RATE)
            
            if eeg_data.shape[1] >= expected_samples:
                # Take the last `expected_samples` samples (most recent)
                epoch = eeg_data[:, -expected_samples:]
                all_epochs.append(epoch)
                all_labels.append(label)
                print(f"  ✓ Recorded {class_name} — {epoch.shape[1]} samples")
            else:
                print(f"  ⚠️  Short trial ({eeg_data.shape[1]} samples), keeping anyway")
                # Pad with zeros if needed
                padded = np.zeros((eeg_data.shape[0], expected_samples))
                padded[:, :eeg_data.shape[1]] = eeg_data
                all_epochs.append(padded)
                all_labels.append(label)
            
            # Rest between trials
            print(f"  Rest for {rest_duration}s...")
            time.sleep(rest_duration)
        
        # Convert to arrays
        epochs = np.array(all_epochs)   # shape: (n_trials, n_channels, n_samples)
        labels = np.array(all_labels)   # shape: (n_trials,)
        
        print(f"\n{'='*60}")
        print(f"  COLLECTION COMPLETE!")
        print(f"{'='*60}")
        print(f"  Total epochs: {len(epochs)}")
        print(f"  Shape: {epochs.shape}")
        
        # Preprocess
        print("\n  Preprocessing (filtering)...")
        for i in range(len(epochs)):
            epochs[i] = preprocess_pipeline(epochs[i])
        
        # Artifact rejection
        clean_epochs, clean_labels = reject_bad_epochs(epochs, labels)
        
        # Save
        os.makedirs(config.DATA_DIR, exist_ok=True)
        
        session_name = args.session or time.strftime('%Y%m%d_%H%M%S')
        save_path = os.path.join(config.DATA_DIR, f"training_{session_name}.npz")
        
        np.savez(save_path,
                 epochs=clean_epochs,
                 labels=clean_labels,
                 epochs_raw=epochs,
                 labels_raw=labels,
                 channel_names=config.CHANNEL_NAMES,
                 class_names=config.CLASS_NAMES,
                 sampling_rate=config.SAMPLING_RATE)
        
        print(f"\n  ✅ Data saved to: {save_path}")
        print(f"     Clean epochs: {len(clean_epochs)}")
        print(f"     Next: python 03_train_classifier.py --data {save_path}")
        
    except KeyboardInterrupt:
        print("\n\n  ⚠️  Collection interrupted by user.")
    
    finally:
        daq.stop()


if __name__ == '__main__':
    main()
