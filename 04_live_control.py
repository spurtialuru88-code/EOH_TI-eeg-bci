#!/usr/bin/env python3
"""
04_live_control.py — Real-Time BCI Hand Control.

This is the main control loop that:
  1. Reads live EEG from the board
  2. Preprocesses in real-time
  3. Extracts features (CSP)
  4. Classifies motor imagery intent
  5. Sends commands to the bionic hand

This is what you'll run during the EOH demo.

Usage:
  python 04_live_control.py              # real board + Arduino
  python 04_live_control.py --synthetic  # test with synthetic data
  python 04_live_control.py --no-hand    # real EEG, no Arduino
"""

import sys
import argparse
import numpy as np
import time

sys.path.insert(0, '.')
import config
from bci.acquisition import EEGAcquisition
from bci.preprocessing import preprocess_pipeline, common_average_reference
from bci.classifier import BCIClassifier
from bci.serial_control import HandController


def main():
    parser = argparse.ArgumentParser(description='Real-time BCI hand control')
    parser.add_argument('--synthetic', action='store_true',
                       help='Use synthetic board')
    parser.add_argument('--no-hand', action='store_true',
                       help='Skip Arduino connection')
    args = parser.parse_args()
    
    print("=" * 60)
    print("  REAL-TIME BCI HAND CONTROL")
    print("=" * 60)
    
    # Load trained model
    print("\n  Loading classifier...")
    clf = BCIClassifier()
    try:
        clf.load()
    except FileNotFoundError:
        print("  ❌ No trained model found!")
        print("     Run 03_train_classifier.py first.")
        sys.exit(1)
    
    # Connect to EEG board
    print("\n  Connecting to EEG board...")
    daq = EEGAcquisition(use_synthetic=args.synthetic)
    daq.connect()
    daq.start_stream()
    
    # Connect to Arduino (hand)
    hand = None
    if not args.no_hand:
        print("\n  Connecting to Arduino...")
        hand = HandController()
        hand.connect()
    
    # Wait for stream to stabilize
    print("\n  Stabilizing stream...")
    time.sleep(2)
    daq.check_signal_quality()
    
    print(f"\n{'='*60}")
    print(f"  LIVE CONTROL ACTIVE")
    print(f"  Classification every {config.CLASSIFICATION_INTERVAL}s")
    print(f"  Confidence threshold: {config.CONFIDENCE_THRESHOLD}")
    print(f"  Smoothing window: {config.PREDICTION_SMOOTHING} predictions")
    print(f"  Press Ctrl+C to stop")
    print(f"{'='*60}\n")
    
    # Main real-time loop
    window_samples = config.WINDOW_LENGTH  # e.g., 250 for 1 second
    interval = config.CLASSIFICATION_INTERVAL
    
    prediction_count = 0
    class_counts = {c: 0 for c in range(config.NUM_CLASSES)}
    
    try:
        while True:
            loop_start = time.time()
            
            # 1. Get recent EEG data
            raw_data = daq.get_recent_data(n_samples=window_samples)
            
            # Check if we have enough data
            if raw_data.shape[1] < window_samples:
                time.sleep(interval)
                continue
            
            # 2. Preprocess
            processed = preprocess_pipeline(raw_data)
            processed = common_average_reference(processed)
            
            # 3. Classify (with smoothing)
            prediction, confidence, raw_pred, raw_conf = clf.predict_smoothed(processed)
            
            prediction_count += 1
            class_counts[prediction] += 1
            
            # 4. Determine action
            class_name = config.CLASS_NAMES[prediction]
            
            # 5. Send command to hand
            if hand is not None:
                hand.execute_prediction(prediction, confidence)
            
            # 6. Display status
            bar_len = int(confidence * 30)
            bar = '█' * bar_len + '░' * (30 - bar_len)
            
            action_icon = "✊" if class_name == "CLOSE" else "🖐️" if class_name == "REST" else "🤌"
            threshold_marker = "✓" if confidence >= config.CONFIDENCE_THRESHOLD else "✗"
            
            hand_state = hand.current_state if hand else "NO HAND"
            
            print(f"  {action_icon} {class_name:>6} [{bar}] "
                  f"{confidence:.0%} {threshold_marker}  "
                  f"Hand: {hand_state:>8}  "
                  f"(raw: {config.CLASS_NAMES[raw_pred]} {raw_conf:.0%})",
                  end='\r')
            
            # 7. Sleep to maintain timing
            elapsed = time.time() - loop_start
            sleep_time = max(0, interval - elapsed)
            time.sleep(sleep_time)
    
    except KeyboardInterrupt:
        print(f"\n\n{'='*60}")
        print(f"  SESSION ENDED")
        print(f"{'='*60}")
        print(f"  Total predictions: {prediction_count}")
        for c, count in class_counts.items():
            pct = count / max(1, prediction_count) * 100
            print(f"    {config.CLASS_NAMES[c]}: {count} ({pct:.1f}%)")
    
    finally:
        if hand:
            hand.disconnect()
        daq.stop()
        print("  ✅ All connections closed.")


if __name__ == '__main__':
    main()
