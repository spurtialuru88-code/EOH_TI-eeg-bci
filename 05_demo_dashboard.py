#!/usr/bin/env python3
"""
05_demo_dashboard.py — EOH Demo Dashboard with Live Visualization.

This is the SHOWPIECE for Engineering Open House.
It combines:
  - Live EEG waveform display
  - Real-time power spectrum
  - Classification output with confidence
  - Hand state indicator
  - Running accuracy tracker

RUNS IN FULLSCREEN — put this on a monitor next to the bionic hand.

Usage:
  python 05_demo_dashboard.py              # real board + Arduino
  python 05_demo_dashboard.py --synthetic  # demo mode (no hardware)
"""

import sys
import argparse
import numpy as np
import time
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for live updates
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.signal import welch
from collections import deque

sys.path.insert(0, '.')
import config
from bci.acquisition import EEGAcquisition
from bci.preprocessing import preprocess_pipeline, common_average_reference
from bci.classifier import BCIClassifier
from bci.serial_control import HandController


class DemoDashboard:
    """
    Real-time visualization dashboard for EOH.
    
    Layout:
    ┌───────────────────────────┬──────────────┐
    │   Live EEG (4 channels)   │  Power       │
    │                           │  Spectrum    │
    ├───────────────────────────┼──────────────┤
    │   Classification History  │  Current     │
    │                           │  State       │
    └───────────────────────────┴──────────────┘
    """
    
    def __init__(self, daq, clf, hand=None):
        self.daq = daq
        self.clf = clf
        self.hand = hand
        
        # Data buffers
        self.eeg_buffer = deque(maxlen=config.SAMPLING_RATE * 5)  # 5 sec display
        self.prediction_history = deque(maxlen=100)
        self.confidence_history = deque(maxlen=100)
        
        # Current state
        self.current_prediction = 0
        self.current_confidence = 0.5
        
        # Setup plot
        self._setup_plot()
    
    def _setup_plot(self):
        """Create the dashboard layout."""
        plt.ion()
        
        self.fig = plt.figure(figsize=(16, 9))
        self.fig.patch.set_facecolor('#1a1a2e')
        
        gs = GridSpec(2, 3, figure=self.fig, hspace=0.3, wspace=0.3)
        
        # Top-left: Live EEG
        self.ax_eeg = self.fig.add_subplot(gs[0, :2])
        self.ax_eeg.set_facecolor('#16213e')
        self.ax_eeg.set_title('Live EEG Signal', color='white', fontsize=14,
                             fontweight='bold')
        self.ax_eeg.tick_params(colors='white')
        
        # Top-right: Power spectrum
        self.ax_psd = self.fig.add_subplot(gs[0, 2])
        self.ax_psd.set_facecolor('#16213e')
        self.ax_psd.set_title('Power Spectrum', color='white', fontsize=14,
                             fontweight='bold')
        self.ax_psd.tick_params(colors='white')
        
        # Bottom-left: Prediction history
        self.ax_history = self.fig.add_subplot(gs[1, :2])
        self.ax_history.set_facecolor('#16213e')
        self.ax_history.set_title('Classification History', color='white',
                                 fontsize=14, fontweight='bold')
        self.ax_history.tick_params(colors='white')
        
        # Bottom-right: Current state
        self.ax_state = self.fig.add_subplot(gs[1, 2])
        self.ax_state.set_facecolor('#16213e')
        self.ax_state.set_title('Hand Command', color='white', fontsize=14,
                               fontweight='bold')
        self.ax_state.set_xlim(0, 1)
        self.ax_state.set_ylim(0, 1)
        self.ax_state.set_xticks([])
        self.ax_state.set_yticks([])
        
        # Main title
        self.fig.suptitle(
            'NeuroGrip — EEG-Based Bionic Hand Control',
            fontsize=18, fontweight='bold', color='#00d4ff',
            y=0.98
        )
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show(block=False)
        plt.pause(0.01)
    
    def update(self):
        """One update cycle: read EEG → classify → update plots."""
        
        # 1. Get data
        raw = self.daq.get_recent_data(n_samples=config.WINDOW_LENGTH)
        
        if raw.shape[1] < config.WINDOW_LENGTH:
            return
        
        # 2. Preprocess
        processed = preprocess_pipeline(raw)
        processed = common_average_reference(processed)
        
        # 3. Classify
        pred, conf, raw_pred, raw_conf = self.clf.predict_smoothed(processed)
        self.current_prediction = pred
        self.current_confidence = conf
        
        self.prediction_history.append(pred)
        self.confidence_history.append(conf)
        
        # 4. Send to hand
        if self.hand:
            self.hand.execute_prediction(pred, conf)
        
        # 5. Update plots
        self._update_eeg_plot(processed)
        self._update_psd_plot(processed)
        self._update_history_plot()
        self._update_state_plot()
        
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.001)
    
    def _update_eeg_plot(self, data):
        """Update the live EEG waveform."""
        self.ax_eeg.clear()
        self.ax_eeg.set_facecolor('#16213e')
        
        n_channels = min(4, data.shape[0])  # show max 4 channels
        t = np.arange(data.shape[1]) / config.SAMPLING_RATE
        
        colors = ['#00d4ff', '#ff6b6b', '#51cf66', '#ffd43b']
        
        for ch in range(n_channels):
            offset = ch * 80  # spread channels vertically
            name = config.CHANNEL_NAMES[ch] if ch < len(config.CHANNEL_NAMES) else f"Ch{ch}"
            self.ax_eeg.plot(t, data[ch] + offset, color=colors[ch],
                            linewidth=0.8, label=name)
        
        self.ax_eeg.set_xlabel('Time (s)', color='white')
        self.ax_eeg.set_ylabel('Amplitude (µV)', color='white')
        self.ax_eeg.legend(loc='upper right', fontsize=8,
                          facecolor='#16213e', edgecolor='white',
                          labelcolor='white')
        self.ax_eeg.set_title('Live EEG Signal', color='white',
                             fontsize=14, fontweight='bold')
        self.ax_eeg.tick_params(colors='white')
    
    def _update_psd_plot(self, data):
        """Update the power spectrum plot."""
        self.ax_psd.clear()
        self.ax_psd.set_facecolor('#16213e')
        
        # Average PSD across channels
        psds = []
        for ch in range(min(4, data.shape[0])):
            f, psd = welch(data[ch], fs=config.SAMPLING_RATE,
                          nperseg=min(config.SAMPLING_RATE // 2, data.shape[1]))
            psds.append(psd)
        
        mean_psd = np.mean(psds, axis=0)
        
        self.ax_psd.semilogy(f, mean_psd, color='#00d4ff', linewidth=1.5)
        
        # Highlight mu and beta bands
        mu_idx = np.where((f >= 8) & (f <= 13))[0]
        beta_idx = np.where((f >= 13) & (f <= 30))[0]
        
        if len(mu_idx) > 0:
            self.ax_psd.fill_between(f[mu_idx], mean_psd[mu_idx],
                                    alpha=0.3, color='#51cf66', label='Mu (8-13 Hz)')
        if len(beta_idx) > 0:
            self.ax_psd.fill_between(f[beta_idx], mean_psd[beta_idx],
                                    alpha=0.3, color='#ffd43b', label='Beta (13-30 Hz)')
        
        self.ax_psd.set_xlim(1, 45)
        self.ax_psd.set_xlabel('Frequency (Hz)', color='white')
        self.ax_psd.set_ylabel('Power', color='white')
        self.ax_psd.set_title('Power Spectrum', color='white',
                             fontsize=14, fontweight='bold')
        self.ax_psd.legend(loc='upper right', fontsize=7,
                          facecolor='#16213e', edgecolor='white',
                          labelcolor='white')
        self.ax_psd.tick_params(colors='white')
    
    def _update_history_plot(self):
        """Update the classification history."""
        self.ax_history.clear()
        self.ax_history.set_facecolor('#16213e')
        
        if len(self.prediction_history) > 0:
            preds = list(self.prediction_history)
            confs = list(self.confidence_history)
            t = range(len(preds))
            
            colors = ['#51cf66' if p == 0 else '#ff6b6b' for p in preds]
            
            self.ax_history.bar(t, confs, color=colors, width=1.0, alpha=0.8)
            self.ax_history.axhline(config.CONFIDENCE_THRESHOLD, color='white',
                                   linestyle='--', alpha=0.5, label='Threshold')
            
            self.ax_history.set_ylim(0, 1)
            self.ax_history.set_ylabel('Confidence', color='white')
            self.ax_history.set_xlabel('Prediction #', color='white')
        
        self.ax_history.set_title('Classification History (green=REST, red=CLOSE)',
                                 color='white', fontsize=14, fontweight='bold')
        self.ax_history.tick_params(colors='white')
    
    def _update_state_plot(self):
        """Update the current state indicator."""
        self.ax_state.clear()
        self.ax_state.set_facecolor('#16213e')
        self.ax_state.set_xlim(0, 1)
        self.ax_state.set_ylim(0, 1)
        self.ax_state.set_xticks([])
        self.ax_state.set_yticks([])
        
        class_name = config.CLASS_NAMES[self.current_prediction]
        
        if class_name == "CLOSE":
            emoji = "✊"
            color = '#ff6b6b'
        elif class_name == "REST":
            emoji = "🖐️"
            color = '#51cf66'
        else:
            emoji = "🤌"
            color = '#ffd43b'
        
        # Big state indicator
        self.ax_state.text(0.5, 0.65, emoji, fontsize=60,
                          ha='center', va='center',
                          transform=self.ax_state.transAxes)
        
        self.ax_state.text(0.5, 0.35, class_name, fontsize=28,
                          ha='center', va='center', color=color,
                          fontweight='bold',
                          transform=self.ax_state.transAxes)
        
        self.ax_state.text(0.5, 0.15, f'{self.current_confidence:.0%}',
                          fontsize=18, ha='center', va='center',
                          color='white', alpha=0.7,
                          transform=self.ax_state.transAxes)
        
        self.ax_state.set_title('Hand Command', color='white',
                               fontsize=14, fontweight='bold')


def main():
    parser = argparse.ArgumentParser(description='EOH Demo Dashboard')
    parser.add_argument('--synthetic', action='store_true')
    parser.add_argument('--no-hand', action='store_true')
    args = parser.parse_args()
    
    print("=" * 60)
    print("  NEURGRIP — EOH DEMO DASHBOARD")
    print("=" * 60)
    
    # Load classifier
    clf = BCIClassifier()
    try:
        clf.load()
    except FileNotFoundError:
        print("  ❌ No trained model! Run 03_train_classifier.py first.")
        sys.exit(1)
    
    # Connect EEG
    daq = EEGAcquisition(use_synthetic=args.synthetic)
    daq.connect()
    daq.start_stream()
    time.sleep(2)
    
    # Connect hand
    hand = None
    if not args.no_hand:
        hand = HandController()
        hand.connect()
    
    # Create dashboard
    dashboard = DemoDashboard(daq, clf, hand)
    
    print("\n  Dashboard running! Press Ctrl+C to stop.\n")
    
    try:
        while True:
            dashboard.update()
            time.sleep(config.CLASSIFICATION_INTERVAL)
    
    except KeyboardInterrupt:
        print("\n\n  Demo ended.")
    
    finally:
        if hand:
            hand.disconnect()
        daq.stop()
        plt.close('all')


if __name__ == '__main__':
    main()
