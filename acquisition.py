"""
acquisition.py — EEG Data Acquisition using BrainFlow.

This module wraps the BrainFlow API to provide a clean interface for:
  - Connecting to your OpenBCI Cyton board
  - Starting/stopping the EEG data stream
  - Reading data in chunks or continuously
  - Handling the ring buffer

WHAT IS BRAINFLOW?
  BrainFlow is a hardware-agnostic library that talks to EEG boards.
  It handles the low-level serial communication, packet parsing, and
  converts raw ADC counts → microvolts for you. All data comes back
  as a numpy array with shape (num_channels_total, num_samples).

WHAT IS A RING BUFFER?
  BrainFlow stores incoming data in a circular buffer. When it's full,
  old data gets overwritten. You "drain" it by calling get_board_data()
  (removes data from buffer) or get_current_board_data(N) (peeks at
  the last N samples without removing).
"""

import numpy as np
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter
import time
import sys

# Import our config
sys.path.insert(0, '..')
import config


class EEGAcquisition:
    """
    Manages the connection to the OpenBCI board and data streaming.
    
    Usage:
        daq = EEGAcquisition()
        daq.connect()
        daq.start_stream()
        
        # In a loop:
        data = daq.get_recent_data(n_samples=250)  # last 1 second
        
        daq.stop()
    """
    
    def __init__(self, board_id=None, serial_port=None, use_synthetic=False):
        """
        Initialize the DAQ system.
        
        Args:
            board_id: BrainFlow board ID. None = use config.
            serial_port: Serial port string. None = use config.
            use_synthetic: If True, use synthetic board (no hardware needed).
                          GREAT for testing your code before the board arrives.
        """
        if use_synthetic:
            self.board_id = BoardIds.SYNTHETIC_BOARD.value  # -1
            serial_port = ""
        else:
            self.board_id = board_id if board_id is not None else config.BOARD_ID
        
        # Set up BrainFlow parameters
        self.params = BrainFlowInputParams()
        if serial_port or (not use_synthetic):
            self.params.serial_port = serial_port or config.SERIAL_PORT
        
        # Get channel info from BrainFlow (it knows the board's data format)
        self.eeg_channel_indices = BoardShim.get_eeg_channels(self.board_id)
        self.sampling_rate = BoardShim.get_sampling_rate(self.board_id)
        self.timestamp_channel = BoardShim.get_timestamp_channel(self.board_id)
        
        # The board object (created on connect)
        self.board = None
        self.is_streaming = False
        
        print(f"[DAQ] Board ID: {self.board_id}")
        print(f"[DAQ] EEG channels available: {len(self.eeg_channel_indices)}")
        print(f"[DAQ] Sampling rate: {self.sampling_rate} Hz")
    
    def connect(self):
        """
        Prepare the board session.
        This opens the serial connection and initializes the hardware.
        """
        print("[DAQ] Connecting to board...")
        self.board = BoardShim(self.board_id, self.params)
        self.board.prepare_session()
        print("[DAQ] Session prepared. Board is ready.")
    
    def start_stream(self, buffer_size=45000):
        """
        Start the data stream.
        
        Args:
            buffer_size: Ring buffer size in samples.
                        45000 = 3 minutes at 250 Hz. Increase if you
                        need longer recordings without draining.
        
        WHAT HAPPENS:
          The board starts sending packets over serial. BrainFlow
          parses them into the ring buffer in a background thread.
          Your main thread can read from the buffer whenever it wants.
        """
        if self.board is None:
            raise RuntimeError("Call connect() first!")
        
        self.board.start_stream(buffer_size)
        self.is_streaming = True
        print(f"[DAQ] Stream started (buffer: {buffer_size} samples)")
        
        # Let the buffer fill a bit before reading
        time.sleep(0.5)
    
    def get_all_data(self):
        """
        Get ALL data from the ring buffer and CLEAR it.
        
        Returns:
            np.ndarray of shape (n_eeg_channels, n_samples)
            Data is in MICROVOLTS (BrainFlow converts automatically).
        
        Use this for: recording training data (you want all of it).
        """
        raw = self.board.get_board_data()  # drains buffer
        # Extract only the EEG channels we care about
        eeg = raw[self.eeg_channel_indices, :]
        
        # If user specified a subset of channels in config, filter further
        if len(config.EEG_CHANNELS) < len(self.eeg_channel_indices):
            eeg = eeg[config.EEG_CHANNELS, :]
        
        return eeg
    
    def get_recent_data(self, n_samples=None):
        """
        Peek at the most recent N samples WITHOUT removing from buffer.
        
        Args:
            n_samples: Number of samples to get. 
                      None = 1 second worth of data.
        
        Returns:
            np.ndarray of shape (n_eeg_channels, n_samples)
        
        Use this for: real-time classification (you want a sliding window).
        """
        if n_samples is None:
            n_samples = self.sampling_rate  # 1 second
        
        raw = self.board.get_current_board_data(n_samples)
        eeg = raw[self.eeg_channel_indices, :]
        
        if len(config.EEG_CHANNELS) < len(self.eeg_channel_indices):
            eeg = eeg[config.EEG_CHANNELS, :]
        
        return eeg
    
    def get_data_chunk(self, duration_sec):
        """
        Record exactly `duration_sec` seconds of data.
        Blocks until the data is collected.
        
        Args:
            duration_sec: How many seconds to record.
        
        Returns:
            np.ndarray of shape (n_eeg_channels, n_samples)
        """
        n_samples_needed = int(duration_sec * self.sampling_rate)
        
        # Clear the buffer first so we get fresh data
        self.board.get_board_data()
        
        # Wait for enough data to accumulate
        time.sleep(duration_sec + 0.1)  # small buffer
        
        raw = self.board.get_board_data()
        eeg = raw[self.eeg_channel_indices, :]
        
        if len(config.EEG_CHANNELS) < len(self.eeg_channel_indices):
            eeg = eeg[config.EEG_CHANNELS, :]
        
        # Trim to exact length
        if eeg.shape[1] > n_samples_needed:
            eeg = eeg[:, :n_samples_needed]
        
        return eeg
    
    def check_signal_quality(self):
        """
        Quick signal quality check. Prints RMS per channel.
        Good EEG should be roughly 10-100 µV RMS.
        
        If you see:
          - < 1 µV: electrode not connected
          - > 200 µV: bad contact, movement artifact
          - ~10-50 µV: good signal
        """
        data = self.get_recent_data(self.sampling_rate * 2)  # 2 seconds
        
        print("\n[DAQ] Signal Quality Check (RMS in µV):")
        print("-" * 45)
        
        for i, ch_name in enumerate(config.CHANNEL_NAMES[:data.shape[0]]):
            rms = np.sqrt(np.mean(data[i, :] ** 2))
            
            if rms < 1:
                status = "⚠️  NO SIGNAL — check electrode"
            elif rms > 200:
                status = "⚠️  NOISY — reseat electrode"
            else:
                status = "✅ Good"
            
            print(f"  {ch_name:>5}: {rms:8.2f} µV  {status}")
        
        print("-" * 45)
    
    def stop(self):
        """Stop streaming and release the board."""
        if self.is_streaming:
            self.board.stop_stream()
            self.is_streaming = False
            print("[DAQ] Stream stopped.")
        
        if self.board is not None:
            self.board.release_session()
            self.board = None
            print("[DAQ] Session released.")
    
    def __enter__(self):
        """Context manager support: `with EEGAcquisition() as daq:`"""
        self.connect()
        self.start_stream()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False
