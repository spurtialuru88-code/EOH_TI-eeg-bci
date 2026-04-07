"""
serial_control.py — Arduino Communication for Hand Control.

Sends classified commands from Python to the Arduino over serial.

Protocol:
  Python sends single-character commands:
    'O' → Open hand
    'C' → Close hand
    'P' → Pinch
    'N' → Neutral (rest)
    'V' → Proportional value (followed by 0-100 integer)

The Arduino reads these and drives the servos accordingly.
"""

import serial
import time
import config


class HandController:
    """
    Controls the bionic hand via serial communication to Arduino.
    
    Usage:
        hand = HandController()
        hand.connect()
        hand.close_hand()
        hand.open_hand()
        hand.disconnect()
    
    Or with context manager:
        with HandController() as hand:
            hand.close_hand()
    """
    
    # Command mapping
    CMD_OPEN = b'O'
    CMD_CLOSE = b'C'
    CMD_PINCH = b'P'
    CMD_NEUTRAL = b'N'
    
    def __init__(self, port=None, baud=None):
        self.port = port or config.ARDUINO_PORT
        self.baud = baud or config.ARDUINO_BAUD
        self.serial = None
        self.current_state = "NEUTRAL"
    
    def connect(self):
        """Open serial connection to Arduino."""
        try:
            self.serial = serial.Serial(self.port, self.baud, timeout=1)
            time.sleep(2)  # Arduino resets on serial connect — wait for it
            print(f"[Hand] Connected to Arduino on {self.port}")
        except serial.SerialException as e:
            print(f"[Hand] ⚠️  Could not connect to Arduino: {e}")
            print(f"[Hand] Running in SIMULATION mode (commands printed to console)")
            self.serial = None
    
    def send_command(self, cmd):
        """
        Send a raw command byte to Arduino.
        
        Args:
            cmd: bytes, e.g. b'C' for close
        """
        if self.serial and self.serial.is_open:
            self.serial.write(cmd)
            self.serial.flush()
        else:
            # Simulation mode
            pass
    
    def open_hand(self):
        """Command: fully open the hand."""
        if self.current_state != "OPEN":
            self.send_command(self.CMD_OPEN)
            self.current_state = "OPEN"
    
    def close_hand(self):
        """Command: fully close the hand (grasp)."""
        if self.current_state != "CLOSE":
            self.send_command(self.CMD_CLOSE)
            self.current_state = "CLOSE"
    
    def pinch(self):
        """Command: pinch grip."""
        if self.current_state != "PINCH":
            self.send_command(self.CMD_PINCH)
            self.current_state = "PINCH"
    
    def neutral(self):
        """Command: return to neutral position."""
        if self.current_state != "NEUTRAL":
            self.send_command(self.CMD_NEUTRAL)
            self.current_state = "NEUTRAL"
    
    def set_proportional(self, value):
        """
        Send a proportional grip command (0-100).
        
        This maps to servo angle:
          0   → fully open
          100 → fully closed
        
        Args:
            value: int, 0-100
        """
        value = max(0, min(100, int(value)))
        cmd = f"V{value}\n".encode()
        self.send_command(cmd)
    
    def execute_prediction(self, prediction, confidence):
        """
        Convert ML prediction into hand action.
        
        Args:
            prediction: int, class label from classifier
            confidence: float, classifier confidence
        
        Only acts if confidence exceeds threshold (prevents jitter).
        """
        if confidence < config.CONFIDENCE_THRESHOLD:
            return  # Not confident enough — hold current position
        
        class_name = config.CLASS_NAMES[prediction]
        
        if class_name == "REST":
            self.open_hand()
        elif class_name == "CLOSE":
            self.close_hand()
        elif class_name == "PINCH":
            self.pinch()
    
    def disconnect(self):
        """Close serial connection."""
        if self.serial and self.serial.is_open:
            self.neutral()  # return to safe position
            time.sleep(0.1)
            self.serial.close()
            print("[Hand] Disconnected from Arduino.")
    
    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, *args):
        self.disconnect()
        return False
