"""
config.py — Central configuration for the BCI system.
Edit THIS file to match your hardware setup. Everything else reads from here.
"""

# =============================================================================
# BOARD CONFIGURATION
# =============================================================================
# OpenBCI Cyton = board_id 0
# OpenBCI Cyton+Daisy = board_id 2
# Synthetic (for testing without hardware) = board_id -1
BOARD_ID = 0

# Serial port for the OpenBCI USB dongle
# Windows: "COM3", "COM4", etc. (check Device Manager)
# Mac:     "/dev/tty.usbserial-XXXXXXXX"
# Linux:   "/dev/ttyUSB0"
SERIAL_PORT = "/dev/ttyUSB0"

# Sampling rate (Cyton default = 250 Hz)
SAMPLING_RATE = 250

# =============================================================================
# ELECTRODE / CHANNEL CONFIGURATION
# =============================================================================
# Map Cyton channels (0-indexed) to 10-20 positions:
#   Channel 0 → C3   (left motor cortex)
#   Channel 1 → C4   (right motor cortex)
#   Channel 2 → FC3  (left frontal-central)
#   Channel 3 → FC4  (right frontal-central)
#   Channel 4 → CP3  (left centro-parietal)
#   Channel 5 → CP4  (right centro-parietal)
#   Channel 6 → Cz   (midline motor)
#   Channel 7 → FCz  (midline frontal-central)
#
# Minimal 4-channel: EEG_CHANNELS = [0, 1, 2, 3]
EEG_CHANNELS = [0, 1, 2, 3, 4, 5, 6, 7]
CHANNEL_NAMES = ["C3", "C4", "FC3", "FC4", "CP3", "CP4", "Cz", "FCz"]

# =============================================================================
# SIGNAL PROCESSING
# =============================================================================
BANDPASS_LOW = 8.0     # Hz — mu rhythm lower bound
BANDPASS_HIGH = 30.0   # Hz — beta rhythm upper bound
FILTER_ORDER = 4       # Butterworth filter order
NOTCH_FREQ = 60.0      # Hz (US=60, Europe/Asia=50)

# =============================================================================
# WINDOWING
# =============================================================================
WINDOW_LENGTH_SEC = 1.0
WINDOW_OVERLAP = 0.5
WINDOW_LENGTH = int(WINDOW_LENGTH_SEC * SAMPLING_RATE)
WINDOW_STEP = int(WINDOW_LENGTH * (1 - WINDOW_OVERLAP))

# =============================================================================
# MOTOR IMAGERY CLASSES
# =============================================================================
NUM_CLASSES = 2
CLASS_NAMES = ["REST", "CLOSE"]
# CLASS_NAMES = ["REST", "CLOSE", "PINCH"]  # 3-class

# =============================================================================
# TRAINING DATA COLLECTION
# =============================================================================
TRIALS_PER_CLASS = 30
TRIAL_DURATION_SEC = 4.0
REST_BETWEEN_SEC = 3.0
BASELINE_SEC = 2.0

# =============================================================================
# MACHINE LEARNING
# =============================================================================
CSP_N_COMPONENTS = 6
TEST_SPLIT = 0.2
RANDOM_SEED = 42

# =============================================================================
# REAL-TIME CONTROL
# =============================================================================
CLASSIFICATION_INTERVAL = 0.25
CONFIDENCE_THRESHOLD = 0.55
PREDICTION_SMOOTHING = 3

# =============================================================================
# ARDUINO / SERIAL CONTROL
# =============================================================================
ARDUINO_PORT = "/dev/ttyACM0"
ARDUINO_BAUD = 9600
HAND_OPEN_ANGLE = 10
HAND_CLOSE_ANGLE = 170
HAND_NEUTRAL_ANGLE = 90

# =============================================================================
# FILE PATHS
# =============================================================================
MODEL_DIR = "models"
DATA_DIR = "data"
MODEL_FILE = f"{MODEL_DIR}/csp_lda_model.joblib"
CSP_FILE = f"{MODEL_DIR}/csp_transform.joblib"
SCALER_FILE = f"{MODEL_DIR}/scaler.joblib"
