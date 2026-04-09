# NeuroGrip — EEG-Based BCI for Bionic Hand Control
## UIUC Engineering Open House (EOH) Project

---

## What This System Does

Translates **motor imagery EEG signals** into real-time bionic hand commands:
- **Imagine closing your hand** → hand closes
- **Rest / relax** → hand opens
- **Imagine left hand** → pinch grip (optional 3rd class)

---

## System Architecture

```
Your Brain (Motor Cortex)
        │
        ▼
 EEG Headset (OpenBCI Cyton, 8ch, 250 Hz)
        │  Electrodes at C3, C4, FC3, FC4 etc.
        ▼
 BrainFlow DAQ (Python)
        │  Raw EEG in microvolts
        ▼
 Preprocessing
        │  Bandpass 8-30 Hz, notch 60 Hz, artifact rejection
        ▼
 Feature Extraction
        │  CSP spatial filters + band power features
        ▼
 ML Classifier (CSP + LDA)
        │  Trained on YOUR motor imagery data
        ▼
 Serial Command
        │  "OPEN" / "CLOSE" / "PINCH"
        ▼
 Arduino / ESP32
        │
        ▼
 Servo Motors → Bionic Hand
```

---

## File Structure

```
bci_bionic_hand/
├── README.md                  ← You are here
├── requirements.txt           ← Python dependencies
├── config.py                  ← All configurable parameters
│
├── 01_test_connection.py      ← Step 1: Verify board connects
├── 02_collect_training.py     ← Step 2: Guided data collection
├── 03_train_classifier.py     ← Step 3: Train CSP+LDA model
├── 04_live_control.py         ← Step 4: Real-time hand control
├── 05_demo_dashboard.py       ← Step 5: EOH demo with live viz
│
├── bci/
│   ├── __init__.py
│   ├── acquisition.py         ← BrainFlow DAQ wrapper
│   ├── preprocessing.py       ← Filters, artifact rejection
│   ├── features.py            ← CSP + band power extraction
│   ├── classifier.py          ← ML training & prediction
│   └── serial_control.py      ← Arduino communication
│
├── models/                    ← Saved trained models go here
├── data/                      ← Recorded EEG sessions go here
└── arduino/
    └── hand_controller.ino    ← Arduino firmware
```

---

## Setup Instructions

### 1. Install Python Dependencies
```bash
pip install brainflow numpy scipy scikit-learn matplotlib pyserial joblib
```

### 2. Hardware Connections
- OpenBCI Cyton → USB dongle → your laptop
- Arduino → USB → your laptop (different port)
- Servos → Arduino PWM pins

### 3. Electrode Placement (10-20 System)
For motor imagery you need electrodes over motor cortex:
- **C3** (left motor cortex — controls right hand)
- **C4** (right motor cortex — controls left hand)
- **FC3, FC4** (frontal-central, supplementary motor area)
- **CP3, CP4** (centro-parietal, somatosensory)
- **Cz** (midline motor, leg area — useful reference)
- **Reference**: Earlobe (A1 or A2)
- **Ground**: AFz or FPz

### 4. Step-by-Step Workflow
```
python 01_test_connection.py          # Verify EEG board works
python 02_collect_training.py         # Record ~5 min of motor imagery
python 03_train_classifier.py         # Train ML model (~60-80% accuracy)
python 04_live_control.py             # Control the hand in real-time
python 05_demo_dashboard.py           # Full EOH demo with visualization
```

---

## Key Neuroscience Concepts

**Motor Imagery**: When you *imagine* moving your hand (without actually moving),
your motor cortex activates similarly to real movement. This causes:
- **Mu rhythm (8-13 Hz) suppression** over contralateral motor cortex
- **Beta desynchronization (13-30 Hz)** during imagery
- These changes are called **ERD** (Event-Related Desynchronization)

**CSP (Common Spatial Pattern)**: A spatial filter that finds the linear combination
of EEG channels that maximally separates two classes. It's the gold standard
for motor imagery BCI.

**LDA (Linear Discriminant Analysis)**: A classifier that finds the optimal
decision boundary between classes. Fast, robust, and works well with CSP features.

---

## Expected Performance
- **Accuracy**: 60-80% (this is GOOD for real-time EEG BCI)
- **Latency**: ~200-500 ms from thought to hand movement
- **Calibration time**: ~5 minutes of training data
- Accuracy improves with practice (neural plasticity!)
