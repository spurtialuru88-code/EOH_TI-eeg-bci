EOH TI-EEG BCI: Bionic Hand Controller
This project implements a Brain-Computer Interface (BCI) designed to control a bionic myoelectric robotic hand. It bridges the gap between high-fidelity neural data acquisition and robotic actuation.

🧠 System Overview

Unlike standard consumer BCI projects, this version leverages Texas Instruments (TI) software and hardware for neural signal evaluation. The system captures raw EEG data from the TI board, processes it through a custom pipeline, and translates neural intent into motor commands for a robotic hand.

🚀 Key Features

TI Hardware Integration: Direct interface with TI EEG evaluation boards for high-precision signal acquisition.

Real-time Processing: A Python-based BCI pipeline (eoh_bci) that handles filtering, feature extraction, and classification.

Live Visualization: Real-time plotting of brainwave data and calibration status using Matplotlib.

Modular Design: Separated modules for hardware bridging, signal preprocessing, and decision-making logic.
