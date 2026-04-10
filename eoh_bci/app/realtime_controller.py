from decision.threshold import decide_action
from hardware.serial_control import SerialController # Make sure your imports match your folder structure!
from io.txt_reader import stream_values # Make sure imports match!
import time

# CHANGE THIS to the actual folder where TI saves its files!
TI_DATA_FOLDER = r"C:\EEG"

def run():
    serial_ctrl = SerialController()

    print(f"🚀 Scanning {TI_DATA_FOLDER} for latest EEG data...")

    # Pass the folder, not the file
    for value in stream_values(TI_DATA_FOLDER):
        action = decide_action(value)

        print(f"Signal: {value:.3f} → Action: {action}")

        if action:
            serial_ctrl.send(action)

        time.sleep(0.01)

if __name__ == "__main__":
    run()