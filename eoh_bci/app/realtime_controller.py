from eoh_bci.streaming.txt_reader import stream_values
from eoh_bci.decision.threshold import decide_action
from eoh_bci.io.serial_control import SerialController
import time

FILE_PATH = "signals.txt"


def run():
    serial_ctrl = SerialController()

    print("🚀 Real-time EEG → Hand control started")

    for value in stream_values(FILE_PATH):
        action = decide_action(value)

        print(f"Signal: {value:.3f} → Action: {action}")

        if action:
            serial_ctrl.send(action)

        time.sleep(0.01)


if __name__ == "__main__":
    run()