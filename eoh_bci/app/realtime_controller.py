import time

from eoh_bci.io.signal_reader import SignalReader
from eoh_bci.decision.threshold import ThresholdClassifier
from eoh_bci.hardware.serial_bridge import SerialBridge
from eoh_bci.ui.live_plot import LivePlot
from eoh_bci.utils.calibration import calibrate


def run():
    reader = SignalReader("signals.txt")

    # AUTO CALIBRATION
    rest, flex = calibrate(reader)

    classifier = ThresholdClassifier(
        rest=rest,
        flex=flex,
        dead_zone=0.05
    )

    bridge = SerialBridge(port="/dev/ttyUSB0")  # CHANGE THIS
    plot = LivePlot()

    last_cmd = None

    print("\n=== RUNNING REAL-TIME BCI ===\n")

    while True:
        value = reader.read_latest()

        cmd = classifier.predict(value)

        # TERMINAL UI
        if value is not None:
            print(f"Signal: {value:.3f} → {cmd}")

       # SEND ONLY IF CHANGE
        if cmd and cmd != last_cmd:
            bridge.send(cmd)
            print(f"→ SENT: {cmd}")
            last_cmd = cmd

        # LIVE GRAPH
        plot.update(value)

        time.sleep(0.02)  # ~50Hz loop


if __name__ == "__main__":
    run()