import time
from eoh_bci.streaming.eeg_stream import EEGStream
from eoh_bci.model.pipeline import RealTimePipeline
from eoh_bci.hardware.serial_control import ArduinoController

class DemoController:
    def __init__(self, pipeline, arduino):
        self.pipeline = pipeline
        self.arduino = arduino

    def run(self, eeg_source):
        stream = EEGStream(window_size=256)

        for window in stream.stream_loop(eeg_source):
            if window is None:
                continue

            start = time.time()

            pred = self.pipeline.process(window)

            if pred == 1:
                self.arduino.open_hand()
            else:
                self.arduino.close_hand()

            latency = (time.time() - start) * 1000
            print(f"Latency: {latency:.2f} ms")