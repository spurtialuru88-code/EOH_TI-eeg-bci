import time
import numpy as np
from collections import deque

class EEGStream:
    def __init__(self, window_size=256):
        self.buffer = deque(maxlen=window_size)

    def add_sample(self, sample):
        self.buffer.append(sample)

    def get_window(self):
        if len(self.buffer) < self.buffer.maxlen:
            return None
        return np.array(self.buffer)

    def stream_loop(self, data_source):
        for sample in data_source:
            self.add_sample(sample)
            yield self.get_window()