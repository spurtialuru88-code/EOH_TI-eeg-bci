import os

class SignalReader:
    def __init__(self, filepath):
        self.filepath = filepath
        self.last_size = 0

    def read_latest(self):
        try:
            if not os.path.exists(self.filepath):
                return None

            with open(self.filepath, "r") as f:
                lines = f.readlines()
                if not lines:
                    return None

                return float(lines[-1].strip())

        except:
            return None