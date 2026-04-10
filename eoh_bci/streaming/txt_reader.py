import time

def read_latest_value(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()
        if not lines:
            return None
        return float(lines[-1].strip())  # LAST VALUE ONLY


def stream_values(file_path, delay=0.01):
    """
    Continuously yield latest EEG value
    """
    last_seen = None

    while True:
        try:
            val = read_latest_value(file_path)

            if val is not None and val != last_seen:
                last_seen = val
                yield val

        except:
            pass

        time.sleep(delay)  # 10ms = fast loop (~100Hz)