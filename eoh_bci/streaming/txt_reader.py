import time
import os
import glob

def get_newest_file(folder_path, extension="*.xls"):
    """Finds the most recently created file in a folder."""
    # Search for all txt files in the folder (change to *.csv if TI outputs CSV)
    search_pattern = os.path.join(folder_path, extension)
    list_of_files = glob.glob(search_pattern)
    
    if not list_of_files:
        return None
        
    # Return the file with the most recent modified time
    return max(list_of_files, key=os.path.getmtime)

def read_latest_value(file_path):
    try:
        with open(file_path, "r") as f:
            lines = f.readlines()
            if not lines:
                return None
            
            # Grab the last line, strip whitespace
            last_line = lines[-1].strip()
            
            # If the TI software outputs a CSV format (like: timestamp, channel1, channel2)
            # This splits by comma and takes the last column. If it's just raw numbers, it still works.
            val_str = last_line.split(',')[-1] 
            
            return float(val_str)
    except:
        return None

def stream_values(folder_path, delay=0.01):
    """Continuously yield latest EEG value from the NEWEST file"""
    last_seen = None

    while True:
        # 1. Constantly check what the newest file is
        latest_file = get_newest_file(folder_path, "*.txt") # Change to *.csv if needed
        
        if latest_file:
            # 2. Read the bottom value of that file
            val = read_latest_value(latest_file)

            if val is not None and val != last_seen:
                last_seen = val
                yield val

        time.sleep(delay)