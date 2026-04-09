import serial

class SerialBridge:
    def __init__(self, port, baudrate=115200):
        self.ser = serial.Serial(port, baudrate, timeout=1)

    def send(self, cmd):
        if cmd in ['R', 'F']:
            self.ser.write(cmd.encode())