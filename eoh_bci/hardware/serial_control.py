import serial
import time

class ArduinoController:
    def __init__(self, port="/dev/ttyUSB0", baud=9600):
        self.ser = serial.Serial(port, baud, timeout=1)
        time.sleep(2)  # allow Arduino reset

    def send_command(self, command: str):
        self.ser.write((command + "\n").encode())

    def open_hand(self):
        self.send_command("OPEN")

    def close_hand(self):
        self.send_command("CLOSE")


class SerialController:
    def __init__(self, port="/dev/tty.usbserial-0001", baud=115200):
        self.ser = serial.Serial(port, baud, timeout=1)

    def send(self, command):
        if command:
            self.ser.write(command.encode())