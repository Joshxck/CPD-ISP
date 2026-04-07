import serial
import time

from cnc_capture import capture_image
from cpd_scan import send_gcode, wait_for_idle

PORT = "COM4"
BAUD = 115200

with serial.Serial(PORT, BAUD, timeout=5) as ser:
        time.sleep(2)  # Wait for grbl to initialize
        ser.flushInput()
        
        # grblHAL sends a startup message — read and print it
        startup = ser.read(ser.in_waiting).decode(errors='ignore')
        print(f'Startup: {startup}')

        send_gcode(ser, "M8\n")

        #send_gcode(ser, "$J=G91 A10 F500\n")

        wait_for_idle(ser)

        path = capture_image(exposure=-8, device=1, frames=30, output_dir="./tests/output_images_2")

        send_gcode(ser, "M9\n")




