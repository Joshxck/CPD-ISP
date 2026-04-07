import serial
import time

from cnc_capture import capture_image
from cpd_scan import send_gcode, wait_for_idle

PORT = "COM4"
BAUD = 115200

set_exposure = -7
set_frames = 15

with serial.Serial(PORT, BAUD, timeout=5) as ser:
        time.sleep(2)  # Wait for grbl to initialize
        ser.flushInput()
        
        # grblHAL sends a startup message — read and print it
        startup = ser.read(ser.in_waiting).decode(errors='ignore')
        print(f'Startup: {startup}')

        #send_gcode(ser, "$J=G91 A8 F100\n")

        wait_for_idle(ser)
        send_gcode(ser, "M8\n")
        path = capture_image(exposure=set_exposure, device=1, frames=set_frames, output_dir="./tests/output_images_5")
        send_gcode(ser, "M9\n")

        send_gcode(ser, "$J=G91 X220 F2000\n") # Move gantry

        wait_for_idle(ser)
        send_gcode(ser, "M8\n")
        path = capture_image(exposure=set_exposure, device=1, frames=set_frames, output_dir="./tests/output_images_5")
        send_gcode(ser, "M9\n")

        send_gcode(ser, "$J=G91 A8 F100\n")

        wait_for_idle(ser)
        send_gcode(ser, "M8\n")
        path = capture_image(exposure=set_exposure, device=1, frames=set_frames, output_dir="./tests/output_images_5")
        send_gcode(ser, "M9\n")

        send_gcode(ser, "$J=G91 X-220 F2000\n") # Move gantry

        wait_for_idle(ser)
        send_gcode(ser, "M8\n")
        path = capture_image(exposure=set_exposure, device=1, frames=set_frames, output_dir="./tests/output_images_5")
        send_gcode(ser, "M9\n")

        send_gcode(ser, "$J=G91 A8 F100\n")

        wait_for_idle(ser)
        send_gcode(ser, "M8\n")
        path = capture_image(exposure=set_exposure, device=1, frames=set_frames, output_dir="./tests/output_images_5")
        send_gcode(ser, "M9\n")

        send_gcode(ser, "$J=G91 X220 F2000\n") # Move gantry

        wait_for_idle(ser)
        send_gcode(ser, "M8\n")
        path = capture_image(exposure=set_exposure, device=1, frames=set_frames, output_dir="./tests/output_images_5")
        send_gcode(ser, "M9\n")

        send_gcode(ser, "$J=G91 A8 F100\n")

        wait_for_idle(ser)
        send_gcode(ser, "M8\n")
        path = capture_image(exposure=set_exposure, device=1, frames=set_frames, output_dir="./tests/output_images_5")
        send_gcode(ser, "M9\n")

        send_gcode(ser, "$J=G91 X-220 F2000\n") # Move gantry

        wait_for_idle(ser)
        send_gcode(ser, "M8\n")
        path = capture_image(exposure=set_exposure, device=1, frames=set_frames, output_dir="./tests/output_images_5")
        send_gcode(ser, "M9\n")

        send_gcode(ser, "$J=G91 A8 F100\n")

        wait_for_idle(ser)
        send_gcode(ser, "M8\n")
        path = capture_image(exposure=set_exposure, device=1, frames=set_frames, output_dir="./tests/output_images_5")
        send_gcode(ser, "M9\n")

        send_gcode(ser, "$J=G91 X220 F2000\n") # Move gantry

        wait_for_idle(ser)
        send_gcode(ser, "M8\n")
        path = capture_image(exposure=set_exposure, device=1, frames=set_frames, output_dir="./tests/output_images_5")
        send_gcode(ser, "M9\n")

        send_gcode(ser, "$J=G91 A8 F100\n")

        wait_for_idle(ser)
        send_gcode(ser, "M8\n")
        path = capture_image(exposure=set_exposure, device=1, frames=set_frames, output_dir="./tests/output_images_5")
        send_gcode(ser, "M9\n")

        send_gcode(ser, "$J=G91 X-220 F2000\n") # Move gantry

        wait_for_idle(ser)
        send_gcode(ser, "M8\n")
        path = capture_image(exposure=set_exposure, device=1, frames=set_frames, output_dir="./tests/output_images_5")
        send_gcode(ser, "M9\n")

        send_gcode(ser, "$J=G91 A8 F100\n")

        wait_for_idle(ser)
        send_gcode(ser, "M8\n")
        path = capture_image(exposure=set_exposure, device=1, frames=set_frames, output_dir="./tests/output_images_5")
        send_gcode(ser, "M9\n")

        send_gcode(ser, "$J=G91 X220 F2000\n") # Move gantry

        wait_for_idle(ser)
        send_gcode(ser, "M8\n")
        path = capture_image(exposure=set_exposure, device=1, frames=set_frames, output_dir="./tests/output_images_5")
        send_gcode(ser, "M9\n")

        send_gcode(ser, "$J=G91 A8 F100\n")

        wait_for_idle(ser)
        send_gcode(ser, "M8\n")
        path = capture_image(exposure=set_exposure, device=1, frames=set_frames, output_dir="./tests/output_images_5")
        send_gcode(ser, "M9\n")

        send_gcode(ser, "$J=G91 X-220 F2000\n") # Move gantry

        wait_for_idle(ser)
        send_gcode(ser, "M8\n")
        path = capture_image(exposure=set_exposure, device=1, frames=set_frames, output_dir="./tests/output_images_5")
        send_gcode(ser, "M9\n")

        send_gcode(ser, "$J=G91 A8 F100\n")

        wait_for_idle(ser)
        send_gcode(ser, "M8\n")
        path = capture_image(exposure=set_exposure, device=1, frames=set_frames, output_dir="./tests/output_images_5")
        send_gcode(ser, "M9\n")

        send_gcode(ser, "$J=G91 X220 F2000\n") # Move gantry

        wait_for_idle(ser)
        send_gcode(ser, "M8\n")
        path = capture_image(exposure=set_exposure, device=1, frames=set_frames, output_dir="./tests/output_images_5")
        send_gcode(ser, "M9\n")

        send_gcode(ser, "$J=G91 A8 F100\n")

        wait_for_idle(ser)
        send_gcode(ser, "M8\n")
        path = capture_image(exposure=set_exposure, device=1, frames=set_frames, output_dir="./tests/output_images_5")
        send_gcode(ser, "M9\n")

        send_gcode(ser, "$J=G91 X-220 F2000\n") # Move gantry

        wait_for_idle(ser)
        send_gcode(ser, "M8\n")
        path = capture_image(exposure=set_exposure, device=1, frames=set_frames, output_dir="./tests/output_images_5")
        send_gcode(ser, "M9\n")

        send_gcode(ser, "$J=G91 A8 F100\n")

        wait_for_idle(ser)
        send_gcode(ser, "M8\n")
        path = capture_image(exposure=set_exposure, device=1, frames=set_frames, output_dir="./tests/output_images_5")
        send_gcode(ser, "M9\n")

        send_gcode(ser, "$J=G91 X220 F2000\n") # Move gantry

        wait_for_idle(ser)
        send_gcode(ser, "M8\n")
        path = capture_image(exposure=set_exposure, device=1, frames=set_frames, output_dir="./tests/output_images_5")
        send_gcode(ser, "M9\n")

        send_gcode(ser, "$J=G91 A8 F100\n")

        wait_for_idle(ser)
        send_gcode(ser, "M8\n")
        path = capture_image(exposure=set_exposure, device=1, frames=set_frames, output_dir="./tests/output_images_5")
        send_gcode(ser, "M9\n")

        send_gcode(ser, "$J=G91 X-220 F2000\n") # Move gantry

        wait_for_idle(ser)
        send_gcode(ser, "M8\n")
        path = capture_image(exposure=set_exposure, device=1, frames=set_frames, output_dir="./tests/output_images_5")
        send_gcode(ser, "M9\n")

        send_gcode(ser, "$J=G91 A8 F100\n")