import serial
import time
import os
import serial.tools.list_ports

BAUD = 115200

def find_port():
    env_port = os.getenv("CNC_PORT")
    if env_port:
        return env_port

    for p in serial.tools.list_ports.comports():
        if "Teensy" in p.description or "USB" in p.description:
            return p.device

    raise RuntimeError("No CNC device found")


def read_response(ser):
    while True:
        line = ser.readline().decode(errors='ignore').strip()
        if not line:
            continue

        print(f'<< {line}')

        if line.lower().startswith('ok') or 'error' in line.lower():
            return line


def send_gcode(ser, line):
    line = line.strip()
    if not line or line.startswith(';'):
        return None

    print(f'>> {line}')
    ser.write((line + '\n').encode())
    return read_response(ser)


def wait_for_idle(ser, poll_interval=0.2):
    while True:
        ser.write(b'?\n')
        line = ser.readline().decode(errors='ignore').strip()

        if line:
            print(f'   status: {line}')
            if line.startswith('<') and 'Idle' in line:
                break

        time.sleep(poll_interval)


def read_startup(ser, timeout=2):
    start = time.time()
    while time.time() - start < timeout:
        line = ser.readline().decode(errors='ignore').strip()
        if line:
            print(f'Startup: {line}')


def run_file(ser, filepath):
    with open(filepath, 'r') as f:
        for line in f:
            response = send_gcode(ser, line)
            if response and 'error' in response.lower():
                print(f'[ERROR] Halting: {response}')
                break


def main():
    port = find_port()

    with serial.Serial(port, BAUD, timeout=1) as ser:
        time.sleep(2)

        ser.reset_input_buffer()
        read_startup(ser)

        run_file(ser, 'my_program.gcode')
        wait_for_idle(ser)

        print('Done.')


if __name__ == '__main__':
    main()