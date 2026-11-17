"""Record the serial output of the Tumbller into a CSV file.

    pip install pyserial
    python tools/tumbller_logger.py --list
    python tools/tumbller_logger.py --port /dev/cu.usbserial-1410 --seconds 20 --out data/run01.csv
    python tools/tumbller_logger.py --port ... --seconds 12 --send Z1.5@3 --out data/zstep.csv
    python tools/tumbller_logger.py --port ... --baud 115200 --send g --out data/motor_step.csv

Works with both course sketches:
  Tumbller_Tuning   (9600 baud)    columns ms,angle,gyro,pwm,speed,integral
  Motor_Step_Test   (115200 baud)  columns ms,pwm,count      (use --send g)

Lines starting with '#' are comments from the robot; they are kept in the file
(tb.load_log skips them). A serial port can be opened by only one program at a
time: close the Arduino Serial Monitor before recording, and use --send for commands.
"""
import argparse
import sys
import time


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="list serial ports and exit")
    ap.add_argument("--port")
    ap.add_argument("--baud", type=int, default=9600)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--out", default="data/run.csv")
    ap.add_argument("--send", action="append", default=[],
                    help="command to send, optionally at a time: --send P60 --send Z1.5@3  "
                         "(Z1.5 three seconds after the recording starts)")
    a = ap.parse_args()

    try:
        import serial
        from serial.tools import list_ports
    except ImportError:
        sys.exit("pyserial is missing:  pip install pyserial")

    if a.list or not a.port:
        for p in list_ports.comports():
            print(p.device, "-", p.description)
        return

    with serial.Serial(a.port, a.baud, timeout=0.2) as ser, open(a.out, "w") as f:
        time.sleep(2.5)                  # opening the port resets the Arduino Nano
        ser.reset_input_buffer()
        pending = []                     # (seconds after start, command)
        for item in a.send:
            cmd, _, at = item.partition("@")
            pending.append((float(at) if at else 0.0, cmd))
        pending.sort()
        t0, n = time.time(), 0
        header_written = False
        while time.time() - t0 < a.seconds:
            while pending and time.time() - t0 >= pending[0][0]:
                ser.write((pending.pop(0)[1] + "\n").encode())
            line = ser.readline().decode(errors="ignore").strip()
            if not line:
                continue
            if line.startswith("ms,"):
                header_written = True
            elif (line[0].isdigit() or line[0] == "-") and not header_written:
                if line.count(",") == 5:
                    f.write("ms,angle,gyro,pwm,speed,integral\n")
                header_written = True
            f.write(line + "\n")
            n += 1
            if line.startswith("# done"):
                break
    print(f"wrote {n} lines to {a.out}")


if __name__ == "__main__":
    main()
