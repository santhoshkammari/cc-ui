import os
import sys
import signal
import subprocess

PID_FILE = os.path.expanduser("~/.ccui.pid")


def start():
    if os.path.exists(PID_FILE):
        with open(PID_FILE) as f:
            pid = int(f.read().strip())
        try:
            os.kill(pid, 0)
            print(f"ccui already running (pid {pid})")
            return
        except ProcessLookupError:
            pass

    app_path = os.path.join(os.path.dirname(__file__), "app.py")
    proc = subprocess.Popen(
        [sys.executable, app_path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    with open(PID_FILE, "w") as f:
        f.write(str(proc.pid))
    print(f"ccui started (pid {proc.pid}) — http://127.0.0.1:7860")


def stop():
    if not os.path.exists(PID_FILE):
        print("ccui is not running")
        return

    with open(PID_FILE) as f:
        pid = int(f.read().strip())

    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
        print(f"ccui stopped (pid {pid})")
    except ProcessLookupError:
        print("ccui process not found (already stopped?)")
    finally:
        os.remove(PID_FILE)


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("start", "stop"):
        print("Usage: ccui start | ccui stop")
        sys.exit(1)

    if sys.argv[1] == "start":
        start()
    else:
        stop()
