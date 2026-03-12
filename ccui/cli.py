import os
import sys
import signal
import subprocess

PID_FILE = os.path.expanduser("~/.ccui.pid")
PORT = 8000


def start():
    if os.path.exists(PID_FILE):
        with open(PID_FILE) as f:
            pid = int(f.read().strip())
        try:
            os.kill(pid, 0)
            print(f"ccui already running (pid {pid}) — http://127.0.0.1:{PORT}")
            return
        except ProcessLookupError:
            os.remove(PID_FILE)

    backend_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "backend.py")
    proc = subprocess.Popen(
        [sys.executable, backend_path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    with open(PID_FILE, "w") as f:
        f.write(str(proc.pid))
    print(f"ccui started (pid {proc.pid}) — http://127.0.0.1:{PORT}")


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
