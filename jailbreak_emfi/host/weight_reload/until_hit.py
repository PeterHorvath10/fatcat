"""Follow the controller log; on the first 'Weight check ...: CHANGED' line print it as 'HIT: ...', stop the
controller (SIGINT, SIGKILL if it does not exit) and exit. Board resets are done by the controller itself.

Usage: until_hit.py CONTROLLER_LOG CONTROLLER_PID"""

import os
import signal
import sys
import time

POLL_S = 1.0
# SIGINT is lost while the controller retries its connection; kill it after this long
SIGINT_GRACE_S = 15


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def main():
    log, pid = sys.argv[1], int(sys.argv[2])
    pos = 0
    while alive(pid):
        with open(log, errors="replace") as f:
            f.seek(pos)
            new = f.read()
            pos = f.tell()
        for line in new.splitlines():
            if line.startswith("Weight check") and ": CHANGED" in line:
                print(time.strftime("%T"), "HIT:", line, flush=True)
                os.kill(pid, signal.SIGINT)
                for _ in range(SIGINT_GRACE_S):
                    time.sleep(1)
                    if not alive(pid):
                        return
                os.kill(pid, signal.SIGKILL)
                return
        time.sleep(POLL_S)
    print("controller gone", flush=True)


if __name__ == "__main__":
    main()
