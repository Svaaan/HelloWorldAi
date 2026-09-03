"""Free a port this project left occupied -- and nothing else.

This used to terminate whatever process held the port, whoever it belonged to.
Two things made that worse than it sounds.

It ran on import. The three calls at the bottom of the file fired the moment
anything imported the module, on ports 8100, 9100 and 3000 hardcoded, on top of
the configured ports app.py then asked for. Importing a helper killed processes.

And it matched on the port alone. A port is not ownership: anything can be
listening there. On a developer's machine port 3000 is one of the most
contested numbers there is, and the thing holding it was Docker Desktop's
backend, publishing a contributor's node dashboard. Terminating it took Docker
down with it -- which reads as Docker crashing, not as this project having
reached over and stopped it.

So the convenience is kept and the danger is not: a process is terminated only
when its command line shows it is one of ours. Anything else is reported and
left alone, which is the honest outcome -- the port really is busy, and the
person running the command is better placed to decide what to do about it than
this file is.
"""

import os
import sys

import psutil

# What one of our own processes looks like from the outside. uvicorn is started
# with the import path of the app it serves, so the app path is the signature.
OURS = (
    "src.app:app",
    "src.backend.coordinator:app",
    "backend.coordinator:app",
    "src.backend.node:app",
    "backend.node:app",
)


def _looks_like_ours(proc) -> bool:
    try:
        cmdline = " ".join(proc.cmdline())
    except (psutil.AccessDenied, psutil.ZombieProcess, psutil.NoSuchProcess):
        # Cannot see what it is, so cannot claim it. Docker's backend and most
        # system services land here, which is exactly the right side to fail on.
        return False

    return any(marker in cmdline for marker in OURS)


def kill_process_on_port(port):
    """Stop our own leftover server on `port`. Never anybody else's."""
    current_pid = os.getpid()

    for conn in psutil.net_connections(kind="inet"):
        if not conn.laddr or conn.laddr.port != port:
            continue
        if not conn.pid or conn.pid == current_pid:
            continue

        try:
            proc = psutil.Process(conn.pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

        if not _looks_like_ours(proc):
            # Plain ASCII on purpose: this prints to whatever console started
            # the process, and on a Windows terminal in cp1252 a warning sign
            # raised UnicodeEncodeError and took the startup down with it.
            print(
                f"[ports] {port} is held by {proc.name()} (pid {proc.pid}), "
                f"which is not part of this project. Leaving it alone -- stop "
                f"it yourself, or start on another port.",
                file=sys.stderr,
            )
            continue

        try:
            print(f"[ports] stopping our own leftover server on {port} "
                  f"(pid {proc.pid})")
            proc.terminate()
            proc.wait(timeout=3)
        except psutil.TimeoutExpired:
            print(f"[ports] pid {proc.pid} did not stop in time.",
                  file=sys.stderr)
        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            print(f"[ports] could not stop pid {proc.pid}: {e}",
                  file=sys.stderr)


# Nothing runs on import. The three unconditional calls that used to sit here
# meant that importing this module was itself destructive, and on ports the
# caller had not asked about.
