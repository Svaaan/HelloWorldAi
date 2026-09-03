"""Freeing a port must never reach outside this project.

This helper used to terminate whatever process held the port. It ran on import,
on ports 8100, 9100 and 3000 hardcoded, before the caller had asked for
anything -- and it matched on the port alone.

A port is not ownership. On a development machine 3000 is one of the most
contested numbers there is, and the process holding it during this project's own
local runs was Docker Desktop's backend, publishing a contributor's node
dashboard. Terminating it took Docker down, which looks like Docker crashing
rather than like this project having stopped it.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from backend import terminate_port                              # noqa: E402


class FakeProc:
    def __init__(self, pid, name, cmdline):
        self.pid = pid
        self._name = name
        self._cmdline = cmdline
        self.terminated = False

    def name(self):
        return self._name

    def cmdline(self):
        return self._cmdline

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        return 0


def test_importing_it_does_nothing():
    """It used to kill three ports as a side effect of the import statement."""
    source = open(terminate_port.__file__, encoding="utf-8").read()

    body = source[source.index("def kill_process_on_port"):]
    after = body[body.index("\n\n\n"):] if "\n\n\n" in body else ""
    assert "kill_process_on_port(" not in after, (
        "something calls kill_process_on_port at module level again")


def test_a_process_that_is_not_ours_is_left_alone():
    docker = FakeProc(3560, "com.docker.backend.exe",
                      ["com.docker.backend.exe", "--watchdog"])
    assert terminate_port._looks_like_ours(docker) is False


def test_our_own_servers_are_recognised():
    for cmd in (
        ["python", "-m", "uvicorn", "src.app:app", "--port", "3000"],
        ["python", "-m", "uvicorn", "src.backend.coordinator:app"],
        ["python", "-m", "uvicorn", "src.backend.node:app"],
    ):
        assert terminate_port._looks_like_ours(FakeProc(1, "python.exe", cmd)), cmd


def test_a_process_we_cannot_inspect_is_not_claimed():
    """Access denied means unknown, and unknown must mean leave it."""
    import psutil

    class Opaque(FakeProc):
        def cmdline(self):
            raise psutil.AccessDenied(self.pid)

    assert terminate_port._looks_like_ours(
        Opaque(4, "System", [])) is False


def test_nothing_it_prints_can_break_a_windows_console():
    """A warning sign raised UnicodeEncodeError on a cp1252 terminal and took
    the whole startup down with it."""
    source = open(terminate_port.__file__, encoding="utf-8").read()

    for line in source.splitlines():
        if "print(" in line or line.strip().startswith('f"[ports]'):
            assert line.isascii(), f"non-ASCII in console output: {line.strip()}"
