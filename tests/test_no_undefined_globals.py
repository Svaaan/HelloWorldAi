"""Every global a function reaches for has to exist.

This exists because of a real mistake. Dead code was being removed from
node.py -- an obsolete push-based task queue -- and the deletion ran from the
`task_queue` declaration down to the next one, quietly taking `task_logs` with
it. `task_logs` was still used by the live training path. The file still
parsed, still imported, and all 409 tests still passed, because no test imports
node.py at all. The failure only appeared when a real job ran on a real node:

    NameError: name 'task_logs' is not defined

A syntax check cannot catch that; Python resolves globals at call time, so a
function referring to a name that no longer exists is perfectly valid code
right up until it runs.

This reads the bytecode instead. Every LOAD_GLOBAL is a name a function will
look up in the module when it runs, so each one must be defined somewhere at
module level, imported, or be a builtin. Reading the compiled form rather than
importing means no module here needs a GPU, a database or a network to be
checked.
"""

import builtins
import dis
import os
import sys

import pytest

HERE = os.path.dirname(__file__)
SRC = os.path.join(HERE, "..", "src")

BUILTINS = set(dir(builtins))


def python_files():
    for folder, _dirs, files in os.walk(SRC):
        if "__pycache__" in folder:
            continue
        for name in sorted(files):
            if name.endswith(".py"):
                yield os.path.join(folder, name)


def module_level_names(code):
    """Everything the module binds at import: assignments, defs, imports."""
    names = set()
    for instr in dis.get_instructions(code):
        if instr.opname in ("STORE_NAME", "IMPORT_NAME", "IMPORT_FROM",
                            "STORE_GLOBAL", "STORE_FAST"):
            names.add(instr.argval)
    return names


def global_loads(code, _seen=None):
    """(name, qualified_function) for every global a nested function reads."""
    found = []
    stack = [(code, code.co_name)]
    while stack:
        current, where = stack.pop()
        for instr in dis.get_instructions(current):
            if instr.opname in ("LOAD_GLOBAL", "STORE_GLOBAL", "DELETE_GLOBAL"):
                found.append((instr.argval, where))
        for const in current.co_consts:
            if hasattr(const, "co_name"):
                stack.append((const, f"{where}.{const.co_name}"))
    return found


FILES = sorted(python_files())


def test_there_are_files_to_check():
    # A guard on the walker: an empty list would make every check below pass
    # for the wrong reason.
    assert len(FILES) >= 25, f"only found {len(FILES)} files; the walker is broken"


@pytest.mark.parametrize("path", FILES, ids=[
    os.path.relpath(p, SRC).replace("\\", "/") for p in FILES])
def test_every_global_a_function_uses_exists(path):
    with open(path, encoding="utf-8") as handle:
        source = handle.read()

    code = compile(source, path, "exec")
    defined = module_level_names(code) | BUILTINS

    missing = {}
    for name, where in global_loads(code):
        if name not in defined:
            missing.setdefault(name, where)

    assert not missing, (
        f"{os.path.relpath(path, SRC)} uses names that are not defined "
        f"anywhere at module level: "
        + ", ".join(f"{n} (in {w})" for n, w in sorted(missing.items()))
    )
