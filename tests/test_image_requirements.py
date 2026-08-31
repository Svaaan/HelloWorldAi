"""Each image's requirements file has to cover what that image actually imports.

A contributor installed a node, both containers started, and the dashboard
crash-looped on:

    File "/app/src/backend/routes/artifacts.py", line 19
    ModuleNotFoundError: No module named 'bson'

Three separate things had to be true for that to happen, and none of them was
visible from the source:

  * src/app.py imported the coordinator app at module level, so that
    run_coordinator() could reach it. Importing the dashboard therefore
    imported the coordinator, its routes, and its database driver.
  * the dashboard was built from requirements-node.txt, which has no driver in
    it, because a node never talks to a database.
  * nothing anywhere compared the two.

Every test passed. The image built. It failed on a stranger's machine, at the
one moment they were trying the project for the first time.

So: for each image, walk the imports reachable from its entry point and check
the requirements file it is built with declares them. Imports inside functions
are deliberately not counted -- that is what makes a lazy import a real fix
rather than a way of hiding a missing dependency, and both run_node() and
run_coordinator() rely on it.

This also pins jinja2, which for a long time was installed only because torch
depends on it. Every template this project renders was leaning on a transitive
dependency of PyTorch.
"""

import ast
import functools
import os
import sys

import pytest

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, ".."))
SRC = os.path.join(ROOT, "src")

# (entry point, the requirements file the image installs, what the image is)
IMAGES = [
    ("app.py", "requirements-dashboard.txt",
     "the dashboard: a proxy and a template renderer"),
    (os.path.join("backend", "coordinator.py"), "requirements.txt",
     "the coordinator: the server image, which the dashboard also runs on"),
    (os.path.join("backend", "node.py"), "requirements-node.txt",
     "the node agent: the only one that needs a graphics card"),
]

# Importing a submodule can need a package the parent does not. This is how
# jinja2 went unlisted for so long: the import reads
# `from fastapi.templating import Jinja2Templates`, which looks like fastapi
# and is not.
IMPLIED_BY_SUBMODULE = {
    "fastapi.templating": "jinja2",
}

# Where a module name and its package name differ and the installed metadata
# cannot say so -- only reached for packages absent from this environment.
FALLBACK_PACKAGE = {
    "dotenv": "python-dotenv",
    "bson": "pymongo",
    "gridfs": "pymongo",
    "yaml": "pyyaml",
    "PIL": "pillow",
}


def normalize(name):
    """pip treats runs of -, _ and . as the same character, and ignores case."""
    out = name.strip().lower()
    for ch in "-_.":
        out = out.replace(ch, "-")
    return out


@functools.lru_cache(maxsize=1)
def installed_modules():
    """Top-level module name -> the distributions providing it.

    Cached because it walks every package in the environment, and this one has
    torch in it. Without the cache the lookups below took a minute and a half.
    """
    try:
        from importlib.metadata import packages_distributions
        return packages_distributions()
    except Exception:
        return {}


def module_to_package(module):
    """The distribution providing a top-level module, named as pip names it."""
    providers = installed_modules().get(module)
    if providers:
        return normalize(sorted(providers)[0])
    return normalize(FALLBACK_PACKAGE.get(module, module))


def declared(requirements_file):
    """The package names a requirements file asks for."""
    path = os.path.join(ROOT, requirements_file)
    names = set()
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.split("#", 1)[0].strip()
            if not line or line.startswith("-"):
                continue
            for separator in ("==", ">=", "<=", "~=", ">", "<", "[", ";"):
                line = line.split(separator, 1)[0]
            if line.strip():
                names.add(normalize(line))
    return names


def local_path(module):
    """The file backing a module inside src/, or None if it is third-party."""
    base = os.path.join(SRC, module.replace(".", os.sep))
    for candidate in (base + ".py", os.path.join(base, "__init__.py")):
        if os.path.exists(candidate):
            return candidate
    return None


def top_level_imports(path):
    """What this file imports at module level, and nothing deeper.

    Module level means it runs on import. An import inside a function body does
    not, which is the whole point of moving one there.
    """
    with open(path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read(), filename=path)

    found = []

    def walk(nodes, inside_function):
        for node in nodes:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                walk(node.body, True)
            elif isinstance(node, ast.Import):
                if not inside_function:
                    found.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                # level > 0 is a relative import: local by definition.
                if not inside_function and node.module and node.level == 0:
                    found.append(node.module)
            elif isinstance(node, (ast.If, ast.Try, ast.With, ast.For,
                                   ast.While, ast.ClassDef)):
                # A conditional import still runs on import.
                for field in ("body", "orelse", "finalbody", "handlers"):
                    branch = getattr(node, field, None)
                    if not branch:
                        continue
                    walk([n for n in branch
                          if not isinstance(n, ast.ExceptHandler)], inside_function)
                    for handler in branch:
                        if isinstance(handler, ast.ExceptHandler):
                            walk(handler.body, inside_function)

    walk(tree.body, False)
    return found


def reachable_third_party(entry):
    """Every third-party package that importing this entry point pulls in."""
    seen_files = set()
    packages = {}                       # package -> the module that asked for it
    queue = [os.path.join(SRC, entry)]

    while queue:
        path = queue.pop()
        if path in seen_files:
            continue
        seen_files.add(path)

        for module in top_level_imports(path):
            if module in IMPLIED_BY_SUBMODULE:
                packages.setdefault(normalize(IMPLIED_BY_SUBMODULE[module]), module)

            local = local_path(module)
            if local:
                queue.append(local)
                continue

            root = module.split(".")[0]
            if root in sys.stdlib_module_names or root in ("src", "tests"):
                continue
            packages.setdefault(module_to_package(root), module)

    return packages


@pytest.mark.parametrize("entry,requirements,description", IMAGES)
def test_requirements_cover_what_the_image_imports(entry, requirements, description):
    needed = reachable_third_party(entry)
    have = declared(requirements)

    missing = {pkg: source for pkg, source in needed.items() if pkg not in have}

    assert not missing, (
        "%s\n"
        "  entry point:  src/%s\n"
        "  built with:   %s\n\n"
        "  imports these, and %s does not list them:\n%s\n\n"
        "  The image will build and then fail to start. Either add the package "
        "to %s, or move the import inside the function that needs it."
        % (description, entry.replace(os.sep, "/"), requirements, requirements,
           "\n".join("    %-22s (imported as %s)" % (pkg, source)
                     for pkg, source in sorted(missing.items())),
           requirements)
    )


def test_the_dashboard_stays_small():
    """The dashboard proxies HTTP. It has no business shipping a training stack.

    It was built from requirements-node.txt once, which put torch, numpy and the
    NVIDIA libraries into a multi-gigabyte image for a service that forwards
    requests. This is here so that stops being an easy mistake to repeat.
    """
    heavy = {"torch", "numpy", "pynvml", "nvidia-ml-py", "motor", "pymongo",
             "safetensors", "cryptography"}
    listed = declared("requirements-dashboard.txt")
    unexpected = sorted(listed & heavy)

    assert not unexpected, (
        "requirements-dashboard.txt lists %s.\n"
        "The dashboard forwards HTTP requests and renders templates -- it holds "
        "no data and trains nothing. If it genuinely needs one of these now, "
        "something about its job has changed enough to be worth saying out loud "
        "here." % ", ".join(unexpected)
    )
