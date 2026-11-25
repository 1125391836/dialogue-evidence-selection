"""Editor entry point: click Run to perform offline verification by default.

Automatically use the project's .venv when launched with a different Python.
Select .venv explicitly in the editor for breakpoint debugging. Optional arguments are
the same as scripts/project.py, for example: test, next, campaign --stage status.
"""

import os
from pathlib import Path
import runpy
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parent
    environment = root / ".venv"
    interpreter = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not interpreter.is_file():
        print(f"Project interpreter is missing: {interpreter}\nSee README.md to restore the project environment.", file=sys.stderr)
        raise SystemExit(2)
    # Compare prefixes, not resolved executable paths: Unix venv binaries can be symlinks.
    if Path(sys.prefix).resolve() != environment.resolve():
        if os.environ.get("CONTEXT_BUNDLES_BOOTSTRAP_ROOT") == str(root):
            print("The project interpreter did not activate its virtual environment. Restore .venv before running.", file=sys.stderr)
            raise SystemExit(2)
        print(f"Switching to project interpreter: {interpreter}", flush=True)
        child_env = os.environ.copy()
        child_env.pop("PYTHONHOME", None)
        child_env["CONTEXT_BUNDLES_BOOTSTRAP_ROOT"] = str(root)
        completed = subprocess.run([str(interpreter), str(root / "run.py"), *sys.argv[1:]],
                                   cwd=root, env=child_env)
        raise SystemExit(completed.returncode)
    os.chdir(root)
    sys.path.insert(0, str(root / "src"))
    arguments = sys.argv[1:] or ["verify"]
    print(f"Interpreter: {sys.executable}", flush=True)
    print(f"Project: {root}", flush=True)
    print(f"Command: {' '.join(arguments)}", flush=True)
    sys.argv = [str(root / "scripts/project.py"), *arguments]
    runpy.run_path(sys.argv[0], run_name="__main__")


if __name__ == "__main__":
    main()
