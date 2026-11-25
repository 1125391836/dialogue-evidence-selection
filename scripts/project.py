"""Convenient entry point that anchors all paths to the project root."""
import os
from pathlib import Path
import runpy
import sys

root = Path(__file__).resolve().parents[1]
os.chdir(root)
if len(sys.argv) < 2:
    print("Usage: python scripts/project.py {verify|next|campaign|audit|review|test} [arguments]")
    raise SystemExit(2)
command = sys.argv.pop(1)
if command == "binding":
    runpy.run_path(str(root / "scripts/binding_pilot.py"), run_name="__main__")
elif command == "recover":
    runpy.run_path(str(root / "scripts/replay_recovery.py"), run_name="__main__")
elif command == "campaign":
    runpy.run_path(str(root / "scripts/run_campaign.py"), run_name="__main__")
elif command == "audit":
    runpy.run_path(str(root / "scripts/audit_data.py"), run_name="__main__")
elif command == "review":
    runpy.run_path(str(root / "scripts/review_campaign.py"), run_name="__main__")
elif command == "verify":
    runpy.run_path(str(root / "scripts/verify_project.py"), run_name="__main__")
elif command == "next":
    runpy.run_path(str(root / "scripts/next_stage.py"), run_name="__main__")
elif command == "test":
    import unittest
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover("tests"))
    raise SystemExit(not result.wasSuccessful())
else:
    raise SystemExit("Unknown project command")
