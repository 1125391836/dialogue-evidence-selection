"""Run interactively, or use --stdin with a private input pipe. No key echo."""
import argparse
import getpass
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from context_bundles.credentials import load_deepseek, save_deepseek

p = argparse.ArgumentParser()
p.add_argument("--stdin", action="store_true")
args = p.parse_args()
key = sys.stdin.readline().strip() if args.stdin else getpass.getpass("DeepSeek API key: ")
save_deepseek(key)
if load_deepseek() != key:
    raise RuntimeError("Credential roundtrip failed")
print("Credential encrypted for the current Windows user; roundtrip verified.")
