#!/usr/bin/env python3
"""Fail if any tracked file names `uv` as a dependency mechanism.

CLAUDE.md blocklists uv for project envs, and the incident that made it
a blocklist entry was Hailo integration — twenty ModuleNotFoundErrors
installed one at a time, no manifest, no lock. This gate keeps that
lesson enforced. Same behaviour as rf-detr-cpp/scripts/check_no_uv.py.
"""
import re, subprocess, sys
from pathlib import Path

TRIP = re.compile(r"(?:\buv\s+(?:run|add|pip|sync|lock|install)\b|\buvx\b|#\s*/// script\b|\{:uv,)")

def tracked_files():
    try:
        out = subprocess.check_output(["git", "ls-files"], text=True, stderr=subprocess.DEVNULL)
        return [Path(p) for p in out.splitlines() if p]
    except Exception:
        return [p for p in Path(".").rglob("*") if p.is_file() and ".pixi" not in p.parts]

def check(files):
    offenders = []
    for f in files:
        if f.suffix not in {".md", ".py", ".sh", ".toml", ".yaml", ".yml", ".txt"}: continue
        if f.name == "check_no_uv.py": continue  # self-exclude — the regex is data here
        try: text = f.read_text(errors="ignore")
        except Exception: continue
        if TRIP.search(text):
            offenders.append(f)
    return offenders

def self_test():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        bad = Path(td) / "bad.md"
        bad.write_text("Run this with `uv run --with torch script.py`")
        assert check([bad]) == [bad], "self-test control failed: uv line not detected"
        good = Path(td) / "good.md"
        good.write_text("Run this with `pixi run script`")
        assert check([good]) == [], "self-test control failed: pixi line falsely flagged"
    print("no-uv self-test PASS")

if __name__ == "__main__":
    if "--self-test" in sys.argv: self_test(); sys.exit(0)
    offenders = check(tracked_files())
    if offenders:
        print("no-uv gate FAIL — these files name uv:")
        for f in offenders: print(f"  {f}")
        sys.exit(1)
    print("no-uv gate PASS")
