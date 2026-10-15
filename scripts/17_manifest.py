"""Write MANIFEST.md (size + SHA-256 of every file in the project), or verify the files against it (--verify)."""
import hashlib
import os
import re
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
SKIP = {"MANIFEST.md"}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def files():
    for d, _, fs in os.walk(ROOT):
        if "__pycache__" in d:
            continue
        for f in sorted(fs):
            p = os.path.join(d, f)
            rel = os.path.relpath(p, ROOT)
            if rel not in SKIP:
                yield rel


def main():
    if "--verify" in sys.argv:
        rows = re.findall(r"^\| `([^`]+)` \| (\d+) \| ([0-9a-f]{64}) \|$", open(os.path.join(ROOT, "MANIFEST.md")).read(), re.M)
        bad = [(r, "missing") for r, _, _ in rows if not os.path.exists(os.path.join(ROOT, r))]
        bad += [(r, "changed") for r, n, h in rows if os.path.exists(os.path.join(ROOT, r)) and sha256(os.path.join(ROOT, r)) != h]
        print(f"{len(rows)} files checked, {len(bad)} problems"); [print("  ", b) for b in bad]
        return 1 if bad else 0
    lines = ["# MANIFEST", "", f"Generated {time.strftime('%Y-%m-%d %H:%M')} by scripts/17_manifest.py. "
             "Verify with `python scripts/17_manifest.py --verify`.", "", "| file | bytes | sha256 |", "|---|---|---|"]
    for rel in sorted(files()):
        p = os.path.join(ROOT, rel)
        lines.append(f"| `{rel}` | {os.path.getsize(p)} | {sha256(p)} |")
    open(os.path.join(ROOT, "MANIFEST.md"), "w").write("\n".join(lines) + "\n")
    print(f"MANIFEST.md: {len(lines) - 6} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
