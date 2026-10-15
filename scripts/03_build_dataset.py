"""Build the paper table: main-track research papers of the six conferences (scripts/config.py).

Keeps dblp records whose booktitle is a main track (config.MAIN_BOOKTITLE): workshops, co-located events, Tiny Papers,
position papers and competition / demo tracks are left out (not research articles of the main conference). Position
papers are also removed by their required title prefix "Position:" (dblp files ICML 2025's under a separate booktitle,
but ICML 2024's and NeurIPS 2025's under the main one). Adds the papers of published proceedings that the dblp dump does
not have yet (data/extra_papers.jsonl from 02_pmlr_additions.py: ICML 2026), with the same filters.
Front matter and similar non-papers are removed by title; duplicates (same venue, year, normalised title) once.
Output: data/papers.csv (venue, year, title, source, uid, booktitle) and results/removed_nonpapers.txt
"""
import csv
import json
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
MAIN = re.compile(config.MAIN_BOOKTITLE)
NONPAPER = re.compile(   # whole-title matches only: "Test of Time: ..." and "Preface: ..." are real paper titles here
    r"^(front matter|back matter|table of contents|author index|preface|foreword|welcome|keynote|invited talk|editorial|"
    r"list of reviewers|organi[sz]ing committee|program committee|test of time award)\s*$|"
    r"^(erratum|errata|corrigendum|correction|retraction|retracted|expression of concern)\b(\s*(to|for|of|:)|\s*$)|^proceedings of\b|"
    r"^notice of retraction\b", re.I)


POSITION = re.compile(r"^position\s*:", re.I)


def norm(t):
    return re.sub(r"[^a-z0-9]+", " ", (t or "").lower()).strip()


def main():
    rows, removed, seen = [], [], set()
    extra = os.path.join(ROOT, "data", "extra_papers.jsonl")
    recs = [dict(r, source="dblp") for r in map(json.loads, open(os.path.join(ROOT, config.DBLP_EXTRACT)))]
    if os.path.exists(extra):
        recs += [dict(r, key=r["uid"], source="pmlr") for r in map(json.loads, open(extra))]
    for r in recs:
        b = r.get("booktitle") or ""
        if r["source"] == "dblp" and not MAIN.match(b):
            continue
        t = (r.get("title") or "").strip().rstrip(".")
        if not t or NONPAPER.search(t) or POSITION.search(t):
            removed.append(f'{r["venue"]} {r["year"]} | {b} | {t}'); continue
        k = (r["venue"], r["year"], norm(t))
        if k in seen:
            removed.append(f'{r["venue"]} {r["year"]} | DUPLICATE | {t}'); continue
        seen.add(k)
        rows.append({"venue": r["venue"], "year": r["year"], "title": t, "source": r["source"], "uid": r["key"], "booktitle": b})
    with open(os.path.join(ROOT, "data", "papers.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["venue", "year", "title", "source", "uid", "booktitle"]); w.writeheader(); w.writerows(rows)
    with open(os.path.join(ROOT, "results", "removed_nonpapers.txt"), "w") as f:
        f.write("\n".join(removed) + "\n")
    c = Counter((r["venue"], r["year"]) for r in rows)
    print("papers kept:", len(rows), " removed:", len(removed))
    print("venue    " + " ".join(f"{y:>6d}" for y in config.YEARS))
    for v in config.VENUES:
        print(f"{v:8s} " + " ".join(f"{c[(v, y)]:6d}" for y in config.YEARS))


if __name__ == "__main__":
    main()
