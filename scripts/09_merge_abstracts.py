"""Merge the per-source abstract caches into data/abstracts.jsonl (one abstract per paper) and print coverage.

Priority when a paper has several sources: the venue's own pages / proceedings first (PMLR, NeurIPS and ICLR
proceedings, ICLR virtual site, CVF, ECVA, ECCV virtual site), then the same pages matched by a near-identical title
(09d_fuzzy_titles.py) or by cleaned title, containment, name before the colon or author list (09e_match_more.py),
every pair read, then OpenAlex, then arXiv (another version of the paper).
"""
import collections
import json
import os
import sys

import fetchlib as fl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402

ORDER = ["pmlr", "neurips", "iclr", "iclr_virtual", "cvf", "cvf_iccv", "ecva", "eccv_virtual", "fuzzy", "more", "openalex", "arxiv"]
# The first pages fetched from the ICLR and NeurIPS proceedings (before the extraction pattern stopped at <script>) carry
# the page's menu script after the abstract; it is cut here (idempotent; the caches are left as fetched).
PAGE_SCRIPT = "$('.dropdown-menu a.dropdown-toggle')"


def clean(a, source=""):
    a = a.split(PAGE_SCRIPT)[0].rstrip()
    if source.startswith("ecva"):
        # ECVA pages hold the abstract as a quoted field; for titles with commas the tail of the title comes first,
        # separated by '","' (23 papers, e.g. "Splat, Shoot: ...","The goal of perception ...")
        i = a.find('","', 0, 400)
        if i >= 0:
            a = a[i + 3:]
        a = a.strip().strip('"').strip()
    return a


def main():
    caches = {}
    for name in ORDER:
        path = os.path.join(fl.ROOT, "data", f"abs_cache_{name}.jsonl")
        if os.path.exists(path):
            for l in open(path):
                r = json.loads(l); caches.setdefault(r["uid"], dict(r, abstract=clean(r["abstract"], r["source"])))
    papers = fl.papers()
    n = collections.Counter((p["venue"], p["year"]) for p in papers)
    h = collections.Counter()
    src = collections.Counter()
    out = os.path.join(fl.ROOT, "data", "abstracts.jsonl")
    with open(out + ".tmp", "w") as f:                 # written aside and renamed: readers never see a partial file
        for p in papers:
            r = caches.get(p["uid"])
            if r:
                f.write(json.dumps({**r, "venue": p["venue"], "year": p["year"]}, ensure_ascii=False) + "\n")
                h[(p["venue"], p["year"])] += 1; src[(p["venue"], r["source"])] += 1
    os.replace(out + ".tmp", out)
    print(f"papers with abstracts: {sum(h.values())} of {len(papers)}")
    print("venue     " + " ".join(f"{y:>12d}" for y in config.YEARS))
    for v in config.VENUES:
        print(f"{v:9s} " + " ".join(f"{h[(v, y)]:5d}/{n[(v, y)]:<5d}  " if n[(v, y)] else f"{'–':>12s}" for y in config.YEARS))
    print("sources:", dict(sorted(src.items())))


if __name__ == "__main__":
    main()
