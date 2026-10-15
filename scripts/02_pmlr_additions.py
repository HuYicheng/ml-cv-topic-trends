"""Papers published in the official proceedings but missing from the dblp dump: ICML 2026.

The dblp dump of 2026-10-13 has no conf/icml/2026 record, while PMLR has published volume 306, "Proceedings of ICML
2026" (checked 2026-10-15). The volume's bibliography.bib gives title, URL and abstract of every paper; 03 adds these
papers with the same filters as the dblp ones (position papers, non-papers, duplicates). uid = pmlr/<volume>/<paper id>.
Output: data/extra_papers.jsonl (venue, year, uid, title, url, booktitle); abstracts go to data/abs_cache_pmlr.jsonl
(source "pmlr-bib", as for ICML 2020-2025).
"""
import json
import os
import re
import sys

import fetchlib as fl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402


def braced(entry, field):
    """Value of a bibtex field written as field = {...}, with nested braces."""
    m = re.search(r"\b" + field + r"\s*=\s*\{", entry)
    if not m:
        return None
    i, depth = m.end() - 1, 0
    for k in range(i, len(entry)):
        depth += entry[k] == "{"; depth -= entry[k] == "}"
        if depth == 0:
            return entry[i + 1:k]
    return None


def clean_title(t):
    for _ in range(2):   # \underline{\text{X}} needs two passes
        t = re.sub(r"\\(?:textit|texttt|textbf|textsc|text|mathrm|mathbf|mathcal|mathsf|underline|emph)\s*\{([^{}]*)\}", r"\1", t)
    t = t.replace("\\&", "&").replace("\\textdegree", "°").replace("\\times", "×")
    return re.sub(r"\s+", " ", t.replace("$", "").replace("{", "").replace("}", "")).strip()


def main():
    pm = fl.Cache("pmlr")
    rows = []
    for (venue, year), vol in sorted(config.PMLR_EXTRA.items()):
        bib = fl.get(f"https://proceedings.mlr.press/{vol}/assets/bib/bibliography.bib") or ""
        n_abs = 0
        for entry in re.split(r"\n@", bib):
            if not entry.lower().startswith("inproceedings"):
                continue
            url, title = braced(entry, "url"), braced(entry, "title")
            if not url or not title:
                continue
            pid = re.search(r"proceedings\.mlr\.press/(v\d+)/([^/.]+)\.html", url)
            uid = f"pmlr/{pid.group(1)}/{pid.group(2)}"
            rows.append({"venue": venue, "year": year, "uid": uid, "title": clean_title(title), "url": url,
                         "booktitle": f"{venue} (PMLR {vol})"})
            n_abs += pm.put(uid, braced(entry, "abstract"), "pmlr-bib")
        fl.log(f"PMLR {vol} ({venue} {year}): {sum(1 for r in rows if r['year'] == year)} papers, {n_abs} new abstracts")
    with open(os.path.join(fl.ROOT, "data", "extra_papers.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
