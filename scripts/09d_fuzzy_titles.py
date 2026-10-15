"""Abstracts, part d: papers whose dblp title differs slightly from the title on the conference's own listing (so the
identical-title match of 09b missed them): the most similar listing title of the same venue and year (difflib ratio on
the compact titles, >= 0.9, each listing entry used once).

Step 1 (default): write logs/fuzzy_pairs.tsv (uid, venue, year, ratio, dblp title, listing title, url) for reading.
Step 2 (--fetch): fetch the pages of the pairs still in that file (rejected pairs are deleted from it by hand first) and
cache the abstracts as source "<site>-fuzzy" in data/abs_cache_fuzzy.jsonl.
Listings: ECVA open access (ECCV 2020/2022/2024), eccv.ecva.net virtual site (ECCV 2026), CVF open access (ICCV),
proceedings.iclr.cc (ICLR 2024-2026), iclr.cc virtual site (ICLR 2020-2023).
"""
import argparse
import csv
import difflib
import html
import importlib.util
import os
import re
import time

import fetchlib as fl

spec = importlib.util.spec_from_file_location("b", os.path.join(os.path.dirname(os.path.abspath(__file__)), "09b_proceedings_pages.py"))
b = importlib.util.module_from_spec(spec); spec.loader.exec_module(b)
TSV = os.path.join(fl.ROOT, "logs", "fuzzy_pairs.tsv")
PATTERN = {"ecva": b.ABS_ECVA, "eccv_virtual": b.ABS_ICLRV, "cvf_iccv": b.ABS_CVF, "iclr": b.ABS_PROC, "iclr_virtual": b.ABS_ICLRV}


def listings():
    """(venue, year) -> list of (site, original title, url)."""
    out = {}
    page = fl.get("https://www.ecva.net/papers.php") or ""
    for href, y, title in re.findall(r'<a href=(papers/eccv_(\d{4})/papers_ECCV/html/[^>\s]+)>\s*([^<]+)</a>', page):
        out.setdefault(("ECCV", int(y)), []).append(("ecva", title.strip(), "https://www.ecva.net/" + href))
    time.sleep(1)
    page = fl.get("https://eccv.ecva.net/virtual/2026/papers.html") or ""
    for href, title in re.findall(r'<a href="(/virtual/2026/(?:poster|oral)/\d+)">([^<]+)</a>', page):
        out.setdefault(("ECCV", 2026), []).append(("eccv_virtual", title.strip(), "https://eccv.ecva.net" + href))
    for y in (2021, 2023, 2025):
        time.sleep(1)
        page = fl.get(f"https://openaccess.thecvf.com/ICCV{y}?day=all") or ""
        for href, title in re.findall(r'<dt class="ptitle"><br><a href="(/?content/ICCV' + str(y) + r'/html/[^"]+)">([^<]+)</a>', page):
            out.setdefault(("ICCV", y), []).append(("cvf_iccv", title.strip(), "https://openaccess.thecvf.com/" + href.lstrip("/")))
    for y in (2024, 2025, 2026):
        time.sleep(1)
        page = fl.get(f"https://proceedings.iclr.cc/paper_files/paper/{y}") or ""
        for href, title in re.findall(r'<a title="paper title" href="(/paper_files/paper/\d+/hash/[^"]+)">([^<]+)</a>', page):
            out.setdefault(("ICLR", y), []).append(("iclr", title.strip(), "https://proceedings.iclr.cc" + href))
    for y in (2020, 2021, 2022, 2023):
        time.sleep(1)
        page = fl.get(f"https://iclr.cc/virtual/{y}/papers.html") or ""
        for href, title in re.findall(r'<a href="(/virtual/\d+/(?:poster|oral)/\d+)">([^<]+)</a>', page):
            out.setdefault(("ICLR", y), []).append(("iclr_virtual", title.strip(), "https://iclr.cc" + href))
    return out


def pairs():
    have = fl.all_cached()
    missing = [p for p in fl.papers() if p["uid"] not in have and p["venue"] in ("ECCV", "ICCV", "ICLR")]
    fl.log("papers without an abstract at ECCV / ICCV / ICLR:", len(missing))
    lst = listings()
    rows = []
    for (v, y), entries in sorted(lst.items()):
        todo = [p for p in missing if p["venue"] == v and p["year"] == y]
        if not todo:
            continue
        keyed = {}
        for site, title, url in entries:
            keyed.setdefault(fl.compact(html.unescape(title)), (site, title, url))
        used = set()
        for p in todo:
            k = fl.compact(p["title"])
            best = difflib.get_close_matches(k, [c for c in keyed if c not in used], n=1, cutoff=0.9)
            if best:
                used.add(best[0])
                site, title, url = keyed[best[0]]
                ratio = difflib.SequenceMatcher(None, k, best[0]).ratio()
                rows.append([p["uid"], v, y, f"{ratio:.3f}", site, p["title"], html.unescape(title), url])
        fl.log(f"{v} {y}: {len(todo)} missing, {sum(1 for r in rows if r[1] == v and r[2] == y)} similar titles found")
    with open(TSV, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["uid", "venue", "year", "ratio", "site", "dblp_title", "listing_title", "url"])
        w.writerows(sorted(rows, key=lambda r: float(r[3])))
    fl.log(f"{len(rows)} pairs -> {TSV}")


def fetch():
    cache = fl.Cache("fuzzy")
    rows = list(csv.DictReader(open(TSV), delimiter="\t"))
    for i, r in enumerate(rows, 1):
        if r["uid"] in cache.got:
            continue
        page = fl.get(r["url"], tries=3, base_sleep=10)
        if page:
            m = PATTERN[r["site"]].search(page)
            cache.put(r["uid"], m.group(1) if m else None, r["site"] + "-fuzzy")
        time.sleep(0.6)
    fl.log(f"fetched; cached {len(cache.got)} of {len(rows)} pairs")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--fetch", action="store_true")
    fetch() if ap.parse_args().fetch else pairs()
