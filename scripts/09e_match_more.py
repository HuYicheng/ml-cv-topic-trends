"""Abstracts, part e: papers still without an abstract after 09a-09d, matched to the conference's own listing by
  a  the dblp title cleaned of Springer markup ("[inline-graphic not available: see fulltext]") and of text appended
     after the title ("Supplementary Material ...", "... also references ..."), similarity >= 0.9
  b  one compact title contains the other (the shorter one has at least 25 letters / digits)
  c  the same name before the colon and similarity >= 0.7
  x  (CVPR only, not covered by 09d) the CVF listing title, similarity >= 0.9
  d  the author list: weighted share of the dblp authors found among the listing's authors, first and last author
     weight 3, the others 1 (user, 2026-10-15: "一作和末作的权重应该提高"); proposed when the share is >= 0.6 and the
     first or the last author is among them
(user, 2026-10-15: "acde都试试", "方法b也可以试试"). Candidates are only listing entries that no other paper of the venue
and year has taken. Every proposed pair goes to logs/match_more_pairs.tsv and is read before --fetch takes the abstracts
from the official pages (source "<site>-<method>"); rejected pairs are deleted from the file first.
Authors of the listings: ECVA open access (in the listing), CVF (in the listing), ECCV 2026 virtual site (the poster
page's schema.org author list; pages of the listing entries no paper has taken, fetched once and kept in
data/eccv2026_posters.jsonl).
"""
import argparse
import csv
import difflib
import html
import json
import os
import re
import threading
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor

import fetchlib as fl

TSV = os.path.join(fl.ROOT, "logs", "match_more_pairs.tsv")
POSTERS = os.path.join(fl.ROOT, "data", "eccv2026_posters.jsonl")
ABS_ECVA = re.compile(r'<div id="abstract"\s*>(.*?)(?:<font size|</div>)', re.S)
ABS_CVF = re.compile(r'<div id="abstract"\s*>(.*?)</div>', re.S)
ABS_VIRT = re.compile(r'class="abstract-text-inner">(.*?)</div>', re.S)
JUNK = re.compile(r"\[inline-graphic not available: see fulltext\]|\s(Supplementary M\w*|Supplemental M\w*|\w+ also references)\b.*$")


def norm_name(n):
    n = unicodedata.normalize("NFKD", html.unescape(n)).encode("ascii", "ignore").decode()
    n = re.sub(r"\b\d{4}\b|[*†‡]", " ", n).lower()
    return " ".join(re.findall(r"[a-z]+", n))


def name_keys(n):
    t = norm_name(n).split()
    # full name, initial + last name, and the name with its parts in any order ("Shah Mubarak" = "Mubarak Shah")
    return {" ".join(t), (t[0][0] + " " + t[-1]) if len(t) > 1 else " ".join(t), " ".join(sorted(t))} if t else set()


def author_score(dblp_authors, listing_authors):
    have = set().union(*[name_keys(a) for a in listing_authors]) if listing_authors else set()
    if not dblp_authors or not have:
        return 0.0, False, False
    w = [3 if i in (0, len(dblp_authors) - 1) else 1 for i in range(len(dblp_authors))]
    hit = [bool(name_keys(a) & have) for a in dblp_authors]
    return sum(x for x, h in zip(w, hit) if h) / sum(w), hit[0], hit[-1]


def listings():
    out = {}
    page = fl.get("https://www.ecva.net/papers.php") or ""
    for href, y, title, auth in re.findall(r'<a href=(papers/eccv_(\d{4})/papers_ECCV/html/[^>\s]+)>\s*([^<]+)</a>\s*</dt>\s*<dd>\s*([^<]*)</dd>', page):
        out.setdefault(("ECCV", int(y)), []).append({"site": "ecva", "title": title.strip(), "url": "https://www.ecva.net/" + href,
                                                     "authors": [a.strip() for a in auth.split(",") if a.strip()]})
    time.sleep(1)
    page = fl.get("https://eccv.ecva.net/virtual/2026/papers.html") or ""
    for href, title in re.findall(r'<a href="(/virtual/2026/(?:poster|oral)/\d+)">([^<]+)</a>', page):
        out.setdefault(("ECCV", 2026), []).append({"site": "eccv_virtual", "title": title.strip(), "url": "https://eccv.ecva.net" + href, "authors": None})
    for y in (2021, 2023, 2025):
        time.sleep(1)
        page = fl.get(f"https://openaccess.thecvf.com/ICCV{y}?day=all") or ""
        for href, title, dd in re.findall(r'<dt class="ptitle"><br><a href="(/?content/ICCV' + str(y) + r'/html/[^"]+)">([^<]+)</a></dt>\s*<dd>(.*?)</dd>', page, flags=re.S):
            out.setdefault(("ICCV", y), []).append({"site": "cvf_iccv", "title": title.strip(), "url": "https://openaccess.thecvf.com/" + href.lstrip("/"),
                                                    "authors": re.findall(r'submit\(\);">([^<]+)</a>', dd)})
    cached = fl.all_cached()
    have_cvpr_gap = sorted({p["year"] for p in fl.papers() if p["venue"] == "CVPR" and p["uid"] not in cached})
    for y in have_cvpr_gap:
        time.sleep(1)
        page = fl.get(f"https://openaccess.thecvf.com/CVPR{y}?day=all") or ""
        for href, title, dd in re.findall(r'<dt class="ptitle"><br><a href="(/?content/CVPR' + str(y) + r'/html/[^"]+)">([^<]+)</a></dt>\s*<dd>(.*?)</dd>', page, flags=re.S):
            out.setdefault(("CVPR", y), []).append({"site": "cvf_cvpr", "title": title.strip(), "url": "https://openaccess.thecvf.com/" + href.lstrip("/"),
                                                    "authors": re.findall(r'submit\(\);">([^<]+)</a>', dd)})
    for y in (2024, 2025, 2026):
        time.sleep(1)
        page = fl.get(f"https://proceedings.iclr.cc/paper_files/paper/{y}") or ""
        for href, title in re.findall(r'<a title="paper title" href="(/paper_files/paper/\d+/hash/[^"]+)">([^<]+)</a>', page):
            out.setdefault(("ICLR", y), []).append({"site": "iclr", "title": title.strip(), "url": "https://proceedings.iclr.cc" + href, "authors": None})
    return out


def taken_keys(papers, have):
    """Compact titles of papers that already have an abstract (their listing entries are taken)."""
    return {(p["venue"], p["year"], fl.compact(p["title"])) for p in papers if p["uid"] in have}


def eccv2026_authors(entries, taken):
    """Authors (and abstract) of the ECCV 2026 poster pages that no paper has taken; fetched once, kept on disk."""
    got = {}
    if os.path.exists(POSTERS):
        for l in open(POSTERS):
            r = json.loads(l); got[r["url"]] = r
    fuzzy_urls = {r["url"] for r in csv.DictReader(open(os.path.join(fl.ROOT, "logs", "fuzzy_pairs.tsv")), delimiter="\t")}
    todo = [e for e in entries if e["url"] not in got and e["url"] not in fuzzy_urls
            and ("ECCV", 2026, fl.compact(html.unescape(e["title"]))) not in taken]
    fl.log(f"ECCV 2026 poster pages to read for authors: {len(todo)}")
    lock = threading.Lock()
    with open(POSTERS, "a") as f:
        def one(e):
            page = fl.get(e["url"], tries=3, base_sleep=10) or ""
            m = re.search(r'"author":\s*\[(.*?)\]', page, re.S)
            a = ABS_VIRT.search(page)
            r = {"url": e["url"], "authors": re.findall(r'"name":\s*"([^"]+)"', m.group(1)) if m else [],
                 "abstract": a.group(1) if a else None}
            with lock:
                got[e["url"]] = r; f.write(json.dumps(r, ensure_ascii=False) + "\n"); f.flush()
            time.sleep(0.6)
        with ThreadPoolExecutor(6) as ex:          # 6 requests at a time, as for the other virtual-site fetches
            list(ex.map(one, todo))
    for e in entries:
        if e["url"] in got:
            e["authors"] = got[e["url"]]["authors"]


def pairs():
    papers, have = fl.papers(), fl.all_cached()
    authors = {r["key"]: r["authors"] for r in map(json.loads, open(os.path.join(fl.ROOT, "data", "dblp_mlcv_authors.jsonl")))}
    missing = [p for p in papers if p["uid"] not in have]
    fl.log("papers without an abstract:", len(missing))
    lst, taken = listings(), taken_keys(papers, have)
    if ("ECCV", 2026) in lst:
        eccv2026_authors(lst[("ECCV", 2026)], taken)
    fuzzy_urls = {r["url"] for r in csv.DictReader(open(os.path.join(fl.ROOT, "logs", "fuzzy_pairs.tsv")), delimiter="\t")}
    rows, used = [], set()
    for p in missing:
        entries = [e for e in lst.get((p["venue"], p["year"]), []) if e["url"] not in fuzzy_urls and e["url"] not in used
                   and (p["venue"], p["year"], fl.compact(html.unescape(e["title"]))) not in taken]
        if not entries:
            rows.append([p["uid"], p["venue"], p["year"], "e", "", p["title"], "", "; ".join(authors.get(p["uid"], [])), "", ""])
            continue
        k_raw, k = fl.compact(p["title"]), fl.compact(JUNK.sub(" ", p["title"]))
        cand = []
        for e in entries:
            c = fl.compact(html.unescape(e["title"]))
            r = difflib.SequenceMatcher(None, k, c).ratio()
            head_p = fl.compact(JUNK.sub(" ", p["title"]).split(":")[0]) if ":" in p["title"] else None
            head_e = fl.compact(html.unescape(e["title"]).split(":")[0]) if ":" in e["title"] else None
            s, f1, fl_ = author_score(authors.get(p["uid"], []), e["authors"] or [])
            if p["venue"] == "CVPR" and r >= 0.9:
                cand.append(("x", r, e))
            elif k != k_raw and r >= 0.9:
                cand.append(("a", r, e))
            elif min(len(k), len(c)) >= 25 and (k in c or c in k):
                cand.append(("b", r, e))
            elif head_p and head_p == head_e and r >= 0.7:
                cand.append(("c", r, e))
            elif s >= 0.6 and (f1 or fl_):
                cand.append(("d", s, e))
        if cand:
            m, sc, e = max(cand, key=lambda x: ("xabcd".index(x[0]) * -1, x[1]))
            used.add(e["url"])
            rows.append([p["uid"], p["venue"], p["year"], m, f"{sc:.2f}", p["title"], html.unescape(e["title"]),
                         "; ".join(authors.get(p["uid"], [])), "; ".join(e["authors"] or []), e["url"] + "\t" + e["site"]])
        else:
            rows.append([p["uid"], p["venue"], p["year"], "none", "", p["title"], "", "; ".join(authors.get(p["uid"], [])), "", ""])
    with open(TSV, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["uid", "venue", "year", "method", "score", "dblp_title", "listing_title", "dblp_authors", "listing_authors", "url", "site"])
        for r in rows:
            url, _, site = r[-1].partition("\t")
            w.writerow(r[:-1] + [url, site])
    fl.log("pairs by method:", {m: sum(1 for r in rows if r[3] == m) for m in ("x", "a", "b", "c", "d", "e", "none")}, "->", TSV)


def fetch():
    cache = fl.Cache("more")
    posters = {}
    if os.path.exists(POSTERS):
        for l in open(POSTERS):
            r = json.loads(l); posters[r["url"]] = r
    pat = {"ecva": ABS_ECVA, "cvf_iccv": ABS_CVF, "cvf_cvpr": ABS_CVF, "iclr": re.compile(r'class="paper-abstract">(.*?)(?:</p>\s*</div>|<h2|<div|<script)', re.S)}
    for r in csv.DictReader(open(TSV), delimiter="\t"):
        if r["method"] not in "xabcd" or not r["url"] or r["uid"] in cache.got:
            continue
        if r["site"] == "eccv_virtual" and r["url"] in posters:
            cache.put(r["uid"], posters[r["url"]]["abstract"], f"{r['site']}-{r['method']}")
            continue
        page = fl.get(r["url"], tries=3, base_sleep=10)
        if page and r["site"] in pat:
            m = pat[r["site"]].search(page)
            cache.put(r["uid"], m.group(1) if m else None, f"{r['site']}-{r['method']}")
        time.sleep(0.6)
    fl.log(f"cached {len(cache.got)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--fetch", action="store_true")
    fetch() if ap.parse_args().fetch else pairs()
