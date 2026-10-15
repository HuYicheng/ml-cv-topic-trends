"""Abstracts, part b: the conferences' own paper pages (public HTML, no robots.txt restrictions, polite rate).

  --site neurips : proceedings.neurips.cc / papers.nips.cc abstract page linked from dblp (all years), and the
                   datasets-benchmarks-proceedings.neurips.cc pages of the 2021 Datasets and Benchmarks track
  --site iclr    : proceedings.iclr.cc (2024-2026); 2026 linked from dblp, 2024-2025 matched by identical compact title
                   from the year listing (dblp links those years to OpenReview, whose API refuses anonymous requests)
  --site cvf     : openaccess.thecvf.com paper page linked from dblp (CVPR; used for CVPR 2026 and for any CVPR
                   paper that OpenAlex left without an abstract)
  --site iclr_virtual : iclr.cc virtual-site poster / oral pages for ICLR 2020-2023, matched by identical compact title
                   from /virtual/<year>/papers.html (robots.txt allows these pages; it disallows /static, whose
                   bulk JSON files are therefore not used)
  --site cvf_iccv: openaccess.thecvf.com ICCV pages, matched by identical compact title from the ICCV<year>?day=all
                   listing (dblp gives ICCV only a DOI; used where OpenAlex has no abstract, mainly ICCV 2025)
  --site ecva    : www.ecva.net open-access ECCV pages (2020, 2022, 2024), matched by identical compact title from the
                   ECVA listing; OpenAlex has no abstracts for ECCV (Springer)
  --site eccv_virtual : eccv.ecva.net virtual-site poster / oral pages for ECCV 2026 (not on the ECVA open-access site
                   yet), matched by identical compact title from /virtual/2026/papers.html (robots.txt allows these pages);
                   fetched also for papers the arXiv search had found, so that every ECCV 2026 abstract is the conference's
Worker threads (default 2, --workers), each waiting 0.6 s between requests: about 1.5-2 requests a second to one host
at the observed page latency.
Cache: data/abs_cache_<site>.jsonl
"""
import argparse
import queue
import re
import threading
import time

import fetchlib as fl

# stops at <script too: the ICLR and newer NeurIPS proceedings pages put a menu script right after the abstract
ABS_PROC = re.compile(r'class="paper-abstract">(.*?)(?:</p>\s*</div>|<h2|<div|<script)', re.S)
ABS_DNB = re.compile(r'<h4>Abstract</h4>\s*(.*?)</div>', re.S)   # NeurIPS 2021 Datasets and Benchmarks pages
ABS_CVF = re.compile(r'<div id="abstract"\s*>(.*?)</div>', re.S)
ABS_ICLRV = re.compile(r'class="abstract-text-inner">(.*?)</div>', re.S)
ABS_ECVA = re.compile(r'<div id="abstract"\s*>(.*?)(?:<font size|</div>)', re.S)


def run(jobs, cache, pattern, source, workers=2, wait=0.6):
    patterns = pattern if isinstance(pattern, tuple) else (pattern,)
    q = queue.Queue()
    for j in jobs:
        q.put(j)
    done = [0]

    def worker():
        while True:
            try:
                uid, url = q.get_nowait()
            except queue.Empty:
                return
            page = fl.get(url, tries=3, base_sleep=10)
            if page:
                m = next((m for m in (rx.search(page) for rx in patterns) if m), None)
                cache.put(uid, m.group(1) if m else None, source)
            done[0] += 1
            if done[0] % 500 == 0:
                fl.log(f"  {done[0]}/{len(jobs)} pages; cached {len(cache.got)}")
            time.sleep(wait)

    ts = [threading.Thread(target=worker) for _ in range(workers)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--site", required=True, choices=["neurips", "iclr", "iclr_virtual", "cvf", "cvf_iccv", "ecva", "eccv_virtual"])
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args(); site = args.site
    papers, dblp = fl.papers(), fl.dblp_records()
    cache = fl.Cache(site)
    have = fl.all_cached()
    jobs = []
    if site == "neurips":
        for p in papers:
            if p["venue"] == "NeurIPS" and p["uid"] not in have:
                url = next((e for e in dblp[p["uid"]]["ee"] if e and ("neurips.cc" in e or "nips.cc" in e) and "Abstract" in e), None)
                if url:
                    jobs.append((p["uid"], url.replace("http://", "https://")))
        pattern, source = (ABS_PROC, ABS_DNB), "neurips-proceedings"
    elif site == "iclr":
        listing = {}
        for y in (2024, 2025, 2026):
            page = fl.get(f"https://proceedings.iclr.cc/paper_files/paper/{y}") or ""
            for href, title in re.findall(r'<a title="paper title" href="(/paper_files/paper/\d+/hash/[^"]+)">([^<]+)</a>', page):
                listing[(y, fl.compact(title))] = "https://proceedings.iclr.cc" + href
            fl.log(f"ICLR {y} listing: {sum(1 for k in listing if k[0] == y)} papers")
        for p in papers:
            if p["venue"] == "ICLR" and p["year"] >= 2024 and p["uid"] not in have:
                url = next((e for e in dblp[p["uid"]]["ee"] if e and "proceedings.iclr.cc" in e), None) \
                    or listing.get((p["year"], fl.compact(p["title"])))
                if url:
                    jobs.append((p["uid"], url))
        pattern, source = ABS_PROC, "iclr-proceedings"
    elif site == "iclr_virtual":
        listing = {}
        for y in (2020, 2021, 2022, 2023):
            page = fl.get(f"https://iclr.cc/virtual/{y}/papers.html") or ""
            for href, title in re.findall(r'<a href="(/virtual/\d+/(?:poster|oral)/\d+)">([^<]+)</a>', page):
                listing.setdefault((y, fl.compact(title)), "https://iclr.cc" + href)
            fl.log(f"ICLR {y} virtual listing: {sum(1 for k in listing if k[0] == y)} papers")
            time.sleep(1)
        for p in papers:
            if p["venue"] == "ICLR" and p["year"] <= 2023 and p["uid"] not in have:
                url = listing.get((p["year"], fl.compact(p["title"])))
                if url:
                    jobs.append((p["uid"], url))
        pattern, source = ABS_ICLRV, "iclr-virtual"
    elif site == "eccv_virtual":
        page = fl.get("https://eccv.ecva.net/virtual/2026/papers.html") or ""
        listing = {}
        for href, title in re.findall(r'<a href="(/virtual/2026/(?:poster|oral)/\d+)">([^<]+)</a>', page):
            listing.setdefault(fl.compact(title), "https://eccv.ecva.net" + href)
        fl.log(f"ECCV 2026 virtual listing: {len(listing)} papers")
        official = {u for u, r in have.items() if r["source"] != "arxiv"}
        for p in papers:
            if p["venue"] == "ECCV" and p["year"] == 2026 and p["uid"] not in official:
                url = listing.get(fl.compact(p["title"]))
                if url:
                    jobs.append((p["uid"], url))
        pattern, source = ABS_ICLRV, "eccv-virtual"
    elif site == "cvf_iccv":
        listing = {}
        for y in (2021, 2023, 2025):
            page = fl.get(f"https://openaccess.thecvf.com/ICCV{y}?day=all") or ""
            for href, title in re.findall(r'<dt class="ptitle"><br><a href="(/?content/ICCV{y}/html/[^"]+)">([^<]+)</a>'.replace("{y}", str(y)), page):
                listing[(y, fl.compact(title))] = "https://openaccess.thecvf.com/" + href.lstrip("/")
            fl.log(f"CVF ICCV {y} listing: {sum(1 for k in listing if k[0] == y)} papers")
            time.sleep(1)
        for p in papers:
            if p["venue"] == "ICCV" and p["uid"] not in have:
                url = listing.get((p["year"], fl.compact(p["title"])))
                if url:
                    jobs.append((p["uid"], url))
        pattern, source = ABS_CVF, "cvf-openaccess"
    elif site == "ecva":
        page = fl.get("https://www.ecva.net/papers.php") or ""
        listing = {}
        for href, title in re.findall(r'<a href=(papers/eccv_(\d{4})/papers_ECCV/html/[^>\s]+)>\s*([^<]+)</a>', page and page or "") and \
                [(h, t) for h, _, t in re.findall(r'<a href=(papers/eccv_(\d{4})/papers_ECCV/html/[^>\s]+)>\s*([^<]+)</a>', page)]:
            y = int(re.search(r"eccv_(\d{4})", href).group(1))
            listing[(y, fl.compact(title))] = "https://www.ecva.net/" + href
        fl.log("ECVA listing:", {y: sum(1 for k in listing if k[0] == y) for y in (2020, 2022, 2024, 2026)})
        for p in papers:
            if p["venue"] == "ECCV" and p["uid"] not in have:
                url = listing.get((p["year"], fl.compact(p["title"])))
                if url:
                    jobs.append((p["uid"], url))
        pattern, source = ABS_ECVA, "ecva-openaccess"
    else:
        for p in papers:
            if p["venue"] == "CVPR" and p["uid"] not in have:
                url = next((e for e in dblp[p["uid"]]["ee"] if e and "openaccess.thecvf.com" in e), None)
                if url:
                    jobs.append((p["uid"], url.replace("http://", "https://")))
        pattern, source = ABS_CVF, "cvf-openaccess"
    fl.log(f"{site}: {len(jobs)} pages to fetch")
    run(jobs, cache, pattern, source, workers=args.workers)
    fl.log(f"{site}: done, cached {len(cache.got)}")


if __name__ == "__main__":
    main()
