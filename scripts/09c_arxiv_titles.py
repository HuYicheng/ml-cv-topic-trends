"""Abstracts, part c: arXiv API title search for papers no other source covered (mainly ICLR 2020-2023, which only
OpenReview hosts and whose API refuses anonymous requests). Accepted only when the compact title (letters and digits)
is identical. arXiv API terms: at most one request every 3 seconds.
  --only-iclr-early : restrict to ICLR 2020-2023 (can start before the other sources finish)
  --venue V --year Y : restrict to one venue / year (ECCV 2026: not yet on the ECVA site, no abstracts in OpenAlex)
Cache: data/abs_cache_arxiv.jsonl
"""
import argparse
import re
import time
import urllib.parse

import fetchlib as fl


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--only-iclr-early", action="store_true")
    ap.add_argument("--venue"); ap.add_argument("--year", type=int)
    args = ap.parse_args()
    cache = fl.Cache("arxiv")
    have = fl.all_cached()
    todo = [p for p in fl.papers() if p["uid"] not in have
            and (not args.only_iclr_early or (p["venue"] == "ICLR" and p["year"] <= 2023))
            and (not args.venue or p["venue"] == args.venue) and (not args.year or p["year"] == args.year)]
    fl.log("arXiv title search:", len(todo))
    for n, p in enumerate(todo, 1):
        tail = p["title"].split(":", 1)[-1]
        words = re.sub(r"[^A-Za-z0-9 ]+", " ", tail if len(tail) > 25 else p["title"]).split()[:14]
        q = urllib.parse.urlencode({"search_query": 'ti:"' + " ".join(words) + '"', "max_results": 10})
        txt = fl.get("http://export.arxiv.org/api/query?" + q, tries=3, base_sleep=15)
        for e in re.findall(r"<entry>(.*?)</entry>", txt or "", flags=re.S):
            t = re.search(r"<title>(.*?)</title>", e, flags=re.S)
            if t and fl.compact(t.group(1)) == fl.compact(p["title"]):
                s = re.search(r"<summary>(.*?)</summary>", e, flags=re.S)
                cache.put(p["uid"], s.group(1) if s else None, "arxiv"); break
        if n % 200 == 0:
            fl.log(f"  {n}/{len(todo)}; found {len(cache.got)}")
        time.sleep(3.1)
    fl.log("arXiv done:", len(cache.got))


if __name__ == "__main__":
    main()
