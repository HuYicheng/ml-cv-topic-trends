"""Shared helpers for the abstract fetchers (09a, 09b, 09c): polite HTTP, per-source caches, title matching.
No personal data in requests (fixed research User-Agent, no e-mail)."""
import html
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
UA = {"User-Agent": "mlcv-trends-study/0.1 (research script)"}
_lock = threading.Lock()


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def get(url, data=None, headers=None, tries=4, base_sleep=10, timeout=60):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, data=data, headers={**UA, **(headers or {})})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            if e.code in (403, 404, 410):
                return None                           # not available: never retried or worked around
            log(f"  HTTP {e.code} on {url[:100]} (attempt {i + 1}/{tries})")
        except Exception as e:  # noqa: BLE001
            log(f"  {type(e).__name__} on {url[:100]} (attempt {i + 1}/{tries})")
        time.sleep(base_sleep * 2 ** i)
    return None


def compact(t):
    return re.sub(r"[^a-z0-9]+", "", html.unescape(t or "").lower())


def strip_tags(s):
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", s or ""))).strip()


def papers():
    import csv
    with open(os.path.join(ROOT, "data", "papers.csv")) as f:
        return [dict(r, year=int(r["year"])) for r in csv.DictReader(f)]


def dblp_records():
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import config
    return {r["key"]: r for r in map(json.loads, open(os.path.join(ROOT, config.DBLP_EXTRACT)))}


class Cache:
    """Append-only cache of {uid, abstract, source} for one source; thread-safe; resumable."""

    def __init__(self, name):
        self.path = os.path.join(ROOT, "data", f"abs_cache_{name}.jsonl")
        self.got = {}
        if os.path.exists(self.path):
            for l in open(self.path):
                r = json.loads(l); self.got[r["uid"]] = r
        self.f = open(self.path, "a")

    def put(self, uid, abstract, source):
        abstract = strip_tags(abstract) if abstract else None
        if not abstract or len(abstract.split()) < 20:
            return False
        with _lock:
            if uid in self.got:
                return False
            r = {"uid": uid, "abstract": abstract, "source": source}
            self.got[uid] = r; self.f.write(json.dumps(r, ensure_ascii=False) + "\n"); self.f.flush()
        return True


def all_cached():
    out = {}
    for fn in sorted(os.listdir(os.path.join(ROOT, "data"))):
        if fn.startswith("abs_cache_") and fn.endswith(".jsonl"):
            for l in open(os.path.join(ROOT, "data", fn)):
                r = json.loads(l); out.setdefault(r["uid"], r)
    return out
