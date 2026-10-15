"""Abstracts, part a: ICML from the PMLR bibliography.bib of each volume (matched by paper URL); CVPR, ICCV and ECCV from
OpenAlex by DOI (100 DOIs per request; anonymous OpenAlex allows 1,000 requests a day, this needs about 275).
Caches: data/abs_cache_pmlr.jsonl, data/abs_cache_openalex.jsonl
"""
import json
import re
import sys
import time
import urllib.parse

import fetchlib as fl

sys.path.insert(0, fl.os.path.dirname(fl.os.path.abspath(__file__)))
import config  # noqa: E402


def oa_abstract(w):
    inv = w.get("abstract_inverted_index") or {}
    words = sorted((p, word) for word, ps in inv.items() for p in ps)
    return " ".join(word for _, word in words) or None


def main():
    papers = fl.papers()
    dblp = fl.dblp_records()
    # ICML: PMLR bib, matched by the proceedings URL in dblp
    pm = fl.Cache("pmlr")
    by_url = {}
    for p in papers:
        if p["venue"] == "ICML" and p["uid"] in dblp:          # ICML 2026 (no dblp record): abstracts cached by 02
            for e in dblp[p["uid"]]["ee"]:
                if e and "proceedings.mlr.press/" in e:
                    by_url[re.sub(r"^https?://", "", e).rstrip("/")] = p["uid"]
    for y, vol in sorted(config.PMLR_ICML.items()):
        bib = fl.get(f"https://proceedings.mlr.press/{vol}/assets/bib/bibliography.bib") or ""
        n = 0
        for entry in re.split(r"\n@", bib):
            url = re.search(r"url\s*=\s*\{(https?://proceedings\.mlr\.press/[^}]+)\}", entry)
            i = entry.find("abstract")
            if not url or i < 0:
                continue
            j = entry.find("{", i); depth, k = 0, j
            while k < len(entry):
                depth += entry[k] == "{"; depth -= entry[k] == "}"
                if depth == 0:
                    break
                k += 1
            uid = by_url.get(re.sub(r"^https?://", "", url.group(1)).rstrip("/"))
            if uid and pm.put(uid, entry[j + 1:k].replace("{", "").replace("}", ""), "pmlr-bib"):
                n += 1
        fl.log(f"PMLR {vol} ({y}): {n} abstracts")
        time.sleep(2)
    # CVPR / ICCV / ECCV: OpenAlex by DOI
    oa = fl.Cache("openalex")
    doi_of = {}
    for p in papers:
        if p["venue"] in ("CVPR", "ICCV", "ECCV") and p["uid"] not in oa.got:
            d = next((e.split("doi.org/")[1].lower() for e in dblp[p["uid"]]["ee"] if e and "doi.org/10." in e), None)
            if d:
                doi_of[d] = p["uid"]
    dois = list(doi_of)
    fl.log("OpenAlex DOIs to query:", len(dois))
    for i in range(0, len(dois), 100):
        chunk = dois[i:i + 100]
        q = urllib.parse.urlencode({"filter": "doi:" + "|".join(chunk), "per_page": 100, "select": "id,doi,abstract_inverted_index"})
        txt = fl.get("https://api.openalex.org/works?" + q)
        if txt:
            for w in json.loads(txt)["results"]:
                d = (w.get("doi") or "").replace("https://doi.org/", "").lower()
                if d in doi_of:
                    oa.put(doi_of[d], oa_abstract(w), "openalex")
        if i // 100 % 20 == 0:
            fl.log(f"  OpenAlex {i + len(chunk)}/{len(dois)}; cached {len(oa.got)}")
        time.sleep(1.0)
    fl.log("done: PMLR", len(pm.got), "OpenAlex", len(oa.got))


if __name__ == "__main__":
    main()
