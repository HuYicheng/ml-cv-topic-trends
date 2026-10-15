"""Data for the interactive page: site/data.json and site/subsets.json (ML / CV study; same format as the robotics page).

Contents: topics (name, family, standardised share per year with 95% interval, change, annual multiplier, call, the same
for 2024 onwards), per-scope results (all venues, ML, Vision, each venue: share, interval, papers k of n per year, trend),
papers of the final measure (title, venue, year, link, topic indices), papers per venue-year, build metadata.
subsets.json: every other combination of venues (scripts/20b_scope_subsets.py), loaded by the page on demand.
No abstract text is published (abstracts were only used to assign topics). Links: the first electronic edition in dblp
(conference proceedings page, DOI, PMLR or OpenReview).
"""
import csv
from collections import Counter
import json
import os
import time

import numpy as np
import pandas as pd
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
OUT = os.path.join(ROOT, "site", "data.json")
VENUES = config.VENUES
JOURNALS = set()                                   # no journals in this study
GROUPS = {"All venues": VENUES, "ML": config.GROUPS["ML"], "Vision": config.GROUPS["Vision"]}


def main():
    sc = pd.read_csv(os.path.join(ROOT, "results", "shares_and_change_ta.csv")).sort_values(f"share_{config.REF_YEAR}", ascending=False)
    topics = list(sc.topic)
    m = pd.read_csv(os.path.join(ROOT, "data", "topic_matrix_ta.csv"))
    with open(os.path.join(ROOT, "data", "papers.csv")) as f:
        papers = {r["uid"]: r for r in csv.DictReader(f)}
    dblp = {r["key"]: r for r in map(json.loads, open(os.path.join(ROOT, config.DBLP_EXTRACT)))}

    def link(uid, src):
        if uid not in dblp:                         # ICML 2026 from PMLR (02_pmlr_additions.py)
            return f"https://proceedings.mlr.press/{uid.split('/', 1)[1]}.html" if uid.startswith("pmlr/") else None
        ee = [e for e in dblp[uid]["ee"] if e]
        return next((e for e in ee if "doi.org/" in e), ee[0] if ee else None)

    rows, missing = [], 0
    cols = [m.columns.get_loc(t) for t in topics]
    for rec in m.itertuples(index=False):
        uid, venue, year = rec[0], rec[1], int(rec[2])
        p = papers[uid]
        url = link(uid, p["source"])
        missing += url is None
        ks = [i for i, c in enumerate(cols) if rec[c] == 1]
        rows.append([p["title"], VENUES.index(venue), year, url or "", ks])
    rows.sort(key=lambda r: (-r[2], r[1], r[0].lower()))
    # 85,000 links would be ~6 MB of text: store each as [prefix index, middle, suffix index] with shared prefix and
    # suffix tables (the page rebuilds the URL); prefixes / suffixes used by fewer than 30 papers stay inline
    def split(u):
        i = max(u.rfind("/"), u.rfind("="), u.rfind("hash/") + 4) + 1
        mid = u[i:]
        j = next((k for k in (mid.find("-Abstract"), mid.rfind(".html")) if k > 0), len(mid))
        return u[:i], mid[:j], mid[j:]
    pcount, scount = Counter(), Counter()
    for r in rows:
        if r[3]:
            a, _, c = split(r[3]); pcount[a] += 1; scount[c] += 1
    LP = [a for a, n in pcount.most_common() if n >= 30]; LS = [c for c, n in scount.most_common() if n >= 30]
    ip, is_ = {a: i for i, a in enumerate(LP)}, {c: i for i, c in enumerate(LS)}
    for r in rows:
        if r[3]:
            a, b_, c = split(r[3])
            if a in ip and c in is_:
                r[3] = [ip[a], b_, is_[c]]
    totals = {f"{v}|{y}": int(n) for (v, y), n in m.groupby(["venue", "year"]).size().items()}
    # full precision (6 decimals): the page rounds once for display, so it shows the same numbers as REPORT.md
    R6 = lambda v: None if v is None or pd.isna(v) or not np.isfinite(float(v)) else round(float(v), 6)
    tp = [{"name": r["topic"], "family": r["family"].split(" ", 1)[1],
           "s": [R6(r[f"share_{y}"]) for y in config.YEARS], "lo": [R6(r[f"share_{y}_lo"]) for y in config.YEARS],
           "hi": [R6(r[f"share_{y}_hi"]) for y in config.YEARS], "dpp": R6(r["change_pp"]), "or": R6(r["annual_OR"]),
           "orlo": R6(r["annual_OR_lo"]), "orhi": R6(r["annual_OR_hi"]), "q": float(f"{r['q']:.3g}"), "call": r["call"],
           "ror": R6(r["recent_OR"]), "rcall": r["recent_call"]} for _, r in sc.iterrows()]
    # per-scope results: the nine named scopes (scripts/20_scope_trends.py) go into data.json; the other 54 venue
    # combinations (scripts/20b_scope_subsets.py) into subsets.json, which the page loads when a combination is chosen
    tindex = {r["name"]: i for i, r in enumerate(tp)}

    def build_scopes(st):
        out = {}
        for scope, g in st.groupby("scope", sort=False):
            years = [int(y) for y in g.years.iloc[0].split(",")]
            srows = []
            for r in g.itertuples():
                def f(c, r=r):
                    v = getattr(r, c, float("nan"))
                    return None if v is None or pd.isna(v) or not np.isfinite(float(v)) else round(float(v), 6)
                srows.append({"i": tindex[r.topic], "s": [f(f"share_{y}") for y in years], "lo": [f(f"lo_{y}") for y in years],
                              "hi": [f(f"hi_{y}") for y in years], "k": [int(getattr(r, f"k_{y}")) for y in years],
                              "n": [int(getattr(r, f"n_{y}")) for y in years], "dpp": f("change_pp"), "or": f("OR"),
                              "orlo": f("OR_lo"), "orhi": f("OR_hi"), "q": f("q"), "call": r.call,
                              "ror": f("rec_OR"), "rcall": (r.rec_call if isinstance(getattr(r, "rec_call", None), str) else "unclear"),
                              "note": (r.note if isinstance(getattr(r, "note", None), str) else None)})
            members = GROUPS.get(scope, scope.split("+"))
            last = config.YEARS[-1]
            missing = [v for v in members if totals.get(f"{v}|{last}", 0) == 0]
            running = [v for v in members if v in JOURNALS]
            partial = [last] if last in years and (missing or running) else []
            estimated = [last] if last in years and missing and len(members) > 1 else []
            out[scope] = {"years": years, "single": len(members) == 1, "members": members, "partial": partial,
                          "estimated": estimated, "rows": srows}
        return out
    scopes = build_scopes(pd.read_csv(os.path.join(ROOT, "results", "scope_trends_ta.csv")))
    subsets = build_scopes(pd.read_csv(os.path.join(ROOT, "results", "scope_trends_subsets_ta.csv")))
    with open(os.path.join(os.path.dirname(OUT), "subsets.json"), "w") as fh:
        json.dump({"scopes": subsets}, fh, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    print(f"subsets.json: {len(subsets)} venue combinations, {os.path.getsize(os.path.join(os.path.dirname(OUT), 'subsets.json')) / 1e6:.2f} MB")
    ab = pd.DataFrame([json.loads(l) for l in open(os.path.join(ROOT, "data", "abstracts.jsonl"))])
    abs_words = {int(y): int(w) for y, w in ab.assign(w=ab.abstract.str.split().str.len()).groupby("year").w.median().items()}
    data = {"built": time.strftime("%Y-%m-%d"), "years": config.YEARS, "ref": config.REF_YEAR, "ref_years": config.REF_YEARS,
            "venues": VENUES, "topics": tp, "lp": LP, "ls": LS,
            "abs_words": abs_words,
            "totals": totals, "papers": rows, "scopes": scopes}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"), allow_nan=False)  # fail on NaN/Infinity
    print(f"{len(rows)} papers, {len(tp)} topics, links missing for {missing}; {os.path.getsize(OUT) / 1e6:.2f} MB")
    by = {}
    for r in rows:
        u = (LP[r[3][0]] + r[3][1] + LS[r[3][2]]) if isinstance(r[3], list) else r[3]
        dom = (u.split("/")[2] if u else "none")
        by[dom] = by.get(dom, 0) + 1
    print("link hosts:", dict(sorted(by.items(), key=lambda x: -x[1])))


if __name__ == "__main__":
    main()
