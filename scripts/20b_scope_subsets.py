"""Every combination of the six venues, for the page's free venue selection (the same page as the robotics study, where the
user asked for it on 2026-10-15: "可以随意多选，但是也保留全选会议，全选期刊的选项"; here the quick choices are all / ML / vision).

The nine named scopes (all venues, ML, Vision, each venue) are computed by 20_scope_trends.py; this script runs the same
function (scope_rows: standardised shares to the selection's 2024 + 2025 venue mix, likelihood-ratio trend over the whole
period and since 2024, Benjamini-Hochberg within the selection) on the other 54 combinations of two to five venues.
Interval draws use a generator seeded from the combination's name, so a combination gives the same numbers whatever the
order of computation. Output: results/scope_trends_subsets_ta.csv (same columns as results/scope_trends_ta.csv;
scope = venues joined by "+", in the order of scripts/config.py).
"""
import importlib.util
import itertools
import multiprocessing as mp
import os
import zlib

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
spec = importlib.util.spec_from_file_location("s20", os.path.join(ROOT, "scripts", "20_scope_trends.py"))
s20 = importlib.util.module_from_spec(spec); spec.loader.exec_module(s20)
NAMED = {frozenset(v) for v in s20.SCOPES.values()}


def combos():
    out = []
    for k in range(2, len(s20.VENUES)):
        for vs in itertools.combinations(s20.VENUES, k):
            if frozenset(vs) not in NAMED:
                out.append(list(vs))
    return out


def work(vs):
    from threadpoolctl import threadpool_limits
    key = "+".join(vs)
    with threadpool_limits(limits=1):
        return s20.scope_rows(key, vs, np.random.default_rng(zlib.crc32(key.encode())))


def main():
    todo = combos()
    print(f"{len(todo)} venue combinations")
    with mp.get_context("fork").Pool(3) as pool:
        parts = pool.map(work, todo, chunksize=1)
    out = pd.DataFrame([r for part in parts for r in part])
    cols = list(pd.read_csv(os.path.join(ROOT, "results", "scope_trends_ta.csv"), nrows=1).columns)
    out = out.reindex(columns=cols)
    out.to_csv(os.path.join(ROOT, "results", "scope_trends_subsets_ta.csv"), index=False)
    print(f"wrote results/scope_trends_subsets_ta.csv: {out.scope.nunique()} combinations x {out.topic.nunique()} topics")


if __name__ == "__main__":
    main()
