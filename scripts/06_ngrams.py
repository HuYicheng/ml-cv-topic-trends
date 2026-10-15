"""Bottom-up: which title n-grams (1-3) rose or fell, without a predefined topic list.

Weighted log-odds ratio with an informative Dirichlet prior (Monroe, Colaresi & Quinn 2008, Political Analysis
16:372), on document frequencies (a title counts once per n-gram). Papers are reweighted so each venue has the
same share of the corpus in both years being compared (venue mix differs between years, e.g. more RSS in 2026).
ML / CV study: comparisons between years in which the same venues met (ICCV odd years, ECCV even years), plus the most
frequent title n-grams of 2025, to check that the topic rules cover what the venues publish.
"""
import csv
import math
import os
import re
from collections import defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
STOP = set("""a an the and or of for to in on with via by from at as into using based towards toward through
under over is are be its it this that we our their can not without new novel approach method methods framework
system systems study analysis toward using use model models learning network networks neural deep via""".split())


def grams(title):
    toks = [t for t in re.sub(r"[^a-z0-9π]+", " ", title.lower().replace("-", " ")).split()]
    out = set()
    for n in (1, 2, 3):
        for i in range(len(toks) - n + 1):
            g = toks[i:i + n]
            if g[0] in STOP or g[-1] in STOP or all(t.isdigit() for t in g):
                continue
            if n == 1 and (len(g[0]) < 3 or g[0] in STOP):
                continue
            out.add(" ".join(g))
    return out


def compare(papers, venues, y_new, y_old, min_df=12):
    papers = [p for p in papers if p["venue"] in venues and p["year"] in (y_new, y_old)]
    n = defaultdict(int)
    for p in papers:
        n[(p["venue"], p["year"])] += 1
    tot = {y: sum(n[(v, y)] for v in venues) for y in (y_new, y_old)}
    # weight so venue v has the same corpus share in both years as in the old year
    w = {(v, y): (n[(v, y_old)] / tot[y_old]) / (n[(v, y)] / tot[y]) for v in venues for y in (y_new, y_old) if n[(v, y)]}
    cnt = {y_new: defaultdict(float), y_old: defaultdict(float)}
    raw = defaultdict(int)
    for p in papers:
        for g in grams(p["title"]):
            cnt[p["year"]][g] += w[(p["venue"], p["year"])]
            raw[g] += 1
    N1, N2 = tot[y_new], tot[y_old]  # weighted totals equal raw totals by construction
    a0 = 0.1 * (N1 + N2)             # prior mass: 10% of the data (moderate shrinkage)
    rows = []
    for g, df in raw.items():
        if df < min_df:
            continue
        y1, y2 = cnt[y_new][g], cnt[y_old][g]
        aw = a0 * (y1 + y2) / (N1 + N2)
        d = math.log((y1 + aw) / (N1 + a0 - y1 - aw)) - math.log((y2 + aw) / (N2 + a0 - y2 - aw))
        z = d / math.sqrt(1 / (y1 + aw) + 1 / (y2 + aw))
        rows.append({"ngram": g, "z": z, "log_odds": d, f"share_{y_new}": 100 * y1 / N1, f"share_{y_old}": 100 * y2 / N2,
                     "titles": df})
    rows.sort(key=lambda r: -r["z"])
    return rows


def main():
    with open(os.path.join(ROOT, "data", "papers.csv")) as f:
        papers = [dict(r, year=int(r["year"])) for r in csv.DictReader(f)]
    V = ["NeurIPS", "ICML", "ICLR", "CVPR", "ICCV", "ECCV"]
    # ICCV only in odd years, ECCV only in even years: compare years in which the same venues met
    jobs = [("2025_vs_2023", V, 2025, 2023), ("2025_vs_2021", V, 2025, 2021), ("2024_vs_2020", V, 2024, 2020),
            ("2025_vs_2020_ml", ["NeurIPS", "ICML", "ICLR"], 2025, 2020), ("2025_vs_2021_cv", ["CVPR", "ICCV"], 2025, 2021)]
    for name, venues, y1, y2 in jobs:
        rows = compare(papers, venues, y1, y2, min_df=25)
        with open(os.path.join(ROOT, "results", f"ngrams_{name}.csv"), "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
        print(f"wrote results/ngrams_{name}.csv ({len(rows)} n-grams)")
    # most frequent n-grams in the latest full year (document frequency), for topic coverage
    from collections import Counter
    c = Counter(g for p in papers if p["year"] == 2025 for g in grams(p["title"]))
    with open(os.path.join(ROOT, "results", "ngrams_top_2025.csv"), "w", newline="") as f:
        wr = csv.writer(f); wr.writerow(["ngram", "titles_2025"]); wr.writerows(c.most_common(3000))
    print("wrote results/ngrams_top_2025.csv")


if __name__ == "__main__":
    main()
