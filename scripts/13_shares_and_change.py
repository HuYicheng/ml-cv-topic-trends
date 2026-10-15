"""Main table (title + abstract measure): share of each topic per year, comparable across years, and its change.

ML / CV study, same method as ~/robotics-trends/scripts/13_shares_and_change.py. Years and venues from scripts/config.py.
Share per year = topic share standardised to the venue mix of the reference years (config.REF_YEARS = 2024 + 2025, so that
both ICCV (odd years) and ECCV (even years) are in it), from the per-topic logistic model  topic ~ C(venue) + C(year).
Years in which a venue did not meet (ICCV in even years, ECCV in odd years, NeurIPS 2026 not yet held) get
that venue's share from the model, i.e. its venue level moved by the year change seen at the venues that met.
95% intervals by 2,000 draws of the coefficients from their estimated covariance. Raw pooled shares are kept alongside.
Trend: logistic topic ~ C(venue) + year over the whole period (annual odds multiplier, 95% CI); the same over 2024-2026
("recent"). Benjamini-Hochberg q over topics for each; 'rising'/'declining' only when q < 0.05.
Output: results/shares_and_change_ta.csv
"""
import importlib.util
import os
import sys
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
spec = importlib.util.spec_from_file_location("t", os.path.join(ROOT, "scripts", "04_topics.py"))
t = importlib.util.module_from_spec(spec); spec.loader.exec_module(t)
family = {n: f for n, f, _ in t.TOPICS}
YEARS, REF = config.YEARS, config.REF_YEAR
m = pd.read_csv(os.path.join(ROOT, "data", "topic_matrix_ta.csv"))
m = m[m.year.isin(YEARS)]
ref = m[m.year.isin(config.REF_YEARS)].venue.value_counts(normalize=True)
rng = np.random.default_rng(0)


def gfit(rhs, d):
    """Binomial GLM of y on venue / year terms, fitted on counts per venue-year cell instead of one row per paper:
    with only categorical venue / year (or linear year) terms the two give the same estimates, standard errors and
    likelihood-ratio statistics (the per-paper log-likelihood differs only by a constant that cancels in LR tests);
    about 40 rows instead of up to 85,000 (verified against the per-paper fits, ANALYSIS_LOG.md)."""
    keys = [c for c in ("venue", "year", "t") if c in d.columns]
    g = d.groupby(keys, as_index=False).agg(k=("y", "sum"), n=("y", "size"))
    g["nk"] = g["n"] - g["k"]
    return smf.glm(f"k + nk ~ {rhs}", data=g, family=sm.families.Binomial()).fit()


def trend(d, years):
    dd = d[d.year.isin(years)].assign(t=lambda x: x.year - years[0])
    keep = dd.groupby("venue").y.sum(); dd = dd[dd.venue.isin(keep[keep > 0].index)]
    f = gfit("C(venue) + t", dd)
    b, se = f.params["t"], f.bse["t"]
    return np.exp(b), np.exp(b - 1.96 * se), np.exp(b + 1.96 * se), f.pvalues["t"]


rows = []
for topic in [n for n, f, _ in t.TOPICS if f != "E term"]:
    d = m[["venue", "year", topic]].rename(columns={topic: "y"})
    keep = d.groupby("venue").y.sum(); keep = keep[keep > 0].index
    dk = d[d.venue.isin(keep)]
    f1 = gfit("C(venue) + C(year)", dk)
    names = list(f1.params.index)
    draws = rng.multivariate_normal(f1.params.values, f1.cov_params().values, size=2000)
    r = {"topic": topic, "family": family[topic]}
    for yr in YEARS:
        X, w = [], []
        for v, share in ref.items():
            if v not in keep:
                continue                                   # venue never has the topic: contributes share 0
            x = np.zeros(len(names)); x[names.index("Intercept")] = 1
            if f"C(venue)[T.{v}]" in names: x[names.index(f"C(venue)[T.{v}]")] = 1
            if f"C(year)[T.{yr}]" in names: x[names.index(f"C(year)[T.{yr}]")] = 1
            X.append(x); w.append(share)
        X, w = np.array(X), np.array(w)
        r[f"share_{yr}"] = 100 * float((w * (1 / (1 + np.exp(-X @ f1.params.values)))).sum())
        sims = ((1 / (1 + np.exp(-draws @ X.T))) * w).sum(axis=1)
        r[f"share_{yr}_lo"], r[f"share_{yr}_hi"] = 100 * np.percentile(sims, 2.5), 100 * np.percentile(sims, 97.5)
        g = m[m.year == yr]
        r[f"raw_{yr}"] = 100 * g[topic].mean(); r[f"n_{yr}"] = int(g[topic].sum())
    r["annual_OR"], r["annual_OR_lo"], r["annual_OR_hi"], r["p"] = trend(d, YEARS)
    r["recent_OR"], r["recent_OR_lo"], r["recent_OR_hi"], r["recent_p"] = trend(d, [y for y in YEARS if y >= 2024])
    r["change_pp"] = r[f"share_{YEARS[-1]}"] - r[f"share_{YEARS[0]}"]
    r["change_pp_recent"] = r[f"share_{YEARS[-1]}"] - r["share_2024"]
    rows.append(r)
out = pd.DataFrame(rows)
for p, q, c, o in (("p", "q", "call", "annual_OR"), ("recent_p", "recent_q", "recent_call", "recent_OR")):
    out[q] = multipletests(out[p], method="fdr_bh")[1]
    out[c] = np.where(out[q] < 0.05, np.where(out[o] > 1, "rising", "declining"), "unclear")
out = out.sort_values(f"share_{REF}", ascending=False)
out.to_csv(os.path.join(ROOT, "results", "shares_and_change_ta.csv"), index=False)
print("papers with abstracts per year:", m.groupby("year").size().to_dict())
show = out[["topic"] + [f"share_{y}" for y in YEARS] + ["change_pp", "annual_OR", "call", "recent_OR", "recent_call"]].copy()
for c in [f"share_{y}" for y in YEARS]:
    show[c] = show[c].map(lambda v: f"{v:.1f}")
show["change_pp"] = show.change_pp.map(lambda v: f"{v:+.1f}")
for c in ("annual_OR", "recent_OR"):
    show[c] = show[c].round(2)
show["topic"] = show.topic.str.slice(0, 40)
pd.set_option("display.width", 250)
print(show.to_string(index=False))
