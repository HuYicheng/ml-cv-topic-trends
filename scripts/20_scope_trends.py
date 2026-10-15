"""Per-venue analysis (title + abstract measure): share per year and change for every topic, by scope.

ML / CV study, same method as ~/robotics-trends/scripts/20_scope_trends.py. Scopes: all six venues, ML (NeurIPS, ICML,
ICLR), Vision (CVPR, ICCV, ECCV), and each venue.
- single venue: raw share per year with Wilson 95% interval; trend = logistic topic ~ year (linear) within the venue.
- group of venues: share standardised to the group's venue mix of the reference years (config.REF_YEARS) from
  topic ~ C(venue) + C(year), with 95% intervals from 2,000 coefficient draws; trend = topic ~ C(venue) + year.
  Years with no venue of the group present are left empty (e.g. NeurIPS + ICCV in 2026).
Trend test = likelihood-ratio test of the year term; the multiplier is reported only when estimable (no separation).
Benjamini-Hochberg within each scope across topics; call only at q < 0.05.
Output: results/scope_trends_ta.csv, results/venue_matrix_ta.md
"""
import importlib.util
import os
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from scipy.stats import chi2
from statsmodels.stats.proportion import proportion_confint

warnings.filterwarnings("ignore")
import sys  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
spec = importlib.util.spec_from_file_location("t", os.path.join(ROOT, "scripts", "04_topics.py"))
t = importlib.util.module_from_spec(spec); spec.loader.exec_module(t)
TOPICS = [n for n, f, _ in t.TOPICS if f != "E term"]
FAMILY = {n: f.split(" ", 1)[1] for n, f, _ in t.TOPICS}
VENUES = config.VENUES
SCOPES = {"All venues": VENUES, "ML": config.GROUPS["ML"], "Vision": config.GROUPS["Vision"],
          **{v: [v] for v in VENUES}}
m = pd.read_csv(os.path.join(ROOT, "data", "topic_matrix_ta.csv"))
m = m[m.year.isin(config.YEARS)]
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


def single(d, topic, years):
    r = {}
    for y in years:
        g = d[d.year == y][topic]; k, n = int(g.sum()), len(g)
        lo, hi = proportion_confint(k, n, method="wilson")
        r.update({f"share_{y}": 100 * k / n, f"lo_{y}": 100 * lo, f"hi_{y}": 100 * hi, f"k_{y}": k, f"n_{y}": n})
    if d[topic].sum() > 0 and d[topic].sum() < len(d):
        dd = d.assign(y=d[topic], t=d.year - config.YEARS[0])
        f = gfit("t", dd[["year", "t", "y"]])
        f0 = gfit("1", dd[["year", "t", "y"]])
        r.update(trend_stats(f, f0))
    return r


def trend_stats(f, f0):
    """LR test of the year term; multiplier and Wald interval only when the fit did not separate."""
    lr = max(0.0, 2 * (f.llf - f0.llf))
    out = {"p": float(chi2.sf(lr, 1)), "p_wald": float(f.pvalues["t"])}
    b, se = f.params["t"], f.bse["t"]
    if np.isfinite(se) and abs(b) < 10 and se < 10:
        out.update(OR=np.exp(b), OR_lo=np.exp(b - 1.96 * se), OR_hi=np.exp(b + 1.96 * se))
    else:
        out["note"] = "not estimable: topic absent in some years"
    return out


def group(d, topic, years, gen=None):
    gen = rng if gen is None else gen
    dd = d[["venue", "year", topic]].rename(columns={topic: "y"})
    keep = dd.groupby("venue").y.sum(); keep = keep[keep > 0].index
    r = {}
    for y in years:
        g = dd[dd.year == y]; r[f"k_{y}"] = int(g.y.sum()); r[f"n_{y}"] = len(g)
    if len(keep) == 0:                            # topic absent from every venue of the group: share 0 in all years
        for y in years:
            r[f"share_{y}"] = r[f"lo_{y}"] = r[f"hi_{y}"] = 0.0
        return r
    dd = dd[dd.venue.isin(keep)]
    ref = d[d.year.isin(config.REF_YEARS)].venue.value_counts(normalize=True)
    try:
        f1 = gfit("C(venue) + C(year)", dd)
        names = list(f1.params.index)
        draws = gen.multivariate_normal(f1.params.values, f1.cov_params().values, size=2000)
        for y in years:
            X, w = [], []
            for v, share in ref.items():
                if v not in keep:
                    continue                      # venue never has the topic: contributes share 0
                x = np.zeros(len(names)); x[names.index("Intercept")] = 1
                if f"C(venue)[T.{v}]" in names: x[names.index(f"C(venue)[T.{v}]")] = 1
                if f"C(year)[T.{y}]" in names: x[names.index(f"C(year)[T.{y}]")] = 1
                X.append(x); w.append(share)
            X, w = np.array(X), np.array(w)
            r[f"share_{y}"] = 100 * float((w * (1 / (1 + np.exp(-X @ f1.params.values)))).sum())
            sims = ((1 / (1 + np.exp(-draws @ X.T))) * w).sum(axis=1)
            r[f"lo_{y}"], r[f"hi_{y}"] = 100 * np.percentile(sims, 2.5), 100 * np.percentile(sims, 97.5)
        dt = dd.assign(t=dd.year - config.YEARS[0])
        rhs0 = "C(venue)" if dt.venue.nunique() > 1 else "1"
        f2 = gfit(f"{rhs0} + t", dt)
        f0 = gfit(rhs0, dt)
        r.update(trend_stats(f2, f0))
    except Exception as e:  # noqa: BLE001
        r["error"] = str(e)[:80]
    return r


def scope_rows(scope, vs, gen=None):
    """All topics for one scope (a set of venues): shares per year, trend over the whole period and since 2024,
    Benjamini-Hochberg across topics within the scope. gen: random generator for the interval draws."""
    d = m[m.venue.isin(vs)]
    years = sorted(d.year.unique())
    part = []
    for topic in TOPICS:
        r = single(d, topic, years) if len(vs) == 1 else group(d, topic, years, gen)
        r.update(scope=scope, topic=topic, family=FAMILY[topic], years=",".join(map(str, years)))
        y0, y1 = years[0], years[-1]
        if f"share_{y0}" in r and f"share_{y1}" in r:
            r["change_pp"] = r[f"share_{y1}"] - r[f"share_{y0}"]
        # recent trend (2024 onwards) with the same test
        rec_years = [y for y in years if y >= 2024]
        if len(rec_years) >= 2:
            dr = d[d.year.isin(rec_years)]
            rr = single(dr, topic, rec_years) if len(vs) == 1 else group(dr, topic, rec_years, gen)
            r.update({"rec_OR": rr.get("OR"), "rec_p": rr.get("p")})
        part.append(r)
    rps = [(i, r["rec_p"]) for i, r in enumerate(part) if pd.notna(r.get("rec_p"))]
    if rps:
        rqs = multipletests([p for _, p in rps], method="fdr_bh")[1]
        for (i, _), q in zip(rps, rqs):
            part[i]["rec_q"] = q
    for r in part:
        q = r.get("rec_q")
        if q is None or q >= 0.05:
            r["rec_call"] = "unclear"
        else:
            last = int(r["years"].split(",")[-1])
            up = r["rec_OR"] > 1 if pd.notna(r.get("rec_OR")) else r[f"share_{last}"] > r["share_2024"]
            r["rec_call"] = "rising" if up else "declining"
    ps = [(i, r["p"]) for i, r in enumerate(part) if pd.notna(r.get("p"))]
    if ps:
        qs = multipletests([p for _, p in ps], method="fdr_bh")[1]
        for (i, _), q in zip(ps, qs):
            part[i]["q"] = q
    for r in part:
        q = r.get("q")
        if q is None or q >= 0.05:
            r["call"] = "unclear"
        else:   # direction from the data when the multiplier is not estimable
            ys = [int(y) for y in r["years"].split(",")]
            up = r["OR"] > 1 if "OR" in r else r[f"share_{ys[-1]}"] > r[f"share_{ys[0]}"]
            r["call"] = "rising" if up else "declining"
    return part


def main():
    rows = []
    for scope, vs in SCOPES.items():
        rows += scope_rows(scope, vs)
    out = pd.DataFrame(rows)
    # The "All venues" scope is the main table's model; its intervals differ from 13's only by simulation noise. Take 13's
    # intervals so that both files, and the page, show identical numbers.
    main = pd.read_csv(os.path.join(ROOT, "results", "shares_and_change_ta.csv")).set_index("topic")
    allv = out.scope == "All venues"
    for y in config.YEARS:
        out.loc[allv, f"lo_{y}"] = out.loc[allv, "topic"].map(main[f"share_{y}_lo"])
        out.loc[allv, f"hi_{y}"] = out.loc[allv, "topic"].map(main[f"share_{y}_hi"])
    cols = ["scope", "topic", "family", "years"] + [c for y in config.YEARS for c in (f"share_{y}", f"lo_{y}", f"hi_{y}", f"k_{y}", f"n_{y}")] \
           + ["change_pp", "OR", "OR_lo", "OR_hi", "p", "p_wald", "q", "call", "rec_OR", "rec_p", "rec_q", "rec_call", "note", "error"]
    out = out.reindex(columns=[c for c in cols if c in out.columns])
    out.to_csv(os.path.join(ROOT, "results", "scope_trends_ta.csv"), index=False)

    # check: the "All venues" scope must reproduce the main table's point estimates
    a = out[out.scope == "All venues"].set_index("topic")
    dev = max(abs(a.loc[tp, f"share_{y}"] - main.loc[tp, f"share_{y}"]) for tp in TOPICS for y in config.YEARS)
    dor = max(abs(a.loc[tp, "OR"] - main.loc[tp, "annual_OR"]) for tp in TOPICS)
    print(f"All-venues scope vs main table: max share difference {dev:.2e} points, max multiplier difference {dor:.2e}")
    diff_calls = [tp for tp in TOPICS if a.loc[tp, "call"] != main.loc[tp, "call"]]
    print("All-venues calls (LR test) differing from the main table (Wald):", diff_calls or "none")

    # matrix: latest-year share with trend mark, topics x scopes
    mark = {"rising": "↑", "declining": "↓", "unclear": ""}
    order = main.sort_values("share_2025", ascending=False).index
    lines = ["# Topic share by venue (title + abstract)", "",
             "Cell = share in the venue's latest year (2026 for ICLR, ICML, CVPR and ECCV; 2025 for NeurIPS and ICCV); "
             "↑/↓ = trend significant within that venue (q < 0.05 across the topics). Generated by scripts/20_scope_trends.py.", "",
             "| topic | " + " | ".join(SCOPES) + " |", "|---|" + "---|" * len(SCOPES)]
    for tp in order:
        cells = []
        for scope in SCOPES:
            r = out[(out.scope == scope) & (out.topic == tp)].iloc[0]
            y = int(r.years.split(",")[-1])
            cells.append(f"{r[f'share_{y}']:.1f}{mark[r.call]}")
        lines.append(f"| {tp} | " + " | ".join(cells) + " |")
    open(os.path.join(ROOT, "results", "venue_matrix_ta.md"), "w").write("\n".join(lines) + "\n")
    calls = out.groupby("scope").call.value_counts().unstack(fill_value=0).reindex(list(SCOPES))
    print(calls.to_string())


if __name__ == "__main__":
    main()
