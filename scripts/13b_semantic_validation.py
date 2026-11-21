"""Checks of the semantic labels (2026-11-20): stability, comparison with the keyword rules, blind re-judgement of the
disagreements, and what changes in the trends. Needs the raw label and re-judgement files, which are kept locally
(data/semantic_labels_v2/, with paper texts in adjudication/cases/); run after 13 on the three matrices
(results/shares_and_change_ta.csv, _keyword.csv, _main.csv), after 20/20b, whose per-venue tables it compares with the
keyword-rule ones archived at the switch, and after 13c (results/semantic_sensitivity.json). Ported from
~/robotics-trends/scripts/13b_semantic_validation.py and from the checking tools of the comparison, which an independent
audit and a re-check recomputed (2026-11-20; same numbers).

Re-judgement scoring: cells where keyword rules and labels agree are taken as right (cells both get wrong are invisible
here, see "cells both miss"); a disagreement cell counts as true with the yes-rate of its (direction, topic) stratum in
the re-judgement ("unsure" = 0.5); strata are weighted by their size. Two definitions: the labels' own (a passing mention
is not a member) and the keyword rules' own (a passing mention counts, as the keyword version's reading notes said).
Output: results/semantic_validation.md, results/semantic_vs_keyword_by_topic.csv, results/semantic_stability.csv,
        results/semantic_score_by_topic.csv
"""
import collections
import csv
import glob
import json
import os
import statistics as st

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
RAW = os.path.join(ROOT, "data", "semantic_labels_v2")
ADJ = os.path.join(RAW, "adjudication")
ARCHIVE_KW = os.path.join(ROOT, "archive", "2020-2026_keyword_pushed_3cda135", "results")   # keyword results at the switch
NT = 106


def load(pat):
    out = {}
    for fn in sorted(glob.glob(os.path.join(RAW, pat))):
        for line in open(fn, encoding="utf-8"):
            if line.strip():
                r = json.loads(line); out[r["pid"]] = r
    return out


def kappa(a, b):
    n = len(a); po = sum(x == y for x, y in zip(a, b)) / n; pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def main():
    ids = {r["pid"]: r for r in csv.DictReader(open(os.path.join(RAW, "ids.csv"), encoding="utf-8"))}
    kw = {r["uid"]: r for r in csv.DictReader(open(os.path.join(ROOT, "data", "topic_matrix_ta_keyword.csv"), encoding="utf-8"))}
    names = [c for c in next(csv.reader(open(os.path.join(ROOT, "data", "topic_matrix_ta_keyword.csv"), encoding="utf-8")))[3:]
             if not c.startswith("[term]")]
    assert len(names) == NT
    M, R = load("b*.jsonl"), load("r*.jsonl")
    pids = [p for p in ids if p in M]
    assert len(pids) == len(ids) == len(kw)
    per = lambda y: "2020-22" if y <= 2022 else "2023-24" if y <= 2024 else "2025-26"   # noqa: E731
    out = []

    # stability (repeat pass)
    rp = [p for p in R if p in M]
    stab = []
    for k, n in enumerate(names, 1):
        a = [int(k in M[p]["involves"]) for p in rp]; b = [int(k in R[p]["involves"]) for p in rp]
        am = [int(k in M[p]["main"]) for p in rp]; bm = [int(k in R[p]["main"]) for p in rp]
        stab.append({"topic": n, "first_pass": sum(a), "second_pass": sum(b), "kappa": round(kappa(a, b), 4),
                     "first_pass_main": sum(am), "second_pass_main": sum(bm),
                     "kappa_main": round(kappa(am, bm), 4) if 0 < sum(am) + sum(bm) < 2 * len(rp) else float("nan")})
    S_ = pd.DataFrame(stab); S_.to_csv(os.path.join(ROOT, "results", "semantic_stability.csv"), index=False)
    ks = list(S_.kappa); kms = [x for x in S_.kappa_main if x == x]
    same = sum(set(R[p]["involves"]) == set(M[p]["involves"]) for p in rp) / len(rp)
    same2 = sum(set(R[p]["involves"]) == set(M[p]["involves"]) and set(R[p]["main"]) == set(M[p]["main"]) for p in rp) / len(rp)
    low = S_.sort_values("kappa").head(2)

    # keyword vs semantic
    rows, dis = [], collections.defaultdict(list)
    for k, n in enumerate(names, 1):
        x = [int(kw[ids[p]["uid"]][n]) for p in pids]; y = [int(k in M[p]["involves"]) for p in pids]
        rows.append({"topic": n, "keyword": sum(x), "semantic": sum(y), "semantic_main": sum(int(k in M[p]["main"]) for p in pids),
                     "both": sum(a & b for a, b in zip(x, y)), "keyword_only": sum(a & (1 - b) for a, b in zip(x, y)),
                     "semantic_only": sum(b & (1 - a) for a, b in zip(x, y)), "kappa": round(kappa(x, y), 4)})
    C = pd.DataFrame(rows); C.to_csv(os.path.join(ROOT, "results", "semantic_vs_keyword_by_topic.csv"), index=False)
    lp = collections.defaultdict(lambda: [0, 0, 0, 0])
    for p in pids:
        v = lp[ids[p]["year"]]; v[0] += len(M[p]["involves"]); v[1] += len(M[p]["main"]); v[2] += 1; v[3] += len(M[p]["main"]) == 3

    # re-judgement
    key = json.load(open(os.path.join(ADJ, "cases_key.json")))
    adj = {}
    for fn in glob.glob(os.path.join(ADJ, "cases", "out_c*.jsonl")):
        for line in open(fn, encoding="utf-8"):
            if line.strip():
                r = json.loads(line); adj[r["cid"]] = r
    assert set(adj) == set(key), "re-judgement incomplete"
    strict = lambda o: 1.0 if o["verdict"] == "yes" else 0.5 if o["verdict"] == "unsure" else 0.0   # noqa: E731
    loose = lambda o: 1.0 if o["verdict"] == "yes" or (o["verdict"] == "no" and o.get("reason") == "passing") else strict(o)   # noqa: E731
    drawn = collections.Counter((k["type"], k["topic"]) for k in key.values())
    reasons, period = collections.defaultdict(collections.Counter), collections.defaultdict(lambda: [0.0, 0.0])
    typ_w = collections.defaultdict(lambda: [0.0, 0.0])
    for cid, k in key.items():
        o = adj[cid]; w = k["n_stratum"] / drawn[(k["type"], k["topic"])]
        reasons[k["type"]]["unsure" if o["verdict"] == "unsure" else f"{o['verdict']}/{o.get('reason')}"] += w
        pp = period[(k["type"], per(int(ids[k["pid"]]["year"])))]; pp[0] += w * strict(o); pp[1] += w
        tw = typ_w[k["type"]]; tw[0] += w * strict(o); tw[1] += w

    def score(val):
        rate = collections.defaultdict(list)
        for cid, k in key.items():
            rate[(k["type"], k["topic"])].append(val(adj[cid]))
        rate = {s: sum(v) / len(v) for s, v in rate.items()}
        acc = {m: collections.Counter() for m in ("keyword", "semantic")}
        per_topic = collections.defaultdict(lambda: {m: collections.Counter() for m in acc})
        for k, n in enumerate(names, 1):
            for p in pids:
                a = int(kw[ids[p]["uid"]][n]); b = int(k in M[p]["involves"])
                t = a if a == b else rate[("kw_only" if a else "sem_only", k)]
                for m, pred in (("keyword", a), ("semantic", b)):
                    for c in (acc[m], per_topic[n][m]):
                        c["tp"] += pred * t; c["fp"] += pred * (1 - t); c["fn"] += (1 - pred) * t
        return acc, per_topic, rate
    pr = lambda c: (c["tp"] / (c["tp"] + c["fp"]), c["tp"] / (c["tp"] + c["fn"]))   # noqa: E731
    acc, per_topic, rate = score(strict)
    acc2, _, _ = score(loose)
    P = pd.DataFrame([{"topic": n, "keyword_precision": round(pr(per_topic[n]["keyword"])[0], 3), "keyword_recall": round(pr(per_topic[n]["keyword"])[1], 3),
                       "semantic_precision": round(pr(per_topic[n]["semantic"])[0], 3), "semantic_recall": round(pr(per_topic[n]["semantic"])[1], 3),
                       "cases_per_direction": drawn.get(("kw_only", k), 0), "true_members_est": round(per_topic[n]["keyword"]["tp"] + per_topic[n]["keyword"]["fn"])}
                      for k, n in enumerate(names, 1)])
    P.to_csv(os.path.join(ROOT, "results", "semantic_score_by_topic.csv"), index=False)
    # cells both miss, seen through the repeat pass (keyword no, first pass no, second pass yes)
    cells = {(p, k) for p in rp for k, n in enumerate(names, 1)
             if kw[ids[p]["uid"]][n] == "0" and k not in M[p]["involves"] and k in R[p]["involves"]}
    seen = [strict(adj[c]) for c, k in key.items() if k["type"] == "pass_split" and (k["pid"], k["topic"]) in cells]
    miss = len(cells) * len(pids) / len(rp) * (sum(seen) / len(seen))
    (kp, kr), (sp, sr) = pr(acc["keyword"]), pr(acc["semantic"])
    (kp2, kr2), (sp2, sr2) = pr(acc2["keyword"]), pr(acc2["semantic"])
    both, so, sem = int(C.both.sum()), int(C.semantic_only.sum()), int(C.semantic.sum())
    kw_half = (both + 0.5 * int(C.keyword_only.sum())) / int(C.keyword.sum())
    # the judge against a second labelling: on the repeat papers, how often the second pass says yes on the first pass's
    # keyword-only and semantic-only cells
    ko = so_ = ko_y = so_y = 0
    for p in rp:
        u = ids[p]["uid"]
        for k, n in enumerate(names, 1):
            a, b, c = kw[u][n] == "1", k in M[p]["involves"], k in R[p]["involves"]
            if a and not b: ko += 1; ko_y += c
            if b and not a: so_ += 1; so_y += c
    halves = collections.defaultdict(list)
    for c, k in key.items():
        if k["type"] == "pass_split":
            p, t = k["pid"], k["topic"]
            halves[(kw[ids[p]["uid"]][names[t - 1]] == "1", t in M[p]["involves"])].append(strict(adj[c]))

    # trends
    T = {m: pd.read_csv(os.path.join(ROOT, "results", f)).set_index("topic") for m, f in
         (("semantic", "shares_and_change_ta.csv"), ("keyword", "shares_and_change_ta_keyword.csv"), ("main", "shares_and_change_ta_main.csv"))}

    def changes(a, b, col):
        x, y = T[a][col], T[b][col].reindex(T[a].index)
        flips = [t for t in x.index if {x[t], y[t]} == {"rising", "declining"}]
        return int((x != y).sum()), flips, [t for t in x.index if x[t] != y[t]]
    venue = {}
    if os.path.isdir(ARCHIVE_KW):
        for f in ("scope_trends_ta.csv", "scope_trends_subsets_ta.csv"):
            K = pd.read_csv(os.path.join(ARCHIVE_KW, f)); S = pd.read_csv(os.path.join(ROOT, "results", f))
            G = K.merge(S, on=["scope", "topic"], suffixes=("_kw", "_sem"))
            assert len(G) == len(K) == len(S)
            for c in ("call", "rec_call"):
                fl = G[((G[f"{c}_kw"] == "rising") & (G[f"{c}_sem"] == "declining")) | ((G[f"{c}_kw"] == "declining") & (G[f"{c}_sem"] == "rising"))]
                venue[(f, c)] = (len(G), int((G[f"{c}_kw"] != G[f"{c}_sem"]).sum()), collections.Counter(fl.topic).most_common())
    sens = json.load(open(os.path.join(ROOT, "results", "semantic_sensitivity.json"))) if os.path.exists(
        os.path.join(ROOT, "results", "semantic_sensitivity.json")) else None
    other = json.load(open(os.path.join(RAW, "labelled_by_other_model.json")))

    tot_kw, tot_sem = int(C.keyword.sum()), int(C.semantic.sum())
    yrs = sorted(lp)
    out += ["# Semantic topic labels: checks (2026-11-20)", "",
            "Generated by scripts/13b_semantic_validation.py. How the labels were made: data/semantic_labelling_prompt.md;",
            "topic definitions: data/semantic_codebook_v2.md; labels: data/semantic_labels.csv.", "",
            "## Stability", "",
            f"A random {len(rp):,} papers were labelled a second time, in other batches by other agents. Agreement per topic",
            f"(Cohen's kappa) has median {st.median(ks):.2f} and minimum {min(ks):.2f} ({low.iloc[0].topic}; {low.iloc[1].topic} {low.iloc[1].kappa:.2f});",
            f"for the \"main\" labels median {st.median(kms):.2f}, minimum {min(kms):.2f}. {100 * same:.0f}% of these papers got exactly the",
            f"same \"involves\" topics both times ({100 * same2:.0f}% the same \"involves\" and \"main\"). This shows that the model is",
            "consistent with itself, not that it is right."]
    if sens:
        rs = sens["repeat_swap"]
        out.append(f"Giving these papers their second-pass labels instead changes {len(rs['whole'])} of the {NT} whole-period trend calls and "
                   f"{len(rs['recent'])} of the 2024-2026 calls" + (": " + "; ".join(f"{d['topic']} ({d['from']} -> {d['to']}, q {d['q_from']:.3f} -> {d['q_to']:.3f})"
                                                                                    for d in rs["whole"] + rs["recent"]) if rs["whole"] + rs["recent"] else "") + ".")
    out += ["", "## Labels compared with the keyword rules", "",
            f"Keyword rules (data/topic_matrix_ta_keyword.csv): {tot_kw:,} paper-topic labels; semantic labels: {tot_sem:,}; both: {both:,};",
            f"keyword only: {tot_kw - both:,}; semantic only: {so:,}. Per topic: results/semantic_vs_keyword_by_topic.csv.",
            "Topics per paper (semantic, involves / main): " + ", ".join(f"{y} {lp[y][0] / lp[y][2]:.2f} / {lp[y][1] / lp[y][2]:.2f}" for y in yrs) + ".",
            f"A paper has at most 3 \"main\" topics; the share of papers at that cap rose from {100 * lp[yrs[0]][3] / lp[yrs[0]][2]:.1f}% ({yrs[0]}) to "
            f"{100 * lp[yrs[-1]][3] / lp[yrs[-1]][2]:.1f}% ({yrs[-1]}), so the \"main\" table (results/shares_and_change_ta_main.csv) may drift down for",
            "topics that often come with others.", "",
            "## Blind re-judgement of the disagreements", "",
            f"{len(key):,} cells were judged again without saying which method said \"yes\": 8 per topic for each direction of",
            "disagreement, and up to 3 per topic where the two semantic passes disagree. Share of each verdict (weighted by stratum size):", "",
            "| cells | involves, words used | involves, other words | passing mention | other sense of the word | absent | unsure |",
            "|---|---|---|---|---|---|---|"]
    for t, lab in (("kw_only", "keyword only"), ("sem_only", "semantic only"), ("pass_split", "the two passes differ")):
        c = reasons[t]; s = sum(c.values())
        out.append(f"| {lab} | " + " | ".join(f"{100 * c[x] / s:.0f}%" for x in ("yes/explicit", "yes/implicit", "no/passing", "no/other_sense", "no/absent", "unsure")) + " |")
    out += ["",
            "Scored against this re-judgement (per topic: results/semantic_score_by_topic.csv; 8 cases per stratum, so a single",
            "topic's numbers are rough: 8 of 8 right still has a 95% lower bound of about 0.63):",
            f"- by the labels' definition (a passing mention does not count): keyword rules precision {kp:.2f}, recall {kr:.2f};",
            f"  semantic labels {sp:.2f} and {sr:.2f};",
            f"- by the keyword version's own definition (a passing mention counts): keyword rules {kp2:.2f} and {kr2:.2f}; semantic",
            f"  labels {sp2:.3f} and {sr2:.2f}.",
            "The two versions answer different questions: how many papers mention a topic, and how many papers' own work involves it.", "",
            "**What this does not show.** The re-judgement was made by the same model (Claude Opus 5.5) with the same codebook, so",
            "none of these numbers is an independent error rate, and no human check has been made. The judge says \"yes\" more often",
            f"than a second labelling does: on the repeat papers, the second pass says yes on {100 * ko_y / ko:.0f}% of the first pass's keyword-only",
            f"cells ({ko:,}) and on {100 * so_y / so_:.0f}% of its semantic-only cells ({so_:,}); the judge on {100 * typ_w['kw_only'][0] / typ_w['kw_only'][1]:.0f}% and "
            f"{100 * typ_w['sem_only'][0] / typ_w['sem_only'][1]:.0f}% (weighted).",
            "Where the two semantic passes disagree, the judge says yes about as often whichever pass said yes (\"first pass no\" means only the second pass said yes; "
            + "; ".join(f"keyword {'yes' if a else 'no'}, first pass {'yes' if b else 'no'}: {100 * sum(v) / len(v):.0f}% of {len(v)}" for (a, b), v in sorted(halves.items()))
            + "): either these cells are mostly members that each pass sometimes misses, or the judge leans to yes when unsure.",
            "More limits:",
            f"- Cells both methods miss are invisible here. The repeat pass shows some ({len(cells):,} cells where the keyword rules and the",
            f"  first pass say no and the second pass says yes; {100 * sum(seen) / len(seen):.0f}% of the {len(seen)} re-judged ones are yes): at that rate at least about",
            f"  {miss:,.0f} such cells in all papers, which caps recall at about {acc['semantic']['tp'] / (acc['semantic']['tp'] + acc['semantic']['fn'] + miss):.2f} (semantic) and "
            f"{acc['keyword']['tp'] / (acc['keyword']['tp'] + acc['keyword']['fn'] + miss):.2f} (keyword).",
            "- If the semantic-only cells were right less often than the re-judgement says, the semantic precision would be",
            "  " + ", ".join(f"{(both + r * so) / sem:.2f} at {int(100 * r)}%" for r in (0.9, 0.8, 0.7))
            + f"; if the keyword-only cells were half right, the keyword precision would be {kw_half:.2f}.", "",
            "Yes-rate of the re-judged cells by period (weighted): " + "; ".join(
                f"{lab} " + " / ".join(f"{period[(t, q)][0] / period[(t, q)][1]:.2f}" for q in ("2020-22", "2023-24", "2025-26"))
                for t, lab in (("kw_only", "keyword only"), ("sem_only", "semantic only"))) + " (2020-22 / 2023-24 / 2025-26).",
            "With 0-8 cases per topic and period, this cannot show whether a single topic's errors change over the years.", "",
            f"**Model.** All labels and re-judgements are Claude Opus 5.5's, except the last {len(other['pids'])} papers of batch {other['batch']}, which "
            f"Claude Opus 5 ({other['model']}) labelled after a model switch inside that agent."]
    if sens:
        om = sens["other_model"]
        out.append(f"Leaving those {om['papers']} papers out changes {len(om['whole'])} of the {NT} whole-period calls and {len(om['recent'])} of the 2024-2026 calls"
                   + (": " + "; ".join(f"{d['topic']} ({d['from']} -> {d['to']}, q {d['q_from']:.3f} -> {d['q_to']:.3f})" for d in om["whole"] + om["recent"])
                      if om["whole"] + om["recent"] else "") + ".")
    out += ["", "## What changes in the trends", ""]
    n1, f1, c1 = changes("keyword", "semantic", "call"); n2, f2, c2 = changes("keyword", "semantic", "recent_call")
    n3, f3, _ = changes("keyword", "main", "call"); n4, f4, _ = changes("keyword", "main", "recent_call")
    out += [f"All venues together: {n1} of {NT} whole-period calls and {n2} of the 2024-2026 calls differ between the keyword rules and",
            f"the semantic labels ({len(f1)} and {len(f2)} from rising to declining or back). Whole period: " + "; ".join(
                f"{t} ({T['keyword'].loc[t, 'call']} -> {T['semantic'].loc[t, 'call']}, q {T['keyword'].loc[t, 'q']:.3f} -> {T['semantic'].loc[t, 'q']:.3f})" for t in c1) + ".",
            "2024-2026: " + "; ".join(f"{t} ({T['keyword'].loc[t, 'recent_call']} -> {T['semantic'].loc[t, 'recent_call']})" for t in c2) + "."]
    for t in f1:
        yk = [T["keyword"].loc[t, f"share_{y}"] for y in yrs]; ys = [T["semantic"].loc[t, f"share_{y}"] for y in yrs]
        out.append(f"- {t}: standardised share peaks in {yrs[yk.index(max(yk))]} (keyword) and {yrs[ys.index(max(ys))]} (semantic); since 2024 "
                   f"{T['keyword'].loc[t, 'recent_call']} (keyword, x{T['keyword'].loc[t, 'recent_OR']:.2f} a year) and {T['semantic'].loc[t, 'recent_call']} "
                   f"(semantic, x{T['semantic'].loc[t, 'recent_OR']:.2f}).")
    out.append(f"With the \"main\" labels (results/shares_and_change_ta_main.csv) {n3} and {n4} calls differ from the keyword rules"
               + (f"; flips: {', '.join(f3)} (whole period), {', '.join(f4)} (2024-2026)." if f3 + f4 else "."))
    for f, lab in (("scope_trends_ta.csv", "each venue and the ML, CV and all-venue groups"), ("scope_trends_subsets_ta.csv", "the other venue combinations")):
        for c, when in (("call", "whole-period"), ("rec_call", "2024-2026")):
            if (f, c) in venue:
                n, ch, fl = venue[(f, c)]
                out.append(f"By {lab}: {ch} of {n} {when} calls differ ({100 * ch / n:.0f}%)"
                           + (f"; {sum(x for _, x in fl)} flips: " + ", ".join(f"{t} {x}" for t, x in fl) + "." if fl else ", no flips."))
    out += ["Which version is right where they differ has no independent evidence here; several changes are q values crossing 0.05.", ""]
    open(os.path.join(ROOT, "results", "semantic_validation.md"), "w", encoding="utf-8").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
