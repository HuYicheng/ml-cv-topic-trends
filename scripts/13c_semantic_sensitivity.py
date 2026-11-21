"""Sensitivity of the trend calls (step 13 refit, six venues pooled) to two things, each in a throwaway copy:
- other_model: leave out the papers whose labels were written by a model other than Claude Opus 5.5
  (data/semantic_labels_v2/labelled_by_other_model.json: the last 50 papers of batch b617);
- repeat_swap: give the 8,536 repeat papers their second-pass labels instead of the first-pass ones.
Run after 12b and 13. Output: results/semantic_sensitivity.json (per variant: papers changed, and every whole-period
or 2024-2026 call that differs from results/shares_and_change_ta.csv, with both q values).
usage: python3 -I 13c_semantic_sensitivity.py [TMP]  (TMP: a folder that does not exist yet; default under /tmp)"""
import csv
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
RAW = os.path.join(ROOT, "data", "semantic_labels_v2")


def main():
    tmp = sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp(prefix="mlcv_sens_")
    os.makedirs(tmp, exist_ok=True)
    assert not os.listdir(tmp), "TMP must be empty"
    ids = {r["pid"]: r for r in csv.DictReader(open(os.path.join(RAW, "ids.csv"), encoding="utf-8"))}
    rows = list(csv.reader(open(os.path.join(ROOT, "data", "topic_matrix_ta.csv"), encoding="utf-8")))
    head, cols = rows[0], rows[0][3:]
    topics = [c for c in cols if not c.startswith("[term]")]
    other = {ids[p]["uid"] for p in json.load(open(os.path.join(RAW, "labelled_by_other_model.json")))["pids"]}
    R = {}
    for fn in glob.glob(os.path.join(RAW, "r*.jsonl")):
        for line in open(fn, encoding="utf-8"):
            if line.strip():
                r = json.loads(line); R[ids[r["pid"]]["uid"]] = set(r["involves"])
    swapped = [head]
    for r in rows[1:]:
        if r[0] in R:
            r = r[:3] + [v if c.startswith("[term]") else str(int(topics.index(c) + 1 in R[r[0]])) for c, v in zip(cols, r[3:])]
        swapped.append(r)
    base = pd.read_csv(os.path.join(ROOT, "results", "shares_and_change_ta.csv")).set_index("topic")
    res = {}
    try:
        for name, out_rows, n in (("other_model", [head] + [r for r in rows[1:] if r[0] not in other], len(other)),
                                  ("repeat_swap", swapped, len(R))):
            d = os.path.join(tmp, name)
            os.makedirs(os.path.join(d, "data")); os.makedirs(os.path.join(d, "results"))
            shutil.copytree(os.path.join(ROOT, "scripts"), os.path.join(d, "scripts"), ignore=shutil.ignore_patterns("__pycache__"))
            with open(os.path.join(d, "data", "topic_matrix_ta.csv"), "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerows(out_rows)
            subprocess.run([sys.executable, "-I", os.path.join(d, "scripts", "13_shares_and_change.py")], check=True, stdout=subprocess.DEVNULL)
            v = pd.read_csv(os.path.join(d, "results", "shares_and_change_ta.csv")).set_index("topic")
            r = {"papers": n, "whole": [], "recent": []}
            for t in base.index:
                for c, q, k in (("call", "q", "whole"), ("recent_call", "recent_q", "recent")):
                    if base.loc[t, c] != v.loc[t, c]:
                        r[k].append({"topic": t, "from": base.loc[t, c], "to": v.loc[t, c], "q_from": float(base.loc[t, q]), "q_to": float(v.loc[t, q])})
            res[name] = r
            print(name, n, "papers; whole-period changed", len(r["whole"]), "; 2024-2026 changed", len(r["recent"]), r["whole"] + r["recent"])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    json.dump(res, open(os.path.join(ROOT, "results", "semantic_sensitivity.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
