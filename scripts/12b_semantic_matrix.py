"""Final measure since 2026-11-20 (user: "把语义版推上去。要的肯定是这篇文章在做的啊。"): semantic topic labels.

A paper counts for a topic when its own work (method, model, theory, task, data or evaluation) involves the topic, as
read from its title and abstract against data/semantic_codebook_v2.md (labels: data/semantic_labels.csv, collected by
scripts/12a_collect_semantic_labels.py). A topic named only as background, motivation, contrast, related or future work
does not count. The 6 "[term]" columns stay word counts and are copied from the keyword matrix
(data/topic_matrix_ta_keyword.csv, scripts/12_title_abstract.py), which also fixes the papers and their order.
A paper of the keyword matrix without semantic labels stops the script: new papers must be labelled first.
Same as ~/robotics-trends/scripts/12b_semantic_matrix.py.

Output: data/topic_matrix_ta.csv       ("involves"; the final measure, same layout as the keyword matrix)
        data/topic_matrix_ta_main.csv  ("main": the 0-3 topics of the main contribution; described only)
"""
import csv
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
NT = 106


def main():
    kw = list(csv.reader(open(os.path.join(ROOT, "data", "topic_matrix_ta_keyword.csv"), encoding="utf-8")))
    head, cols = kw[0], kw[0][3:]
    topics = [c for c in cols if not c.startswith("[term]")]
    assert len(topics) == NT, len(topics)
    lab = {}
    for r in csv.DictReader(open(os.path.join(ROOT, "data", "semantic_labels.csv"), encoding="utf-8")):
        lab[r["uid"]] = tuple({int(k) for k in r[c].split(";") if k} for c in ("involves", "main"))
    missing = [row[0] for row in kw[1:] if row[0] not in lab]
    if missing:
        raise SystemExit(f"{len(missing)} papers have no semantic labels (first: {missing[:5]}); label them first")
    extra = set(lab) - {row[0] for row in kw[1:]}
    if extra:
        print(f"note: {len(extra)} labelled papers are not in the keyword matrix and are left out")
    for name, which in (("topic_matrix_ta.csv", 0), ("topic_matrix_ta_main.csv", 1)):
        out = [head]
        for row in kw[1:]:
            s = lab[row[0]][which]
            out.append(row[:3] + [v if c.startswith("[term]") else str(int(topics.index(c) + 1 in s)) for c, v in zip(cols, row[3:])])
        with open(os.path.join(ROOT, "data", name), "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(out)
        print(f"{len(out) - 1} papers -> data/{name}")


if __name__ == "__main__":
    main()
