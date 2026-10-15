"""ML / CV study (same as ~/robotics-trends/scripts/12_title_abstract.py). Single measure used from here on: a paper belongs to a topic if the title rule matches the title or the abstract
rule matches the abstract (any occurrence). All papers with an abstract (years from scripts/config.py).

Output: data/topic_matrix_ta.csv
        results/audit_abstract_matches.txt   most frequent matched words per rule inside abstracts (all years)
        results/audit_abstract_matches_by_period.txt   the same for 2020-2023 and 2024-2026 side by side (sense drift)
        results/precision_ta_samples/<topic>.txt  random 2024-2026 papers matched by the abstract only, with the matching
            sentence (same draw as the 2024-2026 study: same papers, order and seed)
        results/precision_ta_samples_2020_2023/<topic>.txt  the same for 2020-2023
"""
import collections
import csv
import importlib.util
import json
import os
import random
import re

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
spec = importlib.util.spec_from_file_location("t", os.path.join(ROOT, "scripts", "04_topics.py"))
t = importlib.util.module_from_spec(spec); spec.loader.exec_module(t)


def word_at(a, m):
    s, e = m.start(), m.end()
    while s > 0 and (a[s - 1].isalnum() or a[s - 1] == "-"):
        s -= 1
    while e < len(a) and (a[e].isalnum() or a[e] == "-"):
        e += 1
    prev = re.findall(r"[A-Za-z-]+", a[max(0, s - 25):s])[-1:]
    return ((prev[0].lower() + " ") if prev else "") + a[s:e].lower()


def main():
    with open(os.path.join(ROOT, "data", "papers.csv")) as f:
        papers = {r["uid"]: r for r in csv.DictReader(f)}
    abstracts = {r["uid"]: r["abstract"] for r in map(json.loads, open(os.path.join(ROOT, "data", "abstracts.jsonl")))}
    names = [n for n, _, _ in t.COMPILED]
    title_rx = {n: rx for n, _, rx in t.COMPILED}
    abs_rx = t.ABSTRACT_COMPILED
    rows, abstract_only = [], {n: [] for n in names}
    audit = {n: collections.Counter() for n in names}
    audit_p = {per: {n: collections.Counter() for n in names} for per in ("2020-2023", "2024-2026")}
    for uid, a in abstracts.items():
        p = papers[uid]
        a = t.strip_urls(a)
        row = [uid, p["venue"], p["year"]]
        for n in names:
            in_title = bool(title_rx[n].search(p["title"]))
            hits = list(abs_rx[n].finditer(a))
            per = "2020-2023" if int(p["year"]) <= 2023 else "2024-2026"
            for m in hits:
                audit[n][word_at(a, m)] += 1; audit_p[per][n][word_at(a, m)] += 1
            row.append(int(in_title or bool(hits)))
            if hits and not in_title:
                abstract_only[n].append(uid)
        rows.append(row)
    with open(os.path.join(ROOT, "data", "topic_matrix_ta.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["uid", "venue", "year"] + names); w.writerows(rows)
    with open(os.path.join(ROOT, "results", "audit_abstract_matches.txt"), "w") as f:
        for n in names:
            f.write(f"{n} [{'abstract rule' if n in t.ABSTRACT_OVERRIDES else 'title rule'}] {sum(audit[n].values())} | "
                    + ", ".join(f"{k}:{v}" for k, v in audit[n].most_common(10)) + "\n")
    with open(os.path.join(ROOT, "results", "audit_abstract_matches_by_period.txt"), "w") as f:
        for n in names:
            f.write(f"{n} [{'abstract rule' if n in t.ABSTRACT_OVERRIDES else 'title rule'}]\n")
            for per, a in audit_p.items():
                f.write(f"  {per} {sum(a[n].values())} | " + ", ".join(f"{k}:{v}" for k, v in a[n].most_common(12)) + "\n")
    for sub, keep in (("precision_ta_samples", lambda y: y >= 2024), ("precision_ta_samples_2020_2023", lambda y: y <= 2023)):
        out_dir = os.path.join(ROOT, "results", sub); os.makedirs(out_dir, exist_ok=True)
        rnd = random.Random(int(os.environ.get("SAMPLE_SEED", "0")))
        for n in names:
            uids = [u for u in abstract_only[n] if keep(int(papers[u]["year"]))]
            safe = re.sub(r"[^A-Za-z0-9]+", "_", n).strip("_")
            with open(os.path.join(out_dir, f"{safe}.txt"), "w") as f:
                f.write(f"# {n}: {len(uids)} papers match by abstract only\n")
                for uid in rnd.sample(uids, min(10, len(uids))):
                    a = t.strip_urls(abstracts[uid])
                    sent = next((s for s in re.split(r"(?<=[.!?])\s+", a) if abs_rx[n].search(s, a)), "")
                    f.write(f'{papers[uid]["venue"]} {papers[uid]["year"]} | {papers[uid]["title"][:100]}\n'
                            f'    > {abs_rx[n].mark(sent, a)[:220]}\n')
    print(f"{len(rows)} papers, {len(names)} topics -> data/topic_matrix_ta.csv")


if __name__ == "__main__":
    main()
