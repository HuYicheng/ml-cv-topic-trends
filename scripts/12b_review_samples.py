"""Reading samples for the precision check of the topic rules (results/precision_review.md).

For every topic and period (2020-2023, 2024-2026): up to N random papers matched by the title rule and up to N random
papers matched by the abstract rule only (title does not match), all among papers with an abstract (the population of
the analysis), with the matched words marked [[...]] in the title and in the full abstract, which is printed for every
item so that each judgement can be made from the abstract (user, 2026-10-15: "你不能只按title来判断，你要看abstract"). Same rules and measure as
12_title_abstract.py; the pool sizes are written too, so a precision read on each stratum can be weighted to the topic.
Seed: SAMPLE_SEED (default 0); N: SAMPLE_N (default 10).
Output: results/review_samples/seed<k>/<period>/<topic>.txt, results/review_samples/seed<k>/pools.csv
"""
import csv
import importlib.util
import json
import os
import random
import re

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
spec = importlib.util.spec_from_file_location("t", os.path.join(ROOT, "scripts", "04_topics.py"))
t = importlib.util.module_from_spec(spec); spec.loader.exec_module(t)
PERIODS = {"2020-2023": lambda y: y <= 2023, "2024-2026": lambda y: y >= 2024}


def mark(rx, s):
    return rx.sub(lambda m: "[[" + m.group(0) + "]]", s)


def sentence(rx, a, width=420):
    for s in re.split(r"(?<=[.!?])\s+", a):
        m = rx.search(s, a)
        if m:
            if len(s) > width:
                lo = max(0, min(m.start() - width // 2, len(s) - width))
                s = ("..." if lo else "") + s[lo:lo + width] + ("..." if lo + width < len(s) else "")
            return rx.mark(s, a)
    return ""


def main():
    seed, n = int(os.environ.get("SAMPLE_SEED", "0")), int(os.environ.get("SAMPLE_N", "10"))
    with open(os.path.join(ROOT, "data", "papers.csv")) as f:
        papers = list(csv.DictReader(f))
    abstracts = {r["uid"]: t.strip_urls(r["abstract"]) for r in map(json.loads, open(os.path.join(ROOT, "data", "abstracts.jsonl")))}
    out = os.path.join(ROOT, "results", "review_samples", f"seed{seed}")
    pools = []
    for name, fam, title_rx in t.COMPILED:
        if fam == "E term":
            continue
        abs_rx = t.ABSTRACT_COMPILED[name]
        safe = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")
        for per, keep in PERIODS.items():
            ps = [p for p in papers if keep(int(p["year"]))]
            by_title = [p for p in ps if p["uid"] in abstracts and title_rx.search(p["title"])]
            abs_only = [p for p in ps if p["uid"] in abstracts and not title_rx.search(p["title"])
                        and abs_rx.search(abstracts[p["uid"]])]
            n_abs = sum(1 for p in ps if p["uid"] in abstracts)
            n_title_with_abs = sum(1 for p in by_title if p["uid"] in abstracts)
            pools.append([name, fam, per, len(ps), n_abs, len(by_title), n_title_with_abs, len(abs_only)])
            rnd = random.Random(f"{seed}|{name}|{per}")
            os.makedirs(os.path.join(out, per), exist_ok=True)
            with open(os.path.join(out, per, f"{safe}.txt"), "w") as f:
                f.write(f"# {name} | {per} | papers: {len(ps)} (with abstract {n_abs}) | title matches: {len(by_title)}"
                        f" | abstract-only matches: {len(abs_only)}\n")
                f.write(f"# title rule: {title_rx.pattern}\n# abstract rule: {abs_rx.pattern}\n\n")
                for i, p in enumerate(rnd.sample(by_title, min(n, len(by_title))), 1):
                    f.write(f"[T{i}] {p['uid']} | {p['venue']} {p['year']} | {mark(title_rx, p['title'])}\n"
                            f"     abstract: {abs_rx.mark(abstracts[p['uid']])}\n")
                f.write("\n")
                for i, p in enumerate(rnd.sample(abs_only, min(n, len(abs_only))), 1):
                    f.write(f"[A{i}] {p['uid']} | {p['venue']} {p['year']} | {p['title']}\n"
                            f"     > {sentence(abs_rx, abstracts[p['uid']])}\n"
                            f"     abstract: {abs_rx.mark(abstracts[p['uid']])}\n")
    with open(os.path.join(out, "pools.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["topic", "family", "period", "papers", "papers_with_abstract", "title_matches",
                    "title_matches_with_abstract", "abstract_only_matches"])
        w.writerows(pools)
    print(f"samples for {len(pools) // 2} topics -> {out}")


if __name__ == "__main__":
    main()
