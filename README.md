# ML & CV Topic Trends

**Page:** https://huyicheng.github.io/ml-cv-topic-trends/

What 85,364 main-track papers at NeurIPS, ICML, ICLR, CVPR, ICCV and ECCV (2020 to October 2026) are about, read from
their titles and abstracts: the share and the number of papers of each of 106 topics per year, how they changed, the
same for any combination of venues, one small chart per topic, and a topic word cloud that lists the papers combining
any set of topics, each linked to the original. Same framework and method as
[Robotics Topic Trends](https://huyicheng.github.io/robotics-topic-trends/).

## What is measured

- **Papers**: main-track papers from the dblp XML dump of 2026-10-13, plus ICML 2026 from its PMLR volume (306), which the
  dump did not have yet. Workshops, co-located events, Tiny Papers, position papers (title prefix "Position:"),
  competition / demo tracks and retraction notices are removed (`results/removed_nonpapers.txt`). 85,366 papers,
  85,364 with an abstract. 2026 covers ICLR, ICML, CVPR and ECCV; NeurIPS 2026 meets in December; ICCV meets in odd
  years and ECCV in even years.
- **Topics**: each topic is a written rule (`scripts/04_topics.py`). A paper counts for a topic when the rule matches its
  title or its abstract. Words that are generic inside abstracts have stricter abstract rules, and some phrases count
  only when the same abstract (or sentence) has, or lacks, other words (e.g. "adversarial training" only with
  robustness words). Code links are removed from abstracts before matching. Rules were checked in two rounds of reading
  random matched papers, judged from their abstracts, in 2020-2023 and 2024-2026 (`results/precision_review.md`).
- **Shares**: percentage of papers, standardised to the venue mix of 2024 + 2025 (so that both ICCV and ECCV are in the
  reference); a venue that did not meet in a year contributes its share from the model topic ~ venue + year, with a 95%
  interval. **Numbers of papers**: raw counts. **Trends**: logistic regression of topic on venue and year,
  Benjamini–Hochberg correction across topics; rising or declining only at q < 0.05, over 2020-2026 and separately over
  2024-2026. Per venue: raw shares with Wilson intervals and a likelihood-ratio trend test.
- **Limits**: a share counts papers that name a topic, including in passing; keyword rules miss papers that describe a
  topic in other words; abstracts got longer over the years, which gives every topic slightly more chances to be named
  later (see `results/relative_attention.md`); 2026 is a partial year.

Main tables: `results/REPORT.md`, `results/shares_and_change_ta.csv`, `results/scope_trends_ta.csv` and
`results/venue_matrix_ta.md` (by venue), `results/method_penetration.md` (how often each method appears inside each task
or application domain).

## Data sources

| source | used for | licence / terms |
|---|---|---|
| [dblp](https://dblp.org/xml/) XML dump of 2026-10-13 | list of papers, links | CC0 |
| [PMLR](https://proceedings.mlr.press/) bibliographies | ICML papers 2026, ICML abstracts | CC BY 4.0 |
| NeurIPS and ICLR proceedings, ICLR / ECCV virtual sites, CVF open access, ECVA | abstracts | read only to assign topics |
| [OpenAlex](https://openalex.org/) | CVPR / ICCV abstracts by DOI | CC0 |
| arXiv API | abstracts of a few hundred papers, only when the title is identical | read only to assign topics |

Abstract texts are not redistributed here; `scripts/09*.py` fetch them from the sources above (polite rates, robots.txt
respected; OpenReview's API refuses anonymous requests and was not used).

## Files

`index.html`, `data.json`, `subsets.json`: the page (GitHub Pages). `data/papers.csv`: the paper list.
`data/topic_matrix_ta.csv`: paper × topic matrix of the final measure. `scripts/`: the full pipeline
(`scripts/run_pipeline.sh` after the fetch steps). `results/`, `figures/`: outputs.
