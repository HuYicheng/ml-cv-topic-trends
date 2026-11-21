# ML & CV Topic Trends

**Page:** https://huyicheng.github.io/ml-cv-topic-trends/

**Submission deadlines** of the six venues, one year from 20 October 2026, with official links: https://huyicheng.github.io/ml-cv-topic-trends/deadlines.html

What 85,364 main-track papers at NeurIPS, ICML, ICLR, CVPR, ICCV and ECCV (2020 to October 2026) are about, read from
their titles and abstracts by a language model against a written topic codebook: the share and the number of papers of
each of 106 topics per year, how they changed, the same for any combination of venues, one small chart per topic, and a
topic word cloud that lists the papers combining any set of topics, each linked to the original. Same framework and
method as [Robotics Topic Trends](https://huyicheng.github.io/robotics-topic-trends/).

## What is measured

- **Papers**: main-track papers from the dblp XML dump of 2026-10-13, plus ICML 2026 from its PMLR volume (306), which the
  dump did not have yet. Workshops, co-located events, Tiny Papers, position papers (title prefix "Position:"),
  competition / demo tracks and retraction notices are removed (`results/removed_nonpapers.txt`). 85,366 papers,
  85,364 with an abstract. 2026 covers ICLR, ICML, CVPR and ECCV; NeurIPS 2026 meets in December; ICCV meets in odd
  years and ECCV in even years.
- **Topics** (since 2026-11-20): each paper's title and abstract were read by a language model (Claude Opus 5.5, run as
  agents of about 125 papers each; the last 50 papers of one batch were labelled by Claude Opus 5 after a model switch,
  and leaving them out changes no whole-period trend call) against a written codebook of 106 topics that says what counts and what does not
  (`data/semantic_codebook_v2.md`). A paper counts for a topic when its own work (method, model, theory, task, data or
  evaluation) involves the topic; a topic named only as background, motivation, contrast, related or future work does
  not count. Labels: `data/semantic_labels.csv` ("involves", the measure used, and separately the 0-3 topics of the main
  contribution); how they were made, with the prompts: `data/semantic_labelling_prompt.md`. Checks
  (`results/semantic_validation.md`): a random 10% labelled a second time agrees with a median kappa of 0.96 across
  topics (lowest 0.89). Before 2026-11-20 each topic was a keyword rule, and a share counted papers that named a topic,
  including in passing; the labels count only papers whose own work involves it. A blind re-judgement of 2,004 cells
  where the rules and the labels disagree found, by the labels' definition, that about 16% of the rules' matches did not
  fit and that the rules missed about 36% of members (by the rules' own definition, with passing mentions counted: 8% and
  33%). That re-judgement was made by the same model with the same codebook and says "yes" more often than a second
  labelling does, so it is not an independent check, and no human check has been made yet. With the labels, 10 of the
  106 whole-period trend calls change (4 from rising to declining or back: Synthetic data / data augmentation up;
  Quantization / pruning / compression, Privacy and Neural fields / NeRF down) and 13% of the per-venue calls. The
  keyword-rule version is kept: `data/topic_matrix_ta_keyword.csv`, `results/shares_and_change_ta_keyword.csv`, rules in
  `scripts/04_topics.py`, rule reviews in `results/precision_review.md`.
- **Shares**: percentage of papers, standardised to the venue mix of 2024 + 2025 (so that both ICCV and ECCV are in the
  reference); a venue that did not meet in a year contributes its share from the model topic ~ venue + year, with a 95%
  interval. **Numbers of papers**: raw counts. **Trends**: logistic regression of topic on venue and year,
  Benjamini–Hochberg correction across topics; rising or declining only at q < 0.05, over 2020-2026 and separately over
  2024-2026. Per venue: raw shares with Wilson intervals and a likelihood-ratio trend test.
- **Limits**: a share counts papers whose own work involves a topic, not papers whose main contribution it is; the labels
  are one model's reading, checked only by the same model; later papers involve more topics on average and abstracts got
  longer (see `results/relative_attention.md`), so part of a small rise can come from that; 2026 is a partial year.

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

`index.html`, `data.json`, `subsets.json`: the page (GitHub Pages). `deadlines.html`: submission deadlines of the six venues for one year from 20 October 2026, each date with the official page it was read from (built by `scripts/22_deadlines_page.py`, which holds the dates). `data/papers.csv`: the paper list.
`data/topic_matrix_ta.csv`: paper × topic matrix of the final measure (semantic labels, "involves");
`data/topic_matrix_ta_main.csv`: the same for the main-contribution labels; `data/topic_matrix_ta_keyword.csv`: the
keyword-rule matrix. `data/semantic_labels.csv`, `data/semantic_codebook_v2.md`, `data/semantic_labelling_prompt.md`: the
labels, the codebook and how the labels were made. `scripts/`: the full pipeline (`scripts/run_pipeline.sh` after the
fetch steps; the raw label files with the repeat pass and the re-judgement files are kept locally, so
`scripts/13b_semantic_validation.py` and `scripts/13c_semantic_sensitivity.py` cannot be rerun from this repository; the
re-judgement cases contain paper texts).
`results/`, `figures/`: outputs; `results/semantic_validation.md` and `results/semantic_*.csv`: checks of the labels.
