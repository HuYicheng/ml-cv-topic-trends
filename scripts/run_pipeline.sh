#!/bin/bash
# Full analysis after the abstracts are fetched: merge -> topic matrices -> statistics -> report -> page.
# Usage: bash scripts/run_pipeline.sh   (log in logs/pipeline_<time>.log). Round-2 review samples are written in parallel.
set -euo pipefail
cd "$(dirname "$0")"
PY=~/anaconda3/bin/python
step() { echo "=== $(date +%T) $*"; }
step merge;          $PY 09_merge_abstracts.py
step titles;         $PY 04_topics.py
step title+abstract; $PY 12_title_abstract.py
if [ "${SKIP_SAMPLES:-0}" != 1 ]; then
  step "review samples (seed 1, background)"; SAMPLE_SEED=1 $PY 12b_review_samples.py > ../logs/12b_seed1.log 2>&1 &
  SAMPLES=$!
fi
step shares;         $PY 13_shares_and_change.py > ../logs/13_shares.log; head -n 5 ../logs/13_shares.log
step figures;        $PY 14_figures_ta.py
step scopes;         $PY 20_scope_trends.py > ../logs/20_scopes.log; tail -n 12 ../logs/20_scopes.log
step subsets;        $PY 20b_scope_subsets.py
step report;         $PY 15_report.py
step penetration;    $PY 21_method_penetration.py
step "site data";    $PY 18_build_site_data.py
step site;           $PY 19_build_site.py
if [ -n "${SAMPLES:-}" ]; then wait $SAMPLES && step "review samples done: $(tail -n 1 ../logs/12b_seed1.log)"; fi
step done
