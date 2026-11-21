#!/bin/bash
# Full analysis after the abstracts are fetched: merge -> topic matrices -> statistics -> report -> page.
# Since 2026-11-20 the final matrix comes from semantic labels (12a, 12b); 12 writes the keyword matrix, kept for
# comparison and the [term] columns. New papers must be labelled first (12b stops on papers without labels).
# Usage: bash scripts/run_pipeline.sh   (log in logs/pipeline_<time>.log). Round-2 review samples are written in parallel.
set -euo pipefail
cd "$(dirname "$0")"
PY=~/anaconda3/bin/python
step() { echo "=== $(date +%T) $*"; }
step merge;          $PY 09_merge_abstracts.py
step titles;         $PY 04_topics.py
step "keyword matrix"; $PY 12_title_abstract.py
if [ "${SKIP_SAMPLES:-0}" != 1 ]; then
  step "review samples (seed 1, background)"; SAMPLE_SEED=1 $PY 12b_review_samples.py > ../logs/12b_seed1.log 2>&1 &
  SAMPLES=$!
fi
step "semantic labels"; $PY 12a_collect_semantic_labels.py
step "semantic matrix"; $PY 12b_semantic_matrix.py
step shares;         $PY 13_shares_and_change.py > ../logs/13_shares.log; head -n 5 ../logs/13_shares.log
step "shares keyword"; $PY 13_shares_and_change.py topic_matrix_ta_keyword.csv shares_and_change_ta_keyword.csv > ../logs/13_shares_keyword.log
step "shares main";  $PY 13_shares_and_change.py topic_matrix_ta_main.csv shares_and_change_ta_main.csv > ../logs/13_shares_main.log
step sensitivity;    $PY 13c_semantic_sensitivity.py > ../logs/13c_sensitivity.log; tail -n 2 ../logs/13c_sensitivity.log
step figures;        $PY 14_figures_ta.py
step scopes;         $PY 20_scope_trends.py > ../logs/20_scopes.log; tail -n 12 ../logs/20_scopes.log
step subsets;        $PY 20b_scope_subsets.py
step validation;     $PY 13b_semantic_validation.py > ../logs/13b_validation.log
step report;         $PY 15_report.py
step penetration;    $PY 21_method_penetration.py
step "site data";    $PY 18_build_site_data.py
step site;           $PY 19_build_site.py
if [ -n "${SAMPLES:-}" ]; then wait $SAMPLES && step "review samples done: $(tail -n 1 ../logs/12b_seed1.log)"; fi
step done
