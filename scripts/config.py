"""Single source of the study's scope: ML / CV top conferences, same framework and method as ~/robotics-trends.

User, 2026-10-15: "先做一个对所有ml/cv的顶会的trend。去拆所有的topics。框架就是robotics的框架。完全独立的，方法又完全一样。"
and "不要只看标题，要看摘要". Venues: NeurIPS, ICML, ICLR (machine learning) and CVPR, ICCV, ECCV (computer vision);
AAAI and IJCAI (general AI) are not included. ICCV is held in odd years and ECCV in even years.
"""
YEAR_MIN, YEAR_MAX = 2020, 2026
YEARS = list(range(YEAR_MIN, YEAR_MAX + 1))
VENUES = ["NeurIPS", "ICML", "ICLR", "CVPR", "ICCV", "ECCV"]
GROUPS = {"ML": ["NeurIPS", "ICML", "ICLR"], "Vision": ["CVPR", "ICCV", "ECCV"]}
REF_YEAR = 2025
DBLP_PREFIX = {"conf/nips/": "NeurIPS", "conf/icml/": "ICML", "conf/iclr/": "ICLR",
               "conf/cvpr/": "CVPR", "conf/iccv/": "ICCV", "conf/eccv/": "ECCV"}
DBLP_EXTRACT = f"data/dblp_mlcv_{YEAR_MIN}_{YEAR_MAX}.jsonl"
# main-track proceedings only (dblp booktitle); workshops, co-located events, Tiny Papers, position papers and
# competition / demo tracks are not research articles of the main conference
MAIN_BOOKTITLE = r"^(CVPR|ICCV|NIPS|NeurIPS|NeurIPS Datasets and Benchmarks|ICML|ICLR|ICLR \(Poster\)|ECCV \(\d+\))$"
PMLR_ICML = {2020: "v119", 2021: "v139", 2022: "v162", 2023: "v202", 2024: "v235", 2025: "v267"}
# published proceedings missing from the dblp dump (02_pmlr_additions.py): PMLR volume 306 = Proceedings of ICML 2026
PMLR_EXTRA = {("ICML", 2026): "v306"}
# Reference venue mix for the standardised shares: 2024 + 2025 pooled, because ICCV (odd years) and ECCV (even years)
# never meet in one year; a single reference year would give one of them weight 0. Years in which a venue did not
# meet get that venue's share from the model (venue level + year change), as for IROS / CoRL 2026 in the robotics study.
REF_YEARS = [2024, 2025]
