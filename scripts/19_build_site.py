"""Build the page from site_src/page.html.

site/index.html          full HTML document for GitHub Pages (doctype, head, body), with site/data.json and site/subsets.json beside it
site_preview/index.html  the same content without the document skeleton (the private claude.ai preview adds its own)
"""
import os
import shutil

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
src = open(os.path.join(ROOT, "site_src", "page.html")).read()
head_end = src.index("</style>") + len("</style>")
head, body = src[:head_end], src[head_end:]
full = ("<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1, viewport-fit=cover\">\n"
        "<meta name=\"description\" content=\"Shares, numbers of papers and trends of research topics at NeurIPS, ICML, ICLR, CVPR, ICCV and ECCV, 2020-2026, with per-topic charts and a topic word cloud linking to the papers.\">\n"
        + head + "\n</head>\n<body>\n" + body + "\n</body>\n</html>\n")
os.makedirs(os.path.join(ROOT, "site"), exist_ok=True)
open(os.path.join(ROOT, "site", "index.html"), "w").write(full)
open(os.path.join(ROOT, "site", ".nojekyll"), "w").write("")
os.makedirs(os.path.join(ROOT, "site_preview"), exist_ok=True)
open(os.path.join(ROOT, "site_preview", "index.html"), "w").write(src)
for fn in ("data.json", "subsets.json"):
    shutil.copy(os.path.join(ROOT, "site", fn), os.path.join(ROOT, "site_preview", fn))
print("site/index.html", len(full), "bytes; site_preview/index.html", len(src), "bytes")
