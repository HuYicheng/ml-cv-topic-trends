"""Extract ML / CV conference papers (years and venues in scripts/config.py) from the dblp XML dump.

Input : ~/.cache/robotics-trends/dblp/dblp.xml.gz (+ dblp.dtd), dblp dump of 2026-10-13 (CC0), the same file as the
        robotics study. Venues are selected by dblp key prefix.
Output: data/dblp_mlcv_<first>_<last>.jsonl (one record per paper, workshops included; 03 keeps the main tracks)
"""
import gzip
import json
import os
import sys
from collections import Counter

from lxml import etree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402

DUMP = os.path.expanduser("~/.cache/robotics-trends/dblp/dblp.xml.gz")
OUT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", config.DBLP_EXTRACT))
YEARS = {str(y) for y in config.YEARS}


def text(el):
    return "".join(el.itertext()).strip() if el is not None else None


def main():
    os.chdir(os.path.dirname(DUMP))
    counts = Counter()
    with open(OUT, "w") as out:
        for _, el in etree.iterparse(gzip.open(DUMP), events=("end",), tag=("inproceedings", "article"),
                                     load_dtd=True, resolve_entities=True, huge_tree=True):
            key = el.get("key", "")
            venue = next((v for p, v in config.DBLP_PREFIX.items() if key.startswith(p)), None)
            year = text(el.find("year"))
            if venue and year in YEARS:
                rec = {"key": key, "venue": venue, "type": el.tag, "year": int(year), "title": text(el.find("title")),
                       "booktitle": text(el.find("booktitle")), "pages": text(el.find("pages")),
                       "ee": [text(e) for e in el.findall("ee")], "publtype": el.get("publtype")}
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                counts[(venue, int(year))] += 1
            el.clear()
            while el.getprevious() is not None:
                del el.getparent()[0]
    for k in sorted(counts):
        print(k, counts[k])


if __name__ == "__main__":
    main()
