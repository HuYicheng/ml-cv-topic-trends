"""Author lists (in order) of the papers in data/papers.csv that come from dblp, read from the same dblp dump as
01_extract_dblp.py. Used only to match papers whose title differs between dblp and the conference's own listing
(09e_match_more.py); authors are not published.
Output: data/dblp_mlcv_authors.jsonl ({"key": ..., "authors": [...]})
"""
import csv
import gzip
import json
import os

from lxml import etree

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
DUMP = os.path.expanduser("~/.cache/robotics-trends/dblp/dblp.xml.gz")


def main():
    keys = {r["uid"] for r in csv.DictReader(open(os.path.join(ROOT, "data", "papers.csv"))) if r["source"] == "dblp"}
    out_path = os.path.join(ROOT, "data", "dblp_mlcv_authors.jsonl")
    os.chdir(os.path.dirname(DUMP))
    n = 0
    with open(out_path + ".tmp", "w") as out:
        for _, el in etree.iterparse(gzip.open(DUMP), events=("end",), tag=("inproceedings", "article"),
                                     load_dtd=True, resolve_entities=True, huge_tree=True):
            k = el.get("key", "")
            if k in keys:
                out.write(json.dumps({"key": k, "authors": ["".join(a.itertext()).strip() for a in el.findall("author")]},
                                     ensure_ascii=False) + "\n")
                n += 1
            el.clear()
            while el.getprevious() is not None:
                del el.getparent()[0]
    os.replace(out_path + ".tmp", out_path)
    print(f"{n} of {len(keys)} dblp papers with author lists -> {out_path}")


if __name__ == "__main__":
    main()
