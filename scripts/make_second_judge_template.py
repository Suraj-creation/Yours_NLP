"""Write gold/qrels_second_judge_template.csv: a stratified, blind sample for a second relevance judge.

For each of the 15 queries, up to 3 documents judged relevant and 3 judged not relevant are sampled
(seed 42) and shuffled. The first judgement is NOT included, so the second judge works blind.
Fill `second_judge` with 1 or 0; the notebook then reports Cohen's kappa.
"""
from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nlp_core.config import GOLD, SEED  # noqa: E402
from nlp_core.loaders import read_catalog  # noqa: E402


def main() -> None:
    rng = random.Random(SEED)
    titles = {r["doc_id"]: r["title"] for r in read_catalog()}
    queries = {r["qid"]: r for r in csv.DictReader(open(GOLD / "queries.csv", newline=""))}
    rel: dict[str, dict[str, list[str]]] = {}
    for r in csv.DictReader(open(GOLD / "qrels.csv", newline="")):
        rel.setdefault(r["qid"], {"1": [], "0": []})[r["relevant"]].append(r["doc_id"])
    rows = []
    for qid, groups in sorted(rel.items()):
        picked = rng.sample(groups["1"], min(3, len(groups["1"]))) + rng.sample(groups["0"], min(3, len(groups["0"])))
        rng.shuffle(picked)
        for d in picked:
            rows.append({"qid": qid, "query": queries[qid]["query"], "information_need": queries[qid]["information_need"],
                         "doc_id": d, "title": titles[d], "second_judge": ""})
    out = GOLD / "qrels_second_judge_template.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out} ({len(rows)} pairs)")


if __name__ == "__main__":
    main()
