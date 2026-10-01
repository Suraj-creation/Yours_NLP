"""Run every module and exercise; write results/ (CSV + JSON for the website).

    python scripts/run_experiments.py            # everything
    python scripts/run_experiments.py --only m1  # one part (m1, tok, bpe, pre, pos, ner, ng, ir)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import warnings  # noqa: E402

warnings.filterwarnings("ignore")

from nlp_core import experiments as X  # noqa: E402
from nlp_core.config import GOLD  # noqa: E402
from nlp_core.pipelines import Pipeline, all_run_names  # noqa: E402
from nlp_core.preprocess import write_custom_stopwords  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="all")
    a = ap.parse_args()
    t0 = time.time()
    write_custom_stopwords()
    need_runs = a.only in ("all", "pre", "ng", "ir")
    runs = {}
    if need_runs:
        for n in all_run_names():
            t = time.time()
            runs[n] = Pipeline.from_config(n).run()
            print(f"  run {n:6} {time.time() - t:6.1f}s  tokens={runs[n].summary()['token_count']:,} "
                  f"vocab={runs[n].index.vocabulary:,}")
    winner = None
    if a.only in ("all", "ir", "pre", "ng"):
        cand = GOLD / "ngram_candidates.csv"
        if not (GOLD / "ngram_judgements.csv").exists():
            import csv

            with open(cand, "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["ngram"])
                for g in X.candidate_ngrams(runs):
                    w.writerow([g])
            print(f"wrote {cand} - judge it into gold/ngram_judgements.csv (ngram,meaningful)")
        t = time.time()
        out = X.modules3_6(runs)
        winner = out["winner"]
        print(f"{'Modules 3-6 pipelines + IR':30} done in {time.time() - t:6.1f}s  winner={winner}")
    steps = [("m1", "Module 1 corpus", X.module1), ("tok", "Ex 2, 3, 6 tokenization", X.exercise2_3_6),
             ("bpe", "Ex 4 BPE", X.exercise4), ("pre", "Ex 1, 7-11 preprocessing", lambda: X.preprocessing(runs, winner)),
             ("pos", "Ex 5, 12 POS", X.exercise5_12), ("ner", "Ex 13 NER", X.exercise13),
             ("ng", "Ex 14 n-grams", lambda: X.exercise14(runs, winner))]
    for key, label, fn in steps:
        if a.only in ("all", key):
            t = time.time()
            fn()
            print(f"{label:30} done in {time.time() - t:6.1f}s")
    print(f"total {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
