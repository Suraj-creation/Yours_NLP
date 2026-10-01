"""Scale tier: the two heavy datasets in full, through the same pipelines.

    python scripts/run_scale.py

MathDial: 2,861 tutoring dialogues (one document each, MD00001...).
TalkMoves: 567 K-12 mathematics lesson transcripts (TM0001...).
Outputs go to results/scale/ and results/app/scale.json. The indexes are cached so the
website can search the heavy collections too.
"""
from __future__ import annotations

import json
import pickle
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import warnings  # noqa: E402

warnings.filterwarnings("ignore")

import pandas as pd  # noqa: E402

from nlp_core import bpe as B  # noqa: E402
from nlp_core.cleaning import clean_text, split_sentences  # noqa: E402
from nlp_core.config import CACHE, RESULTS  # noqa: E402
from nlp_core.loaders import load_mathdial_scale, load_talkmoves_scale  # noqa: E402
from nlp_core.ngrams import clean_edges, ngrams  # noqa: E402
from nlp_core.pipelines import Pipeline  # noqa: E402
from nlp_core.preprocess import stopwords  # noqa: E402
from nlp_core.stats import heaps, is_word, zipf  # noqa: E402

OUT = RESULTS / "scale"
OUT.mkdir(parents=True, exist_ok=True)
QUERIES = ["denominator", "common denominator", "fraction AND misconception", "added the numerators",
           "decimal OR percent", "half OR halves", "number line", "\"equivalent fraction\"",
           "divide AND NOT fraction", "percent AND (discount OR interest)"]
RUNS = ["A", "B", "H2"]


def main() -> None:
    t0 = time.time()
    collections = {"mathdial": load_mathdial_scale(), "talkmoves": load_talkmoves_scale()}
    print({k: len(v) for k, v in collections.items()}, f"{time.time() - t0:.1f}s")
    summary, run_rows, query_rows, app = [], [], [], {"collections": {}}
    for cname, docs in collections.items():
        t = time.time()
        corpus = tuple((d.doc_id, clean_text(d.text, d.fmt)[0]) for d in docs)
        meta = {d.doc_id: {"title": d.title, **{k: v for k, v in d.meta.items() if k in ("split", "rows", "qid",
                                                                                           "self_correctness")}}
                for d in docs}
        n_sent = sum(len(split_sentences(t)) for _, t in corpus[:400]) / min(400, len(corpus)) * len(corpus)
        print(f"  {cname}: cleaned in {time.time() - t:.1f}s")
        info = {"documents": len(docs), "characters": sum(len(x) for _, x in corpus),
                "estimated_sentences": int(n_sent)}
        runs = {}
        for rn in RUNS:
            t = time.time()
            r = Pipeline.from_config(rn).run(corpus)
            runs[rn] = r
            secs = time.time() - t
            size = len(json.dumps(r.index.to_json(with_positions=True)))
            eng = r.engine()
            lat = []
            for q in QUERIES:
                res = eng.search(q, repeats=5)
                lat.append(res["time_ms"])
                query_rows.append({"collection": cname, "pipeline": rn, "query": q, "type": res["type"],
                                   "results": res["count"], "time_ms": res["time_ms"],
                                   "top5": " ".join(res["doc_ids"][:5])})
            run_rows.append({"collection": cname, "pipeline": rn, "index_terms": r.summary()["token_count"],
                             "vocabulary": r.index.vocabulary, "build_seconds": round(secs, 1),
                             "index_json_mb": round(size / 1e6, 2), "median_query_ms": round(statistics.median(lat), 3)})
            print(f"    run {rn}: {secs:6.1f}s vocab={r.index.vocabulary:,} terms={r.summary()['token_count']:,}")
            with open(CACHE / f"scale_{cname}_{rn}.pkl", "wb") as fh:
                pickle.dump({"index": r.index, "pipeline": r.pipeline, "surfaces": r.surfaces}, fh)
        toks = [t.lower() for pairs in runs["B"].docs_terms.values() for t, _ in pairs]
        raw = [w.lower() for _, x in corpus for w in x.split() if is_word(w)]
        info.update({"tokens_whitespace": len(raw), "index_terms_B": len(toks)})
        summary.append({"collection": cname, **info})
        # learner vs teacher language: phrases in student turns vs teacher turns (B lemmas)
        speaker_ng = {}
        if cname == "talkmoves":
            stud, teach = [], []
            for d in docs:
                for line in d.text.splitlines():
                    spk, _, utt = line.partition(":")
                    words = [w.lower() for w in utt.split() if is_word(w)]
                    (teach if spk == "Teacher" else stud).append(words)
            stop = stopwords("custom")
            for who, sents in (("student", stud), ("teacher", teach)):
                speaker_ng[who] = {"turns": len(sents), "words": sum(map(len, sents)),
                                   "mean_turn_length": round(sum(map(len, sents)) / max(1, len(sents)), 2),
                                   "bigrams": [{"ngram": " ".join(g), "count": c} for g, c in
                                               clean_edges(ngrams(sents, 2), stop).most_common(15)],
                                   "trigrams": [{"ngram": " ".join(g), "count": c} for g, c in
                                                clean_edges(ngrams(sents, 3), stop).most_common(15)]}
        app["collections"][cname] = {"summary": info, "zipf": zipf(Counter(toks)), "heaps": heaps(toks, 50),
                                     "speakers": speaker_ng,
                                     "documents": [{"doc_id": k, **v} for k, v in list(meta.items())[:3000]]}
    # BPE at scale: train a byte-level BPE (Rust) on both heavy datasets, compare with tiktoken
    t = time.time()
    texts = [d.text for docs in collections.values() for d in docs]
    words_total = sum(len(x.split()) for x in texts)
    bpe_rows = []
    for vs in (2000, 8000, 32000):
        tk = B.train_hf_bpe(texts, vs)
        n = sum(len(tk.encode(x).ids) for x in texts[::10]) * 10
        bpe_rows.append({"method": f"our byte-level BPE, vocab {vs}", "vocabulary_size": tk.get_vocab_size(),
                         "tokens_estimate": n, "tokens_per_word": round(n / words_total, 3)})
    for e in B.ENCODINGS:
        enc = B.encoding(e)
        n = sum(len(enc.encode(x, disallowed_special=())) for x in texts[::10]) * 10
        bpe_rows.append({"method": f"tiktoken {e}", "vocabulary_size": enc.n_vocab, "tokens_estimate": n,
                         "tokens_per_word": round(n / words_total, 3)})
    print(f"  BPE at scale {time.time() - t:.1f}s")
    pd.DataFrame(summary).to_csv(OUT / "scale_summary.csv", index=False)
    pd.DataFrame(run_rows).to_csv(OUT / "scale_pipelines.csv", index=False)
    pd.DataFrame(query_rows).to_csv(OUT / "scale_queries.csv", index=False)
    pd.DataFrame(bpe_rows).to_csv(OUT / "scale_bpe.csv", index=False)
    app.update({"summary": summary, "pipelines": run_rows, "queries": query_rows, "bpe": bpe_rows,
                "query_list": QUERIES, "runs": RUNS, "words_total": words_total})
    (RESULTS / "app" / "scale.json").write_text(json.dumps(app, ensure_ascii=False, default=str))
    print(f"total {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
