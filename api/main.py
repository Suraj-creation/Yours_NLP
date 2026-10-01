"""FastAPI backend: serves precomputed results and runs the NLP pipeline live.

    uvicorn api.main:app --port 8000        (from the project root)

Every page of the website reads /api/results/<name> (precomputed by
scripts/run_experiments.py) and calls the live endpoints for anything a user types.
The built React app (web/dist) is served at / so one process runs the whole site.
"""
from __future__ import annotations

import json
import pickle
import re
import sys
import time
import warnings
from functools import lru_cache
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, File, HTTPException, UploadFile  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.middleware.gzip import GZipMiddleware  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from nlp_core import bpe as B  # noqa: E402
from nlp_core import pos as P  # noqa: E402
from nlp_core import tokenizers as T  # noqa: E402
from nlp_core.cleaning import clean_text, split_sentences  # noqa: E402
from nlp_core.config import CACHE, RESULTS, load_yaml  # noqa: E402
from nlp_core.evaluate import retrieval_scores  # noqa: E402
from nlp_core.loaders import load_bytes, load_corpus  # noqa: E402
from nlp_core.pipelines import Pipeline, all_run_names, cleaned_corpus  # noqa: E402
from nlp_core.preprocess import LEMMATIZERS, STEMMERS, lemma_wordnet, lemmatize_tokens, stem, stopwords  # noqa: E402
from nlp_core.stats import doc_stats, is_word  # noqa: E402

app = FastAPI(title="Learner-Language Text Analysis & Retrieval", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.add_middleware(GZipMiddleware, minimum_size=1000)
MAX_TEXT = 20000


# ------------------------------------------------------------------ cached state
@lru_cache(maxsize=None)
def result(name: str) -> dict:
    p = RESULTS / "app" / f"{name}.json"
    if not p.exists():
        raise HTTPException(404, f"no results for {name}; run scripts/run_experiments.py")
    return json.loads(p.read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def run(name: str):
    if name not in all_run_names():
        raise HTTPException(404, f"unknown pipeline {name}")
    return Pipeline.from_config(name).run()


@lru_cache(maxsize=None)
def scale_run(collection: str, name: str):
    p = CACHE / f"scale_{collection}_{name}.pkl"
    if not p.exists():
        raise HTTPException(404, f"scale index {collection}/{name} missing; run scripts/run_scale.py")
    data = pickle.loads(p.read_bytes())
    from nlp_core.query import QueryEngine

    return data, QueryEngine(data["index"], data["pipeline"].analyzer())


@lru_cache(maxsize=None)
def docs_by_id() -> dict:
    return {d.doc_id: d for d in load_corpus()}


@lru_cache(maxsize=None)
def clean_by_id() -> dict:
    return dict(cleaned_corpus())


@lru_cache(maxsize=None)
def scale_texts(collection: str) -> dict:
    """Titles and cleaned texts for result snippets; cleaned once, then cached on disk."""
    path = CACHE / f"scale_texts_{collection}.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    from nlp_core.loaders import load_mathdial_scale, load_talkmoves_scale

    docs = load_mathdial_scale() if collection == "mathdial" else load_talkmoves_scale()
    out = {d.doc_id: (d.title, clean_text(d.text, d.fmt)[0]) for d in docs}
    path.write_bytes(pickle.dumps(out))
    return out


def _warm() -> None:
    """Load the selected pipeline and the heavy-dataset indexes in the background at start-up,
    so the first search of a visitor does not pay for unpickling 3,427 documents."""
    try:
        run(winner())
        for coll in ("mathdial", "talkmoves"):
            scale_run(coll, "B")
            scale_texts(coll)
    except Exception as exc:  # warming is an optimisation only
        print(f"warm-up skipped: {exc}")


@app.on_event("startup")
def _startup() -> None:
    import threading

    threading.Thread(target=_warm, daemon=True).start()


def winner() -> str:
    try:
        return result("pipelines")["winner"]
    except HTTPException:
        return "B"


def gold_queries() -> dict[str, dict]:
    import csv

    from nlp_core.config import GOLD

    rel: dict[str, set] = {}
    with open(GOLD / "qrels.csv", newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["relevant"] == "1":
                rel.setdefault(r["qid"], set()).add(r["doc_id"])
    with open(GOLD / "queries.csv", newline="", encoding="utf-8") as fh:
        return {r["query"].strip(): {"qid": r["qid"], "need": r["information_need"], "relevant": rel.get(r["qid"], set())}
                for r in csv.DictReader(fh)}


# ------------------------------------------------------------------ models
class TextIn(BaseModel):
    text: str = Field(..., max_length=MAX_TEXT)


class TokenizeIn(TextIn):
    tokenizers: list[str] = ["nltk", "spacy", "custom", "hybrid"]
    rules: list[str] | None = None


class PreprocessIn(TextIn):
    stopwords: str = "custom"
    stemmer: str = "porter"
    lemmatizer: str = "spacy"
    order: str = "stop_then_norm"


class BpeIn(TextIn):
    merges: int = 1000


class SearchIn(BaseModel):
    query: str = Field(..., max_length=500)
    mode: str = "auto"
    pipeline: str | None = None
    collection: str = "assessment"
    limit: int = 30


class PipelineIn(BaseModel):
    tokenizer: str = "hybrid"
    stopwords: str = "custom"
    normalizer: str = "spacy"
    order: str = "norm_then_stop"
    compound_parts: bool = False


# ------------------------------------------------------------------ meta + results
@app.get("/api/health")
def health():
    return {"ok": True}


@app.get("/api/meta")
def meta():
    cfg = load_yaml("pipelines.yaml")
    corpus = result("corpus")
    pl = result("pipelines")
    best = next(r for r in pl["table_h"] if r["pipeline"] == pl["winner"])
    return {"pipelines": [{"name": k, "label": v["label"], "description": v["description"],
                           "posthoc": bool(v.get("posthoc"))} for k, v in cfg.items()],
            "winner": pl["winner"], "summary": corpus["summary"],
            "tiles": {"documents": corpus["summary"]["documents"], "tokens": corpus["summary"]["tokens"],
                      "vocabulary": corpus["summary"]["vocabulary"], "best_f1": best["f1"], "best_pipeline": pl["winner"]},
            "tokenizers": [{"name": k, "label": v} for k, v in T.LABELS.items()], "rules": list(T.RULES),
            "stemmers": STEMMERS, "lemmatizers": LEMMATIZERS, "encodings": B.ENCODINGS}


@app.get("/api/results/{name}")
def get_result(name: str):
    if not re.fullmatch(r"[a-z_A-Z0-9]+", name):
        raise HTTPException(400, "bad name")
    return JSONResponse(result(name))


@app.get("/api/files")
def files():
    out = []
    for p in sorted(RESULTS.glob("*.*")) + sorted((RESULTS / "scale").glob("*.csv")):
        out.append({"name": str(p.relative_to(RESULTS)), "bytes": p.stat().st_size})
    return out


@app.get("/api/files/{path:path}")
def get_file(path: str):
    p = (RESULTS / path).resolve()
    if not str(p).startswith(str(RESULTS.resolve())) or not p.is_file():
        raise HTTPException(404, "not found")
    return FileResponse(p, filename=p.name)


# ------------------------------------------------------------------ corpus
@app.get("/api/documents/{doc_id}")
def document(doc_id: str, chars: int = 6000):
    d = docs_by_id().get(doc_id)
    if not d:
        raise HTTPException(404, "unknown document")
    clean, log = clean_text(d.text, d.fmt)
    return {"doc_id": d.doc_id, "title": d.title, "filename": d.filename, "format": d.fmt, "meta": d.meta,
            "raw": d.text[:chars], "clean": clean[:chars], "raw_length": len(d.text), "clean_length": len(clean),
            "cleaning": log}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    content = await file.read()
    if len(content) > 15_000_000:
        raise HTTPException(413, "file too large (15 MB max)")
    try:
        d = load_bytes(file.filename or "upload.txt", content)
    except ValueError as exc:
        raise HTTPException(415, str(exc)) from exc
    clean, log = clean_text(d.text, d.fmt)
    sents = split_sentences(clean)
    toks = T.hybrid(clean[:200000])
    st = doc_stats("UPLOAD", clean, toks, sents)
    from nlp_core.ner import entities

    return {"filename": file.filename, "format": d.fmt, "title": d.title, "meta": d.meta, "stats": st, "cleaning": log,
            "preview": clean[:4000], "tokens_sample": toks[:200],
            "entities": entities(clean[:5000], True)[:60]}


# ------------------------------------------------------------------ live NLP
@app.post("/api/tokenize")
def tokenize(inp: TokenizeIn):
    out = []
    for n in inp.tokenizers:
        if n not in T.TOKENIZERS:
            raise HTTPException(400, f"unknown tokenizer {n}")
        toks = T.tokenize(inp.text, n, inp.rules)
        spans = T.align(toks, inp.text)
        out.append({"name": n, "label": T.LABELS[n], "count": len(toks),
                    "tokens": [{"text": t, "start": a, "end": b} for t, (a, b) in zip(toks, spans)]})
    return {"tokenizers": out, "rule_hits": T.token_rule_hits(inp.text, inp.rules)}


@app.post("/api/preprocess")
def preprocess(inp: PreprocessIn):
    if inp.stemmer not in STEMMERS or inp.lemmatizer not in LEMMATIZERS:
        raise HTTPException(400, "unknown stemmer or lemmatizer")
    toks = T.hybrid(inp.text)
    stop = stopwords(inp.stopwords)
    lemmas = lemmatize_tokens(toks, inp.lemmatizer) if toks else []
    rows = []
    for tok, lem in zip(toks, lemmas):
        low = tok.lower()
        st = stem(low, inp.stemmer)
        if inp.order == "stop_then_norm":
            removed = low in stop
        else:
            removed = st in stop
        rows.append({"token": tok, "lower": low, "is_word": is_word(tok), "stopword": removed, "stem": st, "lemma": lem,
                     "stem_leak": inp.order == "norm_then_stop" and st not in stop and low in stop})
    kept = [r for r in rows if r["is_word"] and not r["stopword"]]
    return {"rows": rows, "counts": {"tokens": len(rows), "words": sum(r["is_word"] for r in rows), "kept": len(kept),
                                     "unique_stems": len({r["stem"] for r in kept}),
                                     "unique_lemmas": len({r["lemma"] for r in kept})}}


@app.post("/api/stem")
def stem_words(inp: TextIn):
    words = [w for w in re.findall(r"[A-Za-z']+", inp.text)][:200]
    return [{"word": w, **{s: stem(w.lower(), s) for s in STEMMERS}, "wordnet_nopos": lemma_wordnet(w),
             "spacy": lemmatize_tokens([w], "spacy")[0]} for w in words]


@app.post("/api/pos")
def pos(inp: TextIn):
    toks = T.hybrid(inp.text)[:400]
    if not toks:
        return {"tokens": [], "taggers": {}}
    tags = P.tag_all(toks)
    fired = P.rule_tag(toks, "nltk")[1]
    upos = P.spacy_tag(toks)[1]
    return {"tokens": toks, "taggers": tags, "rules_fired": fired, "upos": upos,
            "labels": {k: v[0] for k, v in P.TAGGERS.items()}}


@app.post("/api/syntax")
def syntax(inp: TextIn):
    from nlp_core.nlp_models import full

    doc = full("hybrid")(inp.text[:3000])
    return {"tokens": [{"i": t.i, "text": t.text, "lemma": t.lemma_, "pos": t.pos_, "tag": t.tag_, "dep": t.dep_,
                        "head": t.head.i} for t in doc],
            "noun_phrases": [nc.text for nc in doc.noun_chunks], "verb_object": P.verb_object_pairs(inp.text[:3000])}


@app.post("/api/ner")
def ner(inp: TextIn):
    from nlp_core.ner import entities

    return {"text": inp.text, "model": entities(inp.text, False), "ruler": entities(inp.text, True)}


@lru_cache(maxsize=4)
def our_bpe(merges: int):
    from collections import Counter

    text = "\n".join(t for _, t in cleaned_corpus())
    return B.SimpleBPE().fit(Counter(B.pretokenize(text)), merges)


@app.post("/api/bpe")
def bpe(inp: BpeIn):
    merges = max(50, min(inp.merges, 4000))
    words = inp.text.split()[:80]
    ours = our_bpe(merges)
    out = []
    for w in words:
        row = {"word": w, "ours": [p for part in (B.pretokenize(w) or [w.lower()]) for p in ours.encode_word(part)]}
        for e in B.ENCODINGS:
            row[e] = B.tiktoken_word(w, e)
        out.append(row)
    return {"words": out, "merges": merges, "our_vocab": ours.vocab_size,
            "sentence": {e: B.tiktoken_pieces(inp.text, e) for e in B.ENCODINGS}}


# ------------------------------------------------------------------ index + search
@app.get("/api/index/{pipeline}/term")
def term(pipeline: str, q: str):
    r = run(pipeline)
    an = r.pipeline.analyzer()(q)
    t = an[0][0] if an else q.lower()
    post = r.index.postings.get(t, {})
    surf = {}
    for d in post:
        for s, c in r.surfaces.get(d, {}).get(t, {}).items():
            surf[s] = surf.get(s, 0) + c
    return {"input": q, "term": t, "analysed": [x for x, _ in an], "df": r.index.df(t), "cf": r.index.cf(t),
            "idf": round(r.index.idf(t), 4), "N": r.index.N,
            "postings": [{"doc_id": d, "tf": len(p), "positions": p[:50], "tfidf": round(r.index.tfidf(t, d), 4)}
                         for d, p in sorted(post.items(), key=lambda x: -len(x[1]))],
            "surface_forms": sorted(surf.items(), key=lambda x: -x[1])[:15]}


@app.get("/api/index/{pipeline}/summary")
def index_summary(pipeline: str):
    r = run(pipeline)
    idx = r.index
    df_counts = {}
    for t, p in idx.postings.items():
        df_counts[len(p)] = df_counts.get(len(p), 0) + 1
    top = sorted(((t, sum(len(v) for v in p.values())) for t, p in idx.postings.items()
                  if re.search(r"[a-z]", t) and len(t) > 2), key=lambda x: -x[1])
    stop = stopwords("custom")
    top = [(t, c) for t, c in top if t not in stop][:30]
    return {"pipeline": pipeline, "vocabulary": idx.vocabulary, "documents": idx.N, "doc_length": idx.doc_len,
            "postings": sum(len(p) for p in idx.postings.values()),
            "positions": sum(len(v) for p in idx.postings.values() for v in p.values()),
            "df_histogram": [{"df": k, "terms": v} for k, v in sorted(df_counts.items())],
            "heatmap": {"terms": [t for t, _ in top], "docs": idx.doc_ids,
                        "cells": [[i, j, len(idx.postings[t].get(d, []))] for i, (t, _) in enumerate(top)
                                  for j, d in enumerate(idx.doc_ids)]}}


def _snippet(text: str, forms: list[str], width: int = 110) -> list[dict]:
    if not forms:
        return [{"text": text[: 2 * width], "hit": False}]
    pat = re.compile(r"(?<![\w-])(" + "|".join(re.escape(f) for f in sorted(set(forms), key=len, reverse=True)[:30]) +
                     r")(?![\w-])", re.I)
    m = pat.search(text)
    if not m:
        return [{"text": text[: 2 * width], "hit": False}]
    a, b = max(0, m.start() - width), min(len(text), m.end() + width)
    seg, out, cur = text[a:b], [], 0
    for mm in pat.finditer(seg):
        if mm.start() > cur:
            out.append({"text": seg[cur:mm.start()], "hit": False})
        out.append({"text": mm.group(0), "hit": True})
        cur = mm.end()
    out.append({"text": seg[cur:], "hit": False})
    if a > 0:
        out[0]["text"] = "... " + out[0]["text"]
    if b < len(text):
        out[-1]["text"] += " ..."
    return out


@app.post("/api/search")
def search(inp: SearchIn):
    name = inp.pipeline or winner()
    if inp.collection == "assessment":
        r = run(name)
        eng, surfaces = r.engine(), r.surfaces
        texts = {k: (docs_by_id()[k].title, v) for k, v in clean_by_id().items()}
    elif inp.collection in ("mathdial", "talkmoves"):
        name = name if name in ("A", "B", "H2") else "B"  # only these three were indexed at scale
        data, eng = scale_run(inp.collection, name)
        surfaces = data["surfaces"]
        texts = scale_texts(inp.collection)
    else:
        raise HTTPException(400, "unknown collection")
    try:
        res = eng.search(inp.query, mode=inp.mode, repeats=5)
    except ValueError as exc:
        raise HTTPException(400, f"query error: {exc}") from exc
    pos_terms = {t for a in res["analysed"] if not a["negated"] for t in a["terms"]}
    rows = []
    for d, score in res["results"][: inp.limit]:
        forms = [s for t in pos_terms for s in surfaces.get(d, {}).get(t, {})]
        title, text = texts[d]
        rows.append({"doc_id": d, "title": title, "score": score, "snippet": _snippet(text, forms)})
    out = {"query": inp.query, "type": res["type"], "pipeline": name, "collection": inp.collection,
           "count": res["count"], "time_ms": res["time_ms"], "analysed": res["analysed"], "results": rows,
           "doc_ids": res["doc_ids"]}
    g = gold_queries().get(inp.query.strip()) if inp.collection == "assessment" else None
    if g:
        out["evaluation"] = {"qid": g["qid"], "information_need": g["need"],
                             **retrieval_scores(res["doc_ids"], g["relevant"]), "relevant_docs": sorted(g["relevant"])}
    return out


@app.post("/api/pipelines/run")
def pipeline_run(inp: PipelineIn):
    from nlp_core.experiments import qrels, queries

    if inp.tokenizer not in ("nltk", "spacy", "custom", "hybrid", "tiktoken_cl100k"):
        raise HTTPException(400, "unknown tokenizer")
    if inp.normalizer not in STEMMERS + ["wordnet_nopos", "wordnet_pos", "wordnet_cf", "spacy", "none"]:
        raise HTTPException(400, "unknown normalizer")
    if inp.stopwords not in ("none", "nltk", "spacy", "custom") or inp.order not in ("stop_then_norm", "norm_then_stop"):
        raise HTTPException(400, "bad option")
    t0 = time.perf_counter()
    p = Pipeline(name="custom", tokenizer=inp.tokenizer, stopwords=inp.stopwords, normalizer=inp.normalizer,
                 order=inp.order, compound_parts=inp.compound_parts)
    r = p.run()
    eng = r.engine()
    rel = qrels()
    per = []
    for q in queries():
        res = eng.search(q["query"], repeats=1)
        per.append({"qid": q["qid"], "query": q["query"], **retrieval_scores(res["doc_ids"], rel[q["qid"]])})
    mean = {k: round(sum(x[k] for x in per) / len(per), 4) for k in ("precision", "recall", "f1", "p@5")}
    return {"config": p.config(), "summary": r.summary(), "stages": r.stages, "per_query": per, "mean": mean,
            "seconds": round(time.perf_counter() - t0, 2)}


@app.post("/api/evidence")
def evidence(inp: TextIn):
    from nlp_core.evidence import evidence_record

    return evidence_record(inp.text[:2000])


# ------------------------------------------------------------------ static site
DIST = ROOT / "web" / "dist"
if DIST.exists():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

    @app.get("/{full_path:path}")
    def spa(full_path: str):
        if full_path.startswith("api/"):  # unknown API routes are errors, not the web app
            raise HTTPException(404, "unknown API route")
        f = (DIST / full_path).resolve()
        if not str(f).startswith(str(DIST.resolve())):
            raise HTTPException(404, "not found")
        if full_path and f.is_file():
            return FileResponse(f)
        return FileResponse(DIST / "index.html")
