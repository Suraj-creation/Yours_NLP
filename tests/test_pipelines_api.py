"""Module 3 end to end on the real corpus, the results/ contract, and the web API."""
import csv
import json

import pytest
from fastapi.testclient import TestClient

from nlp_core.config import CACHE, GOLD, RESULTS
from nlp_core.pipelines import Pipeline

REQUIRED = ["preprocessing_results.csv", "tokenization_comparison.csv", "stemming_lemmatization.csv", "pos_tagging_results.csv",
            "ner_results.csv", "unigram_results.csv", "bigram_results.csv", "trigram_results.csv", "ngram_results.csv",
            "bpe_results.csv", "pipeline_comparison.csv", "retrieval_results.csv", "evaluation_results.csv", "inverted_index.json"]


def test_all_required_result_files_exist_and_are_not_empty():
    for f in REQUIRED:
        p = RESULTS / f
        assert p.exists() and p.stat().st_size > 100, f


def test_qrels_cover_every_query_and_document():
    rows = list(csv.DictReader(open(GOLD / "qrels.csv", encoding="utf-8")))
    assert len(rows) == 15 * 30
    assert {r["relevant"] for r in rows} <= {"0", "1"}


def test_selection_rule_reproduces_the_winner():
    pl = json.loads((RESULTS / "app" / "pipelines.json").read_text(encoding="utf-8"))
    eligible = [r for r in pl["table_h"] if not r["posthoc"]]
    best = sorted(eligible, key=lambda r: (-r["f1"], -r["p@5"], -r["domain_term_fidelity"], r["vocabulary_size"]))[0]
    assert best["pipeline"] == pl["winner"]
    assert all(r["pipeline"] != "H2" or r["posthoc"] for r in pl["table_h"])


@pytest.fixture(scope="module")
def run_b():
    return Pipeline.from_config("B").run()


def test_pipeline_b_matches_reported_numbers(run_b):
    pl = json.loads((RESULTS / "app" / "pipelines.json").read_text(encoding="utf-8"))
    row = next(r for r in pl["table_h"] if r["pipeline"] == "B")
    assert run_b.index.vocabulary == row["vocabulary_size"]
    res = run_b.engine().search("denominator", repeats=1)
    assert res["count"] == 15 and res["type"] == "Keyword (unigram)"


def test_query_and_documents_share_the_analyser(run_b):
    an = run_b.pipeline.analyzer()
    assert [t for t, _ in an("denominators")] == [t for t, _ in an("denominator")]


@pytest.fixture(scope="module")
def client():
    from api.main import app

    return TestClient(app)


def test_api_meta_and_results(client):
    m = client.get("/api/meta").json()
    assert m["winner"] == "B" and m["tiles"]["documents"] == 30
    for name in ("corpus", "tokenization", "preprocessing", "bpe", "pos", "ner", "ngrams", "pipelines", "scale"):
        assert client.get(f"/api/results/{name}").status_code == 200
    assert client.get("/api/results/..%2Fsecret").status_code in (400, 404)


def test_api_file_download_is_confined_to_results(client):
    from fastapi import HTTPException

    from api.main import get_file

    assert client.get("/api/files/pipeline_comparison.csv").status_code == 200
    assert client.get("/api/files/../api/main.py").status_code == 404
    with pytest.raises(HTTPException):
        get_file("../api/main.py")  # the guard itself, without URL normalisation
    assert client.get("/api/does-not-exist").status_code == 404


def test_api_live_endpoints(client):
    t = client.post("/api/tokenize", json={"text": "I did 1/2+1/3=2/5", "tokenizers": ["nltk", "hybrid"]}).json()
    assert [x["name"] for x in t["tokenizers"]] == ["nltk", "hybrid"]
    assert client.post("/api/tokenize", json={"text": "x", "tokenizers": ["nope"]}).status_code == 400
    p = client.post("/api/preprocess", json={"text": "This was it", "stopwords": "nltk", "stemmer": "porter", "order": "norm_then_stop"}).json()
    assert any(r["stem_leak"] for r in p["rows"])
    assert client.post("/api/pos", json={"text": "Simplify 12/8."}).json()["tokens"] == ["Simplify", "12/8", "."]
    assert any(e["label"] == "KT_MODEL" for e in client.post("/api/ner", json={"text": "BKT on MathDial"}).json()["ruler"])
    assert client.post("/api/bpe", json={"text": "denominators", "merges": 200}).json()["words"][0]["gpt2"]
    assert client.post("/api/evidence", json={"text": "1/8 is bigger than 1/6"}).json()["interpretation"]["detectors"]


def test_api_search(client):
    r = client.post("/api/search", json={"query": "fraction AND misconception"}).json()
    ev = r["evaluation"]
    assert r["type"] == "Boolean AND" and ev["qid"] == "Q06"
    assert isinstance(ev["relevant_docs"], list) and ev["relevant"] == len(ev["relevant_docs"])
    assert r["results"][0]["snippet"]
    assert client.post("/api/search", json={"query": "fraction AND (misconception"}).status_code == 400
    assert client.post("/api/search", json={"query": "x", "pipeline": "ZZ"}).status_code == 404


@pytest.mark.skipif(not (CACHE / "scale_mathdial_B.pkl").exists(), reason="run scripts/run_scale.py first")
def test_api_search_heavy_datasets(client):
    md = client.post("/api/search", json={"query": "common denominator", "collection": "mathdial", "pipeline": "H"}).json()
    assert md["pipeline"] == "B" and all(x["doc_id"].startswith("MD") for x in md["results"])
    tm = client.post("/api/search", json={"query": "half OR halves", "collection": "talkmoves"}).json()
    assert tm["count"] > 100 and all(x["doc_id"].startswith("TM") for x in tm["results"])


def test_api_upload(client):
    r = client.post("/api/upload", files={"file": ("n.txt", b"Aarav paid Rs 250 on 2026-09-01 for 3 1/2 metres.", "text/plain")}).json()
    assert r["format"] == "TXT" and "3 1/2" in r["tokens_sample"]
    assert client.post("/api/upload", files={"file": ("n.exe", b"MZ", "application/octet-stream")}).status_code == 415
