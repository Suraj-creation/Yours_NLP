# Domain-Specific Text Analysis and Retrieval System

**Learner language about school mathematics** — NLP Assessment-1, Vidyashilp University (due 04.10.2026).

A complete text front end for the research topic *language-grounded learner state modelling for misconception-aware
adaptive learning*: a 30-document, 10-format corpus; cleaning; four tokenizers scored against gold; BPE; stop words,
stemming and lemmatization; default and custom POS taggers; NER with a domain EntityRuler; n-grams; seven pipelines
compared under a rule fixed in advance; a positional inverted index with Boolean and phrase search; and evaluation with
P, R, F1, P@K, R@K and MAP. Two heavy datasets (full MathDial and TalkMoves, 3.4 million words) show the same pipeline at
scale. A website (FastAPI + React) exposes every module live.

| Deliverable | Where |
|---|---|
| Notebook (primary deliverable, executed) | `Domain_Text_Analysis_Retrieval.ipynb` |
| Corpus, 30 documents | `data/` (catalogue: `results/document_catalog.csv`) |
| Results tables A–J and the index | `results/` (14 required files + supporting evidence) |
| Reusable code | `nlp_core/` |
| Website | `api/` (FastAPI) + `web/` (React 19, Vite, TypeScript, Tailwind 4, ECharts) |
| Gold data | `gold/` (tokenization, POS, NER, queries, qrels, stem pairs, n-gram judgements) |
| Tests | `tests/` (99 pytest tests), `web/e2e/` (Playwright) |

## Headline results

| Step | Result |
|---|---|
| Tokenization (40 gold sentences, 651 tokens) | hybrid F1 **0.996**, NLTK 0.949, spaCy 0.892 |
| Lemmatization (220 tokens) | WordNet + POS **98.2%**, spaCy 99.1%, WordNet without POS 86.8% |
| Custom POS (97 held-out tokens) | rules + dictionary **97.9%**, spaCy 93.8%, CRF 92.8%, NLTK 90.7% |
| NER (161 gold entities) | spaCy 0.489 → EntityRuler + spaCy **0.842** F1 |
| Retrieval (15 queries × 30 docs) | selected pipeline **B**, mean F1 **0.679**; hybrid hypothesis H 0.635 (negative result); post-hoc H2 0.668 (not eligible) |

## Quick start

```bash
# Python 3.11
pip install -r requirements-dev.txt          # requirements.txt alone is enough to serve the site
export NLTK_ALLOW_PROXIED_URLOPEN=1          # only needed behind an HTTPS proxy

# 1. Data (already included in data/ and data_scale/; re-download and rebuild if you want)
python scripts/fetch_data.py                 # MathDial, TalkMoves, MaE, OpenStax, ACL Anthology, tiktoken ranks -> sources/
python scripts/build_corpus.py               # sources/ -> data/D01..D30, data_scale/, catalogue

# 2. Experiments (about 2.5 minutes) and the heavy datasets (about 8 minutes)
python scripts/run_experiments.py            # every module and exercise -> results/
python scripts/run_scale.py                  # MathDial + TalkMoves indexes -> .cache/, results/scale/

# 3. Notebook
python scripts/build_notebook.py --execute   # or open the .ipynb in Jupyter / Colab and run all

# 4. Website
cd web && npm ci && npm run build && cd ..
uvicorn api.main:app --port 8000             # open http://localhost:8000
#   development: `uvicorn api.main:app --reload` and, in web/, `npm run dev` (Vite proxies /api)

# 5. Tests
pytest -q                                    # 99 tests: tokenizers, cleaning, loaders, index, queries, metrics, pipelines, API
cd web && npm run e2e && npm run e2e:dark && npm run e2e:mobile && npm run e2e:interact
```

Docker: `docker build -t learner-language-lab . && docker run -p 8000:8000 learner-language-lab`.

**Google Colab.** Upload the project folder to Drive, open the notebook, set `DATA_SOURCE = "drive"` and `DRIVE_PATH` in
the first cell, then run all.

## Corpus

| Layer | IDs | Documents | Format | Licence |
|---|---|---|---|---|
| Concept | D01–D08 | OpenStax *Prealgebra 2e* §4.1, 4.2, 4.4, 4.5, 5.1, 5.3, 5.6, 6.1 | CNXML + MathML | CC BY-NC-SA 4.0 |
| Research | D09–D12 | ACL Anthology abstracts: ACL 2025, EMNLP 2025, BEA 2023–25, dataset papers | ACL XML | CC BY 4.0 |
| Research | D13–D15 | TalkMoves coding manual; MathDial and MaE dataset cards | PDF, Markdown | CC BY-NC-SA 4.0; CC BY-SA 4.0; MIT |
| Learner | D16–D24 | MaE misconceptions (55 × 4 examples); MathDial 60 fraction dialogues and 150 confusions; 6 TalkMoves lessons | JSON, JSONL, CSV, XLSX | MIT; CC BY-SA 4.0; CC BY-NC-SA 4.0 |
| Synthetic | D25–D30 | Student explanations, student questions, teacher notes, tutor chat, class web page, worksheet | TXT, DOCX, JSON, HTML, PDF | created for this project, labelled |

Scale tier (`data_scale/`): all 2,861 MathDial dialogues (CC BY-SA 4.0) and all 566 TalkMoves transcripts (CC BY-NC-SA 4.0).
TalkMoves and OpenStax are non-commercial licences; the project is for coursework only.

## Pipelines (Module 3)

| Run | Tokenizer | Stop words | Normaliser | Order | Mean F1 |
|---|---|---|---|---|---|
| A | NLTK | NLTK | Porter | stop → stem | 0.643 |
| **B (selected)** | spaCy | none | WordNet + POS | — | **0.679** |
| H | hybrid | custom | spaCy lemma | lemma → stop | 0.635 |
| C1 | NLTK | NLTK | Porter | stem → stop | 0.643 |
| C2 | spaCy | none | WordNet, no POS | — | 0.643 |
| C3 | tiktoken cl100k | none | none | — | 0.431 |
| H2 (post-hoc) | hybrid | custom | WordNet context-free + compound parts | lemma → stop | 0.668 |

Selection rule, fixed before any run: highest mean F1, then P@5, then domain-term fidelity, then the smaller vocabulary;
H2 is excluded because it was designed after H's error analysis on the same queries.

## Website

Eighteen pages, one per requirement: Home; Documents (catalogue, raw vs cleaned text, extractor comparison, upload);
Statistics (Zipf, Heaps, per-document); Heavy datasets; Tokenization (live, gold F1, ablation, Table B, numbers and dates);
Preprocessing (live, Table A, stop lists, stemmers, Table C, order experiment); BPE (live with a merges slider, Table G);
POS tagging (live six-tagger comparison, verb–object pairs); Custom POS (accuracy, confusion matrix, Table D); Named
entities (live model vs ruler, Table E); N-grams (n = 1–5, collocations, bigram network, Table F); Inverted index (term
lookup with postings and positions); Search (keyword, phrase and Boolean over the 30 documents or the heavy datasets, gold
metrics for the 15 queries); Pipelines (Table H, error analysis, design-your-own pipeline); Evaluation (Tables I and J,
P@K/R@K curves); Justification; Evidence preview; Downloads. Light and dark themes, phone layout, and a global pipeline
picker. Charts use one fixed colour per pipeline and per corpus layer from a colour-blind-checked palette.

## Honesty notes

- D25–D30 are synthetic, written to cover formats the public data does not offer; they are labelled in the catalogue.
- `gold/qrels.csv` (450 judgements) was drafted with an AI assistant and **must be reviewed by the student**. A blind
  sample for a second judge is in `gold/qrels_second_judge_template.csv` (`python scripts/make_second_judge_template.py`);
  the notebook reports Cohen's kappa once it is filled in.
- The other gold sets were also drafted with assistance and should be spot-checked. They are small; the notebook and the
  website say where differences are within noise.
- The tiktoken rank files are converted from the `js-tiktoken` npm package (`config/tiktoken_ranks/`) because the usual
  download host was not reachable when building; they were verified against known token ids.
- Hugging Face, Kaggle and the gated Eedi/ASSISTments data were not used.

## Layout

```
nlp_core/     config, loaders, cleaning, tokenizers, preprocess, pos, ner, stats, ngrams, bpe, index, query,
              evaluate, pipelines, experiments, evidence
api/main.py   FastAPI: /api/results/*, live /api/tokenize|preprocess|pos|syntax|ner|bpe|search|pipelines/run|evidence,
              /api/index/*, /api/documents/*, /api/upload, /api/files/*; serves web/dist
web/          React app (src/pages/*, src/components/*), Playwright checks in e2e/
scripts/      fetch_data, build_corpus, gold_*, run_experiments, run_scale, build_notebook, make_second_judge_template
config/       pipelines.yaml, domain_lexicon.csv, entity_patterns.yaml, stopwords_custom.txt, tiktoken_ranks/
```
