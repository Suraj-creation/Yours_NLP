"""Build Domain_Text_Analysis_Retrieval.ipynb in the order of the assessment brief.

    python scripts/build_notebook.py            # write the notebook
    python scripts/build_notebook.py --execute  # write and run it (about 5 minutes with warm caches)

The notebook is the primary deliverable. Every number it prints is computed by nlp_core, the
same code the website calls, and every table is also written to results/.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "Domain_Text_Analysis_Retrieval.ipynb"
cells: list = []


def md(s: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))


def code(s: str) -> None:
    cells.append(nbf.v4.new_code_cell(s.strip("\n")))


def why(why: str, where: str, depends: str, moved: str) -> None:
    md(f"""
> **Why this step.** {why}
>
> **Where it sits.** {where}
>
> **What it depends on.** {depends}
>
> **If its position changes.** {moved}
""")


# ----------------------------------------------------------------------------- title
md("""
# Domain-Specific Text Analysis and Retrieval System
### Learner language about school mathematics — NLP Assessment-1, Vidyashilp University

**Research context.** The wider project is *language-grounded learner state modelling for misconception-aware adaptive
learning*: a tutor should read what a learner writes ("I added the tops and the bottoms"), connect it to concept and
misconception documents, and use that as evidence about what the learner believes. This assessment builds the text
front end of that loop: corpus construction, cleaning, tokenization, linguistic analysis, a positional inverted index,
Boolean and phrase retrieval, and evaluation. It builds **no** learner model; the last section only previews the
evidence record the research phase will consume.

**How the notebook is organised.** Sections follow the brief: Module 1 (corpus), Module 2 (Exercises 1–14), Module 3
(pipeline design and justification), Module 4 (inverted index), Module 5 (query processing), Module 6 (evaluation).
Each step follows the brief's loop — *select, order, implement, compare, evaluate, justify* — and each design decision
answers four questions: why the step is needed, where it sits, what it depends on and what happens if it moves.

**Where the code lives.** Reusable code is in the `nlp_core` package next to this notebook; the notebook calls it,
shows intermediate results, and writes every results table to `results/`. The same package powers the website
(`api/` FastAPI + `web/` React), so the numbers in the notebook, the CSV files and the website are identical.

**Honesty notes.** Six documents (D25–D30) are synthetic and labelled as such. Relevance judgements (`gold/qrels.csv`)
were drafted with an AI assistant and must be reviewed by the student; a template for a second judge is provided.
Gold sets are small (40 tokenization sentences, 32 POS sentences, 61 NER sentences, 15 queries), so differences of a
point or two are within noise and are described that way.
""")

code("""
# Setup. Run from the project folder (the one that contains nlp_core/ and data/).
# In Colab: upload or mount the project folder, set DATA_SOURCE = "drive" and DRIVE_PATH below.
import os, sys, json, time, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
os.environ.setdefault("NLTK_ALLOW_PROXIED_URLOPEN", "1")

ROOT = Path.cwd() if (Path.cwd() / "nlp_core").exists() else Path.cwd().parent
sys.path.insert(0, str(ROOT))

DATA_SOURCE = "local"   # "local" or "drive"
DRIVE_PATH = "/content/drive/MyDrive/Domain_Text_Analysis_Retrieval/data"

import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown

from nlp_core.config import resolve_data_dir, RESULTS, GOLD
from nlp_core import experiments as X
DATA_DIR = resolve_data_dir(DATA_SOURCE, DRIVE_PATH)

pd.set_option("display.max_colwidth", 120)
pd.set_option("display.width", 200)
# One fixed colour per pipeline across every chart (colour follows the entity, never its rank).
RUN_COLOR = {"A": "#2a78d6", "B": "#eb6834", "H": "#1baf7a", "C1": "#eda100", "C2": "#e87ba4", "C3": "#008300", "H2": "#4a3aa7"}
LAYER_COLOR = {"concept": "#2a78d6", "research": "#eb6834", "learner": "#1baf7a", "synthetic": "#eda100"}
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e6e4df", "grid.linewidth": 0.6, "font.size": 10})
def table(rows, cols=None, n=None):
    df = pd.DataFrame(rows)
    if cols: df = df[cols]
    return df.head(n) if n else df

import spacy, nltk, sklearn, tiktoken
print("data folder:", DATA_DIR, "| documents:", len(list(DATA_DIR.glob("D*"))))
print("spaCy", spacy.__version__, "| NLTK", nltk.__version__, "| scikit-learn", sklearn.__version__, "| tiktoken", tiktoken.__version__)
""")

# ----------------------------------------------------------------------------- module 1
md("""
---
## Module 1 — Corpus construction and cleaning

**Selection.** The corpus has four layers so that a learner's words can be connected to the knowledge they are about
and to the research that names their misconceptions:

| Layer | Documents | Source (licence) |
|---|---|---|
| Concept knowledge | D01–D08 | OpenStax *Prealgebra 2e*, fractions, decimals, ratios, percents — CNXML with MathML (CC BY-NC-SA 4.0) |
| Research | D09–D15 | ACL Anthology 2022–2025 abstracts on knowledge tracing and tutoring (CC BY 4.0); TalkMoves coding manual (PDF); MathDial and MaE dataset cards (Markdown) |
| Learner language | D16–D24 | MaE misconceptions JSON (55 misconceptions, 220 examples); MathDial tutoring dialogues (JSONL, CSV); six TalkMoves lesson transcripts (XLSX) |
| Synthetic learner artefacts | D25–D30 | Written for this project to cover formats the public data lacks: explanations (TXT), questions (TXT), teacher notes (DOCX), tutor chat (JSON), class web page (HTML), worksheet (PDF) — each line labelled with its misconception |

Thirty documents in ten file formats: XML, PDF, Markdown, JSON, JSONL, CSV, XLSX, TXT, DOCX and HTML.
""")
code("""
from nlp_core.loaders import read_catalog, load_corpus
catalog = pd.DataFrame(read_catalog())
display(catalog[["doc_id", "title", "format", "layer", "source", "licence", "synthetic"]])
print(catalog.groupby("layer").size().to_dict(), "| file formats:", catalog["format"].str.split(" ").str[0].nunique())
""")
md("""
### 1.1 Loading every format

Each format has its own loader. The important decisions are what to *keep out* of the text: OpenStax image alt texts
(they describe pictures, not mathematics), MathDial dialogue-act labels such as `(probing)` and TalkMoves talk-move codes
are moved into metadata so that the index holds only what was said. MathML is linearised (`<mfrac>3 4</mfrac>` → `3/4`).
""")
code("""
corpus = load_corpus(DATA_DIR)
for d in corpus[:1] + [corpus[16], corpus[18], corpus[27]]:
    print(f"--- {d.doc_id} [{d.fmt}] {d.title}\\n{d.text[:420]}\\n")
print({d.doc_id: {k: v for k, v in d.meta.items() if k in ('loader', 'alt_texts_removed', 'dialogues', 'rows')} for d in corpus[:3]})
""")
md("""
### 1.2 Choosing an extractor

PDF and HTML extraction quality varies most, so two extractors were compared for each. For PDFs the measure is the
share of extracted words found in a dictionary (broken words such as "T alk Mo ves" lower it); for HTML it is the number
of navigation, cookie and footer phrases that leak into the text.
""")
code("""
from nlp_core.loaders import compare_pdf_extractors, compare_html_extractors
pdf = compare_pdf_extractors(next(DATA_DIR.glob("D13_*")))
html = compare_html_extractors(next(DATA_DIR.glob("D29_*")))
display(table(pdf, ["method", "pages", "words", "dictionary_word_share", "seconds"]))
display(table(html, ["method", "words", "boilerplate_phrases_found"]))
""")
md("""
pdfplumber keeps D13's words whole and trafilatura drops the page chrome, so both are the loaders' defaults.

### 1.3 Cleaning

Cleaning is an ordered list of steps, each logged per document: vulgar fractions → Unicode NFKC → typography →
de-hyphenation → reference sections → URLs → emoji → table markup → page numbers → digit-word glue → whitespace.
""")
code("""
from nlp_core.cleaning import clean_text, split_sentences
for s in ["He ate 3½ pizzas.", "the denomi-\\nnator is 8", "cut into 1/8pieces", "see https://example.org/page for more"]:
    c, log = clean_text(s, "TXT")
    print(f"{s!r:45} -> {c!r:32} {[k for k, v in log.items() if v]}")
import unicodedata
print("NFKC first would give:", repr(unicodedata.normalize("NFKC", "3½")), "- a different number, so vulgar fractions go first")
""")
why("Tokenizers and indexes see whatever the loader hands them; markup, page numbers, hyphenated line breaks, vulgar fractions and URLs would each become tokens or break real ones.",
    "Straight after extraction and before sentence splitting, so every later module reads the same cleaned text.",
    "The right extractor per format (pdfplumber, trafilatura, MathML linearisation).",
    "Cleaning after tokenization leaves '3½' and '1/8pieces' as single wrong tokens; NFKC before the vulgar-fraction step turns '3½' into '31⁄2'.")
md("### 1.4 Corpus statistics")
code("""
m1 = X.module1()
display(pd.DataFrame([m1["summary"]]))
docs = pd.DataFrame(m1["documents"])
display(docs[["doc_id", "format", "layer", "sentences", "tokens", "word_tokens", "vocabulary", "type_token_ratio", "hapax"]])
display(pd.DataFrame(m1["cleaning"]).set_index("doc_id").loc[:, lambda d: d.sum() > 0])
""")
code("""
fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
ax[0].bar(docs.doc_id, docs.tokens, color=[LAYER_COLOR[l] for l in docs.layer], width=0.7)
ax[0].set_title("Tokens per document (colour = layer)"); ax[0].tick_params(axis="x", rotation=90, labelsize=7)
for l, c in LAYER_COLOR.items(): ax[0].bar([], [], color=c, label=l)
ax[0].legend(frameon=False, fontsize=8)
z = pd.DataFrame(m1["zipf"]["points"]); ax[1].loglog(z["rank"], z["freq"], "o", ms=3, color="#2a78d6")
ax[1].set_title(f"Zipf: slope {m1['zipf']['slope']}"); ax[1].set_xlabel("rank"); ax[1].set_ylabel("frequency")
h = pd.DataFrame(m1["heaps"]["points"]); ax[2].plot(h.tokens, h.vocabulary, color="#2a78d6", lw=2, label="observed")
ax[2].plot(h.tokens, m1["heaps"]["K"] * h.tokens ** m1["heaps"]["beta"], color="#8a877f", lw=1.5, ls="--", label="K·N^β")
ax[2].set_title(f"Heaps: K={m1['heaps']['K']}, β={m1['heaps']['beta']}"); ax[2].legend(frameon=False)
plt.tight_layout(); plt.show()
""")
md("""
Both laws hold, which is a sanity check that cleaning did not distort the text. β ≈ 0.82 is high: fractions like
"240/4" and ratios like "3:5" are types the hybrid tokenizer keeps whole, and most of them occur once.
""")

# ----------------------------------------------------------------------------- module 2 set-up
md("""
---
## Module 2 — Text processing and linguistic analysis

Several exercises report numbers "for the final pipeline", which is only known after Module 3 compares the pipelines.
To keep the brief's order, all seven pipeline runs are built once here (each is cached on disk) and the selection is
computed quietly; Module 3 then presents and justifies it. The selection rule was fixed before any run:
**highest mean F1 over the 15 gold queries, then P@5, then domain-term fidelity, then the smaller vocabulary**, with the
post-hoc pipeline H2 excluded.
""")
code("""
from nlp_core.pipelines import Pipeline, all_run_names
from nlp_core.preprocess import write_custom_stopwords
write_custom_stopwords()
runs = {}
for n in all_run_names():
    t = time.time(); runs[n] = Pipeline.from_config(n).run()
    s = runs[n].summary()
    print(f"{n:3} {s['label']:32} tokens={s['token_count']:>7,}  vocabulary={s['vocabulary_size']:>6,}  ({time.time() - t:.1f}s)")
ir = X.modules3_6(runs)          # Module 3-6 results, explained below
FINAL = ir["winner"]
print("selected pipeline:", FINAL)
""")

# Ex 1
md("""
### Exercise 1 — Preprocessing before and after (Table A)

Table A compares the raw extracted text (NLTK word tokens before cleaning) with the index terms of the final pipeline.
""")
code("""
pre = X.preprocessing(runs, FINAL)
display(table(pre["table_a"]))
display(table(pre["stages"]))
""")

# Ex 2, 3, 6
md("""
### Exercise 2 — Tokenization on the corpus

Four tokenizers are compared as the brief asks (NLTK, spaCy, a custom rule tokenizer, a hybrid), plus four NLTK and
whitespace variants as reference points. The **custom** tokenizer is an ordered regex alternation of 14 protected
patterns (URLs and emails, ISO dates and times, currency with its symbol, mixed numbers, arithmetic expressions,
fractions, percents, ratios, numbers, ordinals, abbreviations, hyphenated compounds, alphanumerics, informal forms).
The **hybrid** keeps spaCy's tokenizer and exception handling but removes its letter-hyphen-letter infix and merges
the protected spans, so POS tagging, lemmas and NER run on the same tokens.
""")
code("""
from nlp_core import tokenizers as T
demo = "I did 1/2+1/3=2/5, idk why; 3 1/2 cups cost $3,650 (a 46% rise) on 2026-09-01 in an LLM-based test."
for name in ["whitespace", "nltk", "spacy", "custom", "hybrid"]:
    print(f"{name:10} {T.tokenize(demo, name)}")
tok = X.exercise2_3_6()
display(table(tok["corpus"]))
display(table(tok["audit"]))
""")
md("""
### Exercise 3 — Scoring against hand-segmented gold (Table B)

Forty sentences (651 gold tokens) were chosen for the hard cases and tagged with a problem type. Scores are token-level
precision, recall and F1 over exact spans. Custom and hybrid share the rules, so the gold spans — written with those
rules in mind — favour them; the fair comparison is on hyphenated terms, acronyms and informal text.
""")
code("""
display(table(tok["scores"], ["label", "precision", "recall", "f1", "sentence_exact_match"] + [c for c in tok["scores"][0] if c.startswith("f1_")]))
display(table(tok["ablation"]))
display(table(tok["table_b"]))
""")
md("""
Removing a rule from the hybrid tokenizer and re-scoring shows which rules earn their place. Seven show no drop: spaCy
already keeps a lone 1/2, 3:5 or 1st whole, and URLs are not in the gold set; they stay because the standalone custom
tokenizer needs them.
""")
why("Every later stage counts tokens: if 1/2 becomes three tokens the index cannot find the fraction, the POS tagger sees a symbol between numbers and NER cannot see a money amount.",
    "First step after cleaning and sentence splitting; the hybrid tokenizer runs inside spaCy.",
    "Cleaned text with vulgar fractions expanded and digit-word glue split.",
    "Tokenizing before cleaning keeps the glue errors. Protecting spans also changes NER: once $3,650 is one token spaCy's model stops tagging it, so the EntityRuler adds MONEY patterns (Exercise 13).")

# Ex 4
md("""
### Exercise 4 — Byte-pair encoding

BPE starts from characters and repeatedly merges the most frequent adjacent pair. Three OpenAI encodings (tiktoken
gpt2, cl100k_base, o200k_base; the rank files are converted from the `js-tiktoken` npm package because the usual download
host is not reachable here, and verified: gpt2 has 50,257 entries and "Hello" encodes to 15496) are compared with our own
BPE trained on this corpus with 500, 1,000 and 2,000 merges, and with a word-level `SimpleTokenizerV2` as in the lab.
""")
code("""
from nlp_core import bpe as B
enc = B.encoding("gpt2"); print("gpt2 n_vocab", enc.n_vocab, "| 'Hello' ->", enc.encode("Hello"))
bp = X.exercise4()
display(table(bp["words"], ["word", "class", "gpt2", "cl100k_base", "o200k_base", "ours_500", "ours_1000", "ours_2000"]))
display(table(bp["per_class"])); display(table(bp["summary"]))
print("first merges:", [m["merged"] for m in bp["first_merges"][:15]])
print("SimpleTokenizerV2:", bp["v2_example"])
c = pd.DataFrame(bp["curve"])
fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
ax[0].plot(c.merges, c.vocabulary_size, "o-", color="#2a78d6", lw=2); ax[0].set_title("vocabulary vs merges")
ax[1].plot(c.merges, c.corpus_tokens, "o-", color="#eb6834", lw=2); ax[1].set_title("corpus tokens vs merges")
plt.tight_layout(); plt.show()
""")
md("""
All methods keep common words whole. Web-trained encodings handle rare general words far better, but our 2,000-merge
vocabulary splits domain words (denominators, misconception, reciprocal) into fewer pieces than any of them. Subword
methods never meet an unknown word; the word-level tokenizer maps unseen words to `<|unk|>`.
""")

# Ex 5
md("""
### Exercise 5 — POS tagging, verbs and noun phrases

Default taggers are trained on newspaper text. In a maths lesson, sentence-initial imperatives (*Simplify*, *Round*),
fraction words (*halves*), the variable *x* and *over* as a fraction bar are tagged wrongly often enough to matter.
Verb–object pairs come from spaCy's dependency parse over hybrid tokens.
""")
code("""
pos = X.exercise5_12()
display(table(pos["identification"]))
for x in pos["ex5"]["verbs_and_nouns"]:
    print(x["text"]); print("   verb-object:", [(v["verb"], v["object"]) for v in x["verb_object"]]); print("   noun phrases:", x["noun_phrases"])
for x in pos["ex5"]["nouns_only"]:
    print(x["text"]); print("   noun phrases:", x["noun_phrases"])
""")
md("""
Verb–object pairs are the most useful output for the research: "add → the tops" and "add → the bottoms" name the
procedure a learner used.

### Exercise 6 — Numbers and dates

Share of each kind of number kept as one token (exact span match), over the whole corpus.
""")
code("display(table(tok['numbers_dates']))")

# Ex 7-11
md("""
### Exercises 7–11 — Stop words, stemming, lemmatization and their order

**Exercise 7.** Standard stop lists delete words that carry meaning in mathematics (*not, more, than, over, under,
same, each*). The custom list starts from NLTK, keeps those, and adds corpus noise (*et, al, fig, um, uh*).
""")
code("""
display(table(pre["stopwords"]))
print("kept back from NLTK:", pre["kept_words"])
""")
md("**Exercises 8 and 9.** Four stemmers and three lemmatizers on 64 word pairs: 40 that should merge and 24 that should not.")
code("""
display(table(pre["stemmers"]))
display(table(pre["stem_errors"]).groupby(["method", "error"]).size().unstack(fill_value=0))
display(table(pre["table_c"]))
""")
md("**Exercise 10.** Lemmatizer accuracy on 220 word tokens with hand-written lemmas.")
code("display(table(pre['lemmas']))")
md("""
WordNet without a POS tag treats every word as a noun ("is", "added" stay; "uses" becomes "us"); with the tag it reaches
98%. **Exercise 11.** Stemming before stop-word removal changes stop words into forms the list no longer recognises:
""")
code("display(table(pre['order']))")
why("Stop words dominate counts and n-grams; stems and lemmas merge the inflected forms a query should match.",
    "After tokenization and lower-casing; in A stop words go first, then Porter; in B each sentence is POS-tagged and WordNet lemmatizes with the tag.",
    "Stemmers need lower-cased tokens; the WordNet lemmatizer needs the POS tagger (Exercise 5).",
    "Stemming first leaks 'thi', 'ha', 'wa' into the index (above); lemmatizing without POS drops accuracy from 98% to 87%; removing stop words before tagging removes the tagger's context.")

# Ex 12
md("""
### Exercise 12 — Custom POS taggers

Two kinds, as the brief asks. A **rules and dictionary** layer over NLTK or spaCy (six ordered rules and a 37-entry domain
lexicon), and **machine-learned** taggers — an NLTK trigram → bigram → unigram → regex backoff chain and a CRF
(sklearn-crfsuite) — trained on the 3,914-sentence Penn Treebank sample plus 20 domain sentences weighted five times.
All are scored on 12 held-out domain sentences (97 tokens).
""")
code("""
display(table(pos["accuracy"]))
display(table(pos["table_d"]))
import numpy as np
cm = np.array(pos["confusion"]["rules_nltk"]); labels = pos["labels"]
fig, ax = plt.subplots(figsize=(6.5, 5.5)); ax.imshow(cm, cmap="Blues"); ax.grid(False)
ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=7); ax.set_yticks(range(len(labels)), labels, fontsize=7)
for i in range(len(labels)):
    for j in range(len(labels)):
        if cm[i, j]: ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7, color="white" if cm[i, j] > cm.max() / 2 else "black")
ax.set_title("Rules over NLTK: gold (rows) vs predicted"); plt.tight_layout(); plt.show()
""")
md("""
The rule layer wins (97.9% vs NLTK 90.7%) because the errors are systematic: the same few words in the same positions.
The CRF learns some of them from features but still calls sentence-initial "Round" a noun — 20 domain sentences are too
few to override the Treebank. With 97 test tokens, one token is one point.
""")

# Ex 13
md("""
### Exercise 13 — Named entities

spaCy's model knows people, places and dates but not concepts, misconceptions, knowledge-tracing models, datasets or
metrics. An EntityRuler placed **before** the model adds those five types from `config/domain_lexicon.csv`, plus money,
ISO-date and product patterns. Both are scored against 161 hand-annotated entities in 61 sentences (exact span and type).
""")
code("""
from nlp_core.ner import entities
s = "Ms. Priya Raman compared BKT and DKT on MathDial using AUC, and bought fraction tiles for Rs 2,500 on 12 August 2026."
print("model:", [(e["text"], e["label"]) for e in entities(s, False)])
print("ruler:", [(e["text"], e["label"]) for e in entities(s, True)])
ner = X.exercise13()
print({k: ner["model"]["overall"][k] for k in ("precision", "recall", "f1")}, "->", {k: ner["ruler"]["overall"][k] for k in ("precision", "recall", "f1")})
display(pd.DataFrame({t: {"model F1": ner["model"]["by_type"][t]["f1"], "ruler F1": ner["ruler"]["by_type"][t]["f1"], "gold": ner["ruler"]["by_type"][t]["gold"]} for t in ner["ruler"]["by_type"]}).T)
display(table(ner["dependency"]))
display(table(ner["table_e"]).head(25))
""")
md("""
The dependency table records an interaction between modules: once the hybrid tokenizer keeps `$3,650` as one token, the
statistical model (trained on its own tokens) stops recognising it, and the ruler's MONEY patterns restore it. EVENT stays
at 0 (no patterns), and PERSON is untouched by the ruler: single Indian first names are missed or mistyped.
""")

# Ex 14
md("""
### Exercise 14 — N-grams and collocations

Unigrams to 5-grams over the final pipeline's terms, counted within sentences. A **meaningful filter** drops n-grams that
start or end with a stop word or punctuation; plain **stop-word removal before counting** is compared with it.
""")
code("""
ng = X.exercise14(runs, FINAL)
display(table(ng["table_f"]))
display(table(ng["dictionary"]))
display(table(ng["collocations"]["bigram"]).head(12)); display(table(ng["collocations"]["trigram"]).head(8))
d = pd.DataFrame(ng["dictionary"])
fig, ax = plt.subplots(figsize=(6, 3.2))
for col, c, lab in [("dictionary_size_all", "#2a78d6", "all"), ("dictionary_size_filtered", "#1baf7a", "meaningful filter"), ("dictionary_size_after_removal", "#eda100", "after stop-word removal")]:
    ax.plot(d.n, d[col], "o-", color=c, lw=2, label=lab)
ax.set_xlabel("n"); ax.set_title("distinct n-grams"); ax.legend(frameon=False); plt.tight_layout(); plt.show()
""")
md("""
Removing stop words first invents n-grams that never occur in the text (for bigrams, "numerator denominator" from
"the numerator and the denominator" is the most frequent). The meaningful filter avoids that, so it is the default.
""")

# ----------------------------------------------------------------------------- module 3
md("""
---
## Module 3 — Pipeline design, comparison and justification

| Run | Tokenizer | Stop words | Normaliser | Order | Purpose |
|---|---|---|---|---|---|
| A | NLTK | NLTK | Porter | stop → stem | the brief's left branch |
| B | spaCy | none | WordNet + POS | — | the brief's right branch |
| H | hybrid | custom | spaCy lemma (in context) | lemma → stop | hybrid hypothesis, proposed before any result |
| C1 | NLTK | NLTK | Porter | stem → stop | order test (Exercise 11) |
| C2 | spaCy | none | WordNet, no POS | — | does POS matter for lemmas? |
| C3 | tiktoken cl100k | none | none | — | BPE pieces as index terms |
| H2 | hybrid | custom | WordNet context-free + compound parts | lemma → stop | **post-hoc**, designed after H's errors; not eligible |
""")
code("""
th = pd.DataFrame(ir["table_h"])
display(th[["pipeline", "tokenizer", "stopwords", "normalizer", "order", "token_count", "vocabulary_size", "domain_term_fidelity",
            "meaningful_ngrams_share", "precision", "recall", "f1", "p@5", "mean_query_time_ms", "posthoc", "selected"]].sort_values("f1", ascending=False))
print("selection rule:", ir["selection_rule"], "| selected:", ir["winner"])
fig, ax = plt.subplots(figsize=(6, 3))
ax.barh(th.pipeline, th.f1, color=[RUN_COLOR[p] for p in th.pipeline], height=0.6); ax.invert_yaxis(); ax.set_xlabel("mean F1 (15 queries)")
for i, v in enumerate(th.f1): ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=8)
plt.tight_layout(); plt.show()
""")
md("""
**Error analysis of the hypothesis.** H differs from B on one query only — Q15 (`LLM AND "knowledge tracing"`). spaCy's
lemmas depend on context: in D10 "Knowledge Tracing" is tagged as a proper noun and keeps the lemma *tracing* while the
query is lemmatized to *trace*, and "LLM-based" is a single hybrid token, so *llm* is not indexed on its own. H2 fixes
both and recovers most of the loss, but it was designed after seeing the errors and is evaluated on the same queries,
so it is reported and not selected. The hypothesis is reported as a negative result.
""")
code("""
ev = pd.DataFrame(ir["evaluation"])
piv = ev.pivot(index="qid", columns="pipeline", values="f1")[["A", "B", "H", "C1", "C2", "C3", "H2"]]
display(piv.style.background_gradient(cmap="Blues", vmin=0, vmax=1).format("{:.3f}"))
display(pd.DataFrame(ir["retrieval"]).query("qid in ['Q10', 'Q15'] and pipeline in ['B', 'H', 'H2']")[["pipeline", "qid", "query", "analysed_terms", "retrieved_documents"]])
""")
md("""
### Justification of the final pipeline

| Stage | Decision | Evidence from this run |
|---|---|---|
| Extraction | pdfplumber, trafilatura, MathML → a/b | fewer broken words; no boilerplate |
| Cleaning | vulgar fractions before NFKC; digit-word glue | keeps 3½ = 3 1/2 |
| Tokenization | spaCy tokens for B; hybrid for the linguistic branch | hybrid gold F1 0.996 vs NLTK 0.949 |
| Stop words | none for retrieval | no query depends on a stop word; standard lists delete *not, more, than* |
| Normalisation | WordNet lemmas with POS | 98.2% lemma accuracy vs 86.8% without POS; no over-stemming |
| Index terms | pipeline B | highest mean F1 under the fixed rule |

Each page of the website carries the same four questions (why, where, depends, if moved) for its step.
""")

# ----------------------------------------------------------------------------- module 4
md("""
---
## Module 4 — Positional inverted index

Each term maps to the documents containing it and the positions where it occurs. A sentence end leaves a one-position
gap, so phrases never match across sentences. Ranking uses tf-idf = (1 + log₁₀ tf) × log₁₀(N / df), summed over the
query's non-negated terms; ties go to the lower document id so P@K is reproducible.
""")
code("""
final = runs[FINAL]; idx = final.index
print(f"pipeline {FINAL}: {idx.vocabulary:,} terms, {idx.N} documents, {sum(len(p) for p in idx.postings.values()):,} postings")
an = final.pipeline.analyzer()
for w in ["denominators", "misconception", "tracing"]:
    t = an(w)[0][0]
    post = sorted(idx.postings.get(t, {}).items(), key=lambda x: -len(x[1]))[:5]
    print(f"{w!r} -> {t!r}: df={idx.df(t)} idf={idx.idf(t):.3f} top postings", [(d, len(p), p[:4]) for d, p in post])
print("inverted_index.json:", (RESULTS / "inverted_index.json").stat().st_size // 1024, "KB")
""")

# ----------------------------------------------------------------------------- module 5
md("""
---
## Module 5 — Query processing and search

Operators are upper-case only (AND, OR, NOT) because lower-case *not* and *and* are ordinary words in this domain
("not a common denominator"). Runs of words and quoted strings are phrase units; adjacent units get an implicit AND;
brackets group; precedence is NOT > AND > OR; NOT is the complement within the collection. The expression goes through
a shunting-yard parser to postfix and is evaluated with set operations.
""")
code("""
from nlp_core.query import lex, to_postfix, classify
for q in ['dialogue AND (tutor OR teacher)', 'LCD OR "least common denominator"', 'denominator AND NOT common', 'fraction "common denominator"']:
    print(f"{q:38} lex={lex(q)}\\n{'':38} postfix={to_postfix(lex(q))}  type={classify(q)}")
eng = final.engine()
for q in ["common denominator", "fraction AND misconception", "denominator AND NOT common"]:
    r = eng.search(q, repeats=5)
    print(f"{q:30} {r['type']:18} {r['count']:>2} docs in {r['time_ms']:.3f} ms  top: {[d for d, _ in r['results'][:5]]}")
""")
code("""
qs = pd.DataFrame(X.queries())
display(qs)
display(pd.DataFrame(ir["retrieval"]).query("pipeline == @FINAL")[["qid", "query", "type", "analysed_terms", "results", "retrieved_documents", "time_ms"]])
""")

# ----------------------------------------------------------------------------- module 6
md("""
---
## Module 6 — Evaluation

Fifteen queries with written information needs, judged against all 30 documents (450 judgements). Set metrics (P, R, F1)
score the Boolean result; P@K, R@K and average precision score its tf-idf order.
""")
code("""
display(pd.DataFrame(ir["overall"]).sort_values("f1", ascending=False))
display(ev.query("pipeline == @FINAL")[["qid", "query", "retrieved", "relevant", "hits", "precision", "recall", "f1", "p@5", "r@5", "ap"]])
fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
for run, pts in ir["curves"].items():
    k = [p["k"] for p in pts]
    ax[0].plot(k, [p["p"] for p in pts], "o-", ms=3, lw=2, color=RUN_COLOR[run], label=run)
    ax[1].plot(k, [p["r"] for p in pts], "o-", ms=3, lw=2, color=RUN_COLOR[run], label=run)
ax[0].set_title("precision at K"); ax[1].set_title("recall at K"); ax[0].legend(frameon=False, ncol=4, fontsize=8)
plt.tight_layout(); plt.show()
""")
md("""
Low-recall queries show the vocabulary gap the research must bridge: Q12 `"student explanation"` misses documents that
contain explanations without naming them, and Q09 misses transcripts that never use the word *dialogue*. Exact-match
retrieval cannot close that gap; it is a motivation for the semantic grounding in the research phase.

**Relevance judgements.** `gold/qrels.csv` was drafted with an AI assistant from the information needs and must be
reviewed. `gold/qrels_second_judge_template.csv` lists a stratified sample to judge blind; Cohen's kappa between the two
judges is computed below once it is filled in.
""")
code("""
from nlp_core.evaluate import cohen_kappa
import csv
tmpl = GOLD / "qrels_second_judge_template.csv"
rows = list(csv.DictReader(open(tmpl))) if tmpl.exists() else []
done = [r for r in rows if r.get("second_judge", "").strip() in ("0", "1")]
if done:
    first = {(r["qid"], r["doc_id"]): r["relevant"] for r in csv.DictReader(open(GOLD / "qrels.csv"))}
    k = cohen_kappa([first[(r["qid"], r["doc_id"])] for r in done], [r["second_judge"] for r in done])
    print(f"Cohen's kappa on {len(done)} double-judged pairs: {k:.3f}")
else:
    print(f"{len(rows)} pairs waiting for a second judge in {tmpl.name}; kappa is computed here once they are filled in.")
""")

# ----------------------------------------------------------------------------- scale + evidence + outputs
md("""
---
## Scale tier — the full MathDial and TalkMoves collections

All exercises and relevance judgements use the 30-document corpus. To show the pipeline holds at scale, the full
MathDial (2,861 dialogues) and TalkMoves (566 transcripts) collections — 3.4 million words — were cleaned, tokenized and
indexed with pipelines A, B and H2 by `scripts/run_scale.py` (about eight minutes). The results are loaded here.
""")
code("""
sc = json.loads((RESULTS / "app" / "scale.json").read_text())
display(pd.DataFrame(sc["summary"])); display(pd.DataFrame(sc["pipelines"])); display(pd.DataFrame(sc["bpe"]))
sp = sc["collections"]["talkmoves"]["speakers"]
print({k: {x: v[x] for x in ("turns", "words", "mean_turn_length")} for k, v in sp.items()})
print("student bigrams:", [b["ngram"] for b in sp["student"]["bigrams"][:8]])
print("teacher bigrams:", [b["ngram"] for b in sp["teacher"]["bigrams"][:8]])
""")
md("""
---
## Preview — from a learner sentence to a candidate evidence record

This is where the assessment hands over to the research. The record bundles the observation, what the pipeline
extracted (tokens, numbers and expressions, concepts, verb–object pairs), transparent detectors that check the arithmetic,
and the nearest misconception descriptions retrieved by tf-idf from MaE (real) and the malrules (synthetic). It is marked
as a prototype for human review; no learner model or probability is computed.
""")
code("""
from nlp_core.evidence import evidence_record
rec = evidence_record("I did 1/2 + 1/3 = 2/5 because you add the tops and then add the bottoms.")
print(json.dumps({k: rec[k] for k in ("evidence_id", "status")}, indent=1))
print("detectors:", [(d["hypothesis"], d["check"]) for d in rec["interpretation"]["detectors"]])
print("grounding:", [(g["id"], g["name"][:50], g["score"]) for g in rec["interpretation"]["retrieved_grounding"][:3]])
""")
md("## Results files")
code("""
required = ["preprocessing_results.csv", "tokenization_comparison.csv", "stemming_lemmatization.csv", "pos_tagging_results.csv",
            "ner_results.csv", "unigram_results.csv", "bigram_results.csv", "trigram_results.csv", "ngram_results.csv",
            "bpe_results.csv", "pipeline_comparison.csv", "retrieval_results.csv", "evaluation_results.csv", "inverted_index.json"]
display(pd.DataFrame([{"file": f, "present": (RESULTS / f).exists(), "KB": round((RESULTS / f).stat().st_size / 1024, 1)} for f in required]))
print(len(list(RESULTS.glob("*.*"))), "files in results/ in total")
""")
md("""
## Conclusion

The selected pipeline B (spaCy tokens, no stop-word removal, WordNet lemmas with POS) reaches mean F1 0.679 on the 15
gold queries under a rule fixed in advance. The hybrid hypothesis H scored 0.635 and is reported as a negative result;
its post-hoc repair H2 (0.668) is labelled and not eligible. The domain work that carried the most weight was elsewhere:
the hybrid tokenizer (gold F1 0.996), the custom stop list that keeps *not, more, than*, the POS rule layer (97.9%) and
the EntityRuler (NER F1 0.49 → 0.84). The remaining recall gap on explanation-style queries is the motivation for the
semantic, learner-state work of the research phase.
""")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                      "language_info": {"name": "python"}}
    if a.execute:
        from nbclient import NotebookClient

        NotebookClient(nb, timeout=1800, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}).execute()
    nbf.write(nb, OUT)
    print(f"wrote {OUT} ({len(cells)} cells{', executed' if a.execute else ''})")


if __name__ == "__main__":
    sys.exit(main())
