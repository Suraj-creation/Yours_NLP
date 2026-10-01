"""Every exercise and module, as functions that compute results and write results/ files.

scripts/run_experiments.py calls these in order. Each function returns a dict that the
website serves as JSON (results/app/<name>.json), and writes the CSV files the brief
asks for. All numbers in the notebook, website and report come from here.
"""
from __future__ import annotations

import csv
import json
import random
import re
import statistics
from collections import Counter, defaultdict

import pandas as pd

from . import bpe as B
from . import ner as NER
from . import pos as P
from . import tokenizers as T
from .cleaning import clean_text, split_sentences
from .config import GOLD, RESULTS, SEED
from .evaluate import boundary_scores, confusion_matrix, curve_at_k, retrieval_scores, tagging_accuracy
from .loaders import compare_html_extractors, compare_pdf_extractors, load_corpus, read_catalog
from .ngrams import clean_edges, collocations, network, ngrams, summary
from .pipelines import Pipeline, all_run_names, cleaned_corpus, tokenized
from .preprocess import (KEEP, LEMMATIZERS, STEMMERS, lemma_wordnet, lemmatize_tokens, normalize_term, stem,
                         stopwords)
from .stats import corpus_summary, doc_stats, heaps, is_word, zipf

APP = RESULTS / "app"
APP.mkdir(parents=True, exist_ok=True)
random.seed(SEED)


def save_app(name: str, data) -> None:
    (APP / f"{name}.json").write_text(json.dumps(data, ensure_ascii=False, default=str))


def write_csv(name: str, rows: list[dict] | pd.DataFrame) -> None:
    df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    df.to_csv(RESULTS / name, index=False)


def queries() -> list[dict]:
    with open(GOLD / "queries.csv", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def qrels() -> dict[str, set[str]]:
    rel = defaultdict(set)
    with open(GOLD / "qrels.csv", newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["relevant"] == "1":
                rel[r["qid"]].add(r["doc_id"])
    return rel


# ====================================================================== Module 1
def module1() -> dict:
    from nltk.tokenize import word_tokenize

    docs = load_corpus()
    cat = {c["doc_id"]: c for c in read_catalog()}
    rows, clean_logs, all_tokens, sent_len = [], [], [], {"punkt": [], "spacy": []}
    for d in docs:
        clean, log = clean_text(d.text, d.fmt)
        s_p = split_sentences(clean, "punkt")
        s_s = split_sentences(clean, "spacy")
        toks = word_tokenize(clean)
        r = doc_stats(d.doc_id, clean, toks, s_p)
        r.update({"sentences_spacy": len(s_s), "raw_characters": len(d.text), "format": cat[d.doc_id]["format"],
                  "layer": cat[d.doc_id]["layer"], "title": d.title, "synthetic": cat[d.doc_id]["synthetic"]})
        rows.append(r)
        clean_logs.append({"doc_id": d.doc_id, **log})
        all_tokens.extend(t.lower() for t in toks if is_word(t))
        sent_len["punkt"].extend(len(word_tokenize(s)) for s in s_p)
        sent_len["spacy"].extend(len(word_tokenize(s)) for s in s_s)
    total = corpus_summary(rows)
    total["vocabulary"] = len(set(all_tokens))
    df = pd.DataFrame(rows)
    order = ["doc_id", "title", "format", "layer", "synthetic", "sentences", "sentences_spacy", "tokens", "word_tokens",
             "characters", "raw_characters", "vocabulary", "type_token_ratio", "hapax"]
    df = df[order]
    tot = {"doc_id": "TOTAL", "title": f"{total['documents']} documents", "sentences": total["sentences"],
           "sentences_spacy": int(df.sentences_spacy.sum()), "tokens": total["tokens"],
           "word_tokens": total["word_tokens"], "characters": total["characters"],
           "raw_characters": int(df.raw_characters.sum()), "vocabulary": total["vocabulary"]}
    avg = {"doc_id": "AVERAGE", "title": "per document", "tokens": total["avg_doc_length_tokens"],
           "characters": total["avg_doc_length_characters"], "sentences": round(total["sentences"] / len(rows), 1)}
    write_csv("document_statistics.csv", pd.concat([df, pd.DataFrame([tot, avg])], ignore_index=True))
    write_csv("cleaning_log.csv", clean_logs)

    pdf_cmp = [dict(r, doc_id="D13") for r in compare_pdf_extractors(next(p for p in (RESULTS.parent / "data").glob("D13_*")))]
    html_cmp = [dict(r, doc_id="D29") for r in compare_html_extractors(next(p for p in (RESULTS.parent / "data").glob("D29_*")))]
    write_csv("extraction_comparison.csv", [{k: v for k, v in r.items() if k != "sample"} for r in pdf_cmp + html_cmp])

    def hist(vals, width=5, cap=80):
        c = Counter(min(v, cap) // width * width for v in vals)
        return [{"bin": b, "count": c[b]} for b in range(0, cap + width, width)]

    out = {"summary": total, "documents": rows, "catalog": list(cat.values()), "cleaning": clean_logs,
           "extraction": {"pdf": pdf_cmp, "html": html_cmp}, "zipf": zipf(Counter(all_tokens)),
           "heaps": heaps(all_tokens), "sentence_lengths": {k: hist(v) for k, v in sent_len.items()},
           "sentence_totals": {k: len(v) for k, v in sent_len.items()}}
    save_app("corpus", out)
    return out


# ====================================================================== Exercises 2, 3, 6
AUDIT = [
    ("fraction split", lambda a, b, c: b == "/" and a.isdigit() and c.isdigit(), "custom rule `fraction`"),
    ("hyphenated word split", lambda a, b, c: b == "-" and a.isalpha() and c.isalpha(), "remove spaCy hyphen infix; rule `compound`"),
    ("currency symbol split", lambda a, b, c: b in "$₹€£" and c[:1].isdigit(), "rule `currency`"),
    ("percent split", lambda a, b, c: b == "%" and re.match(r"\d", a or ""), "rule `percent`"),
    ("abbreviation broken", lambda a, b, c: b in ("e.g", "i.e", "etc") , "rule `abbreviation`"),
    ("contraction split", lambda a, b, c: b in ("n't", "'s", "'m", "'re", "nt") , "rule `compound` / `informal`"),
    ("operator split from expression", lambda a, b, c: b in "+=" and a[:1].isdigit() and c[:1].isdigit(), "rule `expression`"),
]


def exercise2_3_6() -> dict:
    corpus = cleaned_corpus()
    gold = [json.loads(line) for line in open(GOLD / "tokenization_gold.jsonl", encoding="utf-8")]
    names = ["whitespace", "nltk", "nltk_wordpunct", "nltk_treebank", "nltk_tweet", "spacy", "custom", "hybrid"]
    # corpus-level counts
    corpus_rows = []
    per_tok = {}
    for n in names:
        toks = tokenized(corpus, n) if n in ("nltk", "spacy", "custom", "hybrid") else \
            {did: [T.tokenize(s, n) for s in split_sentences(text)] for did, text in corpus}
        flat = [t for sents in toks.values() for s in sents for t in s]
        per_tok[n] = flat
        corpus_rows.append({"tokenizer": n, "label": T.LABELS[n], "tokens": len(flat), "dictionary_size": len(set(flat)),
                            "dictionary_lowercased": len({t.lower() for t in flat}),
                            "punctuation_tokens": sum(1 for t in flat if not is_word(t))})
    # Ex 2 audit of built-in tokenizers
    audit = []
    for n in ("nltk", "spacy"):
        flat = per_tok[n]
        for name, test, fix in AUDIT:
            cnt, ex = 0, []
            for i in range(1, len(flat) - 1):
                if test(flat[i - 1], flat[i], flat[i + 1]):
                    cnt += 1
                    if len(ex) < 3:
                        ex.append(" ".join(flat[i - 1:i + 2]))
            audit.append({"tokenizer": n, "problem": name, "occurrences": cnt, "examples": " | ".join(ex), "fix": fix})
        types = Counter(flat)
        case_dupes = sum(1 for t in types if t != t.lower() and t.lower() in types)
        audit.append({"tokenizer": n, "problem": "case duplicates in dictionary", "occurrences": case_dupes,
                      "examples": "Fraction / fraction", "fix": "lowercase on the index branch only (after NER)"})
    # Ex 3 gold scores
    scores = []
    for n in names:
        s = boundary_scores(lambda t, n=n: T.tokenize(t, n), gold)
        scores.append({"tokenizer": n, "label": T.LABELS[n], **{k: v for k, v in s.items() if k != "by_type"},
                       **{f"f1_{k}": v for k, v in s["by_type"].items()}})
    # rule ablation on the hybrid tokenizer
    ablation = []
    base = boundary_scores(lambda t: T.hybrid(t), gold)["f1"]
    for r in T.DEFAULT_RULES:
        keep = [x for x in T.DEFAULT_RULES if x != r]
        f = boundary_scores(lambda t, keep=keep: T.hybrid(t, keep), gold)["f1"]
        ablation.append({"rule_removed": r, "f1": f, "f1_drop": round(base - f, 4)})
    # Table B examples
    show = ["T10", "T16", "T28", "T31", "T26", "T37", "T13", "T21"]
    table_b = []
    for g in gold:
        if g["id"] in show:
            row = {"id": g["id"], "problem_type": g["problem_type"], "input": g["text"]}
            for n in ("nltk", "spacy", "custom", "hybrid"):
                row[n] = " | ".join(T.tokenize(g["text"], n))
            row["gold"] = " | ".join(g["tokens"])
            table_b.append(row)
    write_csv("tokenization_comparison.csv", table_b)
    write_csv("tokenizer_scores.csv", scores)
    write_csv("tokenizer_corpus_counts.csv", corpus_rows)
    write_csv("tokenizer_audit.csv", audit)
    write_csv("tokenizer_rule_ablation.csv", ablation)
    # Ex 6 numbers and dates: for every numeric/date string (maximal match, rule priority order),
    # did the tokenizer produce exactly that character span as ONE token?
    kinds = ["datetime", "currency", "mixed_number", "expression", "fraction", "percent", "ratio_time", "number"]
    pat = re.compile("|".join(f"(?P<{k}>{T.RULES[k]})" for k in kinds))
    toks_by = {n: tokenized(corpus, n) for n in ("nltk", "spacy", "custom", "hybrid")}
    occ, kept, examples = Counter(), {n: Counter() for n in toks_by}, defaultdict(list)
    for did, text in corpus:
        sents = split_sentences(text)
        for si, sent in enumerate(sents):
            ms = list(pat.finditer(sent))
            if not ms:
                continue
            spans = {n: set(T.align(toks_by[n][did][si], sent)) for n in toks_by}
            for m in ms:
                k = m.lastgroup
                occ[k] += 1
                if len(examples[k]) < 4 and m.group(0) not in examples[k]:
                    examples[k].append(m.group(0))
                for n in toks_by:
                    kept[n][k] += (m.start(), m.end()) in spans[n]
    labels = {"datetime": "ISO date / timestamp", "currency": "currency amount", "mixed_number": "mixed number",
              "expression": "expression (no spaces)", "fraction": "fraction", "percent": "percent",
              "ratio_time": "ratio / clock time", "number": "decimal / thousands"}
    num_rows = [{"kind": labels[k], "occurrences": occ[k], "examples": " | ".join(examples[k]),
                 **{f"kept_whole_{n}": round(kept[n][k] / occ[k], 3) if occ[k] else None for n in toks_by}}
                for k in kinds]
    write_csv("numbers_dates.csv", num_rows)
    out = {"corpus": corpus_rows, "audit": audit, "scores": scores, "ablation": ablation, "table_b": table_b,
           "rules": [{"rule": k, "pattern": v} for k, v in T.RULES.items()], "numbers_dates": num_rows,
           "gold": gold}
    save_app("tokenization", out)
    return out


# ====================================================================== Exercise 4 (BPE)
def exercise4() -> dict:
    corpus = cleaned_corpus()
    text = "\n".join(t for _, t in corpus)
    wc = Counter(B.pretokenize(text))
    stop = stopwords("nltk")
    common = [w for w, _ in wc.most_common(400) if w not in stop and w.isalpha() and len(w) > 3][:10]
    rare_pool = sorted(w for w, c in wc.items() if c == 1 and w.isalpha() and len(w) > 6)
    rare = random.Random(SEED).sample(rare_pool, 10)
    domain = ["denominator", "denominators", "misconception", "numerators", "reciprocal", "knowledge-tracing",
              "Q-MCKT", "ASSIST2009", "1/2+1/3=2/5", "₹50,000"]
    words = [(w, "common") for w in common] + [(w, "rare") for w in rare] + [(w, "domain") for w in domain]
    ours = {m: B.SimpleBPE().fit(wc, m) for m in (500, 1000, 2000)}
    rows = []
    for i, (w, cls) in enumerate(words, 1):
        r = {"s_no": i, "word": w, "class": cls}
        for e in B.ENCODINGS:
            pieces = B.tiktoken_word(w, e)
            r[f"{e}"] = " | ".join(p.replace(" ", "▁") for p in pieces)
            r[f"{e}_n"] = len(pieces)
        for m, b in ours.items():
            pieces = b.encode_word(w.lower()) if re.fullmatch(r"[A-Za-z]+", w) else \
                [p for part in B.pretokenize(w) for p in b.encode_word(part)]
            r[f"ours_{m}"] = " | ".join(pieces)
            r[f"ours_{m}_n"] = len(pieces)
        for n in ("nltk", "spacy", "custom", "hybrid"):
            r[n] = " | ".join(T.tokenize(w, n))
        rows.append(r)
    write_csv("bpe_results.csv", rows)
    # corpus-level: vocabulary and token counts per method
    words_flat = B.pretokenize(text)
    summ = []
    for e in B.ENCODINGS:
        summ.append({"method": f"tiktoken {e}", "vocabulary_size": B.encoding(e).n_vocab,
                     "corpus_tokens": len(B.encoding(e).encode(text, disallowed_special=())), "trained_on": "OpenAI web data"})
    for m, b in ours.items():
        summ.append({"method": f"our BPE {m} merges", "vocabulary_size": b.vocab_size,
                     "corpus_tokens": len(b.encode(words_flat)), "trained_on": "this corpus"})
    for n in ("nltk", "spacy", "custom", "hybrid"):
        toks = [t for sents in tokenized(corpus, n).values() for s in sents for t in s]
        summ.append({"method": T.LABELS[n], "vocabulary_size": len(set(toks)), "corpus_tokens": len(toks),
                     "trained_on": "rules / pretrained"})
    # SimpleTokenizerV2 OOV on unseen text (MathDial train split sample, not in the corpus)
    from .loaders import load_mathdial_scale

    heldout = "\n".join(d.text for d in load_mathdial_scale()[:300])
    v2 = B.SimpleTokenizerV2(text)
    summ.append({"method": "SimpleTokenizerV2 (word-level)", "vocabulary_size": v2.vocab_size,
                 "corpus_tokens": len(v2.encode(text)), "trained_on": "this corpus",
                 "oov_rate_unseen_text": v2.oov_rate(heldout)})
    for s in summ:  # OOV only exists for a fixed word vocabulary; subword BPE cannot be out of vocabulary
        s.setdefault("oov_rate_unseen_text", 0.0 if "BPE" in s["method"] or "tiktoken" in s["method"] else None)
    write_csv("bpe_summary.csv", summ)
    per_class = []
    for cls in ("common", "rare", "domain"):
        rr = [r for r in rows if r["class"] == cls]
        per_class.append({"class": cls, **{m: round(statistics.mean(r[f"{m}_n"] for r in rr), 2)
                                           for m in B.ENCODINGS + [f"ours_{k}" for k in ours]}})
    curve = []
    for m in (100, 250, 500, 1000, 1500, 2000, 3000, 4000):
        b = B.SimpleBPE().fit(wc, m)
        curve.append({"merges": m, "vocabulary_size": b.vocab_size, "corpus_tokens": len(b.encode(words_flat))})
    out = {"words": rows, "summary": summ, "per_class": per_class, "curve": curve, "first_merges": ours[2000].history,
           "v2_example": {"text": "The LCD of 4 and 6 is 12, obviously.",
                          "decoded": v2.decode(v2.encode("The LCD of 4 and 6 is 12, obviously."))}}
    save_app("bpe", out)
    return out


# ====================================================================== Exercises 1, 7-11 (preprocessing)
def preprocessing(runs: dict, final_name: str) -> dict:
    from nltk.tokenize import word_tokenize

    docs = load_corpus()
    raw_tokens = [t for d in docs for t in word_tokenize(d.text)]
    final = runs[final_name]
    base = final.pipeline
    final_terms = [t for pairs in final.docs_terms.values() for t, _ in pairs]
    table_a = [
        {"measure": "Documents", "before": len(docs), "after": len(final.docs_terms)},
        {"measure": "Tokens", "before": len(raw_tokens), "after": len(final_terms)},
        {"measure": "Unique tokens", "before": len(set(raw_tokens)), "after": len(set(final_terms))},
        {"measure": "Vocabulary size (lower-cased word types)",
         "before": len({t.lower() for t in raw_tokens if is_word(t)}), "after": final.index.vocabulary},
        {"measure": "Characters", "before": sum(len(d.text) for d in docs),
         "after": sum(len(t) for _, t in cleaned_corpus())},
    ]
    write_csv("preprocessing_results.csv", table_a)
    # Ex 1: dictionary vs term dictionary at every stage (Final pipeline)
    stages = [{"stage": s["stage"], "tokens": s["tokens"], "dictionary_size": s["vocabulary"]} for s in final.stages]
    write_csv("preprocessing_stages.csv", stages)
    # Ex 7: stop-word lists (Final pipeline with the list swapped), retrieval F1 per list
    qs, rel = queries(), qrels()
    sw_rows = []
    for name in ("none", "nltk", "spacy", "custom"):
        p = Pipeline(name=f"{final_name}_sw_{name}", tokenizer=base.tokenizer, stopwords=name, normalizer=base.normalizer,
                     order=base.order, compound_parts=base.compound_parts)
        run = p.run()
        f1 = _mean_f1(run, qs, rel)
        lst = stopwords(name)
        sw_rows.append({"stopword_list": name, "list_size": len(lst), "index_tokens": run.summary()["token_count"],
                        "vocabulary": run.index.vocabulary, "mean_f1": f1,
                        "domain_words_removed": ", ".join(sorted(w for w in KEEP if w in lst))})
    write_csv("stopword_comparison.csv", sw_rows)
    # Ex 8: stemmers
    with open(GOLD / "stem_pairs.csv", newline="", encoding="utf-8") as fh:
        pairs = list(csv.DictReader(fh))
    vocab = sorted({t for t in final_terms if t.isalpha()})
    st_rows, st_errors = [], []
    for s in STEMMERS:
        over = [p for p in pairs if p["should_merge"] == "0" and stem(p["word_a"], s) == stem(p["word_b"], s)]
        under = [p for p in pairs if p["should_merge"] == "1" and stem(p["word_a"], s) != stem(p["word_b"], s)]
        st_rows.append({"method": s, "kind": "stemmer", "unique_forms": len({stem(w, s) for w in vocab}),
                        "over_stemming_errors": len(over), "under_stemming_errors": len(under),
                        "pairs_tested": len(pairs)})
        for p in over:
            st_errors.append({"method": s, "error": "over", "pair": f"{p['word_a']} / {p['word_b']}",
                              "forms": f"{stem(p['word_a'], s)} / {stem(p['word_b'], s)}"})
        for p in under:
            st_errors.append({"method": s, "error": "under", "pair": f"{p['word_a']} / {p['word_b']}",
                              "forms": f"{stem(p['word_a'], s)} / {stem(p['word_b'], s)}"})
    for lm in ("wordnet_nopos", "spacy"):
        lem = lambda w, lm=lm: lemmatize_tokens([w], lm)[0]
        over = [p for p in pairs if p["should_merge"] == "0" and lem(p["word_a"]) == lem(p["word_b"])]
        under = [p for p in pairs if p["should_merge"] == "1" and lem(p["word_a"]) != lem(p["word_b"])]
        st_rows.append({"method": lm, "kind": "lemmatizer", "unique_forms": len(set(lemmatize_tokens(vocab[:4000], lm))),
                        "over_stemming_errors": len(over), "under_stemming_errors": len(under), "pairs_tested": len(pairs)})
    write_csv("stemmer_scores.csv", st_rows)
    write_csv("stemmer_errors.csv", st_errors)
    # Table C: word | stems | lemmas
    words_c = ["denominators", "numerators", "numerical", "fractions", "fractional", "simplifying", "simplification",
               "multiplication", "multiplied", "dividing", "division", "equivalent", "misconceptions", "conception",
               "generous", "general", "university", "added", "adding", "was", "better", "halves", "thirds", "reciprocal",
               "percentages", "decimals", "studies", "explanations", "tutoring", "comparing"]
    table_c = []
    for w in words_c:
        row = {"word": w}
        for s in STEMMERS:
            row[f"stem_{s}"] = stem(w, s)
        row["lemma_wordnet_nopos"] = lemma_wordnet(w)
        row["lemma_wordnet_pos"] = lemmatize_tokens([w], "wordnet_pos")[0]
        row["lemma_spacy"] = lemmatize_tokens([w], "spacy")[0]
        table_c.append(row)
    write_csv("stemming_lemmatization.csv", table_c)
    # Ex 9: order of stemming and stop-word removal (A vs C1)
    order_rows = []
    nstop = stopwords("nltk")
    stemmed_stop = {stem(w, "porter") for w in nstop} - nstop
    for name in ("A", "C1"):
        r = runs[name]
        terms = Counter(t for pairs in r.docs_terms.values() for t, _ in pairs)
        leaked = {t: c for t, c in terms.items() if t in stemmed_stop}
        order_rows.append({"run": name, "order": r.pipeline.order, "index_tokens": sum(terms.values()),
                           "vocabulary": len(terms), "stop_word_stems_leaked": len(leaked),
                           "leaked_occurrences": sum(leaked.values()),
                           "examples": ", ".join(f"{k} ({v})" for k, v in sorted(leaked.items(), key=lambda x: -x[1])[:8]),
                           "mean_f1": _mean_f1(r, qs, rel)})
    write_csv("order_experiment.csv", order_rows)
    # Ex 10: lemmatizer accuracy on the POS gold (all 32 sentences, alphabetic tokens)
    gold = P.load_gold()
    lem_rows = []
    for lm in LEMMATIZERS:
        tot = cor = 0
        errs = Counter()
        for g in gold:
            pred = lemmatize_tokens(g["tokens"], lm)
            for tok, gl, pl in zip(g["tokens"], g["lemmas"], pred):
                if not tok.isalpha():
                    continue
                tot += 1
                if pl.lower() == gl.lower():
                    cor += 1
                else:
                    errs[f"{tok}->{pl} (gold {gl})"] += 1
        lem_rows.append({"lemmatizer": lm, "accuracy": round(cor / tot, 4), "tokens": tot, "correct": cor,
                         "errors": "; ".join(e for e, _ in errs.most_common(8))})
    write_csv("lemma_scores.csv", lem_rows)
    # Ex 11: before vs after top terms and word-cloud data
    before = Counter(t.lower() for t in raw_tokens if is_word(t))
    after = Counter(final_terms)
    out = {"table_a": table_a, "stages": stages, "stopwords": sw_rows, "stemmers": st_rows, "stem_errors": st_errors,
           "table_c": table_c, "order": order_rows, "lemmas": lem_rows,
           "top_before": [{"term": t, "count": c} for t, c in before.most_common(40)],
           "top_after": [{"term": t, "count": c} for t, c in after.most_common(40)],
           "custom_stoplist": sorted(stopwords("custom")), "kept_words": sorted(KEEP & set(stopwords("nltk"))),
           "final_pipeline": final_name}
    save_app("preprocessing", out)
    return out


# ====================================================================== Exercises 5, 12 (POS)
def exercise5_12() -> dict:
    corpus = cleaned_corpus()
    cands = {"lcd", "bkt", "top", "bottom", "flip", "simplify", "carry", "over", "halves", "thirds", "fourths",
             "reciprocal", "cross", "x", "subtract", "multiply", "round", "plot", "shade"}
    nl, sp = defaultdict(Counter), defaultdict(Counter)
    frac_n, frac_s = Counter(), Counter()
    n_sent = 0
    for did, text in corpus:
        for s in split_sentences(text):
            toks = T.hybrid(s)
            if not any(w.lower() in cands or re.match(r"^\d+/\d+$", w) for w in toks):
                continue
            n_sent += 1
            if n_sent > 1500:
                break
            a, (b, _) = P.nltk_tag(toks), P.spacy_tag(toks)
            for w, x, y in zip(toks, a, b):
                if w.lower() in cands:
                    nl[w.lower()][x] += 1
                    sp[w.lower()][y] += 1
                if re.match(r"^\d+/\d+$", w):
                    frac_n[x] += 1
                    frac_s[y] += 1
    expected = {"lcd": "NN", "simplify": "VB", "multiply": "VB", "subtract": "VB", "x": "SYM", "reciprocal": "NN",
                "plot": "VB", "round": "VB", "shade": "VB", "fourths": "NNS", "halves": "NNS", "thirds": "NNS",
                "top": "JJ", "over": "IN", "bkt": "NNP", "carry": "VB", "cross": "VB", "flip": "VB", "bottom": "JJ"}
    ident = []
    for k in sorted(nl):
        tot = sum(nl[k].values())
        ident.append({"term": k, "occurrences": tot, "expected": expected.get(k, ""),
                      "nltk_tags": ", ".join(f"{t}:{c}" for t, c in nl[k].most_common(4)),
                      "spacy_tags": ", ".join(f"{t}:{c}" for t, c in sp[k].most_common(4)),
                      "nltk_wrong_share": round(1 - nl[k][expected.get(k, "")] / tot, 3) if tot else 0,
                      "spacy_wrong_share": round(1 - sp[k][expected.get(k, "")] / tot, 3) if tot else 0})
    ident.append({"term": "fraction tokens (e.g. 3/4)", "occurrences": sum(frac_n.values()), "expected": "CD",
                  "nltk_tags": ", ".join(f"{t}:{c}" for t, c in frac_n.most_common(4)),
                  "spacy_tags": ", ".join(f"{t}:{c}" for t, c in frac_s.most_common(4)),
                  "nltk_wrong_share": round(1 - frac_n["CD"] / max(1, sum(frac_n.values())), 3),
                  "spacy_wrong_share": round(1 - frac_s["CD"] / max(1, sum(frac_s.values())), 3)})
    ident.sort(key=lambda r: -max(r["nltk_wrong_share"], r["spacy_wrong_share"]))
    write_csv("pos_domain_terms.csv", ident)
    test, allg = P.load_gold("test"), P.load_gold()
    acc_rows, preds, cms = [], {}, {}
    labels = sorted({t for g in test for t in g["penn"]})
    for k, (label, fn) in P.TAGGERS.items():
        pr = [fn(g["tokens"]) for g in test]
        preds[k] = pr
        a = tagging_accuracy([g["penn"] for g in test], pr)
        a_all = tagging_accuracy([g["penn"] for g in allg], [fn(g["tokens"]) for g in allg]) if k in ("nltk", "spacy", "rules_nltk", "rules_spacy") else None
        acc_rows.append({"tagger": k, "label": label, "accuracy_test": a["accuracy"], "tokens_test": a["tokens"],
                         "correct_test": a["correct"],
                         "accuracy_all_32": a_all["accuracy"] if a_all else "",
                         "top_errors": "; ".join(f"{e['gold']}->{e['pred']} x{e['count']}" for e in a["top_errors"][:5])})
        cms[k] = confusion_matrix([g["penn"] for g in test], pr, labels)
    _, spacy_upos = zip(*[P.spacy_tag(g["tokens"]) for g in test])
    upos_acc = tagging_accuracy([g["upos"] for g in test], [[{"ADP": "ADP"}.get(t, t) for t in s] for s in spacy_upos])
    # Table D rows: every test token where any tagger disagrees with gold
    table_d = []
    for gi, g in enumerate(test):
        rules_fired = P.rule_tag(g["tokens"], "nltk")[1]
        for i, w in enumerate(g["tokens"]):
            d_tag, c_tag, gold_tag = preds["nltk"][gi][i], preds["rules_nltk"][gi][i], g["penn"][i]
            if d_tag != gold_tag or c_tag != gold_tag or rules_fired[i]:
                table_d.append({"sentence": g["id"], "word": w, "default_pos_nltk": d_tag, "custom_pos_rules": c_tag,
                                "custom_pos_crf": preds["crf"][gi][i], "gold": gold_tag,
                                "default_correct": d_tag == gold_tag, "custom_correct": c_tag == gold_tag,
                                "rule_fired": rules_fired[i]})
    write_csv("pos_tagging_results.csv", table_d)
    write_csv("pos_tagger_accuracy.csv", acc_rows)
    # Exercise 5 demos
    verb_task = [l for l in open(next((RESULTS.parent / "data").glob("D25_*")), encoding="utf-8").read().splitlines()[:6]]
    np_task = ["To add fractions with a common denominator, add the numerators and place the sum over the common denominator.",
               "Find the least common denominator (LCD) and convert each fraction to an equivalent fraction.",
               "SQKT is a knowledge tracing model that uses students' questions and extracted skill information."]
    ex5 = {"verbs_and_nouns": [{"text": t, "verb_object": P.verb_object_pairs(t), "noun_phrases": P.noun_phrases(t)}
                               for t in verb_task],
           "nouns_only": [{"text": t, "noun_phrases": P.noun_phrases(t)} for t in np_task]}
    rows5 = [{"task": "misconception detection (verbs + noun phrases)", "text": x["text"],
              "verb_object": "; ".join(f"{v['verb']} -> {v['object']}" for v in x["verb_object"]),
              "noun_phrases": "; ".join(x["noun_phrases"])} for x in ex5["verbs_and_nouns"]]
    rows5 += [{"task": "concept tagging / indexing (noun phrases only)", "text": x["text"], "verb_object": "",
               "noun_phrases": "; ".join(x["noun_phrases"])} for x in ex5["nouns_only"]]
    write_csv("ex5_verbs_nounphrases.csv", rows5)
    # distributions default vs custom over the corpus sample
    dist = {"nltk": Counter(), "rules_nltk": Counter()}
    for did, text in corpus[:12]:
        for s in split_sentences(text)[:40]:
            toks = T.hybrid(s)
            dist["nltk"].update(P.nltk_tag(toks))
            dist["rules_nltk"].update(P.rule_tag(toks)[0])
    tags = sorted(set(dist["nltk"]) | set(dist["rules_nltk"]), key=lambda t: -dist["nltk"][t])[:20]
    out = {"identification": ident, "accuracy": acc_rows, "upos_spacy_test": upos_acc["accuracy"], "labels": labels,
           "confusion": cms, "table_d": table_d, "ex5": ex5,
           "distribution": [{"tag": t, "default": dist["nltk"][t], "custom": dist["rules_nltk"][t]} for t in tags],
           "lexicon": P.DOMAIN_LEXICON, "test_sentences": [" ".join(g["tokens"]) for g in test]}
    save_app("pos", out)
    return out


# ====================================================================== Exercise 13 (NER)
def exercise13() -> dict:
    import spacy

    res = {}
    for ruler in (False, True):
        s, rows = NER.evaluate(ruler)
        res["ruler" if ruler else "model"] = {"scores": s, "rows": rows}
    table_e = []
    for r in res["ruler"]["rows"]:
        base = next((b for b in res["model"]["rows"] if b["sentence"] == r["sentence"] and b["entity"] == r["entity"]), None)
        table_e.append({"sentence": r["sentence"], "entity": r["entity"], "gold_type": r["gold_type"],
                        "predicted_type_model": base["predicted_type"] if base else "",
                        "status_model": base["status"] if base else "",
                        "predicted_type_ruler": r["predicted_type"], "status_ruler": r["status"],
                        "domain_type": r["domain_type"]})
    write_csv("ner_results.csv", table_e)
    score_rows = []
    for k in ("model", "ruler"):
        s = res[k]["scores"]
        score_rows.append({"system": k, "type": "ALL", **s["overall"]})
        for t, v in s["by_type"].items():
            score_rows.append({"system": k, "type": t, **v})
    write_csv("ner_scores.csv", score_rows)
    # tokenization -> NER dependency (MONEY)
    plain = spacy.load("en_core_web_sm")
    dep = []
    for g in NER.load_gold():
        for a, b, lab in g["entities"]:
            if lab != "MONEY":
                continue
            ent = g["text"][a:b]
            pl = [e for e in plain(g["text"]).ents if e.start_char < b and a < e.end_char]
            hy = [e for e in NER.pipeline(False)(g["text"]).ents if e.start_char < b and a < e.end_char]
            ru = [e for e in NER.pipeline(True)(g["text"]).ents if e.start_char < b and a < e.end_char]
            dep.append({"entity": ent, "spacy_default_tokens": f"{pl[0].text} ({pl[0].label_})" if pl else "missed",
                        "hybrid_tokens_model_only": f"{hy[0].text} ({hy[0].label_})" if hy else "missed",
                        "hybrid_tokens_plus_ruler": f"{ru[0].text} ({ru[0].label_})" if ru else "missed"})
    write_csv("ner_tokenization_dependency.csv", dep)
    samples = ["Ms. Priya Raman from Greenfield Public School in Bengaluru bought fraction tiles for Rs 2,500 on 12 August 2026.",
               "SQKT, a knowledge tracing model, beat DKT on ASSIST2009 with a higher AUC.",
               "Students who add across show the misconception of adding numerators and denominators."]
    out = {"model": res["model"]["scores"], "ruler": res["ruler"]["scores"], "table_e": table_e, "dependency": dep,
           "samples": [{"text": t, "model": NER.entities(t, False), "ruler": NER.entities(t, True)} for t in samples],
           "types": {"standard": NER.STANDARD, "domain": NER.DOMAIN}, "gold": NER.load_gold()}
    save_app("ner", out)
    return out


# ====================================================================== Exercise 14 (n-grams)
def exercise14(runs: dict, final_name: str) -> dict:
    final = runs[final_name]
    # n-grams are mined on lemma streams, and the edge filter needs a stop list; when the
    # selected pipeline removes none (e.g. B), the custom list is used for the edge filter only.
    stop = final.stop or stopwords("custom")
    sents = [s for did in sorted(final.sent_streams) for s in final.sent_streams[did]]
    words_only = [[t for t in s if is_word(t)] for s in sents]
    after_removal = [[t for t in s if t not in stop] for s in words_only]
    out_rows, dict_sizes = {}, []
    term_dict = final.index.vocabulary
    for n in range(1, 6):
        raw = ngrams(words_only, n)
        edge = clean_edges(raw, stop) if n > 1 else Counter({g: v for g, v in raw.items() if g[0] not in stop})
        rem = ngrams(after_removal, n)
        adjacent_before = set(raw)
        invented = sum(1 for g in rem if g not in adjacent_before) if n > 1 else 0
        out_rows[n] = {"all": summary(raw), "meaningful_filter": summary(edge), "after_stopword_removal": summary(rem)}
        dict_sizes.append({"n": n, "dictionary_size_all": len(raw), "dictionary_size_filtered": len(edge),
                           "dictionary_size_after_removal": len(rem), "invented_by_removal": invented,
                           "singleton_share": out_rows[n]["all"]["singleton_share"], "term_dictionary": term_dict})
    for n, name in ((1, "unigram_results.csv"), (2, "bigram_results.csv"), (3, "trigram_results.csv")):
        rows = []
        for variant in ("all", "meaningful_filter", "after_stopword_removal"):
            s = out_rows[n][variant]
            for rank, t in enumerate(s["top"], 1):
                rows.append({"variant": variant, "rank": rank, "ngram": t["ngram"], "count": t["count"], "frequency": t["freq"],
                             "total": s["total"], "unique": s["unique"]})
        write_csv(name, rows)
    table_f = []
    for n in range(1, 6):
        s = out_rows[n]
        table_f.append({"ngram": {1: "Unigram", 2: "Bigram", 3: "Trigram", 4: "4-Gram", 5: "5-Gram"}[n],
                        "total": s["all"]["total"], "unique": s["all"]["unique"],
                        "top_ngrams_filtered": " | ".join(f"{t['ngram']} ({t['count']})" for t in s["meaningful_filter"]["top"])})
    ng_rows = []
    for n in (4, 5):
        for variant in ("all", "meaningful_filter"):
            for rank, t in enumerate(out_rows[n][variant]["top"], 1):
                ng_rows.append({"n": n, "variant": variant, "rank": rank, "ngram": t["ngram"], "count": t["count"],
                                "frequency": t["freq"]})
    write_csv("ngram_results.csv", pd.concat([pd.DataFrame(table_f).assign(section="Table F"),
                                              pd.DataFrame(dict_sizes).assign(section="dictionary sizes"),
                                              pd.DataFrame(ng_rows).assign(section="4-5 gram top 10")], ignore_index=True))
    coll2 = collocations(words_only, stop, 2)
    coll3 = collocations(words_only, stop, 3)
    write_csv("collocations.csv", [dict(r, n=2) for r in coll2] + [dict(r, n=3) for r in coll3])
    net = network(clean_edges(ngrams(words_only, 2), stop), 70)
    out = {"final_pipeline": final_name, "per_n": {str(k): v for k, v in out_rows.items()}, "table_f": table_f, "dictionary": dict_sizes,
           "collocations": {"bigram": coll2, "trigram": coll3}, "network": net}
    save_app("ngrams", out)
    return out


# ====================================================================== Module 3-6
def _mean_f1(run, qs, rel) -> float:
    eng = run.engine()
    vals = []
    for q in qs:
        res = eng.search(q["query"], repeats=1)
        vals.append(retrieval_scores(res["doc_ids"], rel[q["qid"]])["f1"])
    return round(sum(vals) / len(vals), 4)


def lexicon_fidelity(run) -> dict:
    corpus = dict(cleaned_corpus())
    eng = run.engine()
    rows = []
    for t in NER.lexicon():
        term = t["term"]
        flags = 0 if re.search(r"[A-Z].*[A-Z]|\d", term) else re.I
        pat = re.compile(r"(?<![\w-])" + re.escape(term) + r"(?:s|es)?(?![\w-])", flags)
        truth = {d for d, txt in corpus.items() if pat.search(txt)}
        got = set(eng.search(f'"{term}"', repeats=1)["doc_ids"])
        jac = len(truth & got) / len(truth | got) if truth | got else 1.0
        rows.append({"term": term, "type": t["type"], "docs_true": len(truth), "docs_retrieved": len(got),
                     "jaccard": round(jac, 3), "preserved": jac == 1.0})
    return {"rows": rows, "preserved": sum(r["preserved"] for r in rows), "mean_jaccard": round(
        statistics.mean(r["jaccard"] for r in rows), 4), "terms": len(rows)}


def candidate_ngrams(runs: dict, k: int = 50) -> list[str]:
    cands = []
    for name, r in runs.items():
        sents = [[t for t in s if is_word(t)] for d in sorted(r.sent_streams) for s in r.sent_streams[d]]
        for n in (2, 3):
            c = clean_edges(ngrams(sents, n), r.stop or stopwords("nltk"))
            cands += [" ".join(g) for g, _ in c.most_common(k)]
    return sorted(set(cands))


def meaningful_share(run, judged: dict[str, int], k: int = 50) -> tuple[float, int]:
    sents = [[t for t in s if is_word(t)] for d in sorted(run.sent_streams) for s in run.sent_streams[d]]
    top = []
    for n in (2, 3):
        c = clean_edges(ngrams(sents, n), run.stop or stopwords("nltk"))
        top += [" ".join(g) for g, _ in c.most_common(k)]
    known = [g for g in top if g in judged]
    good = sum(judged[g] for g in known)
    return (round(good / len(known), 4) if known else 0.0), good


def modules3_6(runs: dict) -> dict:
    qs, rel = queries(), qrels()
    judged = {}
    jpath = GOLD / "ngram_judgements.csv"
    if jpath.exists():
        with open(jpath, newline="", encoding="utf-8") as fh:
            judged = {r["ngram"]: int(r["meaningful"]) for r in csv.DictReader(fh)}
    ret_rows, eval_rows, curves = [], [], {}
    overall = []
    table_h = []
    for name, r in runs.items():
        eng = r.engine()
        per_q = []
        times = []
        for q in qs:
            res = eng.search(q["query"], repeats=20)
            sc = retrieval_scores(res["doc_ids"], rel[q["qid"]])
            per_q.append(sc)
            times.append(res["time_ms"])
            ret_rows.append({"pipeline": name, "qid": q["qid"], "query": q["query"], "type": res["type"],
                             "retrieved_documents": " ".join(res["doc_ids"]), "results": res["count"],
                             "time_ms": res["time_ms"],
                             "analysed_terms": " ; ".join(f"{'NOT ' if a['negated'] else ''}{' '.join(a['terms'])}"
                                                          for a in res["analysed"])})
            eval_rows.append({"pipeline": name, "qid": q["qid"], "query": q["query"], **sc})
            curves.setdefault(name, []).append(curve_at_k(res["doc_ids"], rel[q["qid"]]))
        m = {k: round(statistics.mean(s[k] for s in per_q), 4) for k in
             ("precision", "recall", "f1", "p@3", "p@5", "p@10", "r@3", "r@5", "r@10", "ap")}
        fid = lexicon_fidelity(r)
        share, good = meaningful_share(r, judged)
        summ = r.summary()
        overall.append({"pipeline": name, "label": r.pipeline.label, **m,
                        "mean_query_time_ms": round(statistics.mean(times), 4), "index_seconds": summ["index_seconds"]})
        table_h.append({"pipeline": name, "label": r.pipeline.label, "description": r.pipeline.description,
                        "posthoc": r.pipeline.posthoc,
                        **r.pipeline.config(), "token_count": summ["token_count"],
                        "vocabulary_size": summ["vocabulary_size"], "meaningful_ngrams_share": share,
                        "meaningful_ngrams_top100": good, "domain_terms_preserved": fid["preserved"],
                        "domain_terms_total": fid["terms"], "domain_term_fidelity": fid["mean_jaccard"],
                        "precision": m["precision"], "recall": m["recall"], "f1": m["f1"], "p@5": m["p@5"],
                        "mean_query_time_ms": round(statistics.mean(times), 4)})
        (RESULTS / "app" / f"fidelity_{name}.json").write_text(json.dumps(fid))
    # selection rule fixed in advance: F1, then P@5, then domain fidelity, then smaller vocabulary
    eligible = [x for x in table_h if not x["posthoc"]]
    ranked = sorted(eligible, key=lambda x: (-x["f1"], -x["p@5"], -x["domain_term_fidelity"], x["vocabulary_size"]))
    winner = ranked[0]["pipeline"]
    for row in table_h:
        row["selected"] = row["pipeline"] == winner
    write_csv("pipeline_comparison.csv", table_h)
    # the brief's Table H shape: Pipeline A | Pipeline B | Final Pipeline (= the selected run)
    by = {x["pipeline"]: x for x in table_h}
    measures = [("Token Count", "token_count"), ("Vocabulary Size", "vocabulary_size"),
                ("Meaningful N-Grams (share of top 100)", "meaningful_ngrams_share"),
                ("Domain terms preserved (of 60)", "domain_terms_preserved"), ("Precision", "precision"),
                ("Recall", "recall"), ("F1-score", "f1")]
    write_csv("table_h_brief_format.csv", [{"measure": m, "pipeline_a": by["A"][k], "pipeline_b": by["B"][k],
                                             f"final_pipeline ({winner})": by[winner][k]} for m, k in measures])
    write_csv("retrieval_results.csv", ret_rows)
    write_csv("evaluation_results.csv", eval_rows)
    write_csv("overall_performance.csv", overall)
    stage_rows = [{"pipeline": n, **s} for n, r in runs.items() for s in r.stages]
    write_csv("pipeline_stages.csv", stage_rows)
    runs[winner].index.to_json(RESULTS / "inverted_index.json")
    j = [{"method_pipeline": x["pipeline"] if x["pipeline"] != winner else f"Final pipeline ({winner})",
          "precision": x["precision"], "recall": x["recall"], "f1": x["f1"], "time_ms": x["mean_query_time_ms"]}
         for x in table_h if x["pipeline"] in ("A", "B", winner)]
    write_csv("table_j_brief_format.csv", j)
    avg_curves = {}
    for name, cs in curves.items():
        avg_curves[name] = [{"k": k + 1, "p": round(statistics.mean(c[k]["p"] for c in cs), 4),
                             "r": round(statistics.mean(c[k]["r"] for c in cs), 4)} for k in range(10)]
    out = {"table_h": table_h, "winner": winner, "overall": overall, "retrieval": ret_rows, "evaluation": eval_rows,
           "curves": avg_curves, "stages": stage_rows, "queries": qs,
           "qrels": {k: sorted(v) for k, v in rel.items()},
           "selection_rule": "highest mean F1, then P@5, then domain-term fidelity, then the smaller vocabulary"}
    save_app("pipelines", out)
    return out
