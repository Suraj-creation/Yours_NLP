"""All scoring functions in one place, so every number is computed the same way."""
from __future__ import annotations

from collections import Counter, defaultdict


# ---------------------------------------------------------------- generic
def prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4), "tp": tp, "fp": fp, "fn": fn}


def cohen_kappa(a: list, b: list) -> float:
    assert len(a) == len(b) and a
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return round((po - pe) / (1 - pe), 4) if pe != 1 else 1.0


# ---------------------------------------------------------------- tokenization
def boundary_scores(tokenize_fn, gold: list[dict]) -> dict:
    """Exact token-span match against hand-segmented gold (micro-averaged)."""
    from .tokenizers import align

    tot = Counter()
    by_type: dict[str, Counter] = defaultdict(Counter)
    exact_sent = 0
    for g in gold:
        text = g["text"]
        gspans = set(align(g["tokens"], text))
        pred = tokenize_fn(text)
        pspans = set(align(pred, text))
        tp = len(gspans & pspans)
        c = Counter(tp=tp, fp=len(pspans - gspans), fn=len(gspans - pspans))
        tot.update(c)
        by_type[g["problem_type"]].update(c)
        exact_sent += int(gspans == pspans)
    out = prf(tot["tp"], tot["fp"], tot["fn"])
    out["sentence_exact_match"] = round(exact_sent / len(gold), 4)
    out["by_type"] = {k: prf(v["tp"], v["fp"], v["fn"])["f1"] for k, v in sorted(by_type.items())}
    return out


# ---------------------------------------------------------------- retrieval
def retrieval_scores(retrieved: list[str], relevant: set[str], ks=(3, 5, 10)) -> dict:
    """Set metrics on the Boolean result + rank metrics on its TF-IDF order."""
    ret = list(retrieved)
    rs = set(ret)
    tp = len(rs & relevant)
    p = tp / len(rs) if rs else (1.0 if not relevant else 0.0)
    r = tp / len(relevant) if relevant else 1.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    out = {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4),
           "retrieved": len(rs), "relevant": len(relevant), "hits": tp}
    for k in ks:
        top = ret[:k]
        hit = sum(1 for d in top if d in relevant)
        out[f"p@{k}"] = round(hit / k, 4)
        out[f"r@{k}"] = round(hit / len(relevant), 4) if relevant else 1.0
    # average precision (extra; rank quality over the whole list)
    hits, ap = 0, 0.0
    for i, d in enumerate(ret, 1):
        if d in relevant:
            hits += 1
            ap += hits / i
    out["ap"] = round(ap / len(relevant), 4) if relevant else 1.0
    return out


def curve_at_k(retrieved: list[str], relevant: set[str], kmax: int = 10) -> list[dict]:
    rows = []
    for k in range(1, kmax + 1):
        hit = sum(1 for d in retrieved[:k] if d in relevant)
        rows.append({"k": k, "p": round(hit / k, 4), "r": round(hit / len(relevant), 4) if relevant else 1.0})
    return rows


# ---------------------------------------------------------------- tagging
def tagging_accuracy(gold: list[list[str]], pred: list[list[str]]) -> dict:
    tot = cor = 0
    confusion: Counter = Counter()
    for g, p in zip(gold, pred):
        for a, b in zip(g, p):
            tot += 1
            cor += a == b
            if a != b:
                confusion[(a, b)] += 1
    return {"accuracy": round(cor / tot, 4) if tot else 0.0, "tokens": tot, "correct": cor,
            "top_errors": [{"gold": a, "pred": b, "count": c} for (a, b), c in confusion.most_common(15)]}


def confusion_matrix(gold: list[list[str]], pred: list[list[str]], labels: list[str]) -> list[list[int]]:
    idx = {l: i for i, l in enumerate(labels)}
    m = [[0] * len(labels) for _ in labels]
    for g, p in zip(gold, pred):
        for a, b in zip(g, p):
            if a in idx and b in idx:
                m[idx[a]][idx[b]] += 1
    return m


# ---------------------------------------------------------------- NER
def ner_scores(gold: list[list[tuple]], pred: list[list[tuple]]) -> dict:
    """Entities are (start, end, label). correct = exact span + type; incorrect = overlap with
    wrong type or boundary; missed = gold with no overlapping prediction; spurious = prediction
    overlapping no gold entity."""
    tot = Counter()
    by_type: dict[str, Counter] = defaultdict(Counter)
    for gs, ps in zip(gold, pred):
        used = set()
        for g in gs:
            exact = [i for i, p in enumerate(ps) if (p[0], p[1], p[2]) == (g[0], g[1], g[2])]
            if exact:
                used.add(exact[0])
                tot["correct"] += 1
                by_type[g[2]]["correct"] += 1
                continue
            overl = [i for i, p in enumerate(ps) if p[0] < g[1] and g[0] < p[1] and i not in used]
            if overl:
                used.add(overl[0])
                tot["incorrect"] += 1
                by_type[g[2]]["incorrect"] += 1
            else:
                tot["missed"] += 1
                by_type[g[2]]["missed"] += 1
        for i, p in enumerate(ps):
            if i not in used and not any(p[0] < g[1] and g[0] < p[1] for g in gs):
                tot["spurious"] += 1
                by_type[p[2]]["spurious"] += 1
    def _f(c):
        n_gold = c["correct"] + c["incorrect"] + c["missed"]
        n_pred = c["correct"] + c["incorrect"] + c["spurious"]
        p = c["correct"] / n_pred if n_pred else 0.0
        r = c["correct"] / n_gold if n_gold else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        return {"gold": n_gold, "predicted": n_pred, "correct": c["correct"], "incorrect": c["incorrect"],
                "missed": c["missed"], "spurious": c["spurious"], "precision": round(p, 4),
                "recall": round(r, 4), "f1": round(f, 4)}
    return {"overall": _f(tot), "by_type": {k: _f(v) for k, v in sorted(by_type.items())}}
