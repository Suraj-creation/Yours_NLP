"""Module 1: text cleaning, in a fixed and justified order.

Every step logs how many times it fired, so the website and report can show
what cleaning actually did to each document. Case is kept on purpose: NER and
POS need capital letters; lowercasing happens later, on the index branch only.
"""
from __future__ import annotations

import re
import unicodedata

VULGAR = {"½": "1/2", "⅓": "1/3", "⅔": "2/3", "¼": "1/4", "¾": "3/4", "⅕": "1/5", "⅖": "2/5", "⅗": "3/5",
          "⅘": "4/5", "⅙": "1/6", "⅚": "5/6", "⅛": "1/8", "⅜": "3/8", "⅝": "5/8", "⅞": "7/8"}
TYPO = {"’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": " - ",
        "−": "-", "…": "...", " ": " ", " ": " ", "​": "", "﻿": ""}

STEPS = ["vulgar_fractions", "unicode_nfkc", "typography", "dehyphenate", "references", "urls",
         "emoji", "table_markup", "page_numbers", "digit_word_glue", "whitespace"]


def clean_text(text: str, fmt: str = "") -> tuple[str, dict]:
    log = {s: 0 for s in STEPS}

    # 1. Vulgar fractions BEFORE NFKC: NFKC turns "3½" into "31⁄2", which reads as 31/2.
    def _vf(m):
        log["vulgar_fractions"] += 1
        whole = m.group(1)
        return (whole + " " if whole else "") + VULGAR[m.group(2)]
    text = re.sub(r"(\d?)([" + "".join(VULGAR) + "])", _vf, text)

    # 2. Unicode normalisation (full-width digits, ligatures), fraction slash -> "/"
    before = text
    text = unicodedata.normalize("NFKC", text).replace("⁄", "/")
    log["unicode_nfkc"] = sum(1 for a, b in zip(before, text) if a != b) + abs(len(before) - len(text))

    # 3. Typographic quotes, dashes, ellipsis, invisible characters
    for k, v in TYPO.items():
        c = text.count(k)
        if c:
            log["typography"] += c
            text = text.replace(k, v)

    # 4. Re-join words hyphenated across line breaks ("denomi-\nnator")
    text, n = re.subn(r"([a-z])-\n\s*([a-z])", r"\1\2", text)
    log["dehyphenate"] = n

    # 5. Reference lists: author names and venues would inflate vocabulary and NER
    m = re.search(r"\n\s*(References|Bibliography|REFERENCES)\s*\n", text)
    if m and m.start() > len(text) * 0.4:
        log["references"] = len(text) - m.start()
        text = text[: m.start()]

    # 6. URLs are not language
    text, n = re.subn(r"https?://\S+|www\.\S+", " ", text)
    log["urls"] = n

    # 7. Emoji and pictographs (README headers use them)
    text, n = re.subn(r"[\U0001F300-\U0001FAFF☀-➿\U0001F000-\U0001F2FF]", " ", text)
    log["emoji"] = n

    # 8. Table markup left by HTML/Markdown extraction
    text, n1 = re.subn(r"^\s*\|?\s*:?-{3,}.*$", " ", text, flags=re.M)
    text, n2 = re.subn(r"\s\|\s|^\|\s|\s\|$", " ", text, flags=re.M)
    text, n3 = re.subn(r"^\s*[-*]\s+(?=\S)", "", text, flags=re.M)
    log["table_markup"] = n1 + n2 + n3

    # 9. Stand-alone page numbers (PDF)
    if fmt.upper() == "PDF":
        text, n = re.subn(r"^\s*\d{1,3}\s*$", " ", text, flags=re.M)
        log["page_numbers"] = n

    # 10. A number glued to a word by extraction ("1/8pieces"); keeps "2x", "5th"
    text, n = re.subn(r"(\d)([a-z]{3,})", r"\1 \2", text)
    log["digit_word_glue"] = n

    # 11. Whitespace: keep line breaks (dialogue turns are lines), collapse the rest
    lines = [" ".join(ln.split()) for ln in text.splitlines()]
    out, blank = [], False
    for ln in lines:
        if ln:
            out.append(ln)
            blank = False
        elif not blank and out:
            out.append("")
            blank = True
    cleaned = "\n".join(out).strip()
    log["whitespace"] = len(text) - len(cleaned)
    return cleaned, log


def split_sentences(text: str, method: str = "punkt") -> list[str]:
    """Lines are hard boundaries (turns, list items); sentences are split inside lines."""
    sents: list[str] = []
    if method == "punkt":
        from nltk.tokenize import sent_tokenize

        for ln in text.splitlines():
            if ln.strip():
                sents.extend(s for s in sent_tokenize(ln) if s.strip())
        return sents
    if method == "spacy":
        from .nlp_models import sentencizer

        nlp = sentencizer()
        for ln in text.splitlines():
            if ln.strip():
                sents.extend(s.text for s in nlp(ln).sents if s.text.strip())
        return sents
    if method == "naive":
        return [s.strip() for s in re.split(r"[.!?]\s+|\n", text) if s.strip()]
    raise ValueError(method)
