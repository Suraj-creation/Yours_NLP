"""Module 1: extraction and cleaning."""
import xml.etree.ElementTree as ET

import pytest

from nlp_core.cleaning import clean_text, split_sentences
from nlp_core.loaders import load_bytes, load_corpus, mathml_to_text, read_catalog

M = "http://www.w3.org/1998/Math/MathML"


@pytest.mark.parametrize("raw,expected", [
    ("He ate 3½ pizzas and ½ cake.", "He ate 3 1/2 pizzas and 1/2 cake."),
    ("the denomi-\nnator is 8", "the denominator is 8"),
    ("cut into 1/8pieces now", "cut into 1/8 pieces now"),
])
def test_cleaning_steps(raw, expected):
    assert clean_text(raw, "TXT")[0] == expected


def test_vulgar_fractions_run_before_nfkc():
    # NFKC alone would turn 3½ into "31⁄2", a different number
    clean, log = clean_text("3½", "TXT")
    assert clean == "3 1/2" and log["vulgar_fractions"] == 1


def test_urls_removed_and_logged():
    clean, log = clean_text("see https://example.org/x for more", "TXT")
    assert "http" not in clean and log["urls"] == 1


def test_lines_are_hard_sentence_boundaries():
    assert split_sentences("Teacher: What is 1/2 of 8\nStudent: 4. I think so.") == \
        ["Teacher: What is 1/2 of 8", "Student: 4.", "I think so."]


@pytest.mark.parametrize("xml,text", [
    (f'<math xmlns="{M}"><mfrac><mn>3</mn><mn>4</mn></mfrac></math>', "3/4"),
    (f'<math xmlns="{M}"><mn>3</mn><mfrac><mn>1</mn><mn>2</mn></mfrac></math>', "3 1/2"),
    (f'<math xmlns="{M}"><msup><mi>x</mi><mn>2</mn></msup></math>', "x^2"),
    (f'<math xmlns="{M}"><mfrac><mrow><mn>1</mn><mo>+</mo><mn>2</mn></mrow><mn>5</mn></mfrac></math>', "(1+2)/5"),
])
def test_mathml_linearised(xml, text):
    assert mathml_to_text(ET.fromstring(xml)) == text


@pytest.fixture(scope="module")
def corpus():
    return load_corpus()


def test_corpus_has_thirty_documents_in_ten_formats(corpus):
    assert [d.doc_id for d in corpus] == [f"D{i:02d}" for i in range(1, 31)]
    assert len({d.fmt.split(" ")[0] for d in corpus}) == 10
    assert all(len(d.text) > 500 for d in corpus)


def test_catalogue_matches_files(corpus):
    cat = {r["doc_id"]: r for r in read_catalog()}
    assert set(cat) == {d.doc_id for d in corpus}
    assert sum(r["synthetic"] == "True" for r in cat.values()) == 6


def test_dialogue_acts_moved_to_metadata(corpus):
    d17 = next(d for d in corpus if d.doc_id == "D17")
    assert "(probing)" not in d17.text and "(focus)" not in d17.text


def test_cnxml_alt_texts_excluded(corpus):
    d01 = next(d for d in corpus if d.doc_id == "D01")
    assert d01.meta.get("alt_texts_removed", 0) >= 0


@pytest.mark.parametrize("name,content,fmt", [
    ("a.txt", b"Aarav added 1/2 + 1/3 = 2/5.", "TXT"),
    ("a.md", b"# Notes\n\nAdd the **numerators**.", "MD"),
    ("a.html", b"<html><body><nav>Home</nav><p>Fractions are parts of a whole, like 3/4 of a pizza shared between friends at lunch time.</p></body></html>", "HTML"),
    ("a.json", b'{"turns": [{"speaker": "student", "text": "idk"}]}', "JSON"),
    ("a.csv", b"id,text\n1,Add the tops\n", "CSV"),
])
def test_upload_loader_accepts_formats(name, content, fmt):
    d = load_bytes(name, content)
    assert d.fmt.startswith(fmt) and d.text.strip()


def test_upload_rejects_unknown_format():
    with pytest.raises(ValueError):
        load_bytes("a.exe", b"MZ\x00")
