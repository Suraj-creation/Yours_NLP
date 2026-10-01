"""Module 2, Exercise 4: Byte Pair Encoding.

Three views of subword tokenization:
  1. tiktoken (the course's tool): gpt2, cl100k_base, o200k_base, pretrained by OpenAI.
  2. Our own BPE trained from scratch on the corpus (Sennrich et al. 2016), with the
     merge list kept so each merge step can be shown.
  3. SimpleTokenizerV2 (BPE lab): word-level baseline with <|unk|>, showing the OOV
     problem BPE removes.
For the heavy datasets a Rust BPE trainer (Hugging Face tokenizers) trains the same
algorithm on millions of words in seconds.
"""
from __future__ import annotations

import heapq
import json
import re
from collections import Counter, defaultdict
from functools import lru_cache

from .config import CONFIG

ENCODINGS = ["gpt2", "cl100k_base", "o200k_base"]


@lru_cache(maxsize=None)
def encoding(name: str):
    """tiktoken encoding. Uses the bundled official rank files when present (the build
    environment could not reach OpenAI's download host), else tiktoken's own download."""
    import tiktoken

    ranks = CONFIG / "tiktoken_ranks" / f"{name}.tiktoken"
    if ranks.exists():
        from tiktoken.load import load_tiktoken_bpe

        meta = json.loads((CONFIG / "tiktoken_ranks" / f"{name}.meta.json").read_text(encoding="utf-8"))
        return tiktoken.Encoding(name=name, pat_str=meta["pat_str"], mergeable_ranks=load_tiktoken_bpe(str(ranks)),
                                 special_tokens=meta["special_tokens"])
    return tiktoken.get_encoding(name)


def tiktoken_pieces(text: str, name: str = "cl100k_base") -> list[str]:
    enc = encoding(name)
    return [enc.decode_single_token_bytes(i).decode("utf-8", errors="replace") for i in enc.encode(text)]


def tiktoken_word(word: str, name: str) -> list[str]:
    """How a word is split when it appears mid-sentence (leading space, as in running text)."""
    return [p for p in tiktoken_pieces(" " + word, name)]


# ---------------------------------------------------------------- BPE from scratch
EOW = "</w>"


class SimpleBPE:
    """Classic character-level BPE with an end-of-word marker, efficient incremental training."""

    def __init__(self):
        self.merges: list[tuple[str, str]] = []
        self.ranks: dict[tuple[str, str], int] = {}
        self.history: list[dict] = []

    def fit(self, word_counts: Counter, num_merges: int = 2000, record: int = 30) -> "SimpleBPE":
        words = [list(w) + [EOW] for w in word_counts]
        freqs = list(word_counts.values())
        pairs: Counter = Counter()
        where: dict[tuple, set] = defaultdict(set)
        for i, sym in enumerate(words):
            for a, b in zip(sym, sym[1:]):
                pairs[(a, b)] += freqs[i]
                where[(a, b)].add(i)
        heap = [(-c, p) for p, c in pairs.items()]
        heapq.heapify(heap)
        base_vocab = {s for w in words for s in w}
        while len(self.merges) < num_merges and heap:
            c, p = heapq.heappop(heap)
            if pairs.get(p, 0) != -c or -c < 2:
                if pairs.get(p, 0) >= 2 and pairs[p] != -c:
                    heapq.heappush(heap, (-pairs[p], p))
                continue
            a, b = p
            new = a + b
            self.merges.append(p)
            if len(self.history) < record:
                self.history.append({"step": len(self.merges), "pair": f"{a} + {b}", "merged": new, "count": -c})
            for i in list(where[p]):
                sym, f = words[i], freqs[i]
                j, out = 0, []
                changed = False
                while j < len(sym):
                    if j < len(sym) - 1 and sym[j] == a and sym[j + 1] == b:
                        out.append(new)
                        j += 2
                        changed = True
                    else:
                        out.append(sym[j])
                        j += 1
                if not changed:
                    continue
                for x, y in zip(sym, sym[1:]):
                    pairs[(x, y)] -= f
                    where[(x, y)].discard(i)
                for x, y in zip(out, out[1:]):
                    pairs[(x, y)] += f
                    where[(x, y)].add(i)
                    heapq.heappush(heap, (-pairs[(x, y)], (x, y)))
                words[i] = out
            pairs.pop(p, None)
        self.ranks = {m: r for r, m in enumerate(self.merges)}
        self.vocab_size = len(base_vocab) + len(self.merges)
        return self

    def encode_word(self, word: str) -> list[str]:
        sym = list(word) + [EOW]
        while len(sym) > 1:
            cands = [(self.ranks.get((a, b), 1 << 30), i) for i, (a, b) in enumerate(zip(sym, sym[1:]))]
            r, i = min(cands)
            if r == 1 << 30:
                break
            sym = sym[:i] + [sym[i] + sym[i + 1]] + sym[i + 2:]
        return sym

    def encode(self, words: list[str]) -> list[str]:
        out = []
        for w in words:
            out.extend(self.encode_word(w))
        return out


def pretokenize(text: str) -> list[str]:
    """Words for BPE training: lower-cased letters/digits runs (numbers kept, punctuation dropped)."""
    return re.findall(r"[a-z]+|\d+", text.lower())


# ---------------------------------------------------------------- word-level baseline (BPE lab)
class SimpleTokenizerV2:
    """Word-level tokenizer from the BPE lab: fixed vocabulary + <|unk|> + <|endoftext|>."""

    SPLIT = re.compile(r'([,.:;?_!"()\']|--|\s)')

    def __init__(self, training_text: str):
        toks = [t.strip() for t in self.SPLIT.split(training_text) if t.strip()]
        vocab = sorted(set(toks)) + ["<|endoftext|>", "<|unk|>"]
        self.str_to_int = {t: i for i, t in enumerate(vocab)}
        self.int_to_str = {i: t for t, i in self.str_to_int.items()}

    def encode(self, text: str) -> list[int]:
        toks = [t.strip() for t in self.SPLIT.split(text) if t.strip()]
        return [self.str_to_int.get(t, self.str_to_int["<|unk|>"]) for t in toks]

    def decode(self, ids: list[int]) -> str:
        text = " ".join(self.int_to_str[i] for i in ids)
        return re.sub(r'\s+([,.?!"()\'])', r"\1", text)

    @property
    def vocab_size(self) -> int:
        return len(self.str_to_int)

    def oov_rate(self, text: str) -> float:
        ids = self.encode(text)
        unk = self.str_to_int["<|unk|>"]
        return round(sum(1 for i in ids if i == unk) / max(1, len(ids)), 4)


# ---------------------------------------------------------------- Rust trainer for the heavy datasets
def train_hf_bpe(texts, vocab_size: int = 8000):
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers

    tok = Tokenizer(models.BPE(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=["[UNK]"], show_progress=False)
    tok.train_from_iterator(texts, trainer)
    return tok
