"""A small, dependency-free byte-pair encoding tokenizer.

Both merge-learning (`build`) and encoding (`_tokenize`) use a doubly
linked-list token representation with a lazy priority queue, rather than
repeatedly rescanning whole sequences. This matters once vocab sizes and
corpus sizes grow into "upper-mid" territory (up to ~32k vocab, corpora
that are a single large document) -- a naive "rescan everything for every
merge" BPE implementation is easily 100-1000x slower and can turn a
few-second tokenizer build into one that never finishes.
"""

from __future__ import annotations

import heapq
import random
from typing import Dict, List, Optional, Sequence, Tuple

from .char_tokenizer import BOS, EOS, PAD, SPECIAL_TOKENS, UNK


def _subsample_for_training(texts: Sequence[str], max_chars: int, seed: int = 0) -> List[str]:
    """Cap the amount of text used to *learn* merges. Vocabulary/merge
    statistics saturate well before "use the entire multi-GB corpus", so
    sampling keeps training tractable without materially hurting the
    learned vocabulary. All characters in the *full* corpus still end up
    in the base vocabulary (see `build`), so nothing becomes
    unrepresentable -- only the merge statistics are estimated from a
    sample.
    """
    total = sum(len(t) for t in texts)
    if total <= max_chars:
        return list(texts)
    rng = random.Random(seed)
    order = list(range(len(texts)))
    rng.shuffle(order)
    budget = max_chars
    chosen = []
    for i in order:
        if budget <= 0:
            break
        chosen.append(texts[i])
        budget -= len(texts[i])
    return chosen


class _LinkedSeq:
    """A mutable doubly linked list over one tokenized sequence, so a
    merge only touches the handful of nodes around each occurrence
    instead of rebuilding the whole sequence.
    """

    __slots__ = ("tok", "prev", "nxt", "alive")

    def __init__(self, tokens: List[str]):
        n = len(tokens)
        self.tok = list(tokens)
        self.prev = list(range(-1, n - 1))
        self.nxt = list(range(1, n + 1))
        if n:
            self.nxt[-1] = -1
        self.alive = [True] * n

    def ordered_tokens(self) -> List[str]:
        return [t for t, a in zip(self.tok, self.alive) if a]


class BPETokenizer:
    """Character-initialized BPE tokenizer.

    BPE operates on Unicode characters rather than bytes so the resulting
    vocabulary remains portable and decoding preserves the original text.
    """

    def __init__(self, vocab: List[str], merges: List[Tuple[str, str]]):
        self.vocab = list(vocab)
        self.merges = [tuple(pair) for pair in merges]
        self._stoi: Dict[str, int] = {token: i for i, token in enumerate(self.vocab)}
        self._merge_ranks: Dict[Tuple[str, str], int] = {pair: i for i, pair in enumerate(self.merges)}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def pad_id(self) -> int:
        return self._stoi[PAD]

    @property
    def unk_id(self) -> int:
        return self._stoi[UNK]

    @property
    def bos_id(self) -> int:
        return self._stoi[BOS]

    @property
    def eos_id(self) -> int:
        return self._stoi[EOS]

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        texts: Sequence[str],
        vocab_size: int = 1000,
        min_frequency: int = 2,
        max_train_chars: int = 3_000_000,
    ) -> "BPETokenizer":
        if vocab_size < len(SPECIAL_TOKENS):
            raise ValueError(f"vocab_size must be at least {len(SPECIAL_TOKENS)}")
        if min_frequency < 1:
            raise ValueError("min_frequency must be at least 1")

        # Base vocabulary covers every character in the *full* corpus so
        # nothing falls back to <unk> just because it was outside the
        # (possibly subsampled) merge-training set.
        symbols = set(char for text in texts for char in text)
        vocab: List[str] = list(SPECIAL_TOKENS) + sorted(symbols - set(SPECIAL_TOKENS))
        vocab_set = set(vocab)

        train_texts = _subsample_for_training(texts, max_train_chars)
        seqs = [_LinkedSeq(list(t)) for t in train_texts if t]

        pair_count: Dict[Tuple[str, str], int] = {}
        pair_positions: Dict[Tuple[str, str], set] = {}
        heap: List[Tuple[int, Tuple[str, str]]] = []

        def _bump(pair: Tuple[str, str], delta: int, seq_i: int, node_i: int) -> None:
            pos_set = pair_positions.setdefault(pair, set())
            if delta > 0:
                pos_set.add((seq_i, node_i))
            else:
                pos_set.discard((seq_i, node_i))
            new_count = pair_count.get(pair, 0) + delta
            if new_count <= 0:
                pair_count.pop(pair, None)
                pair_positions.pop(pair, None)
            else:
                pair_count[pair] = new_count
                heapq.heappush(heap, (-new_count, pair))

        for si, seq in enumerate(seqs):
            for i in range(len(seq.tok) - 1):
                pair = (seq.tok[i], seq.tok[i + 1])
                _bump(pair, 1, si, i)

        merges: List[Tuple[str, str]] = []

        while len(vocab) < vocab_size and heap:
            neg_count, pair = heapq.heappop(heap)
            if pair_count.get(pair, 0) != -neg_count:
                continue  # stale entry, count has since changed
            if -neg_count < min_frequency:
                break  # heap is max-ordered: nothing left clears the bar
            merged_tok = pair[0] + pair[1]
            if merged_tok in vocab_set:
                continue

            merges.append(pair)
            vocab.append(merged_tok)
            vocab_set.add(merged_tok)

            occurrences = list(pair_positions.get(pair, ()))
            for seq_i, i in occurrences:
                seq = seqs[seq_i]
                if not seq.alive[i]:
                    continue
                j = seq.nxt[i]
                if j == -1 or not seq.alive[j]:
                    continue
                if seq.tok[i] != pair[0] or seq.tok[j] != pair[1]:
                    continue

                p = seq.prev[i]
                nx = seq.nxt[j]

                if p != -1:
                    _bump((seq.tok[p], seq.tok[i]), -1, seq_i, p)
                if nx != -1:
                    _bump((seq.tok[j], seq.tok[nx]), -1, seq_i, j)

                seq.tok[i] = merged_tok
                seq.alive[j] = False
                seq.nxt[i] = nx
                if nx != -1:
                    seq.prev[nx] = i

                if p != -1:
                    _bump((seq.tok[p], seq.tok[i]), 1, seq_i, p)
                if nx != -1:
                    _bump((seq.tok[i], seq.tok[nx]), 1, seq_i, i)

            pair_count.pop(pair, None)
            pair_positions.pop(pair, None)

        return cls(vocab, merges)

    # ------------------------------------------------------------------
    # Encoding / decoding
    # ------------------------------------------------------------------

    def _tokenize(self, text: str) -> List[str]:
        n = len(text)
        if n <= 1:
            return list(text)

        seq = _LinkedSeq(list(text))
        heap: List[Tuple[int, int, str, str]] = []

        def push(i: int) -> None:
            j = seq.nxt[i]
            if j == -1:
                return
            pair = (seq.tok[i], seq.tok[j])
            rank = self._merge_ranks.get(pair)
            if rank is not None:
                heapq.heappush(heap, (rank, i, pair[0], pair[1]))

        for i in range(n - 1):
            push(i)

        while heap:
            rank, i, a, b = heapq.heappop(heap)
            if not seq.alive[i]:
                continue
            j = seq.nxt[i]
            if j == -1 or not seq.alive[j]:
                continue
            if seq.tok[i] != a or seq.tok[j] != b:
                continue  # stale: tokens at these positions changed since this entry was pushed

            merged = a + b
            seq.tok[i] = merged
            seq.alive[j] = False
            nx = seq.nxt[j]
            seq.nxt[i] = nx
            if nx != -1:
                seq.prev[nx] = i

            p = seq.prev[i]
            if p != -1:
                push(p)
            push(i)

        return seq.ordered_tokens()

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        ids = [self._stoi.get(token, self.unk_id) for token in self._tokenize(text)]
        if add_special_tokens:
            ids = [self.bos_id] + ids + [self.eos_id]
        return ids

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        special_ids = {self.pad_id, self.bos_id, self.eos_id} if skip_special_tokens else set()
        result = []
        for index in ids:
            if index in special_ids:
                continue
            if 0 <= index < len(self.vocab):
                token = self.vocab[index]
                if token == UNK:
                    result.append("\ufffd")
                elif token not in SPECIAL_TOKENS:
                    result.append(token)
        return "".join(result)

    def state_dict(self) -> Dict:
        return {
            "tokenizer_type": "bpe",
            "vocab": self.vocab,
            "merges": [list(pair) for pair in self.merges],
        }

    @classmethod
    def from_state_dict(cls, state: Dict) -> "BPETokenizer":
        return cls(list(state["vocab"]), [tuple(pair) for pair in state.get("merges", [])])
