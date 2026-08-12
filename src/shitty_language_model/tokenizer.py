import functools
import itertools
import operator
from dataclasses import dataclass

EOS = "<|EOS|>"
WB = "<|WB|>"


class Tokenizer:
    merges: dict[tuple[str, str], str]
    _splits: dict[str, list[str]]
    vocab: dict[int, str]
    inverse_vocab: dict[str, int]
    eos_token: int

    def __init__(self):
        self.merges = {}
        self._splits = {}

        self.vocab = {}
        self.inverse_vocab = {}
        for i in range(256):
            self._set_token(i, chr(i))
        self._append_token(EOS)
        self.eos_token = self.inverse_vocab[EOS]

        self._append_token(WB)

    def prepare_splits(self, words: list[str]):
        seen_unique = set(self.vocab.values())

        for word in words:
            b = list(word)

            for c in b:
                if c not in seen_unique:
                    self._append_token(c)

            b.append(WB)
            self._splits[word] = b

    def train_on_corpus(self, corpus: list[str], passes: int):
        for text in corpus:
            self.prepare_splits(text.split())

        for pass_n in range(passes):
            pair_freq = {}

            for k in self._splits:
                word = self._splits[k]
                for x, y in itertools.pairwise(word):
                    if (x, y) not in pair_freq:
                        pair_freq[(x, y)] = 0
                    pair_freq[(x, y)] += 1

            if len(pair_freq) == 0:
                continue

            most_freq = max(pair_freq, key=lambda k: pair_freq[k])

            self.update_splits(most_freq[0], most_freq[1])

            self.merges[most_freq] = most_freq[0] + most_freq[1]

            self._append_token(self.merges[most_freq])

    def update_splits(self, lhs: str, rhs: str):
        for word, splits in self._splits.items():
            new_split = []
            i = 0
            while i < len(splits):
                if splits[i] == lhs and i + 1 < len(splits) and splits[i + 1] == rhs:
                    new_split.append(lhs + rhs)
                    i += 2
                else:
                    new_split.append(splits[i])
                    i += 1
            self._splits[word] = new_split

    def tokenize(self, s: str) -> list[int]:
        splits = [list(t) + [WB] for t in s.split(" ")]

        for lhs, rhs in self.merges:
            for idx, split in enumerate(splits):
                new_split = []
                cursor = 0
                while cursor < len(split):
                    if (
                        cursor + 1 < len(split)
                        and split[cursor] == lhs
                        and split[cursor + 1] == rhs
                    ):
                        new_split.append(lhs + rhs)
                        cursor += 2
                    else:
                        new_split.append(split[cursor])
                        cursor += 1
                assert "".join(new_split) == "".join(split)
                splits[idx] = new_split

        tokens_raw: list[str] = functools.reduce(operator.iadd, splits, [])
        return [self.inverse_vocab[t] for t in tokens_raw]

    def untokenize(self, tokens: list[int]) -> str:
        s = "".join(self.vocab[t] for t in tokens)
        return s.replace(WB, " ")

    def _set_token(self, id: int, s: str):
        self.vocab[id] = s
        self.inverse_vocab[s] = id

    def _append_token(self, s: str):
        id = len(self.vocab)
        self._set_token(id, s)


@dataclass
class TokenizerCheckpoint:
    merges: dict[tuple[str, str], str]
    vocab: dict[int, str]
    inverse_vocab: dict[str, int]


def _merges_to_json(merges: dict[tuple[str, str], str]) -> list[list[str]]:
    return [[lhs, rhs, merged] for (lhs, rhs), merged in merges.items()]


def _merges_from_json(data: list[list[str]]) -> dict[tuple[str, str], str]:
    return {(lhs, rhs): merged for lhs, rhs, merged in data}
