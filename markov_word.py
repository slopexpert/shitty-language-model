"""Higher-order word-level Markov model.

Unlike the token-level `markov.Markov` (which strings together BPE tokens and
therefore can't hold grammar), this works on natural-language *words*. The state
is a tuple of the last `order` words, so order-2 conditions on the previous two
words and can actually reconstruct grammatical structure. Generation uses
backoff: the longest seen context wins, falling back to a shorter one until the
empty (sentence-start) context.
"""

import json
import random
import re

# a "word" token is a run of word chars; everything else (punctuation, spaces,
# newlines) is its own token. This keeps grammar and sentence breaks explicit.
WORD_RE = re.compile(r"[A-Za-z0-9_']+|[^\w\s]|\s+")


def words_of(text: str) -> list[str]:
    # collapse runs of whitespace to a single representative so formatting
    # doesn't dominate the distribution
    toks = []
    for t in WORD_RE.findall(text):
        if t.isspace():
            t = "\n" if "\n" in t else " "
        toks.append(t)
    return toks


class WordMarkov:
    def __init__(self, order: int = 2):
        if order < 1:
            raise ValueError("order must be >= 1")
        self.order = order
        # context (tuple of words, maybe empty) -> {next word: count}
        self.next_tokens: dict[tuple[str, ...], dict[str, int]] = {}

    def train_on_corpus(self, corpus: list[str]):
        for text in corpus:
            words = words_of(text)
            for i in range(len(words)):
                start = max(0, i - self.order)
                ctx = tuple(words[start:i])
                nxt = words[i]
                outgoing = self.next_tokens.setdefault(ctx, {})
                outgoing[nxt] = outgoing.get(nxt, 0) + 1

    def continuation(self, context: tuple[str, ...]) -> dict[str, int] | None:
        """Return the outgoing distribution for the longest seen suffix of
        `context` (backoff), or None if none of them are known."""
        for k in range(min(len(context), self.order), -1, -1):
            ctx = tuple(context[-k:]) if k else ()
            out = self.next_tokens.get(ctx)
            if out:
                return out
        return None

    def sample_next(self, context: tuple[str, ...], temperature: float = 1.0,
                    deterministic: bool = False) -> str | None:
        out = self.continuation(context)
        if out is None:
            return None
        words, counts = zip(*out.items())
        if deterministic:
            return words[counts.index(max(counts))]
        if temperature != 1.0:
            t = max(temperature, 1e-3)
            counts = tuple(c ** (1.0 / t) for c in counts)
        return random.choices(words, weights=counts, k=1)[0]

    @property
    def num_transitions(self) -> int:
        return sum(sum(c for c in m.values()) for m in self.next_tokens.values())


def _next_tokens_to_json(
    next_tokens: dict[tuple[str, ...], dict[str, int]],
) -> dict[str, dict[str, int]]:
    # context tuple -> string key via json.dumps of the list
    return {json.dumps(list(ctx)): dict(out) for ctx, out in next_tokens.items()}


def _next_tokens_from_json(
    data: dict[str, dict[str, int]],
) -> dict[tuple[str, ...], dict[str, int]]:
    try:
        return {tuple(json.loads(key)): dict(m) for key, m in data.items()}
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("invalid word-markov checkpoint") from exc
