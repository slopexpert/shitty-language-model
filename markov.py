from tokenizer import Tokenizer


class Markov:
    next_tokens: dict[
        int, dict[int, int]
    ]  # token -> next token -> amount of times this showed up
    tokenizer: Tokenizer

    def __init__(self, tokenizer: Tokenizer):
        self.next_tokens = {}
        self.tokenizer = tokenizer

    def train_on_corpus(self, corpus: list[str]):
        for text in corpus:
            tokens = self.tokenizer.tokenize(text)
            for i in range(len(tokens) - 1):
                if tokens[i] not in self.next_tokens:
                    self.next_tokens[tokens[i]] = {}

                target: int = self.tokenizer.eos_token
                if i + 1 < len(tokens):
                    target = tokens[i + 1]

                if target not in self.next_tokens[tokens[i]]:
                    self.next_tokens[tokens[i]][target] = 0

                self.next_tokens[tokens[i]][target] += 1

    def next_token(self, token: int) -> int:
        candidates = self.next_tokens[token]
        most_freq = max(candidates, key=lambda k: candidates[k])
        return candidates[most_freq]


def _next_tokens_to_json(
    next_tokens: dict[int, dict[int, int]],
) -> dict[str, dict[str, int]]:
    return {str(a): {str(b): c for b, c in m.items()} for a, m in next_tokens.items()}


def _next_tokens_from_json(
    data: dict[str, dict[str, int]],
) -> dict[int, dict[int, int]]:
    out: dict[int, dict[int, int]] = {}
    try:
        for a, m in data.items():
            out[int(a)] = {int(b): c for b, c in m.items()}
        return out
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid markov checkpoint") from exc
