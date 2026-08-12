import argparse
import json

from markov import _next_tokens_from_json
from tokenizer import Tokenizer, _merges_from_json


def label(tokenizer: Tokenizer, token_id: int) -> str:
    if token_id == tokenizer.eos_token:
        return "<EOS>"
    return repr(tokenizer.vocab[token_id])


def show_state(
    tokenizer: Tokenizer, next_tokens: dict[int, dict[int, int]], state: int, limit: int
) -> None:
    outgoing = next_tokens.get(state)
    name = label(tokenizer, state)
    if not outgoing:
        print(f"[{name} (id {state})] no transitions recorded\n")
        return
    total = sum(outgoing.values())
    ranked = sorted(outgoing.items(), key=lambda kv: -kv[1])[:limit]
    print(f"[{name} (id {state})] total transitions: {total}")
    for nxt, cnt in ranked:
        pct = 100.0 * cnt / total if total else 0.0
        print(f"    {cnt:>6} ({pct:5.1f}%)  -> {label(tokenizer, nxt)}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Inspect Markov transitions with token labels"
    )
    parser.add_argument(
        "-t",
        "--tokenizer",
        default="checkpoint/tokenizer/big_corpus.json",
        help="tokenizer checkpoint to load (default: %(default)s)",
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        default="checkpoint/markov/big_corpus.json",
        help="markov checkpoint to load (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--limit",
        type=int,
        default=10,
        help="max transitions to show per state (default: %(default)s)",
    )
    parser.add_argument(
        "query",
        help='token id, or token text to look up (e.g. "return", " ", "\\n")',
    )
    args = parser.parse_args()

    try:
        with open(args.tokenizer, "r") as file:
            ck = json.load(file)
        tokenizer = Tokenizer()
        tokenizer.merges = _merges_from_json(ck["merges"])
        tokenizer.vocab = {int(k): v for k, v in ck["vocab"].items()}
        tokenizer.inverse_vocab = ck["inverse_vocab"]

        with open(args.checkpoint, "r") as file:
            ck2 = json.load(file)
        next_tokens = _next_tokens_from_json(ck2["next_tokens"])
    except (OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
        parser.error(f"could not load checkpoints: {exc}")

    query = args.query
    try:
        state_id = int(query)
    except ValueError:
        state_id = None
    if state_id is not None:
        show_state(tokenizer, next_tokens, state_id, args.limit)
        return

    # exact match first
    if query in tokenizer.inverse_vocab:
        show_state(tokenizer, next_tokens, tokenizer.inverse_vocab[query], args.limit)
        return

    # fuzzy: list tokens containing the query, case-insensitive
    needles = [s for s in tokenizer.vocab.values() if query.lower() in s.lower()]
    needles.sort(key=str.lower)
    if not needles:
        parser.error(f"no token matches {query!r}")
        return
    print(f"{len(needles)} token(s) matching {query!r}:\n")
    for s in needles:
        show_state(tokenizer, next_tokens, tokenizer.inverse_vocab[s], args.limit)


if __name__ == "__main__":
    main()
