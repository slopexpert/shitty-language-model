import argparse
import json
import random
import sys

from shitty_language_model.markov import _next_tokens_from_json
from shitty_language_model.tokenizer import Tokenizer, _merges_from_json


def main():
    parser = argparse.ArgumentParser(
        description="Generate text from a trained Markov model, seeded by some text"
    )
    parser.add_argument(
        "-t",
        "--tokenizer",
        default="checkpoint/tokenizer/big_corpus.json",
        help="tokenizer checkpoint to load (default: %(default)s)",
    )
    parser.add_argument(
        "-m",
        "--markov",
        default="checkpoint/markov/big_corpus.json",
        help="markov checkpoint to load (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--length",
        type=int,
        default=120,
        help="number of tokens to generate (default: %(default)s)",
    )
    parser.add_argument(
        "-s",
        "--seed",
        help="text used to seed the generation (default: read from stdin)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="softmax temperature; higher = more varied, lower = more likely (default: %(default)s)",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="always pick the most frequent next token instead of sampling",
    )
    args = parser.parse_args()

    seed = args.seed
    if seed is None:
        seed = sys.stdin.read()

    try:
        with open(args.tokenizer, "r") as file:
            ck = json.load(file)
        tokenizer = Tokenizer()
        tokenizer.merges = _merges_from_json(ck["merges"])
        tokenizer.vocab = {int(k): v for k, v in ck["vocab"].items()}
        tokenizer.inverse_vocab = ck["inverse_vocab"]

        with open(args.markov, "r") as file:
            ck2 = json.load(file)
        next_tokens = _next_tokens_from_json(ck2["next_tokens"])
    except (OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
        parser.error(f"could not load checkpoints: {exc}")

    # sample from a softmax over the observed transition counts
    def sample(token: int) -> int:
        outgoing = next_tokens.get(token)
        if not outgoing:
            return tokenizer.eos_token
        states, counts = zip(*outgoing.items())
        if args.deterministic:
            return states[counts.index(max(counts))]
        if args.temperature != 1.0:
            t = max(args.temperature, 1e-3)
            counts = tuple(c ** (1.0 / t) for c in counts)
        return random.choices(states, weights=counts, k=1)[0]

    # seed: continue from the last token of the seed text
    tokens = tokenizer.tokenize(seed)
    if not tokens:
        parser.error("empty seed text")
    current = tokens[-1]

    generated = []
    for _ in range(args.length):
        nxt = sample(current)
        if nxt == tokenizer.eos_token:
            break
        generated.append(nxt)
        current = nxt

    print(tokenizer.untokenize(generated))


if __name__ == "__main__":
    main()
