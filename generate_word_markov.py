import argparse
import json
import sys

from markov_word import WordMarkov, _next_tokens_from_json, words_of


def main():
    parser = argparse.ArgumentParser(
        description="Generate text from a trained word-level Markov model"
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        default="checkpoint/word_markov/default.json",
        help="word-markov checkpoint to load (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--length",
        type=int,
        default=100,
        help="number of words to generate (default: %(default)s)",
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
        help="always pick the most frequent next word instead of sampling",
    )
    args = parser.parse_args()

    seed = args.seed
    if seed is None:
        seed = sys.stdin.read()

    try:
        with open(args.checkpoint, "r") as file:
            ck = json.load(file)
        markov = WordMarkov(order=int(ck["order"]))
        markov.next_tokens = _next_tokens_from_json(ck["next_tokens"])
    except (OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
        parser.error(f"could not load checkpoint: {exc}")

    context = tuple(words_of(seed))
    # keep only the last `order` words for the actual context
    context = context[-markov.order:] if context else ()

    generated = []
    for _ in range(args.length):
        nxt = markov.sample_next(
            context,
            temperature=args.temperature,
            deterministic=args.deterministic,
        )
        if nxt is None:
            break
        generated.append(nxt)
        context = (context + (nxt,))[-markov.order:]

    # reconstruct text: strip extra whitespace around punctuation, join sensibly
    out = ""
    for tok in generated:
        if tok.isspace():
            out += tok
        elif out and not out.endswith((" ", "\n")) and tok not in ".,;:!?)]}":
            out += " "
        out += tok
    print(out.strip())


if __name__ == "__main__":
    main()
