import argparse
import json

from corpus_loader import load_corpus
from markov import Markov, _next_tokens_to_json
from tokenizer import Tokenizer, _merges_from_json


def main():
    parser = argparse.ArgumentParser(description="Train the Markov model on the corpus")
    parser.add_argument(
        "--corpus",
        default="corpus/hello_corpus.json",
        help="corpus JSON file to train on (default: %(default)s)",
    )
    parser.add_argument(
        "-t",
        "--tokenizer",
        default="checkpoint/tokenizer/big_corpus.json",
        help="tokenizer checkpoint to load (default: %(default)s)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="checkpoint/markov/big_corpus.json",
        help="checkpoint file to write (default: %(default)s)",
    )
    args = parser.parse_args()

    corpus = load_corpus(args.corpus)

    try:
        with open(args.tokenizer, "r") as file:
            ck = json.load(file)
        tokenizer = Tokenizer()
        tokenizer.merges = _merges_from_json(ck["merges"])
        tokenizer.vocab = {int(k): v for k, v in ck["vocab"].items()}
        tokenizer.inverse_vocab = ck["inverse_vocab"]
    except (OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
        parser.error(f"could not load tokenizer {args.tokenizer!r}: {exc}")

    markov = Markov(tokenizer)
    markov.train_on_corpus(corpus)

    payload = {"next_tokens": _next_tokens_to_json(markov.next_tokens)}
    try:
        with open(args.output, "w") as file:
            json.dump(payload, file, indent=2)
    except (OSError, TypeError) as exc:
        parser.error(f"could not write checkpoint {args.output!r}: {exc}")

    total = sum(
        sum(c for c in outgoing.values())
        for outgoing in markov.next_tokens.values()
    )
    print(
        f"trained Markov on {len(corpus)} texts: "
        f"{len(markov.next_tokens)} states, {total} transitions "
        f"-> {args.output}"
    )


if __name__ == "__main__":
    main()
