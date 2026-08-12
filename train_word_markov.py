import argparse
import json

from corpus_loader import load_corpus
from markov_word import WordMarkov, _next_tokens_to_json


def main():
    parser = argparse.ArgumentParser(
        description="Train a higher-order word-level Markov model on a corpus"
    )
    parser.add_argument(
        "--corpus",
        default="corpus/prose_natural_large.json",
        help="corpus JSON file to train on (default: %(default)s)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="checkpoint/word_markov/default.json",
        help="checkpoint file to write (default: %(default)s)",
    )
    parser.add_argument(
        "--order",
        type=int,
        default=2,
        help="markov order: condition on the last N words (default: %(default)s)",
    )
    args = parser.parse_args()

    corpus = load_corpus(args.corpus)
    markov = WordMarkov(order=args.order)
    markov.train_on_corpus(corpus)

    payload = {
        "order": markov.order,
        "next_tokens": _next_tokens_to_json(markov.next_tokens),
    }
    try:
        with open(args.output, "w") as file:
            json.dump(payload, file, indent=2)
    except (OSError, TypeError) as exc:
        parser.error(f"could not write checkpoint {args.output!r}: {exc}")

    print(
        f"trained word Markov (order {markov.order}) on {len(corpus)} texts: "
        f"{len(markov.next_tokens)} contexts, {markov.num_transitions} transitions "
        f"-> {args.output}"
    )


if __name__ == "__main__":
    main()
