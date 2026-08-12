import argparse
import json

from corpus_loader import load_corpus
from tokenizer import Tokenizer, _merges_to_json


def main():
    parser = argparse.ArgumentParser(description="Train the tokenizer on the corpus")
    parser.add_argument(
        "--corpus",
        default="corpus/hello_corpus.json",
        help="corpus JSON file to train on (default: %(default)s)",
    )
    parser.add_argument(
        "--passes",
        type=int,
        default=10000,
        help="number of BPE merge passes (default: %(default)s)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="checkpoint/tokenizer/big_corpus.json",
        help="checkpoint file to write (default: %(default)s)",
    )
    args = parser.parse_args()

    corpus = load_corpus(args.corpus)

    tokenizer = Tokenizer()
    tokenizer.train_on_corpus(corpus, args.passes)

    payload = {
        "merges": _merges_to_json(tokenizer.merges),
        "vocab": tokenizer.vocab,
        "inverse_vocab": tokenizer.inverse_vocab,
    }
    try:
        with open(args.output, "w") as file:
            json.dump(payload, file)
    except OSError as exc:
        parser.error(f"could not write checkpoint {args.output!r}: {exc}")

    print(
        f"trained {args.passes} passes: "
        f"{len(tokenizer.vocab)} tokens, {len(tokenizer.merges)} merges "
        f"-> {args.output}"
    )


if __name__ == "__main__":
    main()
