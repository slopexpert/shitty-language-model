import argparse
import json
import sys

from shitty_language_model.tokenizer import Tokenizer, _merges_from_json


def main():
    parser = argparse.ArgumentParser(description="Tokenize text with a trained tokenizer")
    parser.add_argument(
        "-c",
        "--checkpoint",
        default="checkpoint/tokenizer/big_corpus.json",
        help="checkpoint file to load (default: %(default)s)",
    )
    parser.add_argument(
        "text",
        nargs="?",
        help="text to tokenize; reads from stdin if omitted",
    )
    args = parser.parse_args()

    text = args.text
    if text is None:
        text = sys.stdin.read()

    try:
        with open(args.checkpoint, "r") as file:
            checkpoint = json.load(file)
        tokenizer = Tokenizer()
        tokenizer.merges = _merges_from_json(checkpoint["merges"])
        tokenizer.vocab = {int(k): v for k, v in checkpoint["vocab"].items()}
        tokenizer.inverse_vocab = checkpoint["inverse_vocab"]
    except (OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
        parser.error(f"could not load checkpoint {args.checkpoint!r}: {exc}")

    tokens = tokenizer.tokenize(text)
    print(tokens)
    for token in tokens:
        print(tokenizer.vocab[token], end=" ")
    print()
    print(tokenizer.untokenize(tokens))


if __name__ == "__main__":
    main()
