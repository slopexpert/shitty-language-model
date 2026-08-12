"""Command-line entry point for the shitty-language-model project.

Dispatches to the individual command modules in the ``cli`` subpackage.
"""

import argparse
import importlib
import sys


def _run(module: str, argv: list[str]) -> None:
    """Import a command module and run its ``main()`` with the given args."""
    mod = importlib.import_module(f"shitty_language_model.cli.{module}")
    sys.argv = [f"slm {module}"] + argv
    mod.main()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="slm",
        description="Train and generate with a tokenizer, Markov models, and a tiny transformer.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    tokenize = sub.add_parser("tokenize", help="train the BPE tokenizer or tokenize text")
    tokenize_sub = tokenize.add_subparsers(dest="action", required=True)
    tokenize_sub.add_parser("train", help="train the tokenizer on a corpus")
    tokenize_sub.add_parser("run", help="tokenize text with a trained tokenizer")

    markov = sub.add_parser("markov", help="token-level Markov model")
    markov_sub = markov.add_subparsers(dest="action", required=True)
    markov_sub.add_parser("train", help="train the Markov model")
    markov_sub.add_parser("inspect", help="inspect Markov transitions")
    markov_sub.add_parser("generate", help="generate text from a Markov model")

    word_markov = sub.add_parser("word-markov", help="word-level Markov model")
    word_markov_sub = word_markov.add_subparsers(dest="action", required=True)
    word_markov_sub.add_parser("train", help="train the word Markov model")
    word_markov_sub.add_parser("generate", help="generate text from a word Markov model")

    transformer = sub.add_parser("transformer", help="tiny transformer model")
    transformer_sub = transformer.add_subparsers(dest="action", required=True)
    transformer_sub.add_parser("train", help="train the transformer on a corpus")
    transformer_sub.add_parser("run", help="generate text from a trained transformer")

    corpus = sub.add_parser("corpus", help="build corpora")
    corpus_sub = corpus.add_subparsers(dest="action", required=True)
    build = corpus_sub.add_parser("build", help="build a corpus")
    build.add_argument("name", choices=["hello", "hello-medium", "hello-large", "gutenberg"])

    args, rest = parser.parse_known_args()

    dispatch = {
        ("tokenize", "train"): ("train_tokenizer", rest),
        ("tokenize", "run"): ("tokenize_text", rest),
        ("markov", "train"): ("train_markov", rest),
        ("markov", "inspect"): ("inspect_markov", rest),
        ("markov", "generate"): ("generate_markov", rest),
        ("word-markov", "train"): ("train_word_markov", rest),
        ("word-markov", "generate"): ("generate_word_markov", rest),
        ("transformer", "train"): ("transformer", rest),
        ("transformer", "run"): ("transformer", rest),
    }

    if args.command == "corpus":
        builders = {
            "hello": "build_hello_corpus",
            "hello-medium": "build_hello_medium",
            "hello-large": "build_hello_large",
            "gutenberg": "build_gutenberg_corpus",
        }
        _run(builders[args.name], rest)
        return

    module, argv = dispatch[(args.command, args.action)]
    _run(module, argv)


if __name__ == "__main__":
    main()