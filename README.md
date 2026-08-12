# shitty-language-model

A tiny, from-scratch language model toolkit. It implements a BPE tokenizer, two
Markov models (token-level and word-level), and a small causal transformer — all
trained on small JSON corpora. The whole thing is a learning project, hence the
name.

## Layout

```
src/shitty_language_model/
├── __init__.py          # `slm` CLI dispatcher
├── __main__.py          # python -m shitty_language_model
├── tokenizer.py         # BPE tokenizer
├── markov.py            # token-level Markov model
├── markov_word.py       # higher-order word-level Markov model
├── corpus_loader.py     # load JSON corpora
└── cli/                 # one module per command
    ├── train_tokenizer.py
    ├── tokenize_text.py
    ├── train_markov.py
    ├── inspect_markov.py
    ├── generate_markov.py
    ├── train_word_markov.py
    ├── generate_word_markov.py
    ├── transformer.py
    └── build_*.py        # corpus builders
```

Data lives in `corpus/` (JSON corpora) and `checkpoint/` (trained models).
Both are gitignored.

## Setup

Requires Python >= 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Usage

Everything is driven by the `slm` CLI. Run it from the project root (data paths
are relative).

```bash
uv run slm --help
```

### Tokenizer

```bash
# Train a BPE tokenizer on a corpus
uv run slm tokenize train --corpus corpus/hello_corpus.json --passes 10000 -o checkpoint/tokenizer/hello_corpus.json

# Tokenize text with a trained tokenizer
uv run slm tokenize run -c checkpoint/tokenizer/hello_corpus.json "hello world"
```

### Token-level Markov

```bash
uv run slm markov train --corpus corpus/hello_corpus.json -t checkpoint/tokenizer/hello_corpus.json -o checkpoint/markov/hello_corpus.json
uv run slm markov inspect -t checkpoint/tokenizer/hello_corpus.json -c checkpoint/markov/hello_corpus.json "return"
uv run slm markov generate -t checkpoint/tokenizer/hello_corpus.json -m checkpoint/markov/hello_corpus.json -n 120 -s "int main"
```

### Word-level Markov

```bash
uv run slm word-markov train --corpus corpus/prose_natural_large.json --order 2 -o checkpoint/word_markov/prose_natural_large_o2.json
uv run slm word-markov generate -c checkpoint/word_markov/prose_natural_large_o2.json -n 100 -s "Once upon a time"
```

### Transformer

```bash
# Train
uv run slm transformer train --corpus corpus/gutenberg_large.json --tokenizer checkpoint/tokenizer/gutenberg_large.json

# Generate from a saved model
uv run slm transformer run --checkpoint checkpoint/transformer/gutenberg_large.pt --prompt "The old man" --n 200 --temperature 0.9

# Resume training from a checkpoint
uv run slm transformer train --corpus corpus/gutenberg_large.json --checkpoint checkpoint/transformer/gutenberg_large.pt --steps 4000
```

Training saves to `checkpoint/transformer/<corpus-name>.pt`. The tokenizer vocab
must match the corpus — the model errors if the checkpoint vocab size differs.

### Building corpora

```bash
uv run slm corpus build hello          # small hello-world/arithmetic C corpus
uv run slm corpus build hello-medium   # more style variants per item
uv run slm corpus build hello-large    # compositionally-generated C programs
uv run slm corpus build gutenberg      # public-domain prose from Project Gutenberg
```

## Notes

- The transformer is a tiny token-level model (subword prediction via BPE), not
  a word-level model.
- Some older checkpoints may be incompatible with the current code (e.g. they
  reference classes that no longer exist); retrain if a checkpoint fails to load.