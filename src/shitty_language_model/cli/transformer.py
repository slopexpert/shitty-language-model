"""A tiny but real transformer LLM with self-attention, built on PyTorch.

Token-LEVEL model: it reuses the existing BPE `Tokenizer` (tokenizer.py) so the
unit of prediction is a subword token, just like a real LLM — not a word.

Each example is a chunk of token ids; the model is trained to predict every
token given only the tokens before it (teacher forcing), with causal
self-attention so it can never peek ahead.

Usage:
  Train (reuses checkpoint/tokenizer/prose_natural_large.json by default):
      uv run python transformer.py --corpus corpus/prose_natural_large.json
  Generate from a saved model:
      uv run python transformer.py --checkpoint checkpoints/transformer/....pt
"""

import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from shitty_language_model.corpus_loader import load_corpus
from shitty_language_model.tokenizer import Tokenizer, _merges_from_json, _merges_to_json

EOS_STR = "<|EOS|>"
WB_STR = "<|WB|>"
PAD_STR = "<|PAD|>"


# --------------------------------------------------------------------------
# Reuse the existing BPE tokenizer for a subword (token-level) vocab
# --------------------------------------------------------------------------


def load_tokenizer(path: str) -> tuple[Tokenizer, int, int]:
    """Load a tokenizer checkpoint; return (tokenizer, pad_id, eos_id)."""
    with open(path) as f:
        ck = json.load(f)
    tok = Tokenizer()
    tok.merges = _merges_from_json(ck["merges"])
    tok.vocab = {int(k): v for k, v in ck["vocab"].items()}
    tok.inverse_vocab = ck["inverse_vocab"]
    pad_id = len(tok.vocab)  # pad is not part of the BPE vocab
    return tok, pad_id, tok.eos_token


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention where each token attends only to prior tokens."""

    def __init__(self, d_model: int, n_heads: int, block_size: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, block_size: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, block_size, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        block_size: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, block_size, dropout) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        assert T <= self.block_size, f"seq {T} > block_size {self.block_size}"
        tok = self.tok_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=idx.device))
        x = tok + pos
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=0)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new: int, temperature: float = 1.0):
        self.eval()
        for _ in range(max_new):
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)
        self.train()
        return idx


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------


def build_token_seqs(texts: list[str], tok: Tokenizer, eos_id: int) -> list[list[int]]:
    """Wrap every text in <EOS> start/stop tokens. Token-level subword ids."""
    seqs = []
    for t in texts:
        seqs.append([eos_id] + tok.tokenize(t) + [eos_id])
    return seqs


def make_batches(seqs: list[list[int]], block_size: int, batch_size: int, pad_id: int):
    """Yield (input, target) batches of cropped+padded token windows."""
    np_rng = np.random.default_rng()
    n = len(seqs)
    while True:
        xs = torch.full((batch_size, block_size), pad_id, dtype=torch.long)
        ys = torch.full((batch_size, block_size), pad_id, dtype=torch.long)
        for b in range(batch_size):
            s = seqs[np_rng.integers(0, n)]
            s = s[: block_size + 1]
            L = min(block_size, len(s) - 1)
            xs[b, :L] = torch.tensor(s[:L])
            ys[b, :L] = torch.tensor(s[1 : L + 1])
        yield xs, ys


def untokenize(ids: list[int], vocab: dict[int, str]) -> str:
    s = "".join(vocab.get(i, "?") for i in ids)
    return s.replace(WB_STR, " ").replace(EOS_STR, " ").strip()


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description="Train / generate with a tiny token-level transformer")
    ap.add_argument("--corpus")
    ap.add_argument("--tokenizer", default="checkpoint/tokenizer/prose_natural_large.json")
    ap.add_argument("--checkpoint")
    ap.add_argument("--prompt", default="", help="text to continue from (empty = fresh <EOS> start)")
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--block-size", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=150, help="tokens to generate")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--eval-every", type=int, default=250)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    if args.checkpoint and not args.corpus:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        vocab = {int(k): v for k, v in ckpt["vocab"].items()}
        eos_id = ckpt["eos_id"]
        model = TinyTransformer(
            vocab_size=ckpt["vocab_size"],
            d_model=ckpt["d_model"],
            n_heads=ckpt["n_heads"],
            n_layers=ckpt["n_layers"],
            block_size=ckpt["block_size"],
        )
        model.load_state_dict(ckpt["state"])
        # reconstruct a tokenizer so we can tokenize the prompt
        if "merges" in ckpt:
            tok = Tokenizer()
            tok.merges = _merges_from_json(ckpt["merges"])
            tok.vocab = vocab
            tok.inverse_vocab = {v: int(k) for k, v in vocab.items()}
        else:
            tok, _, _ = load_tokenizer(args.tokenizer)
        if args.prompt:
            seed_ids = [eos_id] + tok.tokenize(args.prompt)
        else:
            seed_ids = [eos_id]
        idx = torch.tensor([seed_ids])
        out = model.generate(idx, args.n, args.temperature)[0].tolist()
        print("\n" + untokenize(out, vocab))
        return

    if not args.corpus:
        ap.error("need --corpus to train (or --checkpoint to generate)")

    texts = load_corpus(args.corpus)
    tok, pad_id, eos_id = load_tokenizer(args.tokenizer)
    seqs = build_token_seqs(texts, tok, eos_id)
    vocab_size = len(tok.vocab) + 1  # + pad
    print(f"corpus: {len(texts)} texts | tokens: {sum(len(s) for s in seqs):,} | vocab: {vocab_size:,}")

    model = TinyTransformer(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        block_size=args.block_size,
    )
    start_step = 0
    if args.checkpoint and os.path.exists(args.checkpoint):
        ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if ck["vocab_size"] != vocab_size:
            ap.error(f"checkpoint vocab {ck['vocab_size']} != corpus vocab {vocab_size}; retrain")
        model.load_state_dict(ck["state"])
        start_step = ck.get("step", 0)
        print(f"resuming from {args.checkpoint} at step {start_step}")
    print(f"params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    batches = iter(make_batches(seqs, args.block_size, args.batch_size, pad_id))
    sched = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda it: min(1.0, (start_step + it + 1) / 200)
    )

    t0 = time.time()
    for step in range(start_step + 1, args.steps + 1):
        xs, ys = next(batches)
        _, loss = model(xs, ys)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        sched.step()

        if step % args.eval_every == 0 or step == args.steps:
            dt = time.time() - t0
            idx = torch.tensor([[eos_id]])
            gen = model.generate(idx, 70, temperature=0.8)[0].tolist()
            print(f"step {step:5d} loss {loss.item():.4f} [{dt:.0f}s]  ||  {untokenize(gen, tok.vocab)}")

    out_dir = "checkpoint/transformer"
    try:
        os.makedirs(out_dir, exist_ok=True)
    except OSError:
        pass
    out_path = os.path.join(out_dir, os.path.basename(args.corpus).rsplit(".", 1)[0] + ".pt")
    torch.save(
        {
            "state": model.state_dict(),
            "vocab": {str(k): v for k, v in tok.vocab.items()},
            "merges": _merges_to_json(tok.merges),
            "vocab_size": vocab_size,
            "eos_id": eos_id,
            "d_model": args.d_model,
            "n_heads": args.n_heads,
            "n_layers": args.n_layers,
            "block_size": args.block_size,
            "step": args.steps,
        },
        out_path,
    )
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
