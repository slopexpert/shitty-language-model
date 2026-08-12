"""Build a large, copyright-safe prose corpus from Project Gutenberg.

Downloads a set of public-domain novels' plain text, strips the Project
Gutenberg boilerplate header/footer, and splits each book into clean
multi-sentence passages (kept whole). Writes corpus/gutenberg_large.json in
the standard format ({"samples": [str, ...]}).
"""

import json
import os
import re
import urllib.request

# public-domain book ids -> titles
BOOKS = {
    1342: "pride-and-prejudice",
    98: "a-tale-of-two-cities",
    2701: "moby-dick",
    76: "huckleberry-finn",
    11: "alice-in-wonderland",
    1661: "adventures-of-sherlock-holmes",
    35: "the-time-machine",
    74: "tom-sawyer",
    174: "picture-of-dorian-gray",
    84: "frankenstein",
    46: "a-christmas-carol",
    2591: "grimms-fairy-tales",
}

START_RE = re.compile(r"\*\*\* START OF (THE|THIS) PROJECT GUTENBERG.*?\*\*\*", re.S)
END_RE = re.compile(r"\*\*\* END OF (THE|THIS) PROJECT GUTENBERG.*?\*\*\*", re.S)


def fetch(id_: int) -> str:
    url = f"https://www.gutenberg.org/cache/epub/{id_}/pg{id_}.txt"
    with urllib.request.urlopen(url, timeout=60) as resp:
        return resp.read().decode("utf-8", errors="replace")


def clean(text: str) -> str:
    text = START_RE.split(text, maxsplit=1)[-1] if START_RE.search(text) else text
    text = END_RE.split(text, maxsplit=1)[0] if END_RE.search(text) else text
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # collapse any remaining (e.g. notes) horizontal rules / excessive markup
    text = re.sub(r"[_*]{3,}", "", text)
    return text


def split_passages(text: str, min_words: int = 40) -> list[str]:
    """Divide a cleaned book into whole paragraphs of at least min_words."""
    out = []
    for para in re.split(r"\n\s*\n", text):
        s = " ".join(para.split()).strip()
        if len(s.split()) >= min_words:
            out.append(s)
    return out


def main():
    samples: list[str] = []
    try:
        os.makedirs("/tmp/gutenberg", exist_ok=True)
    except OSError:
        pass
    for id_, name in BOOKS.items():
        raw = fetch(id_)
        body = clean(raw)
        passages = split_passages(body)
        if not passages:
            print(f"  !! {name}: no passages parsed ({len(raw)} chars)")
            continue
        try:
            with open(f"/tmp/gutenberg/{name}.txt", "w") as f:
                f.write(body)
        except OSError as exc:
            print(f"  !! {name}: could not write raw ({exc})")
        samples.extend(passages)
        print(f"{name:32s} {len(passages):5d} passages")

    out = "corpus/gutenberg_large.json"
    try:
        with open(out, "w") as f:
            json.dump(
                {
                    "name": "gutenberg-large",
                    "description": "public-domain prose (Project Gutenberg)",
                    "samples": samples,
                },
                f,
            )
    except OSError as exc:
        print(f"!! could not write {out}: {exc}")
        return
    total_words = sum(len(s.split()) for s in samples)
    print(f"\n-> {out}: {len(samples)} passages, {total_words:,} words")


if __name__ == "__main__":
    main()
