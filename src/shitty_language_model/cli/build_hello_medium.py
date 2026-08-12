import json
import random

from shitty_language_model.cli import build_hello_corpus as base

random.seed(2024)


def build():
    samples = []
    # one near-verbatim pass over every item so nothing is missing
    for item in base.ITEMS:
        samples.append(base.render(item))
    # hello gets 1 + 6 = 7 per item; double that to 1 + 13 = 14 per item
    variants = 13
    for item in base.ITEMS:
        for _ in range(variants):
            samples.append(base.render(item))
    return samples


def main():
    samples = build()
    payload = {
        "name": "hello-world-medium",
        "description": (
            "Hello_medium: every hello-world/argv/arithmetic item from hello, "
            "with twice as many style variants per item"
        ),
        "samples": samples,
    }
    out_path = "corpus/hello_medium.json"
    try:
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
    except OSError as exc:
        raise SystemExit(f"could not write {out_path!r}: {exc}") from exc
    print(f"wrote {len(samples)} samples -> {out_path}")


if __name__ == "__main__":
    main()
