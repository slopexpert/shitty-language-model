import json


def load_corpus(path: str) -> list[str]:
    """Load a corpus from a JSON file.

    The JSON may be a top-level array of strings, or an object containing a
    `samples` (or `corpus`) array of strings.
    """
    try:
        with open(path, "r") as file:
            data = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not load corpus {path!r}: {exc}") from exc

    samples = data
    if isinstance(data, dict):
        for key in ("samples", "corpus"):
            if key in data:
                samples = data[key]
                break
    if not isinstance(samples, list) or not all(
        isinstance(s, str) for s in samples
    ):
        raise ValueError(
            f"{path!r} does not contain a list of string samples "
            "(expected a JSON array, or an object with a \"samples\" key)"
        )
    return samples
