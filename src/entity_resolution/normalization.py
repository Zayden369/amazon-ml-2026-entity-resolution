"""Deterministic, country-agnostic text normalization."""

import re
import unicodedata


def normalize_text(value: object) -> str:
    """Normalize names or addresses without using external reference data."""
    text = "" if value is None else str(value)
    text = "".join(
        char
        for char in unicodedata.normalize("NFKD", text.lower())
        if not unicodedata.combining(char)
    )
    text = text.replace("&", " and ")
    return " ".join(re.sub(r"[^\w\s]", " ", text).split())

