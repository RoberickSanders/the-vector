"""Curated remote-friendly company registry + match logic."""
import json
from pathlib import Path
from typing import Optional


def normalize_company(name: str) -> str:
    if not isinstance(name, str):
        return ""
    name = name.lower().strip()
    for suffix in [", inc.", " inc.", ", llc", " llc", ", inc", " inc",
                   " corporation", " corp.", " corp", ", ltd", " ltd",
                   " limited", " co.", " company", ".com"]:
        if name.endswith(suffix):
            name = name[: -len(suffix)].strip()
    return name


def load_remote_friendly_companies(path: Path | str) -> dict:
    """Returns {normalized_name: metadata} for fast lookup."""
    path = Path(path)
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    by_name = {}
    for c in raw:
        title = c.get("title", "").strip().lower()
        slug = c.get("slug", "").strip().lower()
        if title:
            by_name[title] = c
        if slug and slug != title:
            by_name[slug] = c
    return by_name


def find_match(scraped: str, registry: dict) -> Optional[dict]:
    """Match a scraped company name against the curated registry.

    Strategy:
    1. Exact match on normalized name.
    2. Prefix match: scraped name starts with a curated name followed by a
       word boundary (space or punctuation). E.g., "amazon web services"
       matches curated "amazon".
    """
    if not scraped:
        return None
    norm = normalize_company(scraped)
    if norm in registry:
        return registry[norm]
    # Prefix match — only for curated names ≥4 chars to avoid false positives
    for curated_name, meta in registry.items():
        if len(curated_name) < 4:
            continue
        if norm.startswith(curated_name + " ") or \
           norm.startswith(curated_name + "."):
            return meta
    return None
