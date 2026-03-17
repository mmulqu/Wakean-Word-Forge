"""Phonetic analysis and cross-lingual neighbor search.

Uses panphon for articulatory feature distance and epitran for
orthography-to-IPA conversion across languages.
"""

import re
from dataclasses import dataclass
from typing import Optional

try:
    import panphon
    import panphon.distance
    _ft = panphon.FeatureTable()
    _dst = panphon.distance.Distance()
    HAS_PANPHON = True
except ImportError:
    HAS_PANPHON = False

try:
    import epitran
    HAS_EPITRAN = True
except ImportError:
    HAS_EPITRAN = False


@dataclass
class PhoneticNeighbor:
    word: str
    lang_code: str
    ipa: str
    distance: float
    meaning: str = ""


def normalize_ipa(ipa: str) -> str:
    """Strip IPA delimiters and whitespace."""
    return re.sub(r"[/\[\]\s]", "", ipa).strip()


_epitran_cache: dict[str, "epitran.Epitran"] = {}


def word_to_ipa(word: str, lang_code: str = "eng-Latn") -> Optional[str]:
    """Convert a word to IPA using epitran.

    Common lang codes: eng-Latn, deu-Latn, fra-Latn, spa-Latn,
    ita-Latn, nld-Latn, tur-Latn, pol-Latn, hun-Latn

    Note: eng-Latn requires 'flite' (lex_lookup) to be installed.
    If unavailable, English IPA should come from the Wiktionary database instead.
    """
    if not HAS_EPITRAN:
        return None
    try:
        if lang_code not in _epitran_cache:
            _epitran_cache[lang_code] = epitran.Epitran(lang_code)
        result = _epitran_cache[lang_code].transliterate(word)
        return result if result else None
    except Exception:
        return None


def phonetic_distance(ipa_a: str, ipa_b: str) -> float:
    """Compute articulatory feature edit distance between two IPA strings.

    Uses panphon's weighted feature edit distance, which considers
    how similar two sounds are based on articulatory features
    (place, manner, voicing, etc.) rather than just character matching.

    Returns a float where lower = more similar.
    """
    if not HAS_PANPHON:
        # Fallback: normalized Levenshtein on IPA characters
        return _levenshtein_normalized(normalize_ipa(ipa_a), normalize_ipa(ipa_b))

    a = normalize_ipa(ipa_a)
    b = normalize_ipa(ipa_b)

    try:
        return _dst.weighted_feature_edit_distance(a, b)
    except Exception:
        return _levenshtein_normalized(a, b)


def find_phonetic_neighbors(
    target_ipa: str,
    candidates: list[dict],
    max_results: int = 20,
    max_distance: float = 10.0
) -> list[PhoneticNeighbor]:
    """Find words phonetically similar to the target IPA string.

    Args:
        target_ipa: IPA string of the target word.
        candidates: List of dicts with keys: word, lang_code, ipa, meaning (optional).
        max_results: Maximum neighbors to return.
        max_distance: Maximum phonetic distance threshold.

    Returns:
        List of PhoneticNeighbor sorted by distance ascending.
    """
    target_norm = normalize_ipa(target_ipa)
    results = []

    for entry in candidates:
        entry_ipa = entry.get("ipa", "")
        if not entry_ipa:
            continue

        dist = phonetic_distance(target_norm, normalize_ipa(entry_ipa))
        if dist <= max_distance:
            results.append(PhoneticNeighbor(
                word=entry["word"],
                lang_code=entry.get("lang_code", ""),
                ipa=entry_ipa,
                distance=dist,
                meaning=entry.get("meaning", "")
            ))

    results.sort(key=lambda n: n.distance)
    return results[:max_results]


def _levenshtein_normalized(a: str, b: str) -> float:
    """Normalized Levenshtein distance (0 = identical, 1 = completely different)."""
    if not a and not b:
        return 0.0
    if not a or not b:
        return 1.0

    m, n = len(a), len(b)
    dp = list(range(n + 1))

    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp

    return dp[n] / max(m, n)


def get_supported_languages() -> list[dict]:
    """Return list of languages supported by epitran for IPA conversion."""
    # Most commonly useful languages for Wakean portmanteau work
    return [
        {"code": "eng-Latn", "name": "English"},
        {"code": "deu-Latn", "name": "German"},
        {"code": "fra-Latn", "name": "French"},
        {"code": "ita-Latn", "name": "Italian"},
        {"code": "spa-Latn", "name": "Spanish"},
        {"code": "por-Latn", "name": "Portuguese"},
        {"code": "nld-Latn", "name": "Dutch"},
        {"code": "pol-Latn", "name": "Polish"},
        {"code": "hun-Latn", "name": "Hungarian"},
        {"code": "tur-Latn", "name": "Turkish"},
        {"code": "swe-Latn", "name": "Swedish"},
        {"code": "nor-Latn", "name": "Norwegian"},
        {"code": "dan-Latn", "name": "Danish"},
        {"code": "fin-Latn", "name": "Finnish"},
        {"code": "ces-Latn", "name": "Czech"},
        {"code": "ron-Latn", "name": "Romanian"},
        {"code": "cat-Latn", "name": "Catalan"},
        {"code": "gle-Latn", "name": "Irish"},
        {"code": "cym-Latn", "name": "Welsh"},
    ]
