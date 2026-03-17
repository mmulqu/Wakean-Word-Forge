"""The Portmanteau Forge — core fusion engine for Wakean neologism generation.

This is the novel component: given morpheme ingredients and a fusion strategy,
it generates candidate portmanteau words ranked by phonetic smoothness.
"""

import itertools
import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MorphemeInput:
    """A morpheme or word component to be fused."""
    text: str
    lang: str = "en"
    meaning: str = ""
    ipa: str = ""
    role: str = ""  # e.g. "base", "overlay", "sound-donor"


@dataclass
class FusionCandidate:
    """A generated portmanteau candidate with metadata."""
    text: str
    score: float  # 0-1, higher = better
    strategy: str
    components: list[str] = field(default_factory=list)
    explanation: str = ""


def find_overlaps(a: str, b: str, min_len: int = 2) -> list[tuple[int, int, int]]:
    """Find character-level overlaps between the end of `a` and start of `b`.

    Returns list of (a_start, b_end, length) tuples for each overlap found.
    """
    overlaps = []
    a_lower = a.lower()
    b_lower = b.lower()
    max_overlap = min(len(a), len(b))

    for length in range(min_len, max_overlap + 1):
        if a_lower[-length:] == b_lower[:length]:
            overlaps.append((len(a) - length, length, length))

    return overlaps


def find_internal_overlaps(a: str, b: str, min_len: int = 2) -> list[tuple[int, int]]:
    """Find positions where `b` overlaps with a substring inside `a`.

    Returns (position_in_a, overlap_length) pairs.
    """
    overlaps = []
    a_lower = a.lower()
    b_lower = b.lower()

    for i in range(len(a)):
        for length in range(min_len, min(len(b), len(a) - i) + 1):
            if a_lower[i:i + length] == b_lower[:length]:
                overlaps.append((i, length))

    return overlaps


def fuse_overlap(components: list[MorphemeInput]) -> list[FusionCandidate]:
    """Fuse words by finding shared phoneme/character sequences and merging there.

    This is Joyce's primary technique: "Himmel" + "immortality" → "himmertality"
    where "him" overlaps.
    """
    if len(components) < 2:
        return []

    candidates = []

    for i in range(len(components)):
        for j in range(len(components)):
            if i == j:
                continue
            a = components[i]
            b = components[j]

            # End-of-a overlaps start-of-b (classic portmanteau)
            overlaps = find_overlaps(a.text, b.text)
            for _, _, length in overlaps:
                fused = a.text[:len(a.text)] + b.text[length:]
                score = length / max(len(a.text), len(b.text))
                candidates.append(FusionCandidate(
                    text=fused,
                    score=min(score * 1.5, 1.0),
                    strategy="overlap",
                    components=[a.text, b.text],
                    explanation=f"'{a.text}' tail overlaps '{b.text}' head by {length} chars"
                ))

            # Cross-word fusion: find shared substrings between a and b,
            # then graft a's prefix onto b at the shared point.
            # This is how "Himmel" + "immortality" → "himmertality":
            # "im" is shared, so Himmel's "H" prefix grafts onto "immortality" at "im".
            a_lower = a.text.lower()
            b_lower = b.text.lower()
            for sub_len in range(2, min(len(a.text), len(b.text)) + 1):
                for a_pos in range(len(a.text) - sub_len + 1):
                    a_sub = a_lower[a_pos:a_pos + sub_len]
                    for b_pos in range(len(b.text) - sub_len + 1):
                        if b_lower[b_pos:b_pos + sub_len] == a_sub:
                            # Graft: a's prefix up to the shared region + b from shared region onward
                            fused = a.text[:a_pos] + b.text[b_pos:]
                            if (len(fused) > 3
                                    and fused.lower() != a.text.lower()
                                    and fused.lower() != b.text.lower()):
                                # Score: reward longer shared substrings and proportional coverage
                                coverage = sub_len / max(len(a.text), len(b.text))
                                length_ratio = len(fused) / ((len(a.text) + len(b.text)) / 2)
                                score = min(coverage * 1.5 + (0.3 if 0.6 < length_ratio < 1.4 else 0), 1.0)
                                candidates.append(FusionCandidate(
                                    text=fused,
                                    score=score,
                                    strategy="cross_overlap",
                                    components=[a.text, b.text],
                                    explanation=(
                                        f"'{a.text}' and '{b.text}' share '{a_sub}': "
                                        f"grafted a[:{a_pos}] + b[{b_pos}:]"
                                    )
                                ))

    return candidates


def fuse_substitute(components: list[MorphemeInput]) -> list[FusionCandidate]:
    """Substitute a morpheme in one word with a phonetically similar morpheme from another.

    Joyce's technique of swapping a syllable for a foreign near-homophone.
    E.g., replacing "mor" in "mortality" with "Himmer" from "Himmel".
    """
    if len(components) < 2:
        return []

    candidates = []

    for base in components:
        for donor in components:
            if base.text == donor.text:
                continue

            # Find shared substrings that could be substitution points
            base_lower = base.text.lower()
            donor_lower = donor.text.lower()

            for sub_len in range(2, min(len(base.text), len(donor.text)) + 1):
                for b_start in range(len(base.text) - sub_len + 1):
                    base_sub = base_lower[b_start:b_start + sub_len]
                    for d_start in range(len(donor.text) - sub_len + 1):
                        donor_sub = donor_lower[d_start:d_start + sub_len]
                        if base_sub == donor_sub:
                            # Substitute the matched region in base with donor's
                            # extended context
                            fused = base.text[:b_start] + donor.text + base.text[b_start + sub_len:]
                            if len(fused) < len(base.text) * 3:  # sanity limit
                                score = sub_len / max(len(base.text), len(donor.text))
                                candidates.append(FusionCandidate(
                                    text=fused,
                                    score=min(score, 1.0),
                                    strategy="substitute",
                                    components=[base.text, donor.text],
                                    explanation=f"Shared '{base_sub}': '{donor.text}' substituted into '{base.text}' at position {b_start}"
                                ))

    return candidates


def fuse_nest(components: list[MorphemeInput]) -> list[FusionCandidate]:
    """Embed one word inside another at a phonetically plausible point.

    Looks for positions in the base word where the nested word could be
    inserted with minimal disruption — ideally at syllable boundaries.
    """
    if len(components) < 2:
        return []

    candidates = []
    vowels = set("aeiouAEIOU")

    for base in components:
        for nested in components:
            if base.text == nested.text:
                continue

            # Find syllable boundary-ish positions (vowel-consonant transitions)
            for i in range(1, len(base.text)):
                if base.text[i - 1] in vowels and base.text[i] not in vowels:
                    fused = base.text[:i] + nested.text + base.text[i:]
                    score = 0.3  # nesting is lower confidence
                    candidates.append(FusionCandidate(
                        text=fused,
                        score=score,
                        strategy="nest",
                        components=[base.text, nested.text],
                        explanation=f"'{nested.text}' nested into '{base.text}' at syllable boundary position {i}"
                    ))

    return candidates


def fuse_interleave(components: list[MorphemeInput]) -> list[FusionCandidate]:
    """Interleave syllables or morphemes from two words.

    A rougher technique — alternates chunks from each word.
    """
    if len(components) < 2:
        return []

    candidates = []

    def crude_syllables(word: str) -> list[str]:
        """Split a word into crude syllable-like chunks."""
        vowels = set("aeiouAEIOU")
        syllables = []
        current = ""
        for i, ch in enumerate(word):
            current += ch
            # Split after a vowel followed by a consonant (if not at end)
            if (ch in vowels and i + 1 < len(word) and word[i + 1] not in vowels
                    and len(current) >= 2):
                syllables.append(current)
                current = ""
        if current:
            syllables.append(current)
        return syllables if syllables else [word]

    for a, b in itertools.permutations(components, 2):
        syls_a = crude_syllables(a.text)
        syls_b = crude_syllables(b.text)

        # Alternate syllables
        interleaved = []
        for sa, sb in itertools.zip_longest(syls_a, syls_b, fillvalue=""):
            interleaved.append(sa)
            if sb:
                interleaved.append(sb)
        fused = "".join(interleaved)

        if fused.lower() != a.text.lower() and fused.lower() != b.text.lower():
            candidates.append(FusionCandidate(
                text=fused,
                score=0.2,  # interleaving is lowest confidence
                strategy="interleave",
                components=[a.text, b.text],
                explanation=f"Syllables interleaved: {syls_a} + {syls_b}"
            ))

    return candidates


def fuse(components: list[MorphemeInput],
         strategy: Optional[str] = None,
         max_results: int = 10) -> list[FusionCandidate]:
    """Main fusion entry point. Generates portmanteau candidates.

    Args:
        components: List of morpheme inputs to fuse.
        strategy: One of "overlap", "substitute", "nest", "interleave", or None for all.
        max_results: Maximum number of candidates to return.

    Returns:
        List of FusionCandidate objects, sorted by score descending.
    """
    strategies = {
        "overlap": fuse_overlap,
        "substitute": fuse_substitute,
        "nest": fuse_nest,
        "interleave": fuse_interleave,
    }

    candidates = []

    if strategy and strategy in strategies:
        candidates = strategies[strategy](components)
    else:
        for name, func in strategies.items():
            candidates.extend(func(components))

    # Deduplicate by lowercased text
    seen = set()
    unique = []
    for c in candidates:
        key = c.text.lower()
        if key not in seen and len(c.text) > 2:
            seen.add(key)
            unique.append(c)

    # Sort by score descending
    unique.sort(key=lambda c: c.score, reverse=True)
    return unique[:max_results]


def twist_idiom(phrase: str, target_domain: str,
                domain_words: Optional[list[str]] = None) -> list[FusionCandidate]:
    """Recombine a fixed expression with a target semantic domain.

    E.g., "give up the ghost" + agriculture → "gave up his goat"

    This performs simple phonetic substitutions of words in the phrase
    with domain-related words that sound similar.
    """
    if not domain_words:
        return []

    candidates = []
    phrase_words = phrase.split()

    for i, pw in enumerate(phrase_words):
        for dw in domain_words:
            # Check if the domain word sounds vaguely similar
            # (shares first letter or has significant character overlap)
            overlap = len(set(pw.lower()) & set(dw.lower()))
            similarity = overlap / max(len(set(pw.lower())), len(set(dw.lower())))

            if similarity >= 0.3 or pw[0].lower() == dw[0].lower():
                new_phrase = phrase_words[:i] + [dw] + phrase_words[i + 1:]
                result = " ".join(new_phrase)
                if result.lower() != phrase.lower():
                    candidates.append(FusionCandidate(
                        text=result,
                        score=similarity,
                        strategy="idiom_twist",
                        components=[phrase, dw],
                        explanation=f"'{pw}' replaced with '{dw}' from domain '{target_domain}'"
                    ))

    # Deduplicate
    seen = set()
    unique = []
    for c in candidates:
        if c.text not in seen:
            seen.add(c.text)
            unique.append(c)

    unique.sort(key=lambda c: c.score, reverse=True)
    return unique[:10]
