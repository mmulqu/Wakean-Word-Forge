"""Wakean Word Forge — MCP Server

A portmanteau forge for generating Finnegans Wake-style neologisms.
Exposes tools for morpheme lookup, cross-lingual phonetic neighbor search,
word fusion, and idiom recombination.

Run with: fastmcp run server.py
Or install: fastmcp install server.py
"""

import json
from typing import Optional

from fastmcp import FastMCP

from db import lookup_word, search_by_etymology, get_ipa_entries, get_morphemes, search_morphemes
from forge import MorphemeInput, fuse, twist_idiom, FusionCandidate
from phonetics import (
    word_to_ipa, phonetic_distance, find_phonetic_neighbors,
    get_supported_languages, normalize_ipa
)

mcp = FastMCP(
    "Wakean Word Forge",
    instructions=(
        "A portmanteau forge for generating Finnegans Wake-style neologisms. "
        "Provides morpheme lookup, cross-lingual phonetic matching, word fusion, "
        "and idiom recombination tools to circumvent LLM tokenization limitations "
        "when generating Joycean prose."
    ),
)


@mcp.tool()
def lookup_morphemes(
    word: str,
    lang_code: Optional[str] = None,
    convert_ipa: bool = True,
    ipa_lang: str = "eng-Latn",
) -> dict:
    """Look up a word's morphemes, etymology, IPA pronunciation, and semantic fields.

    Queries the Wiktionary-derived database for structured etymological data.
    If no IPA is found in the database and convert_ipa is True, uses epitran
    to generate IPA from the orthographic form.

    Args:
        word: The word to look up.
        lang_code: Optional ISO language code filter (e.g., "en", "de", "fr", "ga").
        convert_ipa: If True, generate IPA via epitran when DB lacks it.
        ipa_lang: Epitran language code for IPA conversion (default: "eng-Latn").

    Returns:
        Dictionary with word entries, each containing etymology, morphemes,
        IPA, and senses.
    """
    entries = lookup_word(word, lang_code)

    results = []
    for entry in entries:
        word_id = entry["id"]
        morpheme_list = get_morphemes(word_id)

        ipa = entry.get("ipa", "")
        if not ipa and convert_ipa:
            ipa = word_to_ipa(word, ipa_lang) or ""

        senses = []
        if entry.get("senses_json"):
            raw_senses = json.loads(entry["senses_json"])
            for s in raw_senses:
                glosses = s.get("glosses", [])
                if glosses:
                    senses.append(glosses[0])

        results.append({
            "word": entry["word"],
            "lang": entry.get("lang", ""),
            "lang_code": entry.get("lang_code", ""),
            "pos": entry.get("pos", ""),
            "ipa": ipa,
            "etymology": entry.get("etymology_text", ""),
            "morphemes": [
                {
                    "morpheme": m["morpheme"],
                    "meaning": m.get("meaning", ""),
                    "lang": m.get("lang", ""),
                    "type": m.get("morph_type", ""),
                }
                for m in morpheme_list
            ],
            "senses": senses,
        })

    if not results:
        # If not in DB, try epitran IPA conversion at minimum
        ipa = ""
        if convert_ipa:
            ipa = word_to_ipa(word, ipa_lang) or ""
        results.append({
            "word": word,
            "lang": "",
            "lang_code": lang_code or "",
            "pos": "",
            "ipa": ipa,
            "etymology": "",
            "morphemes": [],
            "senses": [],
            "note": "Word not found in database. IPA generated via epitran."
        })

    return {"word": word, "entries": results}


@mcp.tool()
def phonetic_neighbors(
    word: str,
    ipa: Optional[str] = None,
    source_lang: str = "eng-Latn",
    target_langs: Optional[list[str]] = None,
    max_results: int = 20,
    max_distance: float = 10.0,
) -> dict:
    """Find words from other languages that sound similar to the given word.

    This is the key tool for Joyce's technique: finding cross-lingual
    near-homophones that can inject a second semantic layer into a portmanteau.
    E.g., finding that German "Himmel" (heaven) sounds like English "him" + "immortal".

    Args:
        word: The word to find phonetic neighbors for.
        ipa: Optional IPA transcription. If not provided, will be generated via epitran.
        source_lang: Epitran language code for IPA conversion of the input word.
        target_langs: Optional list of language codes to search (e.g., ["de", "fr", "ga"]).
                      If None, searches all languages in the database.
        max_results: Maximum number of neighbors to return.
        max_distance: Maximum phonetic distance threshold.

    Returns:
        Dictionary with the target word's IPA and a list of phonetic neighbors
        with their words, languages, IPA, distance scores, and meanings.
    """
    # Get or generate IPA for the target word
    target_ipa = ipa
    if not target_ipa:
        target_ipa = word_to_ipa(word, source_lang) or ""
    if not target_ipa:
        # Try looking it up in the database
        entries = lookup_word(word)
        for e in entries:
            if e.get("ipa"):
                target_ipa = e["ipa"]
                break

    if not target_ipa:
        return {
            "word": word,
            "error": "Could not determine IPA for word. Try providing IPA directly.",
            "neighbors": []
        }

    # Get candidate IPA entries from database
    candidates = []
    if target_langs:
        for lang in target_langs:
            candidates.extend(get_ipa_entries(lang_code=lang, limit=50000))
    else:
        candidates.extend(get_ipa_entries(limit=50000))

    # Find neighbors
    neighbors = find_phonetic_neighbors(
        target_ipa, candidates, max_results=max_results, max_distance=max_distance
    )

    return {
        "word": word,
        "ipa": target_ipa,
        "neighbors": [
            {
                "word": n.word,
                "lang_code": n.lang_code,
                "ipa": n.ipa,
                "distance": round(n.distance, 3),
                "meaning": n.meaning,
            }
            for n in neighbors
        ]
    }


@mcp.tool()
def forge_portmanteau(
    components: list[dict],
    strategy: Optional[str] = None,
    max_results: int = 10,
) -> dict:
    """Fuse word components into portmanteau candidates using various strategies.

    This is the core creative tool. Provide word/morpheme ingredients and
    the forge generates candidate neologisms ranked by phonetic plausibility.

    Strategies:
    - "overlap": Find shared character/phoneme sequences and merge there.
      Joyce's primary technique. E.g., "Himmel" + "immortality" → "himmertality"
    - "substitute": Swap a morpheme in one word with a phonetically similar
      one from another language.
    - "nest": Embed one word inside another at a syllable boundary.
    - "interleave": Alternate syllables from two words.
    - None: Try all strategies and return the best candidates.

    Args:
        components: List of component dictionaries, each with:
            - text (str, required): The word or morpheme
            - lang (str): Language code (default: "en")
            - meaning (str): Semantic content
            - ipa (str): IPA transcription
            - role (str): "base", "overlay", "sound-donor"
        strategy: Fusion strategy or None for all.
        max_results: Maximum candidates to return.

    Returns:
        Dictionary with ranked fusion candidates, each including the fused text,
        confidence score, strategy used, component words, and explanation.
    """
    morpheme_inputs = []
    for comp in components:
        morpheme_inputs.append(MorphemeInput(
            text=comp["text"],
            lang=comp.get("lang", "en"),
            meaning=comp.get("meaning", ""),
            ipa=comp.get("ipa", ""),
            role=comp.get("role", ""),
        ))

    candidates = fuse(morpheme_inputs, strategy=strategy, max_results=max_results)

    return {
        "components": [c["text"] for c in components],
        "strategy": strategy or "all",
        "candidates": [
            {
                "text": c.text,
                "score": round(c.score, 3),
                "strategy": c.strategy,
                "components": c.components,
                "explanation": c.explanation,
            }
            for c in candidates
        ]
    }


@mcp.tool()
def idiom_twist(
    phrase: str,
    target_domain: str,
    domain_words: list[str],
) -> dict:
    """Recombine a fixed expression with words from a target semantic domain.

    Joyce frequently twisted common idioms by substituting words with
    phonetically similar ones from unexpected domains. This tool automates
    that process.

    Args:
        phrase: The source idiom or fixed expression (e.g., "give up the ghost").
        target_domain: The semantic domain to twist toward (e.g., "agriculture").
        domain_words: List of words from the target domain to use as substitution
                      candidates (e.g., ["goat", "graze", "harvest", "herd"]).

    Returns:
        Dictionary with the original phrase and ranked twisted variants.
    """
    candidates = twist_idiom(phrase, target_domain, domain_words)

    return {
        "original_phrase": phrase,
        "target_domain": target_domain,
        "domain_words": domain_words,
        "twists": [
            {
                "text": c.text,
                "score": round(c.score, 3),
                "explanation": c.explanation,
            }
            for c in candidates
        ]
    }


@mcp.tool()
def convert_to_ipa(
    word: str,
    lang_code: str = "eng-Latn",
) -> dict:
    """Convert a word to IPA transcription using epitran.

    Useful for getting the phonetic form of words in various languages
    to plan portmanteau fusions.

    Args:
        word: The word to convert.
        lang_code: Epitran language code (e.g., "eng-Latn", "deu-Latn", "fra-Latn").
                   Use list_languages to see supported codes.

    Returns:
        Dictionary with the word and its IPA transcription.
    """
    ipa = word_to_ipa(word, lang_code)
    return {
        "word": word,
        "lang_code": lang_code,
        "ipa": ipa or "",
        "success": ipa is not None,
    }


@mcp.tool()
def compare_phonetics(
    word_a: str,
    word_b: str,
    ipa_a: Optional[str] = None,
    ipa_b: Optional[str] = None,
    lang_a: str = "eng-Latn",
    lang_b: str = "eng-Latn",
) -> dict:
    """Compare the phonetic similarity of two words.

    Uses articulatory feature distance (panphon) when available, falling
    back to normalized Levenshtein on IPA strings.

    Args:
        word_a: First word.
        word_b: Second word.
        ipa_a: Optional IPA for first word.
        ipa_b: Optional IPA for second word.
        lang_a: Language code for first word (for epitran conversion).
        lang_b: Language code for second word.

    Returns:
        Dictionary with both words' IPA and their phonetic distance.
        Lower distance = more similar.
    """
    if not ipa_a:
        ipa_a = word_to_ipa(word_a, lang_a) or ""
    if not ipa_b:
        ipa_b = word_to_ipa(word_b, lang_b) or ""

    dist = phonetic_distance(ipa_a, ipa_b) if (ipa_a and ipa_b) else -1

    return {
        "word_a": word_a,
        "word_b": word_b,
        "ipa_a": ipa_a,
        "ipa_b": ipa_b,
        "distance": round(dist, 3) if dist >= 0 else None,
        "note": "Lower distance = more phonetically similar"
    }


@mcp.tool()
def search_etymology(
    query: str,
    limit: int = 20,
) -> dict:
    """Search the etymology database by keyword.

    Full-text search on etymology text, useful for finding words that share
    etymological roots across languages.

    Args:
        query: Search query (supports FTS5 syntax).
        limit: Maximum results to return.

    Returns:
        Dictionary with matching word entries and their etymologies.
    """
    entries = search_by_etymology(query, limit=limit)

    return {
        "query": query,
        "results": [
            {
                "word": e["word"],
                "lang": e.get("lang", ""),
                "lang_code": e.get("lang_code", ""),
                "pos": e.get("pos", ""),
                "ipa": e.get("ipa", ""),
                "etymology": e.get("etymology_text", ""),
            }
            for e in entries
        ]
    }


@mcp.tool()
def list_languages() -> dict:
    """List languages supported by epitran for IPA conversion.

    Returns language codes that can be used with convert_to_ipa,
    compare_phonetics, and phonetic_neighbors tools.
    """
    return {"languages": get_supported_languages()}


if __name__ == "__main__":
    mcp.run()
