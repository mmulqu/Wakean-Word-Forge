"""Database layer for the Wakean Word Forge.

Manages a SQLite database seeded from kaikki.org Wiktionary extracts,
indexed for fast morpheme lookup, etymology search, and phonetic neighbor queries.
"""

import sqlite3
import json
import os
from pathlib import Path
from typing import Optional

DB_PATH = Path(os.environ.get("FORGE_DB_PATH", Path(__file__).parent / "forge.db"))


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    """Create tables and indexes if they don't exist."""
    conn = get_connection()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS words (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            word TEXT NOT NULL,
            lang TEXT NOT NULL,
            lang_code TEXT,
            pos TEXT,
            ipa TEXT,
            etymology_text TEXT,
            senses_json TEXT,
            sounds_json TEXT,
            raw_json TEXT
        );

        CREATE TABLE IF NOT EXISTS morphemes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            word_id INTEGER NOT NULL REFERENCES words(id),
            morpheme TEXT NOT NULL,
            meaning TEXT,
            lang TEXT,
            lang_code TEXT,
            morph_type TEXT,  -- prefix, root, suffix, infix
            position INTEGER,
            FOREIGN KEY (word_id) REFERENCES words(id)
        );

        CREATE TABLE IF NOT EXISTS ipa_index (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            word_id INTEGER NOT NULL REFERENCES words(id),
            word TEXT NOT NULL,
            lang_code TEXT,
            ipa TEXT NOT NULL,
            ipa_normalized TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_words_word ON words(word);
        CREATE INDEX IF NOT EXISTS idx_words_lang ON words(lang_code);
        CREATE INDEX IF NOT EXISTS idx_words_word_lang ON words(word, lang_code);
        CREATE INDEX IF NOT EXISTS idx_morphemes_morpheme ON morphemes(morpheme);
        CREATE INDEX IF NOT EXISTS idx_morphemes_word_id ON morphemes(word_id);
        CREATE INDEX IF NOT EXISTS idx_ipa_word ON ipa_index(word);
        CREATE INDEX IF NOT EXISTS idx_ipa_ipa ON ipa_index(ipa);
        CREATE INDEX IF NOT EXISTS idx_ipa_lang ON ipa_index(lang_code);

        -- Tracks ingestion progress so we can stop and resume
        CREATE TABLE IF NOT EXISTS ingest_progress (
            source_file TEXT PRIMARY KEY,  -- the JSONL filename
            lang_code TEXT NOT NULL,
            lines_read INTEGER NOT NULL DEFAULT 0,     -- how many lines we've read through
            entries_inserted INTEGER NOT NULL DEFAULT 0, -- how many actually inserted
            entries_skipped INTEGER NOT NULL DEFAULT 0,
            completed INTEGER NOT NULL DEFAULT 0,  -- 1 = finished the whole file
            last_updated TEXT NOT NULL DEFAULT (datetime('now'))
        );
    """)

    # FTS5 virtual tables for full-text search
    conn.executescript("""
        CREATE VIRTUAL TABLE IF NOT EXISTS words_fts USING fts5(
            word, lang, etymology_text,
            content='words',
            content_rowid='id'
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS morphemes_fts USING fts5(
            morpheme, meaning, lang,
            content='morphemes',
            content_rowid='id'
        );
    """)
    conn.commit()
    conn.close()


def insert_word(conn: sqlite3.Connection, entry: dict) -> Optional[int]:
    """Insert a word entry from kaikki.org JSONL format."""
    word = entry.get("word")
    if not word:
        return None

    lang = entry.get("lang", "")
    lang_code = entry.get("lang_code", "")
    pos = entry.get("pos", "")

    # Extract IPA from sounds
    sounds = entry.get("sounds", [])
    ipa = None
    for s in sounds:
        if "ipa" in s:
            ipa = s["ipa"]
            break

    etymology_text = entry.get("etymology_text", "")

    senses = entry.get("senses", [])
    senses_json = json.dumps(senses) if senses else None

    sounds_json = json.dumps(sounds) if sounds else None

    cursor = conn.execute(
        """INSERT INTO words (word, lang, lang_code, pos, ipa, etymology_text,
           senses_json, sounds_json, raw_json)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (word, lang, lang_code, pos, ipa, etymology_text,
         senses_json, sounds_json, json.dumps(entry))
    )
    word_id = cursor.lastrowid

    # Index IPA for phonetic search
    if ipa:
        ipa_normalized = ipa.replace("/", "").replace("[", "").replace("]", "").strip()
        conn.execute(
            "INSERT INTO ipa_index (word_id, word, lang_code, ipa, ipa_normalized) VALUES (?, ?, ?, ?, ?)",
            (word_id, word, lang_code, ipa, ipa_normalized)
        )

    # Also index any additional IPA variants from sounds
    for s in sounds:
        s_ipa = s.get("ipa", "")
        if s_ipa and s_ipa != ipa:
            ipa_norm = s_ipa.replace("/", "").replace("[", "").replace("]", "").strip()
            conn.execute(
                "INSERT INTO ipa_index (word_id, word, lang_code, ipa, ipa_normalized) VALUES (?, ?, ?, ?, ?)",
                (word_id, word, lang_code, s_ipa, ipa_norm)
            )

    return word_id


def lookup_word(word: str, lang_code: Optional[str] = None) -> list[dict]:
    """Look up a word, optionally filtered by language."""
    conn = get_connection()
    if lang_code:
        rows = conn.execute(
            "SELECT * FROM words WHERE word = ? AND lang_code = ?",
            (word, lang_code)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM words WHERE word = ?", (word,)
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def search_by_etymology(query: str, limit: int = 20) -> list[dict]:
    """Full-text search on etymology text.

    Wraps query terms in double quotes to handle hyphens and special chars
    that FTS5 would otherwise interpret as operators.
    """
    conn = get_connection()
    # Quote the query so hyphens etc. are treated as literal characters
    safe_query = '"' + query.replace('"', '""') + '"'
    rows = conn.execute(
        """SELECT w.* FROM words w
           JOIN words_fts f ON w.id = f.rowid
           WHERE words_fts MATCH ?
           LIMIT ?""",
        (safe_query, limit)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_ipa_entries(lang_code: Optional[str] = None, limit: int = 10000) -> list[dict]:
    """Get IPA-indexed entries for phonetic neighbor search."""
    conn = get_connection()
    if lang_code:
        rows = conn.execute(
            "SELECT * FROM ipa_index WHERE lang_code = ? LIMIT ?",
            (lang_code, limit)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM ipa_index LIMIT ?", (limit,)
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_morphemes(word_id: int) -> list[dict]:
    """Get morphemes for a word entry."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM morphemes WHERE word_id = ? ORDER BY position",
        (word_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def search_morphemes(morpheme: str, limit: int = 50) -> list[dict]:
    """Search for words containing a specific morpheme."""
    conn = get_connection()
    rows = conn.execute(
        """SELECT m.*, w.word, w.lang, w.lang_code, w.ipa
           FROM morphemes m
           JOIN words w ON m.word_id = w.id
           WHERE m.morpheme = ?
           LIMIT ?""",
        (morpheme, limit)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


if __name__ == "__main__":
    init_db()
    print(f"Database initialized at {DB_PATH}")
