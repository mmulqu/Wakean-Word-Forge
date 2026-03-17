"""Data ingestion pipeline for the Wakean Word Forge.

Downloads and processes kaikki.org Wiktionary extracts into the SQLite database.
Supports stop/resume — tracks how many lines of each JSONL file have been
processed so you can Ctrl+C at any time and pick up where you left off.

Usage:
    # Small proof-of-concept: 500 English entries
    python ingest.py --lang en --limit 500

    # Resume an interrupted ingestion (just run the same command again)
    python ingest.py --lang en

    # Ingest FW priority languages with a per-language limit
    python ingest.py --fw-priority --limit 10000

    # Check what's been ingested so far
    python ingest.py --status

    # Reset progress for a language (re-ingest from scratch)
    python ingest.py --reset en
"""

import argparse
import gzip
import json
import os
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

from tqdm import tqdm

from db import get_connection, init_db, insert_word

# kaikki.org base URL for pre-extracted Wiktionary data
KAIKKI_BASE = "https://kaikki.org/dictionary"

# Language name to kaikki.org filename mapping (most relevant for FW)
LANG_MAP = {
    "en": ("English", "kaikki.org-dictionary-English.jsonl"),
    "de": ("German", "kaikki.org-dictionary-German.jsonl"),
    "fr": ("French", "kaikki.org-dictionary-French.jsonl"),
    "it": ("Italian", "kaikki.org-dictionary-Italian.jsonl"),
    "la": ("Latin", "kaikki.org-dictionary-Latin.jsonl"),
    "ga": ("Irish", "kaikki.org-dictionary-Irish.jsonl"),
    "gd": ("Scottish Gaelic", "kaikki.org-dictionary-Scottish_Gaelic.jsonl"),
    "cy": ("Welsh", "kaikki.org-dictionary-Welsh.jsonl"),
    "nl": ("Dutch", "kaikki.org-dictionary-Dutch.jsonl"),
    "da": ("Danish", "kaikki.org-dictionary-Danish.jsonl"),
    "no": ("Norwegian Bokmål", "kaikki.org-dictionary-Norwegian_Bokmål.jsonl"),
    "sv": ("Swedish", "kaikki.org-dictionary-Swedish.jsonl"),
    "es": ("Spanish", "kaikki.org-dictionary-Spanish.jsonl"),
    "pt": ("Portuguese", "kaikki.org-dictionary-Portuguese.jsonl"),
    "el": ("Greek", "kaikki.org-dictionary-Greek.jsonl"),
    "sa": ("Sanskrit", "kaikki.org-dictionary-Sanskrit.jsonl"),
    "ja": ("Japanese", "kaikki.org-dictionary-Japanese.jsonl"),
    "zh": ("Chinese", "kaikki.org-dictionary-Chinese.jsonl"),
    "ar": ("Arabic", "kaikki.org-dictionary-Arabic.jsonl"),
    "he": ("Hebrew", "kaikki.org-dictionary-Hebrew.jsonl"),
    "hi": ("Hindi", "kaikki.org-dictionary-Hindi.jsonl"),
    "ru": ("Russian", "kaikki.org-dictionary-Russian.jsonl"),
    "fi": ("Finnish", "kaikki.org-dictionary-Finnish.jsonl"),
    "hu": ("Hungarian", "kaikki.org-dictionary-Hungarian.jsonl"),
    "tr": ("Turkish", "kaikki.org-dictionary-Turkish.jsonl"),
    "pl": ("Polish", "kaikki.org-dictionary-Polish.jsonl"),
}

# Priority languages for Finnegans Wake (Joyce drew heavily from these)
FW_PRIORITY_LANGS = ["en", "ga", "de", "fr", "it", "la", "nl", "da", "no", "sv"]

DATA_DIR = Path(__file__).parent / "data"


def download_kaikki_file(lang_code: str, force: bool = False) -> Path:
    """Download a kaikki.org JSONL file for a language."""
    if lang_code not in LANG_MAP:
        raise ValueError(f"Unknown language code: {lang_code}. Available: {list(LANG_MAP.keys())}")

    lang_name, filename = LANG_MAP[lang_code]
    DATA_DIR.mkdir(exist_ok=True)

    local_path = DATA_DIR / filename
    gz_path = DATA_DIR / f"{filename}.gz"

    if local_path.exists() and not force:
        print(f"  {filename} already exists, skipping download")
        return local_path

    # Try gzipped first, then uncompressed
    # URL-encode path components to handle non-ASCII chars (e.g., "Bokmål")
    lang_name_encoded = urllib.parse.quote(lang_name)
    filename_encoded = urllib.parse.quote(filename)
    url_gz = f"{KAIKKI_BASE}/{lang_name_encoded}/{filename_encoded}.gz"
    url_plain = f"{KAIKKI_BASE}/{lang_name_encoded}/{filename_encoded}"

    print(f"  Downloading {lang_name} dictionary...")
    try:
        urllib.request.urlretrieve(url_gz, str(gz_path))
        print(f"  Decompressing {gz_path.name}...")
        with gzip.open(str(gz_path), "rt", encoding="utf-8") as gz_in:
            with open(str(local_path), "w", encoding="utf-8") as out:
                for line in gz_in:
                    out.write(line)
        gz_path.unlink()
        return local_path
    except Exception as e:
        print(f"  Gzipped download failed ({e}), trying uncompressed...")
        if gz_path.exists():
            gz_path.unlink()

    try:
        urllib.request.urlretrieve(url_plain, str(local_path))
        return local_path
    except Exception as e:
        print(f"  ERROR: Could not download {lang_name}: {e}")
        raise


def parse_etymology_morphemes(etymology_text: str) -> list[dict]:
    """Extract morphemes from etymology text using pattern matching.

    This is a heuristic parser — kaikki.org etymology_text is free-form
    but often follows patterns like "From Latin X + Y" or "compound of X and Y".
    """
    morphemes = []

    # Pattern: "from X + Y" or "X + Y"
    plus_pattern = re.findall(r'(\w+)\s*\+\s*(\w+)', etymology_text)
    for a, b in plus_pattern:
        morphemes.append({"morpheme": a, "meaning": "", "lang": "", "morph_type": "root"})
        morphemes.append({"morpheme": b, "meaning": "", "lang": "", "morph_type": "root"})

    # Pattern: "prefix X-" or "-suffix X"
    prefix_pattern = re.findall(r'\b(\w+)-\s', etymology_text)
    for p in prefix_pattern:
        if p.lower() not in ("non", "pre", "post", "un", "re", "anti", "dis", "mis", "over", "under"):
            continue
        morphemes.append({"morpheme": p, "meaning": "", "lang": "", "morph_type": "prefix"})

    return morphemes


def get_progress(conn, source_file: str) -> dict:
    """Get ingestion progress for a source file."""
    row = conn.execute(
        "SELECT * FROM ingest_progress WHERE source_file = ?",
        (source_file,)
    ).fetchone()
    if row:
        return dict(row)
    return None


def save_progress(conn, source_file: str, lang_code: str,
                  lines_read: int, entries_inserted: int,
                  entries_skipped: int, completed: bool):
    """Save/update ingestion progress."""
    conn.execute("""
        INSERT INTO ingest_progress
            (source_file, lang_code, lines_read, entries_inserted, entries_skipped, completed, last_updated)
        VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
        ON CONFLICT(source_file) DO UPDATE SET
            lines_read = excluded.lines_read,
            entries_inserted = excluded.entries_inserted,
            entries_skipped = excluded.entries_skipped,
            completed = excluded.completed,
            last_updated = datetime('now')
    """, (source_file, lang_code, lines_read, entries_inserted, entries_skipped, int(completed)))
    conn.commit()


def ingest_file(filepath: Path, conn, lang_code: str,
                limit: int = 0, batch_size: int = 5000) -> int:
    """Ingest a kaikki.org JSONL file into the database, with resume support.

    Tracks how many lines have been processed in the ingest_progress table.
    If interrupted (Ctrl+C), progress is saved. Running the same command again
    skips already-processed lines and continues from where it stopped.

    Args:
        filepath: Path to the JSONL file.
        conn: SQLite connection.
        lang_code: Language code for this file.
        limit: Max NEW entries to ingest in this run (0 = unlimited).
        batch_size: Commit + save progress every N entries.

    Returns:
        Number of new entries ingested in this run.
    """
    source_file = filepath.name

    # Check for prior progress
    progress = get_progress(conn, source_file)
    resume_from_line = 0
    total_inserted = 0
    total_skipped = 0

    if progress:
        if progress["completed"]:
            print(f"  {source_file}: already fully ingested "
                  f"({progress['entries_inserted']} entries). Use --reset {lang_code} to re-do.")
            return 0
        resume_from_line = progress["lines_read"]
        total_inserted = progress["entries_inserted"]
        total_skipped = progress["entries_skipped"]
        print(f"  Resuming {source_file} from line {resume_from_line} "
              f"({total_inserted} already inserted)")
    else:
        print(f"  Starting fresh ingestion of {source_file}")

    new_count = 0  # entries inserted in THIS run
    line_num = resume_from_line  # track even if we exit early

    # Count total lines for the progress bar
    print(f"  Counting lines in {source_file}...")
    total_lines = sum(1 for _ in open(str(filepath), "r", encoding="utf-8"))
    remaining_lines = total_lines - resume_from_line
    if limit:
        desc = f"{lang_code} (limit {limit})"
    else:
        desc = f"{lang_code}"

    try:
        with open(str(filepath), "r", encoding="utf-8") as f:
            pbar = tqdm(
                enumerate(f),
                total=total_lines,
                initial=resume_from_line,
                desc=f"  {desc}",
                unit=" entries",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} lines [{elapsed}<{remaining}, {rate_fmt}]"
            )
            for line_num, line in pbar:
                # Skip lines we've already processed
                if line_num < resume_from_line:
                    continue

                # Check limit (on new entries this run)
                if limit and new_count >= limit:
                    pbar.set_postfix_str(f"limit reached, {new_count} new")
                    save_progress(conn, source_file, lang_code,
                                  line_num, total_inserted, total_skipped,
                                  completed=False)
                    break

                line = line.strip()
                if not line:
                    continue

                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    total_skipped += 1
                    continue

                word_id = insert_word(conn, entry)
                if word_id:
                    # Extract morphemes from etymology
                    etym = entry.get("etymology_text", "")
                    if etym:
                        morphemes = parse_etymology_morphemes(etym)
                        for i, m in enumerate(morphemes):
                            conn.execute(
                                """INSERT INTO morphemes (word_id, morpheme, meaning, lang,
                                   lang_code, morph_type, position)
                                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                                (word_id, m["morpheme"], m["meaning"], m["lang"],
                                 "", m["morph_type"], i)
                            )
                    new_count += 1
                    total_inserted += 1
                else:
                    total_skipped += 1

                # Update progress bar postfix
                if new_count % 100 == 0:
                    pbar.set_postfix_str(f"{total_inserted} total, {new_count} new")

                # Periodic commit + progress save
                if new_count > 0 and new_count % batch_size == 0:
                    save_progress(conn, source_file, lang_code,
                                  line_num + 1, total_inserted, total_skipped,
                                  completed=False)

            else:
                # Loop completed without break = we finished the whole file
                save_progress(conn, source_file, lang_code,
                              line_num + 1, total_inserted, total_skipped,
                              completed=True)
            pbar.close()

    except KeyboardInterrupt:
        # Ctrl+C: save where we got to
        print(f"\n  Interrupted! Saving progress at line {line_num}...")
        save_progress(conn, source_file, lang_code,
                      line_num, total_inserted, total_skipped,
                      completed=False)
        print(f"  Progress saved. Run the same command to resume.")
        sys.exit(0)

    print(f"  Done: {new_count} new entries this run, "
          f"{total_inserted} total, {total_skipped} skipped")
    return new_count


def rebuild_fts(conn):
    """Rebuild FTS5 indexes after bulk ingestion."""
    print("  Rebuilding FTS indexes...")
    conn.executescript("""
        INSERT INTO words_fts(words_fts) VALUES('rebuild');
        INSERT INTO morphemes_fts(morphemes_fts) VALUES('rebuild');
    """)
    conn.commit()
    print("  FTS indexes rebuilt")


def show_status(conn):
    """Show ingestion status for all source files."""
    rows = conn.execute(
        "SELECT * FROM ingest_progress ORDER BY lang_code"
    ).fetchall()

    if not rows:
        print("No ingestion history yet.")
        return

    print(f"\n{'Lang':<6} {'Source File':<50} {'Inserted':>10} {'Skipped':>8} {'Lines':>10} {'Status':<12} {'Last Updated'}")
    print("-" * 120)
    for r in rows:
        status = "COMPLETE" if r["completed"] else "PARTIAL"
        print(f"{r['lang_code']:<6} {r['source_file']:<50} {r['entries_inserted']:>10} "
              f"{r['entries_skipped']:>8} {r['lines_read']:>10} {status:<12} {r['last_updated']}")

    # Also show overall DB stats
    word_count = conn.execute("SELECT COUNT(*) FROM words").fetchone()[0]
    ipa_count = conn.execute("SELECT COUNT(*) FROM ipa_index").fetchone()[0]
    morph_count = conn.execute("SELECT COUNT(*) FROM morphemes").fetchone()[0]
    print(f"\nDatabase totals: {word_count} words, {ipa_count} IPA entries, {morph_count} morphemes")


def reset_progress(conn, lang_code: str):
    """Reset ingestion progress for a language, removing its data."""
    if lang_code not in LANG_MAP:
        print(f"Unknown language code: {lang_code}")
        return

    _, filename = LANG_MAP[lang_code]

    # Delete words + cascading morphemes/ipa for this language
    print(f"  Removing {lang_code} data from database...")
    conn.execute("DELETE FROM ipa_index WHERE lang_code = ?", (lang_code,))
    conn.execute(
        "DELETE FROM morphemes WHERE word_id IN (SELECT id FROM words WHERE lang_code = ?)",
        (lang_code,)
    )
    conn.execute("DELETE FROM words WHERE lang_code = ?", (lang_code,))
    conn.execute("DELETE FROM ingest_progress WHERE source_file = ?", (filename,))
    conn.commit()
    print(f"  Reset complete for {lang_code}")


def main():
    parser = argparse.ArgumentParser(
        description="Ingest kaikki.org Wiktionary data into the Word Forge database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Resume support:
  Ingestion tracks progress per-file. If you Ctrl+C or use --limit,
  running the same command again will skip already-processed lines.

Examples:
  python ingest.py --lang en --limit 500    # small sample
  python ingest.py --lang en                # resume / finish English
  python ingest.py --status                 # see what's been ingested
  python ingest.py --reset en               # wipe English and start over
  python ingest.py --fw-priority --limit 5000  # 5K per FW language
        """
    )
    parser.add_argument("--file", type=str, help="Path to a local kaikki JSONL file")
    parser.add_argument("--lang", type=str, help="Single language code to download and ingest")
    parser.add_argument("--langs", type=str, help="Comma-separated language codes (e.g., en,de,fr)")
    parser.add_argument("--fw-priority", action="store_true",
                        help="Ingest Finnegans Wake priority languages: " + ",".join(FW_PRIORITY_LANGS))
    parser.add_argument("--limit", type=int, default=0,
                        help="Max NEW entries per language in this run (0=unlimited)")
    parser.add_argument("--force-download", action="store_true", help="Re-download even if file exists")
    parser.add_argument("--skip-fts", action="store_true", help="Skip FTS index rebuild")
    parser.add_argument("--status", action="store_true", help="Show ingestion progress and exit")
    parser.add_argument("--reset", type=str, metavar="LANG",
                        help="Reset progress for a language (deletes its data)")

    args = parser.parse_args()

    # Initialize database
    init_db()
    conn = get_connection()

    if args.status:
        show_status(conn)
        conn.close()
        return

    if args.reset:
        reset_progress(conn, args.reset)
        conn.close()
        return

    total = 0

    def ingest_lang(lang_code: str):
        nonlocal total
        try:
            filepath = download_kaikki_file(lang_code, force=args.force_download)
            total += ingest_file(filepath, conn, lang_code, limit=args.limit)
        except Exception as e:
            print(f"  WARNING: Skipping {lang_code}: {e}")

    if args.file:
        filepath = Path(args.file)
        if not filepath.exists():
            print(f"ERROR: File not found: {filepath}")
            sys.exit(1)
        # Guess lang_code from filename or use "unknown"
        lang_code = "unknown"
        for code, (_, fname) in LANG_MAP.items():
            if fname in filepath.name:
                lang_code = code
                break
        total += ingest_file(filepath, conn, lang_code, limit=args.limit)

    elif args.lang:
        ingest_lang(args.lang)

    elif args.langs:
        for lang_code in [l.strip() for l in args.langs.split(",")]:
            ingest_lang(lang_code)

    elif args.fw_priority:
        for lang_code in FW_PRIORITY_LANGS:
            ingest_lang(lang_code)

    else:
        parser.print_help()
        conn.close()
        sys.exit(1)

    # Rebuild FTS indexes
    if not args.skip_fts and total > 0:
        rebuild_fts(conn)

    conn.close()

    from db import DB_PATH
    print(f"\nThis run: {total} new entries ingested into {DB_PATH.absolute()}")


if __name__ == "__main__":
    main()
