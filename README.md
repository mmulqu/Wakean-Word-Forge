# Wakean Word Forge

An MCP server that programmatically generates Finnegans Wake-style portmanteau words. It exists because LLMs *can't do this natively* — BPE tokenization chunks neologisms into rare subword sequences the model has barely seen, and RLHF actively pulls output toward coherent English. The forge separates the literary judgment (which the LLM handles well) from the character-level word surgery (which it doesn't).

Claude (or any LLM) picks the ingredients — which roots, which languages, what semantic layers to target — and the forge does the mechanical fusion, returning ranked candidates. The LLM then selects the best one based on rhythm, meaning, and context.

## How it works

The forge is backed by a SQLite database of 4.2M+ word entries extracted from [kaikki.org](https://kaikki.org) (structured Wiktionary dumps). Each entry includes the word, language, IPA pronunciation, full etymology chain, morpheme breakdown, and sense definitions across 9+ languages.

### Tools

| Tool | Input | What it does |
|------|-------|-------------|
| `lookup_morphemes` | word, language | Returns etymology, morphemes, IPA, senses from the Wiktionary database |
| `phonetic_neighbors` | word or IPA string, target languages | Finds words in other languages that *sound* similar using articulatory feature distance (panphon) |
| `forge_portmanteau` | array of word components + optional strategy | Fuses words together using overlap, cross-overlap, substitute, nest, or interleave strategies. Returns ranked candidates |
| `idiom_twist` | phrase + domain words | Recombines fixed expressions with phonetically similar words from a target domain |
| `convert_to_ipa` | word, language | Converts orthographic text to IPA via epitran |
| `compare_phonetics` | two words | Computes articulatory phonetic distance between two words |
| `search_etymology` | query string | Full-text search across all etymologies in the database |
| `list_languages` | — | Lists supported languages for IPA conversion |

### Example workflow

The LLM wants to create a Joycean portmanteau around "heaven" + "immortality":

1. **Lookup**: calls `lookup_morphemes("Himmel", lang_code="de")` → gets IPA `/hɪml̩/`, etymology from Old High German
2. **Phonetic search**: calls `phonetic_neighbors("Himmel", target_langs=["en"])` → finds "him" sounds similar
3. **Forge**: calls `forge_portmanteau` with:
   ```json
   {
     "components": [
       {"text": "Himmel", "lang": "de", "meaning": "heaven"},
       {"text": "him", "lang": "en", "meaning": "pronoun"},
       {"text": "immortality", "lang": "en", "meaning": "living forever"}
     ]
   }
   ```
   → Returns candidates like **"Himmortality"** (Himmel grafted onto immortality at the shared "im")
4. The LLM picks the candidate that fits the sentence's rhythm and semantic intent

### Fusion strategies

- **overlap**: end of word A matches start of word B → merge at the shared sequence
- **cross_overlap**: shared substring found anywhere in both words → graft A's prefix onto B at the match point. This is Joyce's primary technique
- **substitute**: swap a morpheme in one word for a phonetically similar one from another language
- **nest**: embed one word inside another at a syllable boundary
- **interleave**: alternate syllables from two words

## Setup

### Requirements

- Python 3.10+
- ~15 GB disk space for the full database (or less with fewer languages)

### 1. Clone and install dependencies

```bash
git clone https://github.com/yourusername/Wakean_Word_Forge.git
cd Wakean_Word_Forge
pip install -r requirements.txt
```

Dependencies: `fastmcp`, `panphon`, `epitran`

### 2. Download and ingest Wiktionary data

The forge needs structured dictionary data from [kaikki.org](https://kaikki.org), a project that extracts English Wiktionary into structured JSON. The ingestion script downloads the data, parses it, and loads it into a local SQLite database.

**Start small** — ingest 500 entries to verify everything works:

```bash
python ingest.py --lang en --limit 500
```

This downloads the English JSONL file (~2.7 GB), ingests 500 entries, and creates `forge.db`.

**Add more languages** — Joyce drew heavily from these 10 languages:

```bash
python ingest.py --fw-priority
```

This downloads and ingests English, Irish, German, French, Italian, Latin, Dutch, Danish, Norwegian, and Swedish. The full run ingests ~4.2M entries and takes 10-15 minutes.

**Or pick specific languages:**

```bash
python ingest.py --lang de --limit 10000   # 10K German entries
python ingest.py --langs ga,la,it           # Irish, Latin, Italian (all entries)
```

#### Resume support

Ingestion tracks progress per-file. You can Ctrl+C at any time — running the same command again skips already-processed lines and continues from where it stopped.

```bash
python ingest.py --status                   # see what's been ingested
python ingest.py --lang en                  # resume interrupted English ingestion
python ingest.py --reset en                 # wipe English data and start over
```

#### Available languages

| Code | Language | Approximate entries |
|------|----------|-------------------|
| `en` | English | 1,455,000 |
| `de` | German | 364,000 |
| `fr` | French | 400,000 |
| `it` | Italian | 621,000 |
| `la` | Latin | 885,000 |
| `ga` | Irish | 38,000 |
| `gd` | Scottish Gaelic | varies |
| `cy` | Welsh | varies |
| `nl` | Dutch | 143,000 |
| `da` | Danish | 56,000 |
| `no` | Norwegian | varies |
| `sv` | Swedish | 310,000 |
| `es` | Spanish | varies |
| `pt` | Portuguese | varies |
| `el` | Greek | varies |
| `sa` | Sanskrit | varies |
| `ar` | Arabic | varies |
| `he` | Hebrew | varies |
| `ru` | Russian | varies |
| `fi` | Finnish | varies |
| `hu` | Hungarian | varies |
| `tr` | Turkish | varies |
| `pl` | Polish | varies |

### 3. Windows note

On Windows, panphon's IPA data files require UTF-8 mode. Set this environment variable before running:

```bash
set PYTHONUTF8=1
```

Or on bash/WSL:

```bash
export PYTHONUTF8=1
```

### 4. Run the MCP server

```bash
PYTHONUTF8=1 fastmcp run server.py
```

### 5. Connect to Claude Code

Add to your Claude Code MCP config (`~/.claude.json` or project `.mcp.json`):

```json
{
  "mcpServers": {
    "wakean-word-forge": {
      "command": "python",
      "args": ["path/to/Wakean_Word_Forge/server.py"],
      "env": {
        "PYTHONUTF8": "1"
      }
    }
  }
}
```

Or to connect to a remote server via SSE:

```bash
PYTHONUTF8=1 fastmcp run server.py --transport sse --port 8000
```

### 6. English IPA note

Epitran's English IPA conversion requires `flite` (CMU's text-to-speech engine), which is not easily available on Windows. This is fine — the Wiktionary database already contains IPA for most common English words. Epitran works without flite for all other languages (German, French, Italian, Spanish, Irish, etc.), which is where you need it most for cross-lingual phonetic matching.

On Linux, you can install flite for full English IPA support:

```bash
sudo apt install flite
```

## Architecture

```
┌─────────────────────────────────────────────┐
│  LLM (Claude, etc.)                        │
│  Literary judgment: what to fuse, why,      │
│  which languages, semantic targets          │
│                    │                        │
│                    ▼                        │
│         MCP tool calls                      │
└────────────────────┬────────────────────────┘
                     │
┌────────────────────▼────────────────────────┐
│  Wakean Word Forge (FastMCP server)         │
│                                             │
│  server.py ─── 8 tools exposed via MCP      │
│       │                                     │
│  forge.py ──── fusion engine                │
│       │        (overlap, substitute,        │
│       │         nest, interleave)           │
│       │                                     │
│  phonetics.py ─ panphon distance +          │
│       │         epitran IPA conversion      │
│       │                                     │
│  db.py ──────── SQLite + FTS5               │
│       │                                     │
│  forge.db ───── 4.2M entries, 853K IPA,     │
│                 234K morphemes              │
└─────────────────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        ▼                         ▼
   kaikki.org                  panphon
   (Wiktionary JSONL)     (articulatory features)
```

## Data sources

- **[kaikki.org](https://kaikki.org)** — Structured Wiktionary extracts by Tatu Ylonen's [wiktextract](https://github.com/tatuylonen/wiktextract) project. JSONL format, updated monthly from enwiktionary dumps. Licensed under CC BY-SA 3.0 (same as Wiktionary).
- **[panphon](https://github.com/dmort27/panphon)** — Maps 5,000+ IPA segments to 21 articulatory feature vectors for phonetic distance computation. From CMU's speech research group.
- **[epitran](https://github.com/dmort27/epitran)** — Orthography-to-IPA conversion for 60+ languages. Same CMU research group as panphon.

## Why this exists

LLMs fail at generating Wakean prose for mechanical reasons:

1. **BPE tokenization** chunks portmanteaus into rare subword sequences the model has barely seen during training
2. **RLHF / instruction tuning** actively pulls output toward clear, coherent English
3. The rest of the **training data swamps the signal** — general models can't spontaneously produce Wake-like text because the probability distribution over tokens overwhelmingly favors attested sequences

The forge treats this as a tooling gap, not a capability gap. The LLM's literary judgment about *what* should be fused, which languages carry thematic weight, and what allusions to layer — that works fine. What fails is the final act of producing the mutant word token by token. So we externalize it.
