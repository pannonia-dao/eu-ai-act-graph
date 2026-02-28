# EU AI Act Knowledge Graph Pipeline — Design Decisions

## Architecture

The pipeline is a single-file, modular Python script that doubles as a **Streamlit app** and a **CLI tool**. Running it with `streamlit run eu_ai_act_pipeline.py` launches the interactive UI; running it with `python eu_ai_act_pipeline.py --lang HU` executes the pipeline headlessly.

## Layer-by-Layer Design Decisions

### 0 — Download & Cache
The primary source is always the **official EUR-Lex ELI/HTML endpoint** (`eur-lex.europa.eu/legal-content/{LANG}/TXT/HTML/?uri=OJ:L_202401689`). This guarantees traceability to the authentic legislative text without relying on unofficial mirrors.

A **local JSON cache** (`.cache/eu_ai_act/`) stores the full raw text alongside retrieval metadata. Cache keys are based on an MD5 hash of the language code. Re-download is opt-in via `--force` (CLI) or a checkbox (Streamlit). The cache payload also includes `source_url` and `retrieved_at` so provenance is always traceable.

**PDF fallback** activates only when the HTML endpoint fails. It uses `pdfminer.six` or `pypdf` (detected at runtime) so the dependency footprint is minimal. PDF-extracted text is noisier than HTML but is sufficient for regex-based parsing.

### 1 — Article Parsing
The HTML is pre-processed by BeautifulSoup: boilerplate elements (nav, footer, footnotes) are stripped, block tags are converted to newlines, and excessive blank lines are collapsed. This gives a clean, line-oriented text that works well with regex patterns.

Article boundaries are detected with **language-specific heading regexes** (24 EU official languages defined). For example, Hungarian uses `^\d+\.\s+cikk\r?\n(.+)` while English uses `^Article\s+(\d+)\r?\n(.+)`. All article IDs are **normalized to `Art. N`** regardless of the source language, ensuring graph-node consistency. Annex IDs are normalized to `Annex I`, `Annex II`, etc.

### 2 — Term Extraction
Definition articles are found by scanning article titles for words like "definition", "fogalommeghatározás", "Begriffsbestimmung", etc. If none are found, all articles are searched as a fallback.

Term extraction uses two sequential regex strategies: first, **quoted-phrase patterns** (`"term" means ...`) which are highly reliable, then **unquoted capitalized noun-phrase patterns** as a secondary pass. No LLM or NER is required, keeping the pipeline deterministic and reproducible.

The pipeline intentionally **preserves exact legal wording** and never paraphrases definitions.

### 3 — Obligation Mining
Each article's text is split into sentences using a punctuation-and-capitalization heuristic that works across languages without requiring language-specific NLP libraries. Sentences are then matched against a **per-language modality pattern table** (e.g., `köteles/kell` for HU, `shall/must` for EN). This table is configurable via `OBLIGATION_PATTERNS` at the top of the file.

Obligation IDs are **deterministic**: `obl_art{N}_{seq:02d}`, so the same text always produces the same ID given the same parsing order.

Topic tags are assigned via a **keyword-presence heuristic** across multiple languages (the `_infer_topics` function). This avoids any summarization or interpretation.

### 4 — Role Mapping
A **canonical role vocabulary** (`ROLE_VOCAB`) maps each of the six target roles to surface forms in all supported languages. The matcher sorts surface forms by descending length to handle multi-word roles (e.g., "authorised representative") before their substrings (e.g., "representative").

If no role is found in the obligation sentence, the role is **propagated from the article title** (e.g., Art. 24 "Obligations of distributors" → all obligations in Art. 24 get `distributor`). Truly ambiguous cases fall back to `["unspecified"]`. Raw mentions are always preserved alongside canonical labels for auditability.

### 5 — Cross-Reference Extraction
Cross-reference regexes handle ranges (`Articles 8 to 15`), single articles, and annex references. Ranges are **expanded** into individual edges. The language-specific pattern table covers the most common EU languages; unlisted languages fall back to English patterns, which is acceptable since article/annex reference numerals are invariant across EU language versions.

### 6 — Graph Schema
The graph uses **directed edges** with typed relations: `imposes`, `applies_to`, `defines`, `references`. Node IDs are deterministic (`Art. N`, `Annex N`, `obl_artN_seq`, `term_{slug}`, canonical role names). The schema is directly loadable into NetworkX, Neo4j (via JSON import), PyVis, or `streamlit-agraph`.

## Dependencies
```
requests
beautifulsoup4
lxml
pdfminer.six  # optional PDF fallback
pyvis         # optional graph visualization in Streamlit
streamlit     # for UI only
```

## What the pipeline does NOT do
- It does not translate, interpret, or paraphrase any legal text.
- It does not use LLMs or external APIs beyond EUR-Lex.
- It does not make legal judgements about obligation scope.
