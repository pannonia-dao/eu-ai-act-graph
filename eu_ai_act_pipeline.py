"""
EU AI Act Knowledge Graph Pipeline
===================================
Regulation (EU) 2024/1689 — Multilingual Knowledge Graph Builder

Downloads the official EU AI Act from EUR-Lex, parses articles/annexes,
extracts terms, obligations, roles, cross-references, and builds a
directed knowledge graph suitable for NetworkX / Neo4j / PyVis / streamlit-agraph.

Usage (standalone):
    python eu_ai_act_pipeline.py --lang HU

Usage (Streamlit):
    streamlit run eu_ai_act_pipeline.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Optional Streamlit import — gracefully degrade if running as plain script
# ---------------------------------------------------------------------------
try:
    import streamlit as st
    _STREAMLIT = True
except ImportError:
    _STREAMLIT = False

# ---------------------------------------------------------------------------
# Third-party imports (all commonly available; no heavy ML deps required)
# ---------------------------------------------------------------------------
try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    sys.exit("Please install: pip install requests beautifulsoup4 lxml")

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ===========================================================================
# CONSTANTS
# ===========================================================================

CACHE_DIR = Path(".cache/eu_ai_act")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Official EUR-Lex ELI HTML endpoint for Regulation (EU) 2024/1689
# Language code placeholder: {lang}
EURLEX_HTML_TEMPLATE = (
    "https://eur-lex.europa.eu/legal-content/{lang}/TXT/HTML/"
    "?uri=OJ:L_202401689"
)
# PDF fallback
EURLEX_PDF_TEMPLATE = (
    "https://eur-lex.europa.eu/legal-content/{lang}/TXT/PDF/"
    "?uri=OJ:L_202401689"
)

SUPPORTED_LANGUAGES = [
    "BG", "CS", "DA", "DE", "EL", "EN", "ES", "ET", "FI", "FR",
    "GA", "HR", "HU", "IT", "LT", "LV", "MT", "NL", "PL", "PT",
    "RO", "SK", "SL", "SV",
]

# ---------------------------------------------------------------------------
# Language-specific obligation modality patterns
# ---------------------------------------------------------------------------
OBLIGATION_PATTERNS: dict[str, list[str]] = {
    "EN": [r"\bshall\b", r"\bmust\b", r"\bis required to\b", r"\bare required to\b",
           r"\bshall ensure\b", r"\bshall maintain\b"],
    "HU": [r"\bköteles\b", r"\bkell\b", r"\bbiztosítja\b", r"\bgondoskodik\b",
           r"\bköteles biztosítani\b", r"\bkötelező\b", r"\bkötelezettsége\b"],
    "DE": [r"\bmuss\b", r"\bmüssen\b", r"\bhat zu\b", r"\bhaben zu\b",
           r"\bist verpflichtet\b", r"\bsind verpflichtet\b", r"\bstellt sicher\b"],
    "FR": [r"\bdoit\b", r"\bdoivent\b", r"\bveille\b", r"\bveillent\b",
           r"\best tenu\b", r"\bsont tenus\b", r"\bassure\b"],
    "PL": [r"\bmusi\b", r"\bpowinna\b", r"\bpowinn[iy]\b", r"\bzapewnia\b",
           r"\bjest zobowiązan\b"],
    # Fallback for unlisted languages: generic deontic-ish verbs
    "_default": [r"\bshall\b", r"\bmust\b", r"\bshould\b", r"\bobligation\b"],
}

# ---------------------------------------------------------------------------
# Language-specific article-heading patterns
# ---------------------------------------------------------------------------
# These regexes match article headers in various language forms.
ARTICLE_HEADING_PATTERNS: dict[str, str] = {
    "EN": r"^Article\s+(\d+)\s*\n([^\n]+)",
    "HU": r"^(?:(\d+)\.\s+cikk|Cikk\s+(\d+))[^\n]*\n([^\n]+)",
    "DE": r"^Artikel\s+(\d+)\s*\n([^\n]+)",
    "FR": r"^Article\s+(\d+)\s*\n([^\n]+)",
    "PL": r"^Artykuł\s+(\d+)\s*\n([^\n]+)",
    "IT": r"^Articolo\s+(\d+)\s*\n([^\n]+)",
    "ES": r"^Artículo\s+(\d+)\s*\n([^\n]+)",
    "NL": r"^Artikel\s+(\d+)\s*\n([^\n]+)",
    "PT": r"^Artigo\s+(\d+)[\.º°]?\s*\n([^\n]+)",
    "RO": r"^Articolul\s+(\d+)\s*\n([^\n]+)",
    "CS": r"^Článek\s+(\d+)\s*\n([^\n]+)",
    "SK": r"^Článok\s+(\d+)\s*\n([^\n]+)",
    "SL": r"^Člen\s+(\d+)\s*\n([^\n]+)",
    "HR": r"^Članak\s+(\d+)\s*\n([^\n]+)",
    "BG": r"^Член\s+(\d+)\s*\n([^\n]+)",
    "EL": r"^Άρθρο\s+(\d+)\s*\n([^\n]+)",
    "LT": r"^(\d+)\s+straipsnis\s*\n([^\n]+)",
    "LV": r"^(\d+)\.\s+pants\s*\n([^\n]+)",
    "ET": r"^Artikkel\s+(\d+)\s*\n([^\n]+)",
    "FI": r"^(\d+)\s+artikla\s*\n([^\n]+)",
    "SV": r"^Artikel\s+(\d+)\s*\n([^\n]+)",
    "DA": r"^Artikel\s+(\d+)\s*\n([^\n]+)",
    "MT": r"^Artikolu\s+(\d+)\s*\n([^\n]+)",
    "GA": r"^Airteagal\s+(\d+)\s*\n([^\n]+)",
}

# ---------------------------------------------------------------------------
# Role vocabulary (canonical → per-language surface forms)
# ---------------------------------------------------------------------------
ROLE_VOCAB: dict[str, dict[str, list[str]]] = {
    "provider": {
        "EN": ["provider", "providers"],
        "HU": ["szolgáltató", "szolgáltatók"],
        "DE": ["Anbieter"],
        "FR": ["fournisseur", "fournisseurs"],
        "_default": ["provider"],
    },
    "deployer": {
        "EN": ["deployer", "deployers"],
        "HU": ["üzembe helyező", "üzembe helyezők"],
        "DE": ["Betreiber"],
        "FR": ["déployeur", "déployeurs"],
        "_default": ["deployer"],
    },
    "distributor": {
        "EN": ["distributor", "distributors"],
        "HU": ["forgalmazó", "forgalmazók"],
        "DE": ["Händler"],
        "FR": ["distributeur", "distributeurs"],
        "_default": ["distributor"],
    },
    "importer": {
        "EN": ["importer", "importers"],
        "HU": ["importőr", "importőrök"],
        "DE": ["Einführer"],
        "FR": ["importateur", "importateurs"],
        "_default": ["importer"],
    },
    "authorised_representative": {
        "EN": ["authorised representative", "authorized representative"],
        "HU": ["meghatalmazott képviselő"],
        "DE": ["Bevollmächtigter", "Bevollmächtigte"],
        "FR": ["mandataire"],
        "_default": ["authorised representative"],
    },
    "notified_body": {
        "EN": ["notified body", "notified bodies"],
        "HU": ["bejelentett szervezet", "bejelentett szervek"],
        "DE": ["notifizierte Stelle", "notifizierte Stellen"],
        "FR": ["organisme notifié", "organismes notifiés"],
        "_default": ["notified body"],
    },
}

# ---------------------------------------------------------------------------
# Cross-reference patterns (per language)
# ---------------------------------------------------------------------------
CROSSREF_PATTERNS: dict[str, list[str]] = {
    "EN": [
        r"\bArticles?\s+(\d+)(?:\s+to\s+(\d+))?\b",
        r"\bAnnex\s+(I{1,3}|IV|V|VI|VII|VIII|IX|X|\d+)\b",
        r"\bparagraph\s+(\d+)\s+of\s+Article\s+(\d+)\b",
    ],
    "HU": [
        r"\b(\d+)\.\s+cikk\b",
        r"\b(\d+)–(\d+)\.\s+cikk\b",
        r"\b([IVX]+)\.\s+melléklet\b",
        r"(?:a|az)\s+(\d+)\.\s+cikk\b",
    ],
    "DE": [
        r"\bArtikel\s+(\d+)(?:\s+bis\s+(\d+))?\b",
        r"\bAnhang\s+(I{1,3}|IV|V|VI|VII|VIII|IX|X|\d+)\b",
    ],
    "FR": [
        r"\bArticles?\s+(\d+)(?:\s+à\s+(\d+))?\b",
        r"\bAnnexe\s+(I{1,3}|IV|V|VI|VII|VIII|IX|X|\d+)\b",
    ],
    "_default": [
        r"\bArticles?\s+(\d+)(?:\s+(?:to|bis|à|do|do|до)\s+(\d+))?\b",
        r"\bAnnex\s+(I{1,3}|IV|V|VI|VII|VIII|IX|X|\d+)\b",
    ],
}

ROMAN_TO_INT = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5,
                "VI": 6, "VII": 7, "VIII": 8, "IX": 9, "X": 10}


# ===========================================================================
# 0️⃣  DOWNLOAD LAYER
# ===========================================================================

def _cache_path(language_code: str) -> Path:
    key = hashlib.md5(f"eu_ai_act_{language_code}".encode()).hexdigest()[:10]
    return CACHE_DIR / f"eu_ai_act_{language_code}_{key}.json"


def download_ai_act(language_code: str, force: bool = False) -> tuple[str, dict]:
    """
    Download the EU AI Act from EUR-Lex in the requested language.
    Returns (raw_text, meta_dict).

    Caches result locally to avoid re-downloading.
    """
    lang = language_code.upper()
    cache_file = _cache_path(lang)

    if not force and cache_file.exists():
        log.info("Loading from cache: %s", cache_file)
        cached = json.loads(cache_file.read_text(encoding="utf-8"))
        return cached["raw_text"], {k: v for k, v in cached.items() if k != "raw_text"}

    url = EURLEX_HTML_TEMPLATE.format(lang=lang)
    log.info("Fetching EU AI Act [%s] from: %s", lang, url)

    headers = {"User-Agent": "EUAIActResearch/1.0 (legal-tech pipeline; non-commercial)"}
    raw_text = ""
    content_type = "html"
    actual_url = url

    try:
        resp = requests.get(url, headers=headers, timeout=60)
        resp.raise_for_status()
        raw_text = _parse_html(resp.text)
    except Exception as exc:
        log.warning("HTML fetch failed (%s); trying PDF fallback…", exc)
        pdf_url = EURLEX_PDF_TEMPLATE.format(lang=lang)
        try:
            resp = requests.get(pdf_url, headers=headers, timeout=90)
            resp.raise_for_status()
            raw_text = _extract_pdf_text(resp.content)
            content_type = "pdf"
            actual_url = pdf_url
        except Exception as exc2:
            raise RuntimeError(
                f"Could not download EU AI Act for language '{lang}': {exc2}"
            ) from exc2

    meta = {
        "language": lang,
        "source_url": actual_url,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "content_type": content_type,
    }

    # Persist to cache
    payload = {**meta, "raw_text": raw_text}
    cache_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                          encoding="utf-8")
    log.info("Saved to cache: %s", cache_file)
    return raw_text, meta


def _parse_html(html: str) -> str:
    """Extract clean text from EUR-Lex HTML response."""
    soup = BeautifulSoup(html, "lxml")

    # Remove nav / header / footer / footnotes boilerplate
    for tag in soup.select("nav, header, footer, .footnote, .note, #toc, "
                           ".oj-hd-text, .oj-footer, script, style"):
        tag.decompose()

    # EUR-Lex wraps document body in #document or article element
    body = soup.select_one("#document") or soup.select_one("article") or soup.body
    if body is None:
        body = soup

    # Convert <br> / <p> to newlines for downstream regex parsing
    for br in body.find_all("br"):
        br.replace_with("\n")
    for p in body.find_all(["p", "div"]):
        p.insert_before("\n")
        p.insert_after("\n")

    text = body.get_text(separator="\n")
    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove page numbers (e.g. "L 200/12  EN  Official Journal…")
    text = re.sub(r"L\s+\d+/\d+\s+\w{2}\s+Official Journal.*?\n", "", text)
    return text.strip()


def _extract_pdf_text(pdf_bytes: bytes) -> str:
    """Extract plain text from PDF bytes using pdfminer if available."""
    try:
        from pdfminer.high_level import extract_text as pm_extract
        import io
        return pm_extract(io.BytesIO(pdf_bytes))
    except ImportError:
        pass
    try:
        import pypdf
        import io
        reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)
    except ImportError:
        pass
    raise RuntimeError(
        "No PDF parser available. Install pdfminer.six or pypdf: "
        "pip install pdfminer.six"
    )


# ===========================================================================
# 1️⃣  PARSING LAYER
# ===========================================================================

def parse_articles(raw_text: str, language_code: str) -> tuple[list[dict], list[dict]]:
    """
    Parse article and annex structure from raw text.
    Returns (articles, annexes) each as list of dicts.
    """
    lang = language_code.upper()
    articles: list[dict] = []
    annexes: list[dict] = []

    # --- Article parsing ---
    # Build a robust multi-language article splitter
    art_patterns = _build_article_split_pattern(lang)
    article_blocks = _split_by_heading(raw_text, art_patterns)

    for num, title, body in article_blocks:
        articles.append({
            "article_id": f"Art. {num}",
            "title": title.strip(),
            "text": _clean_body(body),
            "language": lang,
        })

    # --- Annex parsing ---
    annex_pattern = _build_annex_split_pattern(lang)
    annex_blocks = _split_annexes(raw_text, annex_pattern)
    for ann_id, ann_title, ann_body in annex_blocks:
        annexes.append({
            "annex_id": ann_id,
            "title": ann_title.strip(),
            "text": _clean_body(ann_body),
            "language": lang,
        })

    log.info("Parsed %d articles, %d annexes [%s]", len(articles), len(annexes), lang)
    return articles, annexes


def _build_article_split_pattern(lang: str) -> re.Pattern:
    """
    Returns a compiled regex that matches article headings in the given language.
    Group 1 = article number, Group 2 = title (may be on next line).
    """
    # Universal fallback patterns in addition to language-specific ones
    patterns = []

    # Language-specific patterns
    specific = {
        "EN": r"(?m)^Article\s+(\d+)\r?\n(.+)",
        "HU": r"(?m)^(\d+)\.\s+cikk\r?\n(.+)",
        "DE": r"(?m)^Artikel\s+(\d+)\r?\n(.+)",
        "FR": r"(?m)^Article\s+(\d+)\r?\n(.+)",
        "PL": r"(?m)^Artykuł\s+(\d+)\r?\n(.+)",
        "IT": r"(?m)^Articolo\s+(\d+)\r?\n(.+)",
        "ES": r"(?m)^Artículo\s+(\d+)\r?\n(.+)",
        "NL": r"(?m)^Artikel\s+(\d+)\r?\n(.+)",
        "PT": r"(?m)^Artigo\s+(\d+)[\.º°]?\r?\n(.+)",
        "RO": r"(?m)^Articolul\s+(\d+)\r?\n(.+)",
        "CS": r"(?m)^Článek\s+(\d+)\r?\n(.+)",
        "SK": r"(?m)^Článok\s+(\d+)\r?\n(.+)",
        "SL": r"(?m)^Člen\s+(\d+)\r?\n(.+)",
        "HR": r"(?m)^Članak\s+(\d+)\r?\n(.+)",
        "BG": r"(?m)^Член\s+(\d+)\r?\n(.+)",
        "EL": r"(?m)^Άρθρο\s+(\d+)\r?\n(.+)",
        "LT": r"(?m)^(\d+)\s+straipsnis\r?\n(.+)",
        "LV": r"(?m)^(\d+)\.\s+pants\r?\n(.+)",
        "ET": r"(?m)^Artikkel\s+(\d+)\r?\n(.+)",
        "FI": r"(?m)^(\d+)\s+artikla\r?\n(.+)",
        "SV": r"(?m)^Artikel\s+(\d+)\r?\n(.+)",
        "DA": r"(?m)^Artikel\s+(\d+)\r?\n(.+)",
        "MT": r"(?m)^Artikolu\s+(\d+)\r?\n(.+)",
        "GA": r"(?m)^Airteagal\s+(\d+)\r?\n(.+)",
    }
    pattern_str = specific.get(lang, r"(?m)^Article\s+(\d+)\r?\n(.+)")
    return re.compile(pattern_str)


def _split_by_heading(text: str, pattern: re.Pattern) -> list[tuple[str, str, str]]:
    """
    Split text into (number, title, body) tuples using a heading pattern.
    """
    matches = list(pattern.finditer(text))
    if not matches:
        return []
    blocks = []
    for i, m in enumerate(matches):
        num = m.group(1)
        title = m.group(2) if m.lastindex and m.lastindex >= 2 else ""
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end]
        blocks.append((num, title, body))
    return blocks


def _build_annex_split_pattern(lang: str) -> re.Pattern:
    annex_words = {
        "EN": "ANNEX", "HU": "MELLÉKLET", "DE": "ANHANG", "FR": "ANNEXE",
        "PL": "ZAŁĄCZNIK", "IT": "ALLEGATO", "ES": "ANEXO", "NL": "BIJLAGE",
        "PT": "ANEXO", "RO": "ANEXA", "CS": "PŘÍLOHA", "SK": "PRÍLOHA",
        "SL": "PRILOGA", "HR": "PRILOG", "BG": "ПРИЛОЖЕНИЕ", "EL": "ΠΑΡΑΡΤΗΜΑ",
        "LT": "PRIEDAS", "LV": "PIELIKUMS", "ET": "LISA", "FI": "LIITE",
        "SV": "BILAGA", "DA": "BILAG", "MT": "ANNESS", "GA": "IARSCRÍBHINN",
    }
    word = annex_words.get(lang, "ANNEX")
    return re.compile(
        rf"(?m)^(?:{word}|Annex|ANNEX)\s+(I{{1,3}}|IV|V{{1,3}}|VI{{1,2}}|VII|VIII|IX|X|\d+)\b",
        re.IGNORECASE,
    )


def _split_annexes(text: str, pattern: re.Pattern) -> list[tuple[str, str, str]]:
    matches = list(pattern.finditer(text))
    if not matches:
        return []
    blocks = []
    for i, m in enumerate(matches):
        raw_id = m.group(1).strip()
        # Normalize to "Annex I", "Annex II", etc.
        ann_id = f"Annex {raw_id.upper()}"
        start = m.end()
        # Title: rest of heading line
        line_end = text.find("\n", start)
        title = text[start:line_end].strip() if line_end != -1 else ""
        body_start = line_end + 1 if line_end != -1 else start
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[body_start:body_end]
        blocks.append((ann_id, title, body))
    return blocks


def _clean_body(text: str) -> str:
    # Remove lines that look like page numbers / OJ headers
    text = re.sub(r"(?m)^[\s\d]+$", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ===========================================================================
# 2️⃣  TERM EXTRACTION
# ===========================================================================

def extract_terms(articles: list[dict], language_code: str) -> list[dict]:
    """
    Extract legally defined terms from definition articles.
    Returns list of term dicts.
    """
    lang = language_code.upper()

    # Definition trigger phrases by language
    defn_triggers: dict[str, list[str]] = {
        "EN": [r"means\b", r"shall mean\b", r"refers to\b", r"is defined as\b"],
        "HU": [r"azt jelenti\b", r"jelenti\b", r"alatt értendő\b", r"fogalmán\b.*érteni kell"],
        "DE": [r"bedeutet\b", r"bezeichnet\b", r"ist\b.*definiert als\b", r"versteht man\b"],
        "FR": [r"désigne\b", r"entend\b", r"s'entend\b", r"signifie\b"],
        "PL": [r"oznacza\b", r"rozumie się\b"],
        "_default": [r"means\b", r"refers to\b", r"shall mean\b"],
    }
    triggers = defn_triggers.get(lang, defn_triggers["_default"])
    trigger_re = re.compile("|".join(triggers), re.IGNORECASE)

    # Find definition article(s) — typically Art. 3 in EU AI Act
    def_articles = [
        a for a in articles
        if any(kw in (a.get("title") or "").lower()
               for kw in ["definition", "définition", "begriffsbestimmung",
                           "fogalommeghatározás", "definicj", "definizioni"])
    ]
    if not def_articles:
        # Fallback: search all articles
        def_articles = articles

    terms: list[dict] = []
    seen: set[str] = set()

    for art in def_articles:
        art_id = art["article_id"]
        text = art["text"]

        # Split into numbered sub-paragraphs
        # Pattern: lines starting with a number + period or parenthesis
        para_splits = re.split(r"(?m)(?=^\s*\(\d+\)|\s*\d+\.\s)", text)

        for para in para_splits:
            para = para.strip()
            if not para:
                continue

            # Look for: "term" means ...  or  ... means "term" ...
            # Try quoted term first
            m = re.search(
                r'[«""„\'"]([^«""„\'"]{2,60})[«""„\'"](?:[^a-z]|\s)?' +
                r"(?:" + "|".join(triggers) + r")\s+(.+?)(?=\n\n|\n\d+\.|\Z)",
                para, re.IGNORECASE | re.DOTALL
            )
            if m:
                term_text = m.group(1).strip()
                defn_text = m.group(2).strip()
            else:
                # Unquoted: "SomeNounPhrase means ..."
                m2 = re.search(
                    r"^([A-ZÁÉÍÓÖŐÚÜŰ][a-záéíóöőúüűA-ZÁÉÍÓÖŐÚÜŰ\s\-]{2,60}?)"
                    r"\s+(?:" + "|".join(triggers) + r")\s+(.+?)(?=\n\n|\n\d+\.|\Z)",
                    para, re.IGNORECASE | re.DOTALL
                )
                if m2:
                    term_text = m2.group(1).strip()
                    defn_text = m2.group(2).strip()
                else:
                    continue

            key = term_text.lower()
            if key in seen:
                continue
            seen.add(key)

            terms.append({
                "term": term_text,
                "definition": defn_text[:1000],  # cap for readability
                "defined_in": art_id,
                "language": lang,
            })

    log.info("Extracted %d defined terms [%s]", len(terms), lang)
    return terms


# ===========================================================================
# 3️⃣  OBLIGATION MINING
# ===========================================================================

def extract_obligations(articles: list[dict], language_code: str) -> list[dict]:
    """
    Extract obligation sentences from articles.
    Returns list of obligation dicts.
    """
    lang = language_code.upper()
    patterns = OBLIGATION_PATTERNS.get(lang, OBLIGATION_PATTERNS["_default"])
    combined_re = re.compile("|".join(patterns), re.IGNORECASE)

    obligations: list[dict] = []

    for art in articles:
        art_id = art["article_id"]  # e.g. "Art. 24"
        art_num = art_id.replace("Art. ", "")
        text = art["text"]

        # Split into sentences (rough but robust across languages)
        sentences = _split_sentences(text)

        obl_idx = 0
        for sent in sentences:
            sent = sent.strip()
            if len(sent) < 20:
                continue
            if not combined_re.search(sent):
                continue

            obl_idx += 1
            obl_id = f"obl_art{art_num}_{obl_idx:02d}"

            keywords = _extract_keywords(sent, lang)
            topic_tags = _infer_topics(sent)

            obligations.append({
                "obligation_id": obl_id,
                "source_article": art_id,
                "text": sent,
                "keywords": keywords,
                "topic_tags": topic_tags,
                "language": lang,
            })

    log.info("Extracted %d obligations [%s]", len(obligations), lang)
    return obligations


def _split_sentences(text: str) -> list[str]:
    """Heuristic sentence splitter that works across languages."""
    # Split on sentence-ending punctuation followed by whitespace + capital
    parts = re.split(r"(?<=[.;])\s+(?=[A-ZÁÉÍÓÖŐÚÜŰA-Z\u00C0-\u024F])", text)
    # Also split on newlines that introduce numbered paragraphs
    result = []
    for part in parts:
        sub = re.split(r"\n(?=\s*\(\d+\))", part)
        result.extend(sub)
    return result


def _extract_keywords(text: str, lang: str) -> list[str]:
    """Extract candidate keywords (simple noun-phrase heuristic)."""
    # Tokenize on word boundaries, keep tokens ≥ 4 chars
    words = re.findall(r"\b\w{4,}\b", text)
    # Deduplicate preserving order
    seen: set[str] = set()
    kws = []
    for w in words:
        lw = w.lower()
        if lw not in seen:
            seen.add(lw)
            kws.append(lw)
    return kws[:10]


def _infer_topics(text: str) -> list[str]:
    """Heuristic topic tagging based on keyword presence."""
    tag_map = {
        "documentation": ["document", "record", "log", "register", "dokumentáci",
                          "Dokumentation", "documentation", "registr"],
        "monitoring": ["monitor", "surveil", "oversight", "felügyel", "überwach",
                       "surveill"],
        "transparency": ["transparent", "inform", "disclose", "notif", "tájékoztat",
                         "Transparenz"],
        "risk_management": ["risk", "kockázat", "Risiko", "risque", "riesgo"],
        "conformity": ["conform", "compli", "megfelel", "Konformität", "conformité"],
        "training_data": ["training data", "training set", "tanítóadatok"],
        "human_oversight": ["human oversight", "emberi felügyelet", "menschliche Aufsicht"],
        "accuracy": ["accura", "robust", "pontos", "Genauigkeit"],
    }
    tags = []
    tl = text.lower()
    for tag, kws in tag_map.items():
        if any(kw.lower() in tl for kw in kws):
            tags.append(tag)
    return tags


# ===========================================================================
# 4️⃣  ROLE MAPPING
# ===========================================================================

def map_roles(
    obligations: list[dict],
    articles: list[dict],
    language_code: str,
) -> list[dict]:
    """
    Annotate each obligation with the legal roles it applies to.
    Returns obligations enriched with 'applies_to' and 'role_mentions_raw'.
    """
    lang = language_code.upper()

    # Build role surface-form index for this language
    role_index: dict[str, str] = {}  # surface_form_lower → canonical_role
    for canonical, lang_map in ROLE_VOCAB.items():
        forms = lang_map.get(lang, lang_map.get("_default", []))
        for form in forms:
            role_index[form.lower()] = canonical

    # Build article-title → roles lookup (for propagation)
    article_title_roles: dict[str, list[str]] = {}
    for art in articles:
        roles_found = _find_roles_in_text(
            (art.get("title") or ""), role_index
        )
        if roles_found:
            article_title_roles[art["article_id"]] = [r for r, _ in roles_found]

    enriched = []
    for obl in obligations:
        roles_in_text = _find_roles_in_text(obl["text"], role_index)
        raw_mentions = [raw for _, raw in roles_in_text]
        canonical_roles = list({r for r, _ in roles_in_text})

        # Propagate from article title if obligation has none
        if not canonical_roles:
            canonical_roles = article_title_roles.get(
                obl["source_article"], ["unspecified"]
            )

        enriched.append({
            **obl,
            "applies_to": canonical_roles,
            "role_mentions_raw": raw_mentions,
        })

    return enriched


def _find_roles_in_text(
    text: str, role_index: dict[str, str]
) -> list[tuple[str, str]]:
    """Return list of (canonical_role, raw_mention) pairs found in text."""
    results = []
    tl = text.lower()
    # Sort by length descending so multi-word matches take priority
    for surface, canonical in sorted(
        role_index.items(), key=lambda x: -len(x[0])
    ):
        if surface in tl:
            results.append((canonical, surface))
            # Replace matched span to avoid double-matching substrings
            tl = tl.replace(surface, " " * len(surface), 1)
    return results


# ===========================================================================
# 5️⃣  CROSS-REFERENCE EXTRACTION
# ===========================================================================

def extract_crossrefs(
    articles: list[dict],
    obligations: list[dict],
    language_code: str,
) -> list[dict]:
    """
    Extract cross-reference edges between articles, annexes, and obligations.
    Returns list of edge dicts.
    """
    lang = language_code.upper()
    patterns = CROSSREF_PATTERNS.get(lang, CROSSREF_PATTERNS["_default"])
    edges: list[dict] = []
    edge_set: set[tuple[str, str, str]] = set()  # dedup

    def add_edge(src: str, tgt: str, etype: str) -> None:
        key = (src, tgt, etype)
        if key not in edge_set:
            edge_set.add(key)
            edges.append({"source": src, "target": tgt, "type": etype})

    def parse_refs_from_text(text: str, source_id: str) -> None:
        for pat in patterns:
            for m in re.finditer(pat, text, re.IGNORECASE):
                full_match = m.group(0)
                groups = [g for g in m.groups() if g]

                # Annex reference
                if re.search(r"annex|melléklet|anhang|annexe|bijlage|allegato|"
                             r"prilog|liite|bilaga|iarscríbhinn|ΠΑΡΑΡΤΗΜΑ",
                             full_match, re.IGNORECASE):
                    ann_id = f"Annex {groups[0].upper()}"
                    add_edge(source_id, ann_id, "references")

                # Article range
                elif len(groups) == 2 and groups[0].isdigit() and groups[1].isdigit():
                    start_n, end_n = int(groups[0]), int(groups[1])
                    for n in range(start_n, end_n + 1):
                        tgt = f"Art. {n}"
                        if tgt != source_id:
                            add_edge(source_id, tgt, "references")

                # Single article
                elif groups and groups[0].isdigit():
                    tgt = f"Art. {groups[0]}"
                    if tgt != source_id:
                        add_edge(source_id, tgt, "references")

    # Article → Article / Annex
    for art in articles:
        parse_refs_from_text(art["text"], art["article_id"])

    # Obligation → Article (for refs within obligation sentences)
    for obl in obligations:
        parse_refs_from_text(obl["text"], obl["obligation_id"])

    log.info("Extracted %d cross-reference edges [%s]", len(edges), lang)
    return edges


# ===========================================================================
# 6️⃣  GRAPH BUILDER
# ===========================================================================

def build_graph(
    articles: list[dict],
    annexes: list[dict],
    terms: list[dict],
    obligations: list[dict],
    edges: list[dict],
    meta: dict,
) -> dict:
    """
    Assemble the final directed knowledge graph JSON.
    """
    nodes: dict[str, dict] = {}

    def add_node(node_id: str, node_type: str, label: str, **extra) -> None:
        if node_id not in nodes:
            nodes[node_id] = {"id": node_id, "type": node_type,
                              "label": label, **extra}

    # Article nodes
    for art in articles:
        add_node(art["article_id"], "Article",
                 label=art["title"] or art["article_id"],
                 title=art["title"])

    # Annex nodes
    for ann in annexes:
        add_node(ann["annex_id"], "Annex",
                 label=ann["title"] or ann["annex_id"])

    # Obligation nodes + Article→Obligation edges
    graph_edges: list[dict] = []
    obl_edge_set: set[tuple[str, str, str]] = set()

    def add_edge(src: str, tgt: str, etype: str) -> None:
        key = (src, tgt, etype)
        if key not in obl_edge_set:
            obl_edge_set.add(key)
            graph_edges.append({"source": src, "target": tgt, "type": etype})

    for obl in obligations:
        short_label = obl["text"][:80] + ("…" if len(obl["text"]) > 80 else "")
        add_node(obl["obligation_id"], "Obligation",
                 label=short_label,
                 full_text=obl["text"],
                 topic_tags=obl.get("topic_tags", []))

        add_edge(obl["source_article"], obl["obligation_id"], "imposes")

        # Role → Obligation edges
        for role in obl.get("applies_to", ["unspecified"]):
            add_node(role, "Role", label=role)
            add_edge(obl["obligation_id"], role, "applies_to")

    # Term nodes
    for term in terms:
        term_id = f"term_{_slugify(term['term'])}"
        add_node(term_id, "Term",
                 label=term["term"],
                 definition=term["definition"][:200])
        add_edge(term["defined_in"], term_id, "defines")

    # Cross-reference edges
    for e in edges:
        add_edge(e["source"], e["target"], e["type"])

    graph = {
        "meta": meta,
        "nodes": list(nodes.values()),
        "edges": graph_edges,
        "stats": {
            "n_articles": len(articles),
            "n_annexes": len(annexes),
            "n_obligations": len(obligations),
            "n_terms": len(terms),
            "n_edges": len(graph_edges),
        },
    }
    log.info(
        "Graph built: %d nodes, %d edges",
        len(nodes), len(graph_edges)
    )
    return graph


def _slugify(text: str) -> str:
    return re.sub(r"\W+", "_", text.lower()).strip("_")[:40]


# ===========================================================================
# FULL PIPELINE RUNNER
# ===========================================================================

def run_pipeline(language_code: str, force_download: bool = False) -> dict:
    """End-to-end pipeline. Returns graph JSON dict."""
    lang = language_code.upper()
    assert lang in SUPPORTED_LANGUAGES, f"Unsupported language: {lang}"

    raw_text, meta = download_ai_act(lang, force=force_download)
    articles, annexes = parse_articles(raw_text, lang)
    terms = extract_terms(articles, lang)
    obligations = extract_obligations(articles, lang)
    obligations = map_roles(obligations, articles, lang)
    edges = extract_crossrefs(articles, obligations, lang)
    graph = build_graph(articles, annexes, terms, obligations, edges, meta)
    return graph


# ===========================================================================
# STREAMLIT UI
# ===========================================================================

def streamlit_app() -> None:
    """Streamlit entry point."""
    st.set_page_config(
        page_title="EU AI Act Knowledge Graph",
        page_icon="⚖️",
        layout="wide",
    )
    st.title("⚖️ EU AI Act — Knowledge Graph Pipeline")
    st.caption(
        "Regulation (EU) 2024/1689 · Official EUR-Lex source · "
        "Multilingual parsing & graph extraction"
    )

    # --- Sidebar controls ---
    with st.sidebar:
        st.header("⚙️ Settings")
        lang = st.selectbox(
            "Document language",
            options=SUPPORTED_LANGUAGES,
            index=SUPPORTED_LANGUAGES.index("EN"),
        )
        force_dl = st.checkbox("Force re-download (ignore cache)", value=False)
        run_btn = st.button("▶ Run Pipeline", type="primary", use_container_width=True)

    if not run_btn:
        st.info(
            "Select a language and click **Run Pipeline** to begin. "
            "The document will be fetched directly from EUR-Lex."
        )
        return

    with st.spinner(f"Running pipeline for language: **{lang}**…"):
        try:
            graph = run_pipeline(lang, force_download=force_dl)
        except Exception as exc:
            st.error(f"Pipeline failed: {exc}")
            return

    meta = graph["meta"]
    stats = graph["stats"]

    # --- Metadata ---
    st.success("✅ Pipeline complete!")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Articles", stats["n_articles"])
    col2.metric("Annexes", stats["n_annexes"])
    col3.metric("Obligations", stats["n_obligations"])
    col4.metric("Terms", stats["n_terms"])
    col5.metric("Graph Edges", stats["n_edges"])

    with st.expander("📋 Document Metadata"):
        st.json(meta)

    # --- Graph JSON download ---
    graph_json_str = json.dumps(graph, ensure_ascii=False, indent=2)
    st.download_button(
        "⬇ Download Knowledge Graph JSON",
        data=graph_json_str,
        file_name=f"eu_ai_act_graph_{lang}.json",
        mime="application/json",
    )

    # --- Tabs for exploration ---
    tab_art, tab_obl, tab_terms, tab_graph = st.tabs(
        ["📜 Articles", "📌 Obligations", "🔤 Terms", "🕸 Graph Preview"]
    )

    nodes = {n["id"]: n for n in graph["nodes"]}
    articles_data = [n for n in graph["nodes"] if n["type"] == "Article"]
    obligations_data = [n for n in graph["nodes"] if n["type"] == "Obligation"]
    terms_data = [n for n in graph["nodes"] if n["type"] == "Term"]

    with tab_art:
        st.subheader(f"Articles ({len(articles_data)})")
        search = st.text_input("🔍 Filter articles", key="art_search")
        for a in articles_data:
            if search and search.lower() not in a["label"].lower():
                continue
            with st.expander(f"**{a['id']}** — {a['label']}"):
                st.write(a.get("full_text", a.get("label", "")))

    with tab_obl:
        st.subheader(f"Obligations ({len(obligations_data)})")
        search_o = st.text_input("🔍 Filter obligations", key="obl_search")
        for o in obligations_data:
            if search_o and search_o.lower() not in o["label"].lower():
                continue
            art_src = next(
                (e["source"] for e in graph["edges"]
                 if e["target"] == o["id"] and e["type"] == "imposes"),
                "?"
            )
            roles = [
                e["target"] for e in graph["edges"]
                if e["source"] == o["id"] and e["type"] == "applies_to"
            ]
            with st.expander(f"**{o['id']}** [{art_src}] → {', '.join(roles)}"):
                st.write(o.get("full_text", o["label"]))
                if o.get("topic_tags"):
                    st.caption("Topics: " + ", ".join(o["topic_tags"]))

    with tab_terms:
        st.subheader(f"Defined Terms ({len(terms_data)})")
        for t in terms_data:
            with st.expander(f"**{t['label']}**"):
                st.write(t.get("definition", ""))

    with tab_graph:
        st.subheader("Graph Preview")
        try:
            from pyvis.network import Network
            import streamlit.components.v1 as components

            try:
                net = Network(height="600px", width="100%", directed=True,
                              bgcolor="#1a1a2e", font_color="white",
                              cdn_resources="in_line")
            except TypeError:
                # pyvis < 0.3.0 fallback
                net = Network(height="600px", width="100%", directed=True,
                              bgcolor="#1a1a2e", font_color="white")

            COLOR_MAP = {
                "Article": "#4e9af1",
                "Obligation": "#f1a84e",
                "Role": "#4ef196",
                "Term": "#e74ef1",
                "Annex": "#f14e4e",
            }

            shown_nodes: set[str] = set()
            # Show at most 200 nodes for performance
            for n in graph["nodes"][:200]:
                color = COLOR_MAP.get(n["type"], "#cccccc")
                net.add_node(n["id"], label=n["label"][:40],
                             color=color, title=n.get("full_text", n["label"]))
                shown_nodes.add(n["id"])

            for e in graph["edges"]:
                if e["source"] in shown_nodes and e["target"] in shown_nodes:
                    net.add_edge(e["source"], e["target"], title=e["type"])

            net.set_options("""{"physics":{"stabilization":{"iterations":100}}}""")
            try:
                html_str = net.generate_html(notebook=False)
            except TypeError:
                html_str = net.generate_html()
            if not html_str:
                import tempfile, os
                with tempfile.NamedTemporaryFile(suffix=".html", delete=False,
                                                mode="w", encoding="utf-8") as f:
                    net.save_graph(f.name)
                    tmp = f.name
                with open(tmp, encoding="utf-8") as f:
                    html_str = f.read()
                os.unlink(tmp)
            components.html(html_str, height=620, scrolling=True)
        except ImportError:
            st.info(
                "Install `pyvis` for interactive graph visualization: "
                "`pip install pyvis`\n\nShowing raw edge list instead:"
            )
            st.dataframe(
                [{"from": e["source"], "to": e["target"], "type": e["type"]}
                 for e in graph["edges"][:100]],
                use_container_width=True,
            )


# ===========================================================================
# CLI ENTRY POINT
# ===========================================================================

def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="EU AI Act Knowledge Graph Pipeline (CLI)"
    )
    parser.add_argument("--lang", default="EN",
                        help=f"Language code. Supported: {SUPPORTED_LANGUAGES}")
    parser.add_argument("--output", default=None,
                        help="Output JSON file path (default: stdout)")
    parser.add_argument("--force", action="store_true",
                        help="Force re-download, ignore cache")
    args = parser.parse_args()

    graph = run_pipeline(args.lang, force_download=args.force)

    graph_json = json.dumps(graph, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(graph_json, encoding="utf-8")
        print(f"Graph written to: {args.output}")
    else:
        # Print excerpt to stdout
        excerpt = {
            "meta": graph["meta"],
            "stats": graph["stats"],
            "nodes_sample": graph["nodes"][:5],
            "edges_sample": graph["edges"][:10],
        }
        print(json.dumps(excerpt, ensure_ascii=False, indent=2))


# ===========================================================================
# MAIN
# ===========================================================================

if __name__ == "__main__":
    if _STREAMLIT:
        streamlit_app()
    else:
        _cli()
