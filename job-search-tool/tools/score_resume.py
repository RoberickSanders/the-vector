"""ATS-style keyword score for a tailored resume PDF.

Two extraction modes:
  - LLM mode (default when ANTHROPIC_API_KEY is available): Claude extracts
    skill / tech / methodology keywords from the JD. Filters generic English,
    company / customer names, JD boilerplate, contractions automatically.
  - Heuristic fallback (--no-llm or no API key): TF-IDF-style frequency
    analysis on tokens minus stopwords + n-grams. Faster, free, but noisier.

Both modes feed the same matcher + scorer.

Score = matched / total * 100.

Designed as a ship-gate. Exit 0 when score >= threshold, exit 1 below.
Lets shell scripts gate uploads on a passing score.

CLI:
    .venv/bin/python -m tools.score_resume --pdf <path> --company "Buildkite"
    .venv/bin/python -m tools.score_resume --pdf <path> --jd-file <path>
    .venv/bin/python -m tools.score_resume --pdf <path> --jd-file <path> \\
        --threshold 70
    .venv/bin/python -m tools.score_resume --pdf <path> --company X --no-llm
"""
import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

import pandas as pd
from anthropic import Anthropic
from dotenv import load_dotenv
from pypdf import PdfReader

from tools.profile import load_profile

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"

# Auto-load workspace .env so ANTHROPIC_API_KEY is available.
WORKSPACE_ENV = PROJECT_ROOT.parent.parent.parent / ".env"
if WORKSPACE_ENV.exists():
    load_dotenv(WORKSPACE_ENV, override=True)  # override Claude Code's empty-string exports

DEFAULT_THRESHOLD = 70
MAX_KEYWORDS = 30
LLM_MODEL = "claude-sonnet-4-6"

# ~50 common English stopwords. We keep this list short and inline so we
# don't drag in nltk just for tokenization.
STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "if", "while", "with", "without",
    "to", "of", "in", "on", "at", "by", "for", "from", "as", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "do", "does",
    "did", "this", "that", "these", "those", "it", "its", "they", "them",
    "their", "there", "than", "then", "so", "such", "into", "out", "up",
    "down", "over", "under", "about", "across", "after", "before",
})

# Domain-noise terms that show up in nearly every JD and don't discriminate.
JD_NOISE = frozenset({
    "looking", "experience", "role", "team", "you", "we", "our", "your",
    "company", "build", "work", "time", "year", "years", "job", "position",
    "candidate", "candidates", "skills", "responsibilities", "requirements",
    "qualifications", "us", "will", "can", "able", "must",
})


def extract_pdf_text(pdf_path: Path) -> str:
    """Concatenate text from every page of a PDF.

    Returns "" on read failure (corrupt PDF, no text layer, etc.) — the
    caller's score will then drop to 0 and the ship-gate will reject.
    """
    try:
        reader = PdfReader(str(pdf_path))
    except Exception:
        return ""
    parts: list[str] = []
    for page in reader.pages:
        try:
            parts.append(page.extract_text() or "")
        except Exception:
            continue
    return "\n".join(parts)


def _tokenize(text: str) -> list[str]:
    """Lowercase + tokenize on word boundaries. Keeps alphanumerics + hyphens."""
    return re.findall(r"[a-z0-9][a-z0-9\-]+", text.lower())


def _is_useful(token: str) -> bool:
    """Drop stopwords, JD noise, very-short tokens, pure numbers."""
    if len(token) < 3:
        return False
    if token in STOPWORDS:
        return False
    if token in JD_NOISE:
        return False
    if token.isdigit():
        return False
    return True


def _ngram_phrases(tokens: list[str], n: int, min_count: int = 2) -> Counter:
    """Build n-gram phrases that appear at least min_count times.

    Skips:
      - n-grams that begin or end with a stopword (phrase noise like "with the team")
      - n-grams whose first/last tokens are JD noise ("looking for the role")
      - pure-repetition n-grams ("agentic agentic agentic") — those are
        always artifacts of repeated unigrams in the source text
    """
    grams: Counter = Counter()
    if len(tokens) < n:
        return grams
    for i in range(len(tokens) - n + 1):
        chunk = tokens[i:i + n]
        if chunk[0] in STOPWORDS or chunk[-1] in STOPWORDS:
            continue
        if chunk[0] in JD_NOISE or chunk[-1] in JD_NOISE:
            continue
        if not any(_is_useful(t) for t in chunk):
            continue
        # Drop pure-repetition n-grams: every token is the same word.
        if len(set(chunk)) == 1:
            continue
        phrase = " ".join(chunk)
        grams[phrase] += 1
    # Drop anything that didn't repeat — ad-hoc phrases aren't keyword candidates.
    return Counter({p: c for p, c in grams.items() if c >= min_count})


# ---------- LLM-based keyword extraction ----------

LLM_KEYWORD_PROMPT = """You are extracting ATS-relevant keywords from a job description for resume-matching.

Return ONLY skills, technologies, methodologies, frameworks, tools, and concepts that a candidate's resume should contain to match this role.

DO NOT return:
- Generic English words (how, what, where, build, work, time, team, role, looking)
- Company names or customer / logo names mentioned in the JD
- Boilerplate phrases ("at [Company]", "we value", "looking for", "About [Company]")
- Email addresses, URLs, accommodations contacts
- Filler verbs without object (helps, makes, creates, builds)
- Contractions or partial tokens ("ve built", "ll need")

DO return:
- Specific skills (Python, SQL, TypeScript, agentic workflows)
- Specific tools (Clay, Salesforce, Smartlead, Customer.io, Metaflow)
- Methodologies (segmentation framework, quality gate, ABM, intent systems)
- Domain concepts (deliverability, suppression discipline, lifecycle modeling)
- Multi-word phrases that are field-of-art terms (orchestration engine, stage plays, data foundation)

Return the {n} most-important keywords for ATS matching, ranked by importance to the role.

Job description:
---
{jd}
---

Return STRICT JSON (no prose, no markdown fences):
{{"keywords": ["keyword1", "keyword2", ...]}}
"""


def make_anthropic_client() -> Optional[Anthropic]:
    """Build an Anthropic client. Returns None if no key (caller falls back)."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return None
    return Anthropic(api_key=key)


def extract_keywords_via_llm(
    jd: str,
    client: Anthropic,
    max_keywords: int = MAX_KEYWORDS,
) -> list[str]:
    """Use Claude to extract skill / tech / methodology keywords from a JD.

    Filters generic English, company names, JD boilerplate, contractions
    automatically — far higher signal than the heuristic frequency-based
    extractor.

    Returns lowercased keywords for case-insensitive matching. Returns
    empty list on API error or unparseable response (caller can fall back).
    """
    prompt = LLM_KEYWORD_PROMPT.format(jd=jd[:12000], n=max_keywords)
    try:
        resp = client.messages.create(
            model=LLM_MODEL,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as e:
        print(f"  LLM keyword extraction failed: {str(e)[:200]}", file=sys.stderr)
        return []

    raw = resp.content[0].text if resp.content else ""
    if not raw:
        return []
    # Strip optional markdown fences
    raw = re.sub(r"^```(?:json)?\n?", "", raw.strip())
    raw = re.sub(r"\n?```$", "", raw).strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        print(f"  LLM returned non-JSON: {raw[:200]}", file=sys.stderr)
        return []

    keywords = parsed.get("keywords", [])
    if not isinstance(keywords, list):
        return []

    # Lowercase + dedupe + cap
    seen: set[str] = set()
    out: list[str] = []
    for kw in keywords:
        if not isinstance(kw, str):
            continue
        kw_lc = kw.strip().lower()
        if not kw_lc or kw_lc in seen:
            continue
        seen.add(kw_lc)
        out.append(kw_lc)
        if len(out) >= max_keywords:
            break
    return out


# ---------- Heuristic-based keyword extraction (fallback) ----------


def extract_keywords(jd: str, max_keywords: int = MAX_KEYWORDS) -> list[str]:
    """Return up to `max_keywords` candidate keywords ranked by importance.

    Strategy:
      1. tokenize + filter stopwords / noise
      2. score unigrams by raw frequency, bigrams by count*1.5, trigrams by count*2
      3. dedupe-by-coverage: a shorter phrase is dropped if any phrase in the
         pool fully contains it as a whole-word substring (so "agentic" loses
         to "agentic workflows" even when the unigram has higher raw count)
      4. sort survivors by score desc, take top N

    Phrases are returned lower-cased; the matcher is case-insensitive.
    """
    tokens = _tokenize(jd)
    if not tokens:
        return []
    useful = [t for t in tokens if _is_useful(t)]

    unigram_counts = Counter(useful)
    bigrams = _ngram_phrases(tokens, 2, min_count=2)
    trigrams = _ngram_phrases(tokens, 3, min_count=2)

    # Score: unigrams = count; bigrams = count*1.5; trigrams = count*2.
    # The length bonus surfaces multi-word phrases like "segmentation framework"
    # ahead of a same-count single token like "agentic".
    scored: dict[str, float] = {}
    for tok, c in unigram_counts.items():
        scored[tok] = float(c)
    for ph, c in bigrams.items():
        scored[ph] = c * 1.5
    for ph, c in trigrams.items():
        scored[ph] = c * 2.0

    # Coverage-dedupe: drop a phrase that's a strict whole-word substring of
    # another phrase in the pool — but only if the covering phrase has all-
    # distinct tokens (so a fragment like "agentic workflows agentic" with a
    # repeated word can't kill the real signal "agentic workflows").
    # Order-independent; handles the "agentic" vs "agentic workflows" case
    # where the unigram out-scores the bigram in raw count.
    def _covers(short: str, long: str) -> bool:
        if not _is_subphrase(short, long):
            return False
        long_tokens = long.split()
        return len(set(long_tokens)) == len(long_tokens)

    all_phrases = list(scored.keys())
    survivors: dict[str, float] = {}
    for phrase, score in scored.items():
        covered = any(_covers(phrase, other) for other in all_phrases)
        if not covered:
            survivors[phrase] = score

    # Sort by score desc, then alphabetically for stable output.
    ranked = sorted(survivors.items(), key=lambda kv: (-kv[1], kv[0]))
    return [phrase for phrase, _ in ranked[:max_keywords]]


def _is_subphrase(short: str, long: str) -> bool:
    """True if `short` is a whole-word substring of `long` and short != long."""
    if short == long:
        return False
    pattern = r"(?:^|\s)" + re.escape(short) + r"(?:$|\s)"
    return re.search(pattern, long) is not None


def _tokenize_for_match(text: str) -> set[str]:
    """Lowercase + extract word/number tokens. Used for token-level matching.

    Splits on whitespace, slashes, dots, hyphens, parens, etc. so multi-word
    inputs like 'intent systems' tokenize to {'intent', 'systems'} for an
    'are all halves present in the haystack?' check.
    """
    return set(re.findall(r"[a-z0-9]+", text.lower()))


# Brand-style punctuation. A keyword containing any of these chars gets a
# strict (exact-substring) match instead of token-level — otherwise tokens
# like 'io' from 'Customer.io' would match against unrelated 'Gong.io'.
_BRAND_PUNCTUATION = (".", "+", "#")


def _is_brand_keyword(keyword: str) -> bool:
    """True if keyword should require strict substring match (brand name).

    Brand markers: '.' (Customer.io, reo.dev, .NET), '+' (C++, A+), '#' (C#).
    Everything else gets token-level matching for phrasing-variation tolerance.
    """
    return any(c in keyword for c in _BRAND_PUNCTUATION)


def match_keywords(
    keywords: list[str],
    pdf_text: str,
    strict: bool = False,
) -> tuple[list[str], list[str]]:
    """Return (matched, missing).

    Default (token-level for non-brand, strict for brand): a keyword matches
    when EVERY word-token appears in the PDF. Handles slight phrasing variation
    between JD and resume — e.g., LLM 'intent systems' matches resume's
    'intent/ABM systems'; LLM 'crm architecture' matches 'CRM data
    architecture'.

    Brand-style keywords (containing '.', '+', or '#') require a strict
    contiguous-substring match. This catches the failure mode where token
    matching would mark 'Customer.io' as matched against a resume containing
    'Gong.io' (sharing the 'io' token). Brand identity requires the full
    name as a unit.

    Strict mode (for substring-style ATS simulators): force ALL keywords to
    require contiguous substring matches. More conservative; produces lower
    scores; catches some real ATS behavior but generates many false
    negatives on legitimate phrasing variation.
    """
    haystack_lower = pdf_text.lower()
    haystack_tokens = _tokenize_for_match(pdf_text)
    matched: list[str] = []
    missing: list[str] = []
    for kw in keywords:
        kw_lower = kw.lower()
        # Strict: fully contiguous substring required.
        if strict or _is_brand_keyword(kw):
            if kw_lower in haystack_lower:
                matched.append(kw)
            else:
                missing.append(kw)
            continue
        # Token-level: every kw token must appear in haystack tokens.
        kw_tokens = _tokenize_for_match(kw)
        if not kw_tokens:
            missing.append(kw)
            continue
        if kw_tokens.issubset(haystack_tokens):
            matched.append(kw)
        else:
            missing.append(kw)
    return matched, missing


def compute_score(matched: int, total: int) -> int:
    if total == 0:
        return 0
    return round(matched / total * 100)


def find_company_jd_for_score(
    master_file: Path,
    company: str,
) -> Optional[str]:
    """Look up a company's JD in the master Excel.

    Lighter-weight twin of tools.tailor.find_company_jd: just company match
    (no role filter), pick highest-fit_score row.
    """
    if not master_file.exists():
        return None
    try:
        df = pd.read_excel(master_file)
    except Exception:
        return None
    if df.empty or "company" not in df.columns:
        return None

    company_col = df["company"].astype(str).str.lower()
    matches = df[company_col.str.contains(company.lower(), na=False, regex=False)]
    if matches.empty:
        return None
    if "fit_score" in matches.columns and matches["fit_score"].notna().any():
        ranked = matches[matches["fit_score"].notna()].sort_values(
            "fit_score", ascending=False
        )
        row = ranked.iloc[0]
    else:
        row = matches.iloc[0]
    if "description" not in row.index:
        return None
    desc = row.get("description")
    if pd.isna(desc):
        return None
    desc = str(desc).strip()
    return desc or None


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="ATS-style keyword score for a tailored resume PDF."
    )
    parser.add_argument("--pdf", required=True, help="Path to the resume PDF.")
    parser.add_argument(
        "--company",
        default=None,
        help="Company name — looks up JD in master Excel.",
    )
    parser.add_argument(
        "--jd-file",
        default=None,
        help="Read JD from this file. Overrides --company.",
    )
    parser.add_argument("--profile", default="example")
    parser.add_argument("--threshold", type=int, default=DEFAULT_THRESHOLD)
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Use heuristic keyword extractor instead of Claude API "
             "(faster, free, but noisier — generic words and JD boilerplate "
             "leak into the keyword set).",
    )
    parser.add_argument(
        "--strict-match",
        action="store_true",
        help="Require keywords to appear as contiguous substrings in the "
             "PDF (legacy behavior). Default is token-level matching — "
             "all word-tokens of the keyword must appear, not necessarily "
             "adjacent. Token-level catches 'intent systems' against a "
             "resume that has 'intent/ABM systems', etc.",
    )
    args = parser.parse_args(argv)

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"PDF not found: {pdf_path}", file=sys.stderr)
        return 2

    # Resolve JD
    if args.jd_file:
        jd_path = Path(args.jd_file)
        if not jd_path.exists():
            print(f"--jd-file not found: {jd_path}", file=sys.stderr)
            return 2
        jd = jd_path.read_text().strip()
    elif args.company:
        profile = load_profile(PROJECT_ROOT / "profiles" / f"{args.profile}.yaml")
        master_file = OUTPUT_DIR / profile.drive.master_filename
        jd = find_company_jd_for_score(master_file, args.company) or ""
    else:
        print("Pass --company or --jd-file.", file=sys.stderr)
        return 2

    if not jd or len(jd) < 50:
        print("JD missing or too short.", file=sys.stderr)
        return 2

    pdf_text = extract_pdf_text(pdf_path)
    if not pdf_text.strip():
        print(
            f"Could not extract text from {pdf_path} — corrupt PDF or no text layer.",
            file=sys.stderr,
        )
        return 1

    # Try LLM extraction first (default); fall back to heuristic when
    # disabled, no API key, or LLM error.
    keywords: list[str] = []
    extractor_used = "heuristic"
    if not args.no_llm:
        client = make_anthropic_client()
        if client is not None:
            llm_keywords = extract_keywords_via_llm(jd, client)
            if llm_keywords:
                keywords = llm_keywords
                extractor_used = "llm"
    if not keywords:
        keywords = extract_keywords(jd)

    matched, missing = match_keywords(keywords, pdf_text, strict=args.strict_match)
    score = compute_score(len(matched), len(keywords))

    match_mode = "strict" if args.strict_match else "token-level"
    print(f"Score: {score}/100  (extractor: {extractor_used}, match: {match_mode})")
    print(f"\nMatched ({len(matched)}): {', '.join(matched) if matched else '(none)'}")
    print(f"\nMissing ({len(missing)}): {', '.join(missing) if missing else '(none)'}")
    print()
    if score >= args.threshold:
        print(f"Ship gate: PASS (>= {args.threshold} threshold)")
        return 0
    print(f"Ship gate: BELOW THRESHOLD (< {args.threshold})")
    return 1


if __name__ == "__main__":
    sys.exit(main())
