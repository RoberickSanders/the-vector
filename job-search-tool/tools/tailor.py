"""JD-aware resume tailoring.

Pulls a job description (from --jd-file or by company-lookup in the master
Excel), parses it into a structured priority-ranked requirements list (so
high-priority asks surface achievements first), asks Claude to identify JD
keywords missing from resume.yaml, gates each suggestion against the
proof-point library in positioning.md (no fabrication), applies surviving
rewrites in-memory, writes a variant YAML, and renders a per-application
PDF via the rendercv CLI.

Output convention: every application gets a flat-file pair under
`the-vector/applications/`:
    applications/<slug>.md   (the package — written by the human)
    applications/<slug>.pdf  (the tailored resume — written by this tool)
where <slug> = "<company>-<role>" lower-kebab, or just "<company>" when role
is omitted. Mirrors existing buildkite-staff-gtm-engineer.md / .pdf pair.

Variant YAML lives one level above the project root (alongside resume.yaml)
so rendercv resolves any sibling design / locale paths the same way the
base resume does.

CLI:
    .venv/bin/python -m tools.tailor --profile example --company "Buildkite"
    .venv/bin/python -m tools.tailor --profile example --company "Buildkite" \\
        --role "Staff GTM Engineer"
    .venv/bin/python -m tools.tailor --profile example --company "Buildkite" \\
        --jd-file path/to/jd.txt
"""
import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml
from anthropic import Anthropic
from dotenv import load_dotenv

from tools.profile import load_profile

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
UMBRELLA_ROOT = PROJECT_ROOT.parent  # the-vector/
APPLICATIONS_DIR = UMBRELLA_ROOT / "applications"
RESUME_YAML_PATH = UMBRELLA_ROOT / "resume.yaml"
POSITIONING_MD_PATH = UMBRELLA_ROOT / "positioning.md"

# Auto-load workspace .env so ANTHROPIC_API_KEY is available.
# Mirrors the pattern in tools/find_managers.py and tools/score.py.
WORKSPACE_ENV = PROJECT_ROOT.parent.parent.parent / ".env"
if WORKSPACE_ENV.exists():
    load_dotenv(WORKSPACE_ENV, override=True)  # override empty strings exported by Claude Code

CLAUDE_MODEL = "claude-sonnet-4-6"

logger = logging.getLogger(__name__)

# Anchor markers in positioning.md. The proof-point library is a markdown
# section between "## Proof-point library" and the next "## " heading.
PROOF_POINT_HEADING = "## Proof-point library"


# ---------- JD priority ranking ----------
#
# Inspired by javiera-vasquez/claude-code-job-tailor: BEFORE we ask Claude
# to extract missing keywords, we structure the JD into a ranked list of
# requirements so high-priority asks (mentioned often, early, with strong
# language like "required") surface their matching achievements first.
#
# This is a deterministic heuristic ranker (no LLM call). It is cheap,
# debuggable, and unit-testable. Output is then passed into the tailor
# prompt as additional structured context so the model can use it to bias
# what it suggests.

# Categories
CATEGORY_MUST_HAVE = "must-have"
CATEGORY_NICE_TO_HAVE = "nice-to-have"
CATEGORY_RED_HERRING = "red-herring"

# Priority levels
PRIORITY_HIGH = "high"
PRIORITY_MEDIUM = "medium"
PRIORITY_LOW = "low"

# Phrases that signal a hard requirement. Order: longest-first so we don't
# pre-empt a longer match (e.g. "must have" before "have").
MUST_HAVE_PHRASES = (
    "is required",
    "are required",
    "must have",
    "must be able",
    "required:",
    "requirement:",
    "you must",
    "minimum qualifications",
    "minimum requirements",
    "essential:",
    "essential ",
    "required ",
)

NICE_TO_HAVE_PHRASES = (
    "preferred:",
    "preferred ",
    "nice to have",
    "nice-to-have",
    "bonus:",
    "bonus ",
    "plus:",
    "a plus",
    "ideal candidate",
    "ideally",
    "would be helpful",
    "helpful:",
)

# Generic filler that hiring managers paste in but rarely actually screen
# on. We tag these red-herring so the resume tailor doesn't burn rewrite
# budget chasing them.
RED_HERRING_PATTERNS = (
    "passion for",
    "self-starter",
    "team player",
    "work in a fast-paced",
    "fast-paced environment",
    "wear many hats",
    "rockstar",
    "ninja",
)


@dataclass
class RankedRequirement:
    """One JD requirement with a category + priority + signal trail.

    `signals` is a list of one-line strings explaining why we ranked this
    way (used in the tailor.py log so the ranking is debuggable).
    """
    text: str
    category: str
    priority: str
    score: float = 0.0
    frequency: int = 1
    signals: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "category": self.category,
            "priority": self.priority,
            "score": round(self.score, 2),
            "frequency": self.frequency,
            "signals": list(self.signals),
        }


def _split_jd_paragraphs(jd: str) -> list[str]:
    """Split a JD into rough paragraphs (blank-line delimited).

    Falls back to single-line splitting when no blank lines are present
    (common in scraped JDs from LinkedIn that come as one wall of text).
    """
    if not jd:
        return []
    # Normalize Windows line endings.
    text = jd.replace("\r\n", "\n").replace("\r", "\n")
    # Try blank-line paragraphs first.
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(paras) >= 2:
        return paras
    # Fall back to single-line bullets/sentences (LinkedIn-style walls).
    return [ln.strip() for ln in text.split("\n") if ln.strip()]


def _extract_bullet_items(jd: str) -> list[tuple[str, int, str]]:
    """Pull bullet-like lines out of a JD.

    Returns tuples of (text, line_index, surrounding_paragraph). Line_index
    is the position in the source JD's line list (earlier lines get a
    position bonus).
    """
    if not jd:
        return []
    text = jd.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")
    out: list[tuple[str, int, str]] = []
    bullet_re = re.compile(r"^\s*(?:[-*•·]+|\d+[.)])\s+(.+?)\s*$")
    for i, ln in enumerate(lines):
        m = bullet_re.match(ln)
        if m:
            body = m.group(1).strip()
            if 4 <= len(body) <= 240:
                # surrounding context = 1 line before for category hints
                ctx_start = max(0, i - 2)
                ctx = "\n".join(lines[ctx_start:i + 1]).lower()
                out.append((body, i, ctx))
    # If JD has no bullets, fall back to short sentences inside paragraphs.
    if not out:
        for p_i, para in enumerate(_split_jd_paragraphs(jd)):
            for sent in re.split(r"(?<=[.!?])\s+", para):
                sent = sent.strip()
                if 8 <= len(sent) <= 240:
                    out.append((sent, p_i, para.lower()))
    return out


def _category_for(item_text: str, surrounding_lower: str) -> str:
    """Decide must-have / nice-to-have / red-herring for one bullet.

    Uses the bullet text AND the immediately-preceding context (so a
    bullet directly under a 'Required:' header inherits must-have).
    """
    item_lower = item_text.lower()

    for phrase in RED_HERRING_PATTERNS:
        if phrase in item_lower:
            return CATEGORY_RED_HERRING

    # Surrounding context (heading directly above bullets) wins, but a
    # bullet that says "preferred" overrides a "Required:" heading.
    for phrase in NICE_TO_HAVE_PHRASES:
        if phrase in item_lower:
            return CATEGORY_NICE_TO_HAVE
    for phrase in MUST_HAVE_PHRASES:
        if phrase in item_lower:
            return CATEGORY_MUST_HAVE

    # Headings above the bullet
    for phrase in NICE_TO_HAVE_PHRASES:
        if phrase in surrounding_lower:
            return CATEGORY_NICE_TO_HAVE
    for phrase in MUST_HAVE_PHRASES:
        if phrase in surrounding_lower:
            return CATEGORY_MUST_HAVE

    # Default: treat as must-have. Most JDs front-load actual requirements
    # without explicit "Required:" tags.
    return CATEGORY_MUST_HAVE


def _normalize_for_dedup(text: str) -> str:
    """Lowercase + collapse whitespace + strip punctuation for fuzzy dedup."""
    s = text.lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _title_or_first_para(jd: str) -> str:
    """The first non-empty paragraph (often title + summary). Lowercased."""
    paras = _split_jd_paragraphs(jd)
    if not paras:
        return ""
    return paras[0].lower()


def parse_jd_requirements(jd: str) -> list[RankedRequirement]:
    """Parse a JD into a deduped, categorized list of RankedRequirements.

    No priority ranking happens here (see `rank_requirements` below).
    Frequency in the JD is captured for the ranker.
    """
    items = _extract_bullet_items(jd)
    if not items:
        return []

    # Group by normalized form so duplicates ("Strong Python" appearing
    # twice) collapse and bump frequency.
    grouped: dict[str, RankedRequirement] = {}
    line_pos: dict[str, int] = {}
    for raw_text, line_idx, ctx in items:
        key = _normalize_for_dedup(raw_text)
        if not key:
            continue
        category = _category_for(raw_text, ctx)
        if key in grouped:
            grouped[key].frequency += 1
            # Preserve the earliest position (lower = better).
            line_pos[key] = min(line_pos[key], line_idx)
        else:
            grouped[key] = RankedRequirement(
                text=raw_text,
                category=category,
                priority=PRIORITY_LOW,
                frequency=1,
            )
            line_pos[key] = line_idx

    out: list[RankedRequirement] = []
    for key, req in grouped.items():
        # Stash position for the ranker (private attr, not serialized).
        req.signals.append(f"line:{line_pos[key]}")
        out.append(req)
    return out


def rank_requirements(
    requirements: list[RankedRequirement],
    jd: str,
) -> list[RankedRequirement]:
    """Score each requirement and assign priority high/medium/low.

    Score factors:
      - Category: must-have +3, nice-to-have +1, red-herring -2
      - Frequency: +1 per repeat (cap +3)
      - Position: bonus if appears in title/first paragraph (+2) or in
        the first half of the JD bullets (+1)
      - Strong language: bumps already handled via category

    Top-tercile -> high, middle -> medium, bottom -> low. Red-herrings are
    always low regardless of score.
    """
    if not requirements:
        return []

    title_lower = _title_or_first_para(jd)
    total = len(requirements)

    # Compute scores
    for req in requirements:
        score = 0.0

        # Category contribution
        if req.category == CATEGORY_MUST_HAVE:
            score += 3.0
            req.signals.append("category:must-have+3")
        elif req.category == CATEGORY_NICE_TO_HAVE:
            score += 1.0
            req.signals.append("category:nice-to-have+1")
        else:  # red-herring
            score -= 2.0
            req.signals.append("category:red-herring-2")

        # Frequency contribution (capped at +3)
        if req.frequency > 1:
            bump = min(req.frequency - 1, 3)
            score += bump
            req.signals.append(f"frequency:{req.frequency}+{bump}")

        # Position contribution: extract the line:X signal we stashed earlier
        line_pos = 9999
        for sig in req.signals:
            if sig.startswith("line:"):
                try:
                    line_pos = int(sig.split(":", 1)[1])
                except ValueError:
                    pass
                break

        # Title/first-paragraph bonus
        body_lower = _normalize_for_dedup(req.text)
        if body_lower and body_lower[:25] and body_lower[:25] in title_lower:
            score += 2.0
            req.signals.append("position:in-title+2")
        elif line_pos < 8:
            score += 1.5
            req.signals.append("position:top-of-jd+1.5")
        elif line_pos < total // 2:
            score += 0.5
            req.signals.append("position:upper-half+0.5")

        req.score = score

    # Assign priority from score buckets.
    sorted_reqs = sorted(requirements, key=lambda r: r.score, reverse=True)
    n = len(sorted_reqs)
    high_cut = max(1, n // 3)
    mid_cut = max(2, (2 * n) // 3)
    for i, req in enumerate(sorted_reqs):
        if req.category == CATEGORY_RED_HERRING:
            req.priority = PRIORITY_LOW
            continue
        if i < high_cut:
            req.priority = PRIORITY_HIGH
        elif i < mid_cut:
            req.priority = PRIORITY_MEDIUM
        else:
            req.priority = PRIORITY_LOW

    return sorted_reqs


def format_ranked_requirements(ranked: list[RankedRequirement]) -> str:
    """Pretty multi-line block for log output and prompt context.

    Format:
        [HIGH | must-have | freq=2 | score=5.5] Strong Python proficiency
            signals: category:must-have+3, frequency:2+1, position:top-of-jd+1.5

    Used both for the tailor.py CLI log and for the prompt context block
    sent to Claude (so the model can see the ranking).
    """
    if not ranked:
        return "(no structured requirements parsed)"
    lines = []
    for r in ranked:
        head = (
            f"[{r.priority.upper()} | {r.category} | "
            f"freq={r.frequency} | score={r.score:.1f}] {r.text}"
        )
        lines.append(head)
        if r.signals:
            lines.append(f"    signals: {', '.join(r.signals)}")
    return "\n".join(lines)


def slugify(company: str, role: Optional[str]) -> str:
    """company='Buildkite', role='Staff GTM Engineer' -> 'buildkite-staff-gtm-engineer'.

    Lowercases, replaces any run of non-[a-z0-9-] with a single hyphen,
    collapses repeats, strips trailing/leading hyphens.
    """
    parts = [company]
    if role:
        parts.append(role)
    s = "-".join(parts).lower()
    s = re.sub(r"[^a-z0-9-]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s


def find_company_jd(
    master_file: Path,
    company: str,
    role: Optional[str] = None,
) -> Optional[str]:
    """Look up a company's JD in the master Excel.

    Filter rules:
      - case-insensitive substring match on `company` column
      - if `role` given, also require case-insensitive substring on `title`
      - among matches, pick the highest fit_score with non-null fit_score
        (falls back to first match when no fit_score column or all null)

    Returns the row's `description` column as the JD text, or None if no
    matching row was found / no description present.
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
    mask = company_col.str.contains(company.lower(), na=False, regex=False)
    if role and "title" in df.columns:
        title_col = df["title"].astype(str).str.lower()
        mask = mask & title_col.str.contains(role.lower(), na=False, regex=False)

    matches = df[mask]
    if matches.empty:
        return None

    # Pick highest fit_score among matches (non-null), else first.
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


def extract_proof_point_section(positioning_md: str) -> str:
    """Slice positioning.md from `## Proof-point library` to the next `## `.

    Returns the heading + the body of the section. If the heading isn't
    present, returns the entire file (degraded but safe — Claude still has
    enough signal to reject fabrications).
    """
    idx = positioning_md.find(PROOF_POINT_HEADING)
    if idx < 0:
        return positioning_md
    rest = positioning_md[idx:]
    # Find the next "## " heading after this one's body. The heading itself
    # starts with "## ", so look from after the first newline to skip it.
    body_start = rest.find("\n", len(PROOF_POINT_HEADING))
    if body_start < 0:
        return rest
    next_heading = rest.find("\n## ", body_start)
    if next_heading < 0:
        return rest
    return rest[:next_heading].rstrip() + "\n"


TAILOR_PROMPT = """You are tailoring a resume to a specific job description.

JOB DESCRIPTION:
{jd}

RANKED JD REQUIREMENTS (parsed and prioritized; focus on HIGH first):
{ranked_requirements}

CURRENT RESUME (YAML, source of truth):
{resume_yaml}

PROOF-POINT LIBRARY (the only claims this candidate is allowed to make):
{proof_points}

TASK:
1. For each HIGH-priority requirement above, check whether the resume
   already surfaces a matching achievement. If a matching achievement
   exists deeper in the resume, prefer rewrites that pull it forward.
2. Identify keywords / phrases from the JD that are NOT present in the
   resume. Prioritize keywords from HIGH-priority requirements first,
   then MEDIUM. Skip RED-HERRING items entirely.
3. For each missing keyword, decide if it's TRUTHFUL to add (i.e., the
   keyword is supported by an existing proof-point above, or is a synonym
   for an existing proof-point). If no proof-point supports it, SKIP it.
   This gate applies even to HIGH-priority requirements: never fabricate
   evidence for a high-priority ask we can't back up.
4. For truthful keywords, propose specific bullet rewrites that weave the
   keyword into an existing bullet naturally. Do NOT invent new bullets.
   Each rewrite must reference an existing bullet substring (so we can
   find-and-replace it).

CRITICAL RULES:
- DO NOT fabricate. If a keyword has no proof-point, skip it with a reason.
- DO NOT add metrics, dates, employers, or claims that don't appear in
  the proof-point library or the current resume.
- old_bullet_substring must be a substring of an actual bullet currently
  in the resume YAML — long enough to be unique, short enough to be
  exact (typically 30-80 characters).

Return STRICT JSON (no prose, no markdown fences) with exactly:
{{
  "added": ["<keyword1>", "<keyword2>"],
  "skipped": [
    {{"keyword": "<kw>", "reason": "<why no proof-point supports it>"}}
  ],
  "rewrites": [
    {{
      "old_bullet_substring": "<exact substring from current bullet>",
      "new_bullet": "<full replacement bullet text>"
    }}
  ]
}}
"""


def build_tailor_prompt(
    jd: str,
    resume_yaml: str,
    proof_points: str,
    ranked_requirements: Optional[list[RankedRequirement]] = None,
) -> str:
    """Build the prompt sent to Claude.

    `ranked_requirements` is optional for backwards compatibility: if not
    provided, we render an empty placeholder so callers from before the
    ranking feature still work.
    """
    ranked_block = (
        format_ranked_requirements(ranked_requirements)
        if ranked_requirements
        else "(no ranked requirements supplied)"
    )
    return TAILOR_PROMPT.format(
        jd=jd,
        resume_yaml=resume_yaml,
        proof_points=proof_points,
        ranked_requirements=ranked_block,
    )


def parse_tailor_response(raw: str) -> Optional[dict]:
    """Parse Claude's tailor response. Tolerate markdown fences."""
    if not raw:
        return None
    cleaned = re.sub(r"^```(?:json)?\n?", "", raw.strip())
    cleaned = re.sub(r"\n?```$", "", cleaned).strip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    parsed.setdefault("added", [])
    parsed.setdefault("skipped", [])
    parsed.setdefault("rewrites", [])
    return parsed


def make_anthropic_client() -> Anthropic:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set. Add it to your .env or export it."
        )
    return Anthropic(api_key=api_key)


def call_claude_for_tailor(
    client: Anthropic,
    jd: str,
    resume_yaml: str,
    proof_points: str,
    model: str = CLAUDE_MODEL,
    ranked_requirements: Optional[list[RankedRequirement]] = None,
) -> Optional[dict]:
    """Send the tailor prompt to Claude and parse the JSON response.

    Returns the parsed dict ({added, skipped, rewrites}) or None on failure.
    """
    prompt = build_tailor_prompt(
        jd, resume_yaml, proof_points,
        ranked_requirements=ranked_requirements,
    )
    system = (
        "You are tailoring resume YAML to a job description. Only suggest "
        "rewrites that are supported by the candidate's proof-point library. "
        "Never fabricate, even for high-priority requirements."
    )
    resp = client.messages.create(
        model=model,
        max_tokens=2048,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = resp.content[0].text if resp.content else ""
    return parse_tailor_response(raw)


def apply_rewrites_to_yaml(resume_dict: dict, rewrites: list[dict]) -> str:
    """Walk every string under resume_dict and apply rewrites in place.

    For each rewrite:
      - find any string value that contains old_bullet_substring
      - replace it with new_bullet (full string, not a substring patch —
        this is the contract: Claude returns the full replacement bullet)

    If old_bullet_substring isn't found anywhere, log a warning and skip
    that rewrite. Returns the rewritten YAML as a string.
    """

    def _walk(node):
        if isinstance(node, str):
            return node
        if isinstance(node, list):
            return [_walk(v) for v in node]
        if isinstance(node, dict):
            return {k: _walk(v) for k, v in node.items()}
        return node

    # We need to mutate strings in place, so walk with parent context.
    def _apply_one(rewrite: dict) -> bool:
        old = rewrite.get("old_bullet_substring", "") or ""
        new = rewrite.get("new_bullet", "") or ""
        if not old or not new:
            return False
        return _replace_in_strings(resume_dict, old, new)

    def _replace_in_strings(node, old: str, new: str) -> bool:
        """DFS: replace the FIRST string containing `old` with `new`. Return
        True if a replacement happened. Mutates node in place when node is
        a list or dict."""
        if isinstance(node, dict):
            for k in list(node.keys()):
                v = node[k]
                if isinstance(v, str):
                    if old in v:
                        node[k] = new
                        return True
                else:
                    if _replace_in_strings(v, old, new):
                        return True
            return False
        if isinstance(node, list):
            for i, v in enumerate(node):
                if isinstance(v, str):
                    if old in v:
                        node[i] = new
                        return True
                else:
                    if _replace_in_strings(v, old, new):
                        return True
            return False
        return False

    for rewrite in rewrites or []:
        applied = _apply_one(rewrite)
        if not applied:
            old = rewrite.get("old_bullet_substring", "")
            logger.warning(
                "rewrite skipped — old_bullet_substring not found in resume: %r",
                (old[:80] + "...") if len(old) > 80 else old,
            )

    # Round-trip via yaml.safe_dump. allow_unicode preserves em dashes / etc.
    # default_flow_style=False keeps the file in block style like the source.
    # sort_keys=False preserves the original section ordering.
    return yaml.safe_dump(
        resume_dict,
        allow_unicode=True,
        default_flow_style=False,
        sort_keys=False,
        width=10000,
    )


def render_variant_pdf(
    variant_yaml_path: Path,
    output_pdf_path: Path,
    rendercv_bin: Optional[Path] = None,
) -> bool:
    """Shell out to `rendercv render <yaml>` and move the PDF to its final home.

    rendercv writes outputs under <yaml-parent>/rendercv_output/<Name>_CV.pdf.
    We move that PDF to applications/<slug>.pdf.

    Returns True on success, False otherwise. Never raises.
    """
    if rendercv_bin is None:
        # Prefer the venv's rendercv (matches the rest of the toolchain).
        candidate = PROJECT_ROOT / ".venv" / "bin" / "rendercv"
        rendercv_bin = candidate if candidate.exists() else Path("rendercv")

    cmd = [str(rendercv_bin), "render", str(variant_yaml_path)]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(variant_yaml_path.parent),
            capture_output=True,
            text=True,
            timeout=120,
        )
    except Exception as e:
        logger.error("rendercv subprocess failed: %s", e)
        return False
    if proc.returncode != 0:
        logger.error(
            "rendercv returncode=%d stderr=%s",
            proc.returncode,
            (proc.stderr or "")[-500:],
        )
        return False

    # rendercv writes outputs to <yaml-parent>/rendercv_output/. Find the
    # most recently-modified PDF in there and move it to output_pdf_path.
    rendered_dir = variant_yaml_path.parent / "rendercv_output"
    if not rendered_dir.exists():
        logger.error("rendercv_output dir not produced at %s", rendered_dir)
        return False

    pdfs = sorted(rendered_dir.glob("*.pdf"), key=lambda p: p.stat().st_mtime)
    if not pdfs:
        logger.error("no PDF produced in %s", rendered_dir)
        return False
    src_pdf = pdfs[-1]
    output_pdf_path.parent.mkdir(parents=True, exist_ok=True)
    # Copy (don't move) so the base PDF stays in rendercv_output/ as the
    # canonical generic version Simplify Copilot pulls autofill data from.
    # If we moved, the next `tailor` run for a different company would
    # re-overwrite rendercv_output, but in the meantime Simplify would
    # autofill from missing data.
    shutil.copy2(str(src_pdf), str(output_pdf_path))

    # Re-render the BASE resume so rendercv_output/ holds the canonical
    # generic PDF, not the just-rendered variant. Cheap (~1 sec) and
    # ensures Simplify always autofills from up-to-date base content.
    base_yaml = variant_yaml_path.parent / "resume.yaml"
    if base_yaml.exists():
        try:
            subprocess.run(
                [str(rendercv_bin), "render", str(base_yaml)],
                check=True,
                capture_output=True,
                timeout=120,
                cwd=str(variant_yaml_path.parent),
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            logger.warning(
                "Variant PDF written, but failed to re-render base: %s. "
                "Run `rendercv render resume.yaml` manually if rendercv_output "
                "is missing the base.",
                e,
            )
    return True


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="JD-aware resume tailoring for the Vector."
    )
    parser.add_argument("--profile", required=True)
    parser.add_argument("--company", required=True)
    parser.add_argument("--role", default=None)
    parser.add_argument(
        "--jd-file",
        default=None,
        help="Read JD from this file. Overrides master-Excel lookup.",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Skip rendercv invocation (write variant YAML only).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    profile = load_profile(PROJECT_ROOT / "profiles" / f"{args.profile}.yaml")
    master_file = OUTPUT_DIR / profile.drive.master_filename

    # Step 1: get JD
    if args.jd_file:
        jd_path = Path(args.jd_file)
        if not jd_path.exists():
            print(f"--jd-file not found: {jd_path}", file=sys.stderr)
            return 2
        jd = jd_path.read_text().strip()
        jd_source = f"file:{jd_path}"
    else:
        jd = find_company_jd(master_file, args.company, role=args.role) or ""
        jd_source = (
            f"master:{master_file.name} "
            f"(company={args.company!r}, role={args.role!r})"
        )
    if not jd or len(jd) < 50:
        print(
            f"No usable JD found ({jd_source}). "
            "Pass --jd-file or score the company first.",
            file=sys.stderr,
        )
        return 2

    # Step 2: load base resume + positioning
    if not RESUME_YAML_PATH.exists():
        print(f"resume.yaml not found at {RESUME_YAML_PATH}", file=sys.stderr)
        return 2
    if not POSITIONING_MD_PATH.exists():
        print(f"positioning.md not found at {POSITIONING_MD_PATH}", file=sys.stderr)
        return 2

    resume_yaml_str = RESUME_YAML_PATH.read_text()
    resume_dict = yaml.safe_load(resume_yaml_str)
    positioning_md = POSITIONING_MD_PATH.read_text()
    proof_points = extract_proof_point_section(positioning_md)

    # Step 2.5: parse + rank JD requirements (deterministic, no LLM call).
    # The ranked list is also passed through to Claude so it can prioritize
    # rewrites for HIGH-priority asks first. The proof-point gate still
    # applies; even high-priority requirements get skipped if there's no
    # backing claim in positioning.md.
    raw_requirements = parse_jd_requirements(jd)
    ranked = rank_requirements(raw_requirements, jd)
    print("\nRanked JD requirements:")
    print(format_ranked_requirements(ranked))

    # Step 3: ask Claude
    client = make_anthropic_client()
    result = call_claude_for_tailor(
        client=client,
        jd=jd,
        resume_yaml=resume_yaml_str,
        proof_points=proof_points,
        ranked_requirements=ranked,
    )
    if result is None:
        print("Claude returned no parseable result.", file=sys.stderr)
        return 1

    # Step 4: apply rewrites
    variant_yaml_str = apply_rewrites_to_yaml(resume_dict, result.get("rewrites", []))

    # Step 5: write variant YAML
    slug = slugify(args.company, args.role)
    variant_yaml_path = UMBRELLA_ROOT / f"resume-{slug}.yaml"
    variant_yaml_path.write_text(variant_yaml_str)

    # Step 6: render PDF
    pdf_path = APPLICATIONS_DIR / f"{slug}.pdf"
    rendered = False
    if not args.no_render:
        rendered = render_variant_pdf(variant_yaml_path, pdf_path)

    # Step 7: report
    print(f"Company: {args.company}")
    print(f"Role: {args.role or '(none)'}")
    print(f"JD source: {jd_source}")
    added = result.get("added", [])
    skipped = result.get("skipped", [])
    rewrites = result.get("rewrites", [])
    print(f"\nKeywords added ({len(added)}): {', '.join(added) if added else '(none)'}")
    if skipped:
        print(f"\nKeywords skipped ({len(skipped)}):")
        for s in skipped:
            print(f"  - {s.get('keyword', '?')}: {s.get('reason', '?')}")
    print(f"\nRewrites applied: {len(rewrites)}")
    print(f"Variant YAML: {variant_yaml_path}")
    if rendered:
        print(f"Variant PDF: {pdf_path}")
    elif args.no_render:
        print("PDF render skipped (--no-render)")
    else:
        print("PDF render FAILED — check rendercv stderr above")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
