"""Batch generation of per-application packages from top-N master candidates.

For each unfilled candidate (fit_score set, manager_email set, no existing
applications/<slug>.md):

  1. Run tools.tailor via subprocess -> variant resume YAML + PDF.
  2. Run tools.score_resume via subprocess -> ATS keyword score.
  3. LLM-draft email + DM + A-F evaluation grounded in positioning.md.
  4. Write applications/<slug>.md mirroring the Buildkite template format.

Plus a batch-<YYYY-MM-DD>.md summary listing all packages generated.

CLI:
    .venv/bin/python -m tools.batch_apply --profile example --top 5
    .venv/bin/python -m tools.batch_apply --profile example --top 5 --regenerate
    .venv/bin/python -m tools.batch_apply --profile example --top 5 --skip-tailor
    .venv/bin/python -m tools.batch_apply --profile example --company "LangChain"
"""
import argparse
import datetime
import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
from anthropic import Anthropic
from dotenv import load_dotenv

from tools.profile import load_profile
from tools.tailor import slugify

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
UMBRELLA_ROOT = PROJECT_ROOT.parent  # the-vector/
APPLICATIONS_DIR = UMBRELLA_ROOT / "applications"
RESUME_YAML_PATH = UMBRELLA_ROOT / "resume.yaml"
POSITIONING_MD_PATH = UMBRELLA_ROOT / "positioning.md"

# Auto-load workspace .env so ANTHROPIC_API_KEY is available.
# Same pattern as tailor.py / score_resume.py — handles Claude Code's
# empty-string ANTHROPIC_API_KEY export.
WORKSPACE_ENV = PROJECT_ROOT.parent.parent.parent / ".env"
if WORKSPACE_ENV.exists():
    load_dotenv(WORKSPACE_ENV, override=True)

CLAUDE_MODEL = "claude-sonnet-4-6"

# Email length targets per positioning.md "Email length rules".
# Hiring-manager-after-application: 100-175 words, NOT cold-email's 75-cap.
EMAIL_WORDS_MIN = 100
EMAIL_WORDS_MAX = 175
DM_WORDS_MIN = 30
DM_WORDS_MAX = 50

logger = logging.getLogger(__name__)


# ---------- Candidate selection ----------


def select_candidates(
    df: pd.DataFrame,
    top_n: int,
    applications_dir: Path,
    regenerate: bool = False,
    company_filter: Optional[str] = None,
) -> tuple[list[pd.Series], list[str]]:
    """Pick top N rows from master that meet the prep gate.

    Filter rules:
      - fit_score not null
      - manager_email not null (skip rows w/o managers — find_managers first)
      - No existing applications/<slug>.md (unless regenerate)
      - If company_filter: case-insensitive substring on company column
      - Sort by fit_score desc

    Returns (selected_rows, skipped_warnings) — warnings is a list of
    human-readable strings about rows that *would* have qualified by
    fit_score order but lacked a manager_email.
    """
    if "fit_score" not in df.columns:
        return [], ["fit_score column missing — run tools/score.py first"]

    work = df.copy()

    # Apply company filter early so the top-N is computed within the filter.
    if company_filter:
        if "company" not in work.columns:
            return [], ["company column missing"]
        company_col = work["company"].astype(str).str.lower()
        work = work[company_col.str.contains(
            company_filter.lower(), na=False, regex=False
        )]
        if work.empty:
            return [], [f"no rows matched --company {company_filter!r}"]

    # Drop rows missing fit_score early — they can't compete for "top N".
    work = work[work["fit_score"].notna()]
    if work.empty:
        return [], ["no rows with fit_score"]

    # Sort by fit_score desc — applies to BOTH the manager-skip warnings
    # and the final selection. We walk this ordered list, skipping rows
    # that fail the gate, until we've collected top_n.
    ordered = work.sort_values("fit_score", ascending=False)

    selected: list[pd.Series] = []
    warnings: list[str] = []
    for _, row in ordered.iterrows():
        if len(selected) >= top_n:
            break

        company = str(row.get("company") or "").strip()
        title = row.get("title")
        title_str = str(title).strip() if pd.notna(title) else ""

        # Gate 1: manager_email present?
        manager_email = row.get("manager_email")
        if pd.isna(manager_email) or not str(manager_email).strip():
            warnings.append(
                f"[skipped] {company} — no manager_email; run find_managers first"
            )
            continue

        # Gate 2: existing .md unless --regenerate?
        slug = slugify(company, title_str if title_str else None)
        md_path = applications_dir / f"{slug}.md"
        if md_path.exists() and not regenerate:
            warnings.append(
                f"[skipped] {company} — applications/{slug}.md exists "
                "(use --regenerate to overwrite)"
            )
            continue

        selected.append(row)

    return selected, warnings


# ---------- Subprocess wrappers ----------


def run_tailor(
    profile: str, company: str, role: Optional[str], python_bin: Optional[Path] = None
) -> tuple[bool, str]:
    """Run `python -m tools.tailor --profile X --company Y [--role Z]`.

    Returns (success, stdout_or_stderr). Subprocess; never raises.
    """
    if python_bin is None:
        python_bin = PROJECT_ROOT / ".venv" / "bin" / "python"
        if not python_bin.exists():
            python_bin = Path(sys.executable)

    cmd = [str(python_bin), "-m", "tools.tailor", "--profile", profile, "--company", company]
    if role:
        cmd.extend(["--role", role])

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=180,
        )
    except Exception as e:
        return False, f"tailor subprocess error: {e}"

    if proc.returncode != 0:
        return False, (proc.stderr or proc.stdout or "")[-1000:]
    return True, proc.stdout


def run_score_resume(
    pdf_path: Path, company: str, profile: str = "example", python_bin: Optional[Path] = None
) -> tuple[Optional[int], str]:
    """Run score_resume on a PDF for a company.

    Returns (score_or_None, full_stdout). score_resume's exit code reflects
    ship-gate (0 = pass, 1 = below threshold) — we capture the score from
    stdout regardless of exit code.
    """
    if python_bin is None:
        python_bin = PROJECT_ROOT / ".venv" / "bin" / "python"
        if not python_bin.exists():
            python_bin = Path(sys.executable)

    cmd = [
        str(python_bin), "-m", "tools.score_resume",
        "--pdf", str(pdf_path),
        "--company", company,
        "--profile", profile,
    ]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=120,
        )
    except Exception as e:
        return None, f"score_resume subprocess error: {e}"

    score = parse_score_from_output(proc.stdout or "")
    return score, proc.stdout or ""


def parse_score_from_output(stdout: str) -> Optional[int]:
    """Pull the integer score out of a score_resume.py stdout dump.

    Format: 'Score: 67/100  (extractor: llm, match: token-level)'
    """
    if not stdout:
        return None
    m = re.search(r"Score:\s*(\d+)\s*/\s*100", stdout)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


# ---------- LLM email + evaluation drafting ----------


PACKAGE_PROMPT = """You are drafting an application package for a candidate who has JUST submitted their job application via the company's ATS and is now writing a follow-up email + LinkedIn DM to the hiring manager.

CANDIDATE:
- Name: Example User
- LinkedIn: https://www.linkedin.com/in/your-profile/
- Phone: +1-555-555-5555
- Email (sender): your-email@example.com

COMPANY: {company}
ROLE: {role}
HIRING MANAGER: {manager_name} ({manager_title}) <{manager_email}>
JD URL: {jd_url}

JOB DESCRIPTION:
---
{jd}
---

CANDIDATE RESUME (resume.yaml — source of truth for claims):
---
{resume_yaml}
---

POSITIONING PLAYBOOK (the candidate's vocabulary, proof-points, anti-patterns, and email-length rules):
---
{positioning_md}
---

CRITICAL RULES:
1. EMAIL is for a HIRING MANAGER WHO REVIEWS APPLICATIONS — NOT a cold prospecting email.
   - Word count: {email_min}-{email_max} words. NOT capped at 75. Don't compress.
   - Subject explicitly names the role + the candidate's name (so it threads in their inbox).
   - Open with: "I just submitted my application for [role] through [ATS] and wanted to introduce myself directly..."
   - Include one short background paragraph (most-recent roles — one line each).
   - Include one proof-points-and-JD-vocabulary callback paragraph.
   - CTA: SPECIFIC day window + time window (e.g., "Tuesday-Thursday 1-5pm ET"), not vague "next week".
   - Sign-off includes LinkedIn URL + phone.
2. LINKEDIN DM: {dm_min}-{dm_max} words. Different angle than the email. Conversational.
3. NEVER name-drop specific people, frameworks, or proprietary methodology in the email body or DM unless they appear in the candidate's resume. Use abstract framing.
4. NEVER claim tools or skills that aren't in the proof-point library or resume.yaml. If the JD lists Metaflow, TypeScript, Customer.io, reo.dev, Clay, Apollo, etc., and the candidate doesn't have direct experience per the proof-points, do NOT claim it. Acknowledge the gap honestly per the playbook.
5. Reference at least one specific JD phrase verbatim somewhere in the email or DM.

DELIVERABLES — return STRICT JSON (no prose, no markdown fences). Schema:
{{
  "subject": "Just submitted: <Role> — Example User",
  "email_body": "Hi <Name>,\\n\\n...",
  "email_word_count": 142,
  "linkedin_dm": "<short DM text>",
  "linkedin_dm_word_count": 30,
  "evaluation": {{
    "executive_summary": "<2-4 sentences: archetype, seniority, verdict>",
    "background_match": [
      {{"jd_requirement": "<paraphrase of JD line>", "candidate_evidence": "<concrete proof-point evidence>"}}
    ],
    "positioning_strategy": "<2-4 bullet points compressed into a paragraph: lead-with, emphasize, de-emphasize, veteran angle if relevant>",
    "tailoring_plan": ["<resume edit 1>", "<edit 2>", "..."],
    "star_stories": [
      {{"jd_bullet": "<JD line being addressed>", "story": {{"S": "<situation>", "T": "<task>", "A": "<action>", "R": "<result with metric>"}}}}
    ]
  }}
}}

Aim for 5-7 background_match rows, 3-5 tailoring_plan items, 3-5 star_stories.
"""


def make_anthropic_client() -> Anthropic:
    """Same shape as tailor.make_anthropic_client — explicit error if no key."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set. Add it to your .env or export it."
        )
    return Anthropic(api_key=api_key)


def build_package_prompt(
    company: str,
    role: str,
    jd: str,
    jd_url: str,
    manager_name: str,
    manager_title: str,
    manager_email: str,
    resume_yaml: str,
    positioning_md: str,
) -> str:
    return PACKAGE_PROMPT.format(
        company=company,
        role=role or "(unspecified)",
        jd=jd,
        jd_url=jd_url or "(not in master)",
        manager_name=manager_name or "(unknown)",
        manager_title=manager_title or "(unknown)",
        manager_email=manager_email,
        resume_yaml=resume_yaml,
        positioning_md=positioning_md,
        email_min=EMAIL_WORDS_MIN,
        email_max=EMAIL_WORDS_MAX,
        dm_min=DM_WORDS_MIN,
        dm_max=DM_WORDS_MAX,
    )


def parse_package_response(raw: str) -> Optional[dict]:
    """Parse Claude's JSON package response. Tolerate markdown fences."""
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
    # Fill required keys with safe defaults so downstream MD writer
    # never KeyErrors on a partial response.
    parsed.setdefault("subject", "")
    parsed.setdefault("email_body", "")
    parsed.setdefault("email_word_count", 0)
    parsed.setdefault("linkedin_dm", "")
    parsed.setdefault("linkedin_dm_word_count", 0)
    eval_section = parsed.get("evaluation") or {}
    if not isinstance(eval_section, dict):
        eval_section = {}
    eval_section.setdefault("executive_summary", "")
    eval_section.setdefault("background_match", [])
    eval_section.setdefault("positioning_strategy", "")
    eval_section.setdefault("tailoring_plan", [])
    eval_section.setdefault("star_stories", [])
    parsed["evaluation"] = eval_section
    return parsed


def call_claude_for_package(
    client: Anthropic,
    company: str,
    role: str,
    jd: str,
    jd_url: str,
    manager_name: str,
    manager_title: str,
    manager_email: str,
    resume_yaml: str,
    positioning_md: str,
    model: str = CLAUDE_MODEL,
) -> Optional[dict]:
    prompt = build_package_prompt(
        company=company, role=role, jd=jd, jd_url=jd_url,
        manager_name=manager_name, manager_title=manager_title,
        manager_email=manager_email, resume_yaml=resume_yaml,
        positioning_md=positioning_md,
    )
    system = (
        "You draft application packages (email + LinkedIn DM + A-F evaluation) "
        "for hiring managers. Hiring-manager-after-application emails are "
        "100-175 words, NOT cold-email's 75-word cap. Never name modern "
        "cold-email operators or historical copywriting canon in external "
        "copy. Never fabricate tool experience that isn't in the proof-point "
        "library."
    )
    resp = client.messages.create(
        model=model,
        max_tokens=4096,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = resp.content[0].text if resp.content else ""
    return parse_package_response(raw)


# ---------- Application package MD writer ----------


def render_application_md(
    company: str,
    role: str,
    fit_score,
    keyword_score: Optional[int],
    location: str,
    apply_url: str,
    submitted_on: str,
    manager_name: str,
    manager_title: str,
    manager_email: str,
    manager_linkedin: str,
    manager_source: str,
    package: dict,
) -> str:
    """Render the per-application .md mirroring the Buildkite template format.

    Sections (must include for tests):
      - Header: apply URL, comp, location, fit, submitted-on
      - Hiring manager (table)
      - A-F Evaluation (executive summary, background match, positioning,
        compensation, tailoring plan, STAR stories)
      - Outreach -- cold email (subject + body + word-count + meta)
      - Outreach -- LinkedIn DM
      - Status log
    """
    eval_section = package.get("evaluation") or {}
    background_match = eval_section.get("background_match") or []
    star_stories = eval_section.get("star_stories") or []
    tailoring_plan = eval_section.get("tailoring_plan") or []

    # Header
    fit_display = f"{fit_score}/10" if fit_score is not None else "n/a"
    score_line = f" | ATS keyword score: {keyword_score}/100" if keyword_score is not None else ""
    role_label = role if role else "(role unspecified)"

    parts: list[str] = []
    parts.append(f"# {company} — {role_label}\n")
    parts.append(f"**Apply via:** {apply_url or '(not in master)'}")
    parts.append("**Compensation:** _check JD; band varies by company stage_")
    parts.append(f"**Location:** {location or '(unspecified)'}")
    parts.append(f"**Fit:** {fit_display} (Vector master Excel){score_line}")
    parts.append(f"**Submitted on:** {submitted_on} _(pending — review before sending)_")
    parts.append("\n---\n")

    # Hiring manager
    parts.append("## Hiring manager (parallel outreach)\n")
    parts.append("| Field | Value |")
    parts.append("|---|---|")
    parts.append(f"| Name | {manager_name or '(unknown)'} |")
    parts.append(f"| Title | {manager_title or '(unknown)'} |")
    parts.append(f"| Email | {manager_email} |")
    parts.append(f"| LinkedIn | {manager_linkedin or '_not yet found_'} |")
    parts.append(f"| Source | {manager_source or '(Vector cascade)'} |")
    parts.append("\n---\n")

    # A-F Evaluation
    parts.append("## A–F Evaluation\n")

    parts.append("### A. Executive Summary")
    parts.append(eval_section.get("executive_summary", "_pending_") or "_pending_")
    parts.append("")

    parts.append("### B. Background match (every JD requirement → the candidate's experience)\n")
    if background_match:
        parts.append("| JD requirement | Candidate's evidence |")
        parts.append("|---|---|")
        for item in background_match:
            req = (item.get("jd_requirement") or "").replace("|", "\\|").replace("\n", " ").strip()
            ev = (item.get("candidate_evidence") or "").replace("|", "\\|").replace("\n", " ").strip()
            parts.append(f"| {req} | {ev} |")
    else:
        parts.append("_(no matches drafted)_")
    parts.append("")

    parts.append("### C. Positioning strategy")
    parts.append(eval_section.get("positioning_strategy", "_pending_") or "_pending_")
    parts.append("")

    parts.append("### D. Compensation")
    parts.append(
        "_Check the JD for a published band. the candidate's floor $140K, target $150-220K._"
    )
    parts.append("")

    parts.append("### E. Tailoring plan (resume edits)")
    if tailoring_plan:
        for i, edit in enumerate(tailoring_plan, 1):
            parts.append(f"{i}. {edit}")
    else:
        parts.append("_(no tailoring plan drafted)_")
    parts.append("")

    parts.append("### F. STAR stories for interview prep\n")
    if star_stories:
        parts.append("| JD bullet | STAR story |")
        parts.append("|---|---|")
        for s in star_stories:
            jd_bullet = (s.get("jd_bullet") or "").replace("|", "\\|").replace("\n", " ").strip()
            story = s.get("story") or {}
            star_text = (
                f"**S:** {story.get('S', '')} "
                f"**T:** {story.get('T', '')} "
                f"**A:** {story.get('A', '')} "
                f"**R:** {story.get('R', '')}"
            ).replace("|", "\\|").replace("\n", " ").strip()
            parts.append(f"| {jd_bullet} | {star_text} |")
    else:
        parts.append("_(no STAR stories drafted)_")
    parts.append("\n---\n")

    # Outreach — cold email
    parts.append(f"## Outreach — cold email to {manager_name or 'hiring manager'}\n")
    subject = package.get("subject", "") or f"Just submitted: {role_label} — Example User"
    parts.append(f"**Subject:** `{subject}`\n")
    parts.append("```")
    parts.append(package.get("email_body", "") or "_(no body drafted)_")
    parts.append("```\n")
    word_count = package.get("email_word_count", 0)
    parts.append(f"**Word count:** {word_count}.")
    if word_count and (word_count < EMAIL_WORDS_MIN or word_count > EMAIL_WORDS_MAX):
        parts.append(
            f"\n_Warning: word count outside the {EMAIL_WORDS_MIN}-{EMAIL_WORDS_MAX} "
            "target for hiring-manager-after-application emails. Review before sending._"
        )
    parts.append("")
    parts.append(
        "**Send window:** 30-60 min after submitting the application. "
        "From `your-email@example.com`."
    )
    parts.append("\n---\n")

    # Outreach — LinkedIn DM
    parts.append("## Outreach — LinkedIn DM (when URL is found)\n")
    parts.append("```")
    parts.append(package.get("linkedin_dm", "") or "_(no DM drafted)_")
    parts.append("```\n")
    dm_count = package.get("linkedin_dm_word_count", 0)
    parts.append(f"**Word count:** {dm_count}.")
    if dm_count and (dm_count < DM_WORDS_MIN or dm_count > DM_WORDS_MAX):
        parts.append(
            f"\n_Warning: DM word count outside the {DM_WORDS_MIN}-{DM_WORDS_MAX} "
            "target. Review before sending._"
        )
    parts.append("\n---\n")

    # Status log
    parts.append("## Status log\n")
    parts.append("| Date | Action | Status |")
    parts.append("|---|---|---|")
    parts.append(f"| {submitted_on} | Vector batch_apply generated package | ✅ |")
    parts.append(f"| _pending_ | Submit application via {apply_url or 'ATS'} | _todo_ |")
    parts.append(
        f"| _pending_ | Send email to {manager_email} (subject \"{subject}\") | _todo_ |"
    )
    parts.append("| _pending_ | Find LinkedIn URL + send DM | _todo_ |")
    parts.append("| _pending_ | 7-day follow-up email if no reply | _todo_ |")
    parts.append("")

    return "\n".join(parts)


# ---------- Batch summary writer ----------


def render_batch_summary_md(
    rows: list[dict], date_str: str, regenerate: bool = False
) -> str:
    """Render applications/batch-<date>.md.

    rows: list of {company, role, fit, score, manager, apply_url, package_path}
    """
    parts: list[str] = []
    parts.append(f"# Batch run — {date_str}\n")
    parts.append(f"Generated {len(rows)} application package(s).\n")
    if regenerate:
        parts.append("_Mode: --regenerate (overwrote existing .md files)_\n")

    parts.append("| # | Company | Role | Fit | ATS score | Manager | Apply URL | Package |")
    parts.append("|---|---|---|---|---|---|---|---|")
    for i, r in enumerate(rows, 1):
        company = r.get("company", "")
        role = r.get("role", "") or "(unspecified)"
        fit = r.get("fit", "")
        score = r.get("score")
        score_str = f"{score}/100" if score is not None else "n/a"
        manager = r.get("manager", "") or "(unknown)"
        apply_url = r.get("apply_url", "") or "(not in master)"
        path = r.get("package_path", "")
        parts.append(
            f"| {i} | {company} | {role} | {fit} | {score_str} | {manager} | "
            f"{apply_url} | {path} |"
        )
    parts.append("")

    parts.append("## Next actions\n")
    parts.append("Spot-check each package BEFORE sending:\n")
    parts.append("1. Open each `applications/<slug>.md` and read the email body — does it sound like the candidate?")
    parts.append("2. Verify no operator names (Plascencia, Nowoslawski, Oliverify, Schwartz, Hopkins, Cialdini, Masterson, Halbert) leaked into copy.")
    parts.append("3. Verify no fabricated tool claims (TypeScript, Metaflow, Customer.io, reo.dev, Clay, Apollo) — if mentioned, must be honest gap acknowledgement.")
    parts.append("4. Confirm the JD-phrase callback feels natural, not stuffed.")
    parts.append("5. Submit the application via the apply URL.")
    parts.append("6. Send the email 30-60 min after submitting.")
    parts.append("7. Send the LinkedIn DM once the manager's URL is found.")
    parts.append("8. Update each package's Status log table after each step.")
    parts.append("")

    return "\n".join(parts)


# ---------- Main loop ----------


def process_candidate(
    row: pd.Series,
    profile_name: str,
    applications_dir: Path,
    resume_yaml: str,
    positioning_md: str,
    client: Anthropic,
    skip_tailor: bool = False,
    submitted_on: Optional[str] = None,
) -> Optional[dict]:
    """Process a single candidate end-to-end.

    Returns a summary dict for batch-summary rendering, or None on hard fail.
    Logs warnings (non-fatal) to logger; never raises.
    """
    company = str(row.get("company") or "").strip()
    title = row.get("title")
    role = str(title).strip() if pd.notna(title) else ""
    slug = slugify(company, role if role else None)

    fit_score = row.get("fit_score")
    fit_display = fit_score if pd.notna(fit_score) else None
    location = str(row.get("location") or "")

    # Apply URL preference: job_url_direct > job_url > company_url
    apply_url = ""
    for col in ("job_url_direct", "job_url", "company_url"):
        v = row.get(col)
        if pd.notna(v) and str(v).strip():
            apply_url = str(v).strip()
            break

    description = str(row.get("description") or "")

    manager_name = str(row.get("manager_name") or "").strip()
    manager_email = str(row.get("manager_email") or "").strip()
    manager_linkedin = str(row.get("manager_linkedin") or "").strip()
    manager_source = str(row.get("manager_source") or "").strip()

    # Step 1: tailor (variant YAML + PDF) unless --skip-tailor.
    if not skip_tailor:
        ok, tailor_out = run_tailor(profile_name, company, role if role else None)
        if not ok:
            logger.warning("tailor failed for %s: %s", company, tailor_out[-300:])
            # Continue anyway — score/email can still proceed if PDF exists.

    pdf_path = applications_dir / f"{slug}.pdf"

    # Step 2: score_resume (only if PDF exists).
    keyword_score: Optional[int] = None
    if pdf_path.exists():
        keyword_score, _ = run_score_resume(pdf_path, company, profile=profile_name)
    else:
        logger.warning("PDF not found for %s at %s — skipping score", company, pdf_path)

    # Step 3: LLM draft email + DM + evaluation.
    try:
        package = call_claude_for_package(
            client=client,
            company=company,
            role=role,
            jd=description,
            jd_url=apply_url,
            manager_name=manager_name,
            manager_title="",  # not in master Excel; LLM can leave blank in copy
            manager_email=manager_email,
            resume_yaml=resume_yaml,
            positioning_md=positioning_md,
        )
    except Exception as e:
        logger.warning("Anthropic call failed for %s: %s", company, str(e)[:200])
        return None

    if package is None:
        logger.warning("Anthropic returned no parseable package for %s — skipping", company)
        return None

    # Word-count sanity check (warn, don't fail).
    email_wc = package.get("email_word_count", 0)
    if email_wc and (email_wc < EMAIL_WORDS_MIN or email_wc > EMAIL_WORDS_MAX):
        logger.warning(
            "%s — email word count %d outside %d-%d target",
            company, email_wc, EMAIL_WORDS_MIN, EMAIL_WORDS_MAX,
        )

    # Step 4: write the per-application .md.
    submitted_on = submitted_on or datetime.date.today().isoformat()
    md_text = render_application_md(
        company=company,
        role=role,
        fit_score=fit_display,
        keyword_score=keyword_score,
        location=location,
        apply_url=apply_url,
        submitted_on=submitted_on,
        manager_name=manager_name,
        manager_title="",
        manager_email=manager_email,
        manager_linkedin=manager_linkedin,
        manager_source=manager_source,
        package=package,
    )
    md_path = applications_dir / f"{slug}.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(md_text)

    return {
        "company": company,
        "role": role,
        "fit": fit_display,
        "score": keyword_score,
        "manager": f"{manager_name} <{manager_email}>" if manager_name else manager_email,
        "apply_url": apply_url,
        "package_path": f"applications/{slug}.md",
        "slug": slug,
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Batch-generate per-application packages for top-N candidates."
    )
    parser.add_argument("--profile", required=True)
    parser.add_argument(
        "--top", type=int, default=5,
        help="How many candidates to process (default: 5)",
    )
    parser.add_argument(
        "--regenerate", action="store_true",
        help="Force overwrite existing applications/<slug>.md files.",
    )
    parser.add_argument(
        "--skip-tailor", action="store_true",
        help="Skip tailor.py — use existing variant PDFs (faster, for re-runs).",
    )
    parser.add_argument(
        "--company", default=None,
        help="Single-company mode: only process rows matching this company.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    profile = load_profile(PROJECT_ROOT / "profiles" / f"{args.profile}.yaml")
    master_file = OUTPUT_DIR / profile.drive.master_filename

    if not master_file.exists():
        print(f"Master Excel not found: {master_file}", file=sys.stderr)
        return 2

    if not RESUME_YAML_PATH.exists():
        print(f"resume.yaml not found at {RESUME_YAML_PATH}", file=sys.stderr)
        return 2
    if not POSITIONING_MD_PATH.exists():
        print(f"positioning.md not found at {POSITIONING_MD_PATH}", file=sys.stderr)
        return 2

    df = pd.read_excel(master_file)
    selected, warnings = select_candidates(
        df,
        top_n=args.top,
        applications_dir=APPLICATIONS_DIR,
        regenerate=args.regenerate,
        company_filter=args.company,
    )

    for w in warnings:
        print(w)

    if not selected:
        print("No candidates qualify for batch processing.")
        return 1

    resume_yaml_str = RESUME_YAML_PATH.read_text()
    positioning_md_str = POSITIONING_MD_PATH.read_text()

    client = make_anthropic_client()
    submitted_on = datetime.date.today().isoformat()

    summary_rows: list[dict] = []
    for i, row in enumerate(selected, 1):
        company = str(row.get("company") or "")
        title = row.get("title")
        role = str(title).strip() if pd.notna(title) else ""
        result = process_candidate(
            row=row,
            profile_name=args.profile,
            applications_dir=APPLICATIONS_DIR,
            resume_yaml=resume_yaml_str,
            positioning_md=positioning_md_str,
            client=client,
            skip_tailor=args.skip_tailor,
            submitted_on=submitted_on,
        )
        if result is None:
            print(f"[{i}/{len(selected)}] {company} {role} -> SKIPPED (LLM/tailor error)")
            continue
        score_str = f"{result['score']}/100" if result["score"] is not None else "n/a"
        print(
            f"[{i}/{len(selected)}] {company} {role} -> "
            f"{result['package_path']} (score: {score_str})"
        )
        summary_rows.append(result)

    # Batch summary
    if summary_rows:
        batch_path = APPLICATIONS_DIR / f"batch-{submitted_on}.md"
        batch_path.write_text(
            render_batch_summary_md(summary_rows, submitted_on, regenerate=args.regenerate)
        )
        print(f"\nBatch summary: {batch_path}")
        print(
            f"Generated {len(summary_rows)} application package(s) in applications/. "
            "Run python -m tools.batch_apply --regenerate to refresh."
        )
    else:
        print("\nNo packages generated (all candidates failed).")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
