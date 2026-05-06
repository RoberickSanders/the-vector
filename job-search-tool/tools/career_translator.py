"""career_translator.py — Translate work history into skills + adjacent titles + resume bullets.

V1 is Kimi-driven. Government API hooks (O*NET, CareerOneStop) are noted in TODOs
for V2 once you register for free API keys.

Reuses the Kimi-via-Anthropic-SDK pattern from tools/score.py. Self-contained;
no external imports beyond pinned third-party packages.

CLI:
    python -m tools.career_translator --profile example
    python -m tools.career_translator --profile example --target-jd /path/to/jd.txt
    python -m tools.career_translator --profile example --target-jd https://job.url --output-name custom.md
"""
import argparse
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from anthropic import Anthropic

from tools.profile import load_profile, Profile

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
COMPANIES_DB_FILE = PROJECT_ROOT / "config" / "companies-db.json"

# Output dir lives one level up from job-search-tool/, alongside it under
# 01-Projects/the-vector/translations/
TRANSLATIONS_DIR = PROJECT_ROOT.parent / "translations"

# Load environment from workspace .env (where KIMI_API_KEY lives)
WORKSPACE_ENV = PROJECT_ROOT.parent.parent.parent / ".env"
if WORKSPACE_ENV.exists():
    load_dotenv(WORKSPACE_ENV, override=True)  # override empty strings exported by Claude Code

# Kimi exposes an Anthropic-compatible API at this endpoint.
# Mirrored from tools/score.py / the upstream llm_router.py — do not import from upstream.
KIMI_BASE_URL = "https://api.kimi.com/coding"


# ============================================================================
# PROMPTS — separate functions for testability
# ============================================================================

def build_skills_prompt(resume: str, rubric: str) -> str:
    """Asks Kimi to extract hard/soft/tool skills from resume, grounded in rubric."""
    return f"""You are extracting structured skills from a candidate's resume, grounded in
the rubric of the kinds of jobs they are pursuing.

CANDIDATE RUBRIC (kinds of roles + signals that matter):
{rubric}

CANDIDATE RESUME:
{resume}

Return STRICT JSON (no prose, no markdown fences) with exactly these fields:
{{
  "hard_skills": ["<technical / domain skill 1>", "<skill 2>", ...],
  "soft_skills": ["<leadership / communication / etc 1>", ...],
  "tools": ["<specific software / SaaS / API 1>", ...]
}}

Rules:
- hard_skills = technical or domain skills (Python, SQL, lead scoring, deliverability, etc.)
- soft_skills = leadership, communication, decision-making, mentorship
- tools = specific named software, SaaS, or APIs (Salesforce, Smartlead, Anthropic SDK)
- Pull only skills that the resume actually demonstrates — do not invent.
- 8-15 items per list is ideal.
"""


def build_adjacent_titles_prompt(resume: str, rubric: str) -> str:
    """Asks Kimi for 5-8 adjacent job titles + why each matches."""
    return f"""You identify ADJACENT job titles a candidate qualifies for that are NOT
already on their resume — roles their experience translates to, framed against
the kind of jobs they are actually pursuing.

CANDIDATE RUBRIC (the kinds of roles + signals that matter):
{rubric}

CANDIDATE RESUME:
{resume}

Identify 5-8 adjacent job titles the candidate could realistically target. Quality
over quantity — each must be a defensible match given their actual experience.

Return STRICT JSON (no prose, no markdown fences) — a JSON array of objects:
[
  {{
    "title": "<adjacent job title>",
    "why_qualified": "<one or two sentences citing specific resume evidence>",
    "gap_to_close": "<one sentence: the thing they would still need to learn / show>"
  }},
  ...
]

Rules:
- Skip titles already on the resume — only ADJACENT ones.
- Anchor each "why_qualified" in a concrete resume fact (numbers, companies, tools).
- Keep "gap_to_close" honest — if the gap is small, say so; if large, name it.
"""


def build_car_bullets_prompt(resume: str, rubric: str, target_titles: list[str]) -> str:
    """Asks Kimi for 5-8 CAR-format resume bullets tailored to target titles."""
    titles_str = ", ".join(target_titles) if target_titles else "(no explicit targets — use rubric)"
    return f"""You write strong, metric-driven resume bullets in CAR format
(Challenge / Action / Result), tailored to specific target roles.

CANDIDATE RUBRIC:
{rubric}

CANDIDATE RESUME:
{resume}

TARGET TITLES TO TAILOR FOR:
{titles_str}

Produce 5-8 CAR-format resume bullets that re-frame the candidate's existing work
for these target titles. Each bullet must follow this exact pattern:

  Challenge: [1 sentence stating the problem / context].
  Action: [1 sentence with the specific action the candidate took].
  Result: [1 sentence with a quantified outcome — number, %, $ amount, ranking, time saved].

Return STRICT JSON (no prose, no markdown fences) — a JSON array:
[
  {{
    "role": "<which target title this bullet best supports>",
    "bullet": "Challenge: ... Action: ... Result: ..."
  }},
  ...
]

Rules:
- Use ONLY facts present in the resume — do not fabricate metrics.
- If a metric is missing, leave the Result vague rather than invent a number.
- Prefer bullets that emphasize the technical / GTM-engineering skills in the rubric.
"""


def build_gap_analysis_prompt(resume: str, jd_text: str, rubric: str) -> str:
    """Asks Kimi to compare candidate skills vs JD requirements + recommend close-gap actions."""
    return f"""You assess fit gap honestly between a candidate and a target job description.

CANDIDATE RUBRIC (their target market):
{rubric}

CANDIDATE RESUME:
{resume}

TARGET JOB DESCRIPTION:
{jd_text}

Return STRICT JSON (no prose, no markdown fences) with exactly these fields:
{{
  "jd_required_skills": ["<skill 1>", "<skill 2>", ...],
  "candidate_has": ["<skill the candidate already has from JD list>", ...],
  "gap": ["<skill from JD list the candidate is missing or weak on>", ...],
  "close_gap_actions": ["<concrete, time-boxed action 1>", "<action 2>", ...]
}}

Rules:
- Pull jd_required_skills directly from the JD (max ~12).
- candidate_has = subset of jd_required_skills the resume clearly proves.
- gap = subset of jd_required_skills the resume does NOT prove.
- close_gap_actions = 3-6 concrete actions (build a demo, take a short course, ship a
  side project, get a cert) — each should be doable in 1-4 weeks.
- Be honest, not generous.
"""


# ============================================================================
# RESPONSE PARSING
# ============================================================================

def parse_kimi_json(raw: str) -> Optional[dict | list]:
    """Strip markdown fences, parse JSON. Tolerant of malformed output."""
    if not raw:
        return None
    cleaned = re.sub(r"^```(?:json)?\n?", "", raw.strip())
    cleaned = re.sub(r"\n?```$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None


# ============================================================================
# COMPANIES CROSS-REFERENCE
# ============================================================================

def cross_reference_companies(adjacent_titles: list[dict], db_path: Path | str = COMPANIES_DB_FILE,
                              limit: int = 15) -> list[dict]:
    """Cross-reference adjacent titles with companies-db.json.

    V1 heuristic: surface tier-1 d100 companies first (the primary target list)
    since we don't yet have per-company current-job-postings data here. The scoring
    pipeline (tools/score.py) is where actual JD-to-company matching happens; this
    is the "where to look next" map.
    """
    db_path = Path(db_path)
    if not db_path.exists():
        return []
    try:
        db = json.loads(db_path.read_text())
    except (json.JSONDecodeError, OSError):
        return []
    companies = db.get("companies", [])
    # Tier-1 first, then tier-2, then rest
    tier1 = [c for c in companies if c.get("tier") == 1]
    tier2 = [c for c in companies if c.get("tier") == 2]
    rest = [c for c in companies if c.get("tier") not in (1, 2)]
    ordered = tier1 + tier2 + rest
    # Trim to fields useful for the report
    return [
        {"name": c.get("name", ""), "ats": c.get("ats", ""),
         "tier": c.get("tier"), "notes": c.get("notes", "")}
        for c in ordered[:limit] if c.get("name")
    ]


# ============================================================================
# KIMI CLIENT
# ============================================================================

def make_client() -> Anthropic:
    api_key = os.environ.get("KIMI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "KIMI_API_KEY not set. Add to your .env or export it."
        )
    return Anthropic(api_key=api_key, base_url=KIMI_BASE_URL)


def call_kimi(client: Anthropic, model: str, system: str, prompt: str,
              max_tokens: int = 1500) -> str:
    """Single Kimi call with retry. Returns raw text or empty string on failure."""
    for attempt in range(3):
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text if resp.content else ""
        except Exception as e:
            print(f"  Kimi attempt {attempt + 1} failed: {str(e)[:120]}")
            time.sleep(1.5 ** attempt)
    return ""


# ============================================================================
# REPORT ASSEMBLY
# ============================================================================

def render_markdown_report(profile, skills: dict, adjacent: list,
                           bullets: list, gap: Optional[dict], jd_text: Optional[str],
                           companies_hiring: list = None) -> str:
    """Compose the final markdown output from the parsed Kimi responses."""
    companies_hiring = companies_hiring or []
    name = getattr(profile, "name", "Candidate")
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines: list[str] = []

    # Header
    lines.append(f"# Career Translation — {name}")
    lines.append("")
    lines.append(f"_Generated {generated_at} by `tools/career_translator.py` (Kimi-driven, V1)._")
    lines.append("")

    # Dynamic section counter so omitted optional sections don't leave gaps
    section = [0]
    def heading(title: str) -> str:
        section[0] += 1
        return f"## {section[0]}. {title}"

    # Section — Skills
    lines.append(heading("Inferred Skills"))
    lines.append("")
    hard = skills.get("hard_skills", []) or []
    soft = skills.get("soft_skills", []) or []
    tools = skills.get("tools", []) or []
    lines.append("### Hard Skills (Technical / Domain)")
    if hard:
        for s in hard:
            lines.append(f"- {s}")
    else:
        lines.append("- _(none extracted)_")
    lines.append("")
    lines.append("### Soft Skills")
    if soft:
        for s in soft:
            lines.append(f"- {s}")
    else:
        lines.append("- _(none extracted)_")
    lines.append("")
    lines.append("### Tools / Tech Stack")
    if tools:
        for s in tools:
            lines.append(f"- {s}")
    else:
        lines.append("- _(none extracted)_")
    lines.append("")

    # Section — Adjacent Titles
    lines.append(heading("Adjacent Job Titles"))
    lines.append("")
    lines.append("_Roles the candidate could realistically target that are not on their resume._")
    lines.append("")
    if adjacent:
        for item in adjacent:
            if not isinstance(item, dict):
                continue
            title = item.get("title", "(untitled)")
            why = item.get("why_qualified", "")
            gap_to_close = item.get("gap_to_close", "")
            lines.append(f"### {title}")
            if why:
                lines.append(f"- **Why qualified:** {why}")
            if gap_to_close:
                lines.append(f"- **Gap to close:** {gap_to_close}")
            lines.append("")
    else:
        lines.append("_(no adjacent titles inferred)_")
        lines.append("")

    # Section — CAR Bullets
    lines.append(heading("CAR-Format Resume Bullets"))
    lines.append("")
    lines.append("_Challenge / Action / Result bullets re-tailored for target titles._")
    lines.append("")
    if bullets:
        for item in bullets:
            if not isinstance(item, dict):
                continue
            role = item.get("role", "(general)")
            bullet = item.get("bullet", "")
            lines.append(f"**Target role:** {role}")
            lines.append(f"- {bullet}")
            lines.append("")
    else:
        lines.append("_(no bullets generated)_")
        lines.append("")

    # Section — Gap Analysis (only when JD provided)
    if jd_text and gap:
        lines.append(heading("Gap Analysis (vs Target JD)"))
        lines.append("")
        lines.append("**JD-required skills:**")
        for s in gap.get("jd_required_skills", []) or []:
            lines.append(f"- {s}")
        lines.append("")
        lines.append("**Candidate already has:**")
        for s in gap.get("candidate_has", []) or []:
            lines.append(f"- {s}")
        lines.append("")
        lines.append("**Gap:**")
        for s in gap.get("gap", []) or []:
            lines.append(f"- {s}")
        lines.append("")
        lines.append("**Recommended actions to close the gap:**")
        for s in gap.get("close_gap_actions", []) or []:
            lines.append(f"- {s}")
        lines.append("")

    # Section — Companies Hiring (cross-reference w/ companies-db.json)
    if companies_hiring:
        lines.append(heading("Companies Worth Targeting"))
        lines.append("")
        lines.append("_Cross-referenced with `config/companies-db.json` (d100 tier-1 first)._")
        lines.append("")
        lines.append("| Company | ATS | Tier | Notes |")
        lines.append("| --- | --- | --- | --- |")
        for c in companies_hiring:
            name_c = c.get("name", "")
            ats = c.get("ats", "")
            tier = c.get("tier", "")
            notes = c.get("notes", "")
            lines.append(f"| {name_c} | {ats} | {tier} | {notes} |")
        lines.append("")

    # Section 6 — V2 hooks
    lines.append("## V2 Notes")
    lines.append("")
    lines.append(
        "Government data integration is V2. Once free API keys are activated, "
        "wire in:"
    )
    lines.append("- **O*NET** (onetonline.org) — official skill-to-occupation crosswalks; "
                 "use to validate adjacent titles against BLS-recognized occupations.")
    lines.append("- **CareerOneStop** (careeronestop.org) — wage / employment outlook + "
                 "training/cert recommendations to enrich `close_gap_actions`.")
    lines.append("- **BLS Occupational Employment Statistics (OES)** — pay-band sanity-check "
                 "for adjacent titles.")
    lines.append("")
    lines.append("Until then, this report is grounded only in: the candidate's resume, "
                 "the scoring rubric, and the in-house `companies-db.json`.")
    lines.append("")

    return "\n".join(lines)


# ============================================================================
# JD LOADING (file or master Excel by job_url)
# ============================================================================

def _load_jd(target_jd: str, profile: Profile) -> Optional[str]:
    """Load JD text from a file path OR look up by job_url in the master Excel."""
    p = Path(target_jd)
    if p.exists() and p.is_file():
        return p.read_text(errors="ignore")
    # Treat as a job_url — look up in the master Excel
    if target_jd.startswith("http"):
        try:
            import pandas as pd
            master_file = OUTPUT_DIR / profile.drive.master_filename
            if not master_file.exists():
                return None
            df = pd.read_excel(master_file)
            if "job_url" not in df.columns or "description" not in df.columns:
                return None
            row = df[df["job_url"].astype(str) == target_jd]
            if row.empty:
                return None
            return str(row.iloc[0].get("description", "") or "")
        except Exception as e:
            print(f"  JD lookup failed: {e}")
            return None
    return None


# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Translate work history into skills + adjacent titles + resume bullets."
    )
    parser.add_argument("--profile", required=True, help="Profile name (e.g. 'example')")
    parser.add_argument("--target-jd",
                        help="Path to a JD text file OR a job_url present in the master Excel")
    parser.add_argument("--output-name",
                        help="Custom output filename (default: {profile}-{timestamp}.md)")
    args = parser.parse_args()

    profile = load_profile(PROJECT_ROOT / "profiles" / f"{args.profile}.yaml")
    resume_path = PROJECT_ROOT / profile.identity.resume_path
    rubric_path = PROJECT_ROOT / profile.scoring.rubric_path

    resume_text = resume_path.read_text() if resume_path.exists() else ""
    rubric_text = rubric_path.read_text() if rubric_path.exists() else ""

    # Resume placeholder fallback — load from canonical 05-Goals location.
    if not resume_text.strip():
        raise RuntimeError("No resume text found. Check profile.identity.resume_path.")
    if not rubric_text.strip():
        raise RuntimeError("No rubric text found. Check profile.scoring.rubric_path.")

    target_jd_text = _load_jd(args.target_jd, profile) if args.target_jd else None
    if args.target_jd and not target_jd_text:
        print(f"WARNING: could not load --target-jd '{args.target_jd}' — running without gap analysis.")

    client = make_client()
    model = profile.scoring.llm_model

    # Truncate to keep prompts reasonable (rubric ~3K, resume ~5K, JD ~6K)
    resume_for_prompt = resume_text[:8000]
    rubric_for_prompt = rubric_text[:4000]
    jd_for_prompt = target_jd_text[:6000] if target_jd_text else None

    # Step 1 — Skill inference
    print("Inferring skills...")
    skills_raw = call_kimi(
        client, model,
        "You extract structured skills from resumes. Return only valid JSON.",
        build_skills_prompt(resume_for_prompt, rubric_for_prompt),
    )
    skills = parse_kimi_json(skills_raw) or {"hard_skills": [], "soft_skills": [], "tools": []}
    if not isinstance(skills, dict):
        skills = {"hard_skills": [], "soft_skills": [], "tools": []}

    # Step 2 — Adjacent titles
    print("Finding adjacent titles...")
    adj_raw = call_kimi(
        client, model,
        "You identify adjacent career paths. Return only valid JSON.",
        build_adjacent_titles_prompt(resume_for_prompt, rubric_for_prompt),
    )
    adjacent = parse_kimi_json(adj_raw) or []
    if not isinstance(adjacent, list):
        adjacent = []
    target_title_strs = [
        t.get("title", "") for t in adjacent if isinstance(t, dict)
    ][:5]

    # Step 3 — CAR bullets
    print("Generating CAR-format resume bullets...")
    bullets_raw = call_kimi(
        client, model,
        "You write strong resume bullets. Return only valid JSON.",
        build_car_bullets_prompt(resume_for_prompt, rubric_for_prompt, target_title_strs),
        max_tokens=2000,
    )
    bullets = parse_kimi_json(bullets_raw) or []
    if not isinstance(bullets, list):
        bullets = []

    # Step 4 — Gap analysis (optional)
    gap = None
    if jd_for_prompt:
        print("Computing skills gap vs target JD...")
        gap_raw = call_kimi(
            client, model,
            "You assess fit gap honestly. Return only valid JSON.",
            build_gap_analysis_prompt(resume_for_prompt, jd_for_prompt, rubric_for_prompt),
        )
        gap = parse_kimi_json(gap_raw)
        if not isinstance(gap, dict):
            gap = None

    # Step 5 — Cross-reference companies
    companies_hiring = cross_reference_companies(adjacent, COMPANIES_DB_FILE, limit=15)

    # Render + write
    TRANSLATIONS_DIR.mkdir(parents=True, exist_ok=True)
    output_name = args.output_name or f"{args.profile}-{datetime.now():%Y-%m-%d-%H%M}.md"
    output_path = TRANSLATIONS_DIR / output_name

    md = render_markdown_report(
        profile=profile,
        skills=skills,
        adjacent=adjacent,
        bullets=bullets,
        gap=gap,
        jd_text=target_jd_text,
        companies_hiring=companies_hiring,
    )
    output_path.write_text(md)
    print(f"Wrote translation to {output_path}")


if __name__ == "__main__":
    main()
