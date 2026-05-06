"""LLM fit scoring against a profile's rubric. Uses Kimi via Anthropic-compatible API.

Reads master Excel, finds rows without fit_score, scores each via Kimi,
writes results back to the master.

Note: Constants (KIMI_BASE_URL, SDK choice) are duplicated from upstream's
llm_router.py per the "the upstream is hands-off" rule. Do NOT import from upstream.
"""
import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from anthropic import Anthropic

from tools.profile import load_profile
from tools.master import reorder_columns
from tools.drive import upload_to_drive

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
DRIVE_STATE_FILE = PROJECT_ROOT / "config" / "drive-state.json"
COMPANIES_DB_FILE = PROJECT_ROOT / "config" / "companies-db.json"

# Load environment variables from workspace .env (where KIMI_API_KEY lives)
WORKSPACE_ENV = PROJECT_ROOT.parent.parent.parent / ".env"
if WORKSPACE_ENV.exists():
    load_dotenv(WORKSPACE_ENV, override=True)  # override empty strings exported by Claude Code

# Kimi exposes an Anthropic-compatible API at this endpoint.
# Mirrored from the upstream llm_router.py — do not import from upstream.
KIMI_BASE_URL = "https://api.kimi.com/coding"

SCORING_INSTRUCTIONS = """
You are scoring how well a job posting matches a candidate's profile.

CANDIDATE PROFILE RUBRIC:
{rubric}

JOB DESCRIPTION:
{jd}

Return STRICT JSON (no prose, no markdown fences) with exactly these fields:
{{
  "fit_score": <integer 1-10>,
  "why_match": "<one or two sentences explaining the score>",
  "recommended_action": "<one of: apply, dm, skip>",
  "missing_skills": ["<skill1>", "<skill2>"]
}}

Be strict on the rubric's hard NO signals. Be honest, not generous.
"""


def build_scoring_prompt(rubric: str, jd: str) -> str:
    return SCORING_INSTRUCTIONS.format(rubric=rubric, jd=jd)


def parse_score_response(raw: str) -> Optional[dict]:
    """Parse a Kimi/Claude scoring response. Tolerate markdown fences."""
    if not raw:
        return None
    cleaned = re.sub(r"^```(?:json)?\n?", "", raw.strip())
    cleaned = re.sub(r"\n?```$", "", cleaned).strip()
    try:
        parsed = json.loads(cleaned)
        if "fit_score" not in parsed:
            return None
        return parsed
    except json.JSONDecodeError:
        return None


def load_d100_company_names(path: Path | str = COMPANIES_DB_FILE) -> set[str]:
    """Return lowercase d100 company names for prioritization. Empty set if missing."""
    path = Path(path)
    if not path.exists():
        return set()
    try:
        db = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return set()
    return {
        str(c.get("name", "")).strip().lower()
        for c in db.get("companies", [])
        if c.get("name")
    }


TARGET_ROLE_KEYWORDS = (
    "gtm engineer", "growth engineer", "revops engineer",
    "revenue operations engineer", "applied ai engineer",
    "forward deployed engineer", "marketing operations engineer",
    "sales engineer", "solutions engineer", "sales operations",
)


def round_robin_by_company(df: pd.DataFrame) -> pd.DataFrame:
    """Reorder so first occurrence of each company precedes second, etc.

    E.g., if MongoDB has rows [M1, M2, M3] and Notion has rows [N1, N2],
    output: [M1, N1, M2, N2, M3]. Preserves within-company order.

    Why: if a single big company (MongoDB, ~433 jobs) sits at the top of
    a tier, naively scoring `limit=100` jobs will burn 100 calls on
    MongoDB alone. Round-robin spreads attention across companies.
    """
    if df.empty or "company" not in df.columns:
        return df
    df = df.copy()
    # cumcount() yields 0,1,2... per company. Stable sort by cumcount keeps
    # within-company ordering AND interleaves across companies.
    df["_co_pos"] = df.groupby("company").cumcount()
    df = df.sort_values(by=["_co_pos"], kind="stable").drop(columns=["_co_pos"])
    return df.reset_index(drop=True)


def prioritize_d100(unscored: pd.DataFrame, d100_names: set[str],
                    limit: int) -> pd.DataFrame:
    """3-tier priority: d100+target_role > d100 > rest, capped at limit.

    Surfaces the highest-signal jobs first: d100 companies AND the title
    contains an actual user target-role keyword (GTM Engineer, Forward
    Deployed Engineer, etc.). Then d100-only (any title). Then everything
    else. Within each tier, round-robin by company so no single company
    dominates the scoring batch.
    """
    if not d100_names or "company" not in unscored.columns:
        return unscored.head(limit)
    is_d100 = unscored["company"].astype(str).str.lower().isin(d100_names)
    if "title" in unscored.columns:
        title_lower = unscored["title"].astype(str).str.lower()
        is_target_role = title_lower.apply(
            lambda t: any(kw in t for kw in TARGET_ROLE_KEYWORDS)
        )
    else:
        is_target_role = pd.Series(False, index=unscored.index)
    tier_1 = round_robin_by_company(unscored[is_d100 & is_target_role])
    tier_2 = round_robin_by_company(unscored[is_d100 & ~is_target_role])
    tier_3 = round_robin_by_company(unscored[~is_d100])
    return pd.concat([tier_1, tier_2, tier_3]).head(limit)


def make_client() -> Anthropic:
    api_key = os.environ.get("KIMI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "KIMI_API_KEY not set. Add it to your .env or export it."
        )
    return Anthropic(api_key=api_key, base_url=KIMI_BASE_URL)


def score_job(client: Anthropic, model: str, rubric: str, jd: str) -> Optional[dict]:
    """Score a single JD against the rubric. Returns parsed dict or None."""
    system = (
        "You are scoring how well a job posting matches a candidate's profile. "
        "Be strict on hard NO signals. Be honest, not generous."
    )
    user_prompt = build_scoring_prompt(rubric, jd)
    for attempt in range(3):
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=512,
                system=system,
                messages=[{"role": "user", "content": user_prompt}],
            )
            raw = resp.content[0].text if resp.content else ""
            parsed = parse_score_response(raw)
            if parsed is not None:
                return parsed
        except Exception as e:
            print(f"  Score attempt {attempt + 1} failed: {str(e)[:120]}")
            time.sleep(1.5 ** attempt)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    parser.add_argument("--limit", type=int, default=200,
                        help="Max jobs to score per run (cost guardrail)")
    parser.add_argument("--no-upload", action="store_true")
    args = parser.parse_args()

    profile = load_profile(PROJECT_ROOT / "profiles" / f"{args.profile}.yaml")
    master_file = OUTPUT_DIR / profile.drive.master_filename
    rubric = (PROJECT_ROOT / profile.scoring.rubric_path).read_text()

    df = pd.read_excel(master_file)
    if "fit_score" not in df.columns:
        df["fit_score"] = pd.NA
        df["why_match"] = pd.NA
        df["recommended_action"] = pd.NA
        df["missing_skills"] = pd.NA

    unscored_all = df[df["fit_score"].isna()]
    d100_names = load_d100_company_names()
    if d100_names and "company" in unscored_all.columns:
        is_d100 = unscored_all["company"].astype(str).str.lower().isin(d100_names)
        d100_count = int(is_d100.sum())
        other_count = int((~is_d100).sum())
        unscored = prioritize_d100(unscored_all, d100_names, args.limit)
        print(
            f"Prioritizing d100: {d100_count} d100 + {other_count} other "
            f"(scoring {len(unscored)} this run, capped at {args.limit})"
        )
    else:
        unscored = unscored_all.head(args.limit)
        print(f"Scoring {len(unscored)} unscored jobs (limit: {args.limit})")
    if unscored.empty:
        print("Nothing to score.")
        return

    client = make_client()
    for idx, row in unscored.iterrows():
        jd = str(row.get("description", "") or "")
        if not jd or len(jd) < 50:
            print(f"  [{idx}] skipping (no/short JD)")
            df.at[idx, "fit_score"] = 0
            df.at[idx, "why_match"] = "No job description available"
            df.at[idx, "recommended_action"] = "skip"
            df.at[idx, "missing_skills"] = "[]"
            continue
        result = score_job(client, profile.scoring.llm_model, rubric, jd[:8000])
        if result is None:
            print(f"  [{idx}] FAILED to score: {row.get('company')}")
            continue
        df.at[idx, "fit_score"] = result["fit_score"]
        df.at[idx, "why_match"] = result["why_match"]
        df.at[idx, "recommended_action"] = result["recommended_action"]
        df.at[idx, "missing_skills"] = json.dumps(result.get("missing_skills", []))
        print(f"  [{idx}] {row.get('company')}: {result['fit_score']}/10 — {result['recommended_action']}")

    df = reorder_columns(df)
    df.to_excel(master_file, index=False)
    print(f"Updated {master_file}")

    if not args.no_upload:
        upload_to_drive(master_file, profile_name=args.profile,
                        state_file=DRIVE_STATE_FILE)


if __name__ == "__main__":
    main()
