"""
Job scraper for the user's job search — supports three source modes.

By default scrapes target ATS sites (Greenhouse / Lever / Ashby) directly
from `config/companies-db.json` (`--source direct`, the new default).
JobSpy was historically the primary path but returns mostly noise on
keyword searches, so it is now supplementary (`--source jobspy`). Use
`--source both` to run them together — results are deduped against the
master Excel via `merge_with_master`.

Usage:
    .venv/bin/python -m tools.scrape --profile example                     # direct (default)
    .venv/bin/python -m tools.scrape --profile example --source jobspy     # jobspy keyword search
    .venv/bin/python -m tools.scrape --profile example --source both       # both, deduped
    .venv/bin/python -m tools.scrape --profile example --quick             # smaller test run (jobspy modes only)
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from jobspy import scrape_jobs

from tools.companies import normalize_company, find_match, load_remote_friendly_companies
from tools.drive import upload_to_drive
from tools.master import merge_with_master, reorder_columns
from tools.profile import load_profile
from tools.scrape_direct import run_direct

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
COMPANIES_FILE = PROJECT_ROOT / "config" / "remote-friendly-companies.json"
DRIVE_STATE_FILE = PROJECT_ROOT / "config" / "drive-state.json"

SITES = ["indeed", "linkedin", "zip_recruiter", "google"]


def flag_remote_friendly(df: pd.DataFrame, registry: dict) -> pd.DataFrame:
    """Add columns flagging if company is on the curated remote-friendly list."""
    matches = []
    policies = []
    for company in df.get("company", []):
        meta = find_match(str(company), registry)
        matches.append("Yes" if meta else "")
        policies.append(meta.get("remote_policy", "") if meta else "")
    df = df.copy()
    df["remote_friendly_verified"] = matches
    df["curated_remote_policy"] = policies
    return df


def run_jobspy_scrape(profile, quick: bool = False) -> pd.DataFrame:
    registry = load_remote_friendly_companies(COMPANIES_FILE)
    print(f"Loaded {len(registry)} curated remote-friendly company aliases")

    results = []
    terms = profile.search.terms[:2] if quick else profile.search.terms
    locs = profile.search.locations[:1] if quick else profile.search.locations
    per_call = 15 if quick else 40

    for term in terms:
        for loc_cfg in locs:
            loc = loc_cfg.location
            is_remote = loc_cfg.is_remote
            label = f"{term} | {loc or 'anywhere'} | remote={is_remote}"
            try:
                kwargs = dict(
                    site_name=SITES,
                    search_term=term,
                    google_search_term=f"{term} jobs {loc or 'remote'} since yesterday",
                    results_wanted=per_call,
                    hours_old=profile.search.hours_old,
                    country_indeed="USA",
                )
                if loc:
                    kwargs["location"] = loc
                if is_remote is not None:
                    kwargs["is_remote"] = is_remote
                df = scrape_jobs(**kwargs)
                if df is not None and len(df):
                    df["search_term"] = term
                    df["search_location"] = loc or "Remote (anywhere)"
                    results.append(df)
                    print(f"  {label}: {len(df)} jobs")
                else:
                    print(f"  {label}: 0 jobs")
            except Exception as e:
                err = str(e)[:120]
                print(f"  {label}: ERROR {err}", file=sys.stderr)

    if not results:
        return pd.DataFrame()
    combined = pd.concat(results, ignore_index=True)
    combined = flag_remote_friendly(combined, registry)
    combined["scraped_at"] = datetime.now().isoformat(timespec="seconds")
    return combined


def run_direct_scrape(profile) -> pd.DataFrame:
    """Run direct ATS scrapers via tools.scrape_direct.run_direct().

    Wrapper that flags remote-friendly companies (matching the JobSpy path)
    so the merged master schema stays consistent regardless of source.
    """
    print("Running direct ATS scrapers (Greenhouse/Lever/Ashby/Workday)...")
    df = run_direct(profile)
    if df is None or df.empty:
        return pd.DataFrame()
    registry = load_remote_friendly_companies(COMPANIES_FILE)
    df = flag_remote_friendly(df, registry)
    if "scraped_at" not in df.columns:
        df["scraped_at"] = datetime.now().isoformat(timespec="seconds")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True,
                        help="Profile name (e.g., 'example'); reads profiles/{name}.yaml")
    parser.add_argument(
        "--source",
        choices=["direct", "jobspy", "both"],
        default="direct",
        help=("Source mode. 'direct' (default): scrape target ATSes from "
              "companies-db.json. 'jobspy': run keyword JobSpy scrape. "
              "'both': run both and merge."),
    )
    parser.add_argument("--quick", action="store_true",
                        help="Small JobSpy run (2 terms, 1 loc). Direct mode ignores --quick.")
    parser.add_argument("--no-upload", action="store_true",
                        help="Skip Google Drive upload step")
    args = parser.parse_args()

    profile_path = PROJECT_ROOT / "profiles" / f"{args.profile}.yaml"
    profile = load_profile(profile_path)
    master_file = PROJECT_ROOT / "output" / profile.drive.master_filename
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Profile: {profile.name}")
    print(f"Source: {args.source}")

    frames: list[pd.DataFrame] = []
    if args.source in ("direct", "both"):
        df_direct = run_direct_scrape(profile)
        if not df_direct.empty:
            frames.append(df_direct)
            print(f"Direct: {len(df_direct)} jobs")
        else:
            print("Direct: 0 jobs")
    if args.source in ("jobspy", "both"):
        df_jobspy = run_jobspy_scrape(profile, quick=args.quick)
        if not df_jobspy.empty:
            frames.append(df_jobspy)
            print(f"JobSpy: {len(df_jobspy)} jobs")
        else:
            print("JobSpy: 0 jobs")

    if not frames:
        print("No jobs returned. Exiting.")
        return
    new_jobs = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]

    merged = merge_with_master(new_jobs, master_file)
    sort_keys = [k for k in ("remote_friendly_verified", "scraped_at") if k in merged.columns]
    if sort_keys:
        merged = merged.sort_values(
            by=sort_keys,
            ascending=[False] * len(sort_keys),
        ).reset_index(drop=True)
    merged = reorder_columns(merged)
    merged.to_excel(master_file, index=False)
    print(f"Wrote {len(merged)} total jobs to {master_file}")
    if "remote_friendly_verified" in merged.columns:
        print(f"  Remote-friendly verified: "
              f"{(merged['remote_friendly_verified'] == 'Yes').sum()}")

    if not args.no_upload:
        upload_to_drive(master_file, profile_name=args.profile,
                        state_file=DRIVE_STATE_FILE)


if __name__ == "__main__":
    main()
