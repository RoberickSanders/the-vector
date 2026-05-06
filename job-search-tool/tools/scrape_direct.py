"""Direct careers-page scraping for d100 target companies.

Loads config/companies-db.json, runs the appropriate ATS scraper for each
company that has an ATS+slug, normalizes results, and merges into the
profile's master Excel via merge_with_master.

Usage:
    .venv/bin/python -m tools.scrape_direct --profile example
    .venv/bin/python -m tools.scrape_direct --profile example --no-upload
    .venv/bin/python -m tools.scrape_direct --profile example --ats greenhouse,lever
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from tools.direct_scrapers import (
    scrape_greenhouse,
    scrape_lever,
    scrape_ashby,
    scrape_workday,
)
from tools.drive import upload_to_drive
from tools.master import merge_with_master, reorder_columns
from tools.profile import load_profile

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
COMPANIES_DB_FILE = PROJECT_ROOT / "config" / "companies-db.json"
DRIVE_STATE_FILE = PROJECT_ROOT / "config" / "drive-state.json"

SCRAPER_DISPATCH = {
    "greenhouse": scrape_greenhouse,
    "lever": scrape_lever,
    "ashby": scrape_ashby,
    "workday": scrape_workday,
}

SUPPORTED_ATS = set(SCRAPER_DISPATCH.keys())


def load_companies_db(path: Path | str = COMPANIES_DB_FILE) -> list[dict]:
    """Load companies-db.json and return the companies list."""
    path = Path(path)
    if not path.exists():
        return []
    raw = json.loads(path.read_text())
    return raw.get("companies", [])


def run_direct(profile, ats_filter: set[str] | None = None) -> pd.DataFrame:
    """Top-level entry point used by both scrape_direct.main() and scrape.main().

    Loads companies-db, runs ATS scrapers, returns a normalized DataFrame.
    Returns empty DataFrame if no companies or no jobs found. The `profile`
    arg is currently unused (no per-profile direct-scrape config yet) but
    is accepted so callers stay symmetric with run_jobspy_scrape(profile).
    """
    companies = load_companies_db()
    if not companies:
        print("ERROR: no companies in config/companies-db.json", file=sys.stderr)
        return pd.DataFrame()
    return run_direct_scrape(companies, ats_filter=ats_filter)


def run_direct_scrape(
    companies: list[dict],
    ats_filter: set[str] | None = None,
) -> pd.DataFrame:
    """Run the appropriate scraper for each company with a known ATS+slug.

    Per-company errors are caught + logged so one bad slug can't crash
    the whole run.
    """
    all_rows: list[dict] = []
    scraped_at = datetime.now().isoformat(timespec="seconds")
    skipped_unsupported = 0
    skipped_no_slug = 0
    err_count = 0

    for c in companies:
        name = c.get("name") or ""
        ats = (c.get("ats") or "").lower()
        slug = c.get("ats_slug")
        if ats not in SUPPORTED_ATS:
            skipped_unsupported += 1
            continue
        if ats_filter and ats not in ats_filter:
            continue
        if not slug:
            skipped_no_slug += 1
            continue

        scraper = SCRAPER_DISPATCH[ats]
        try:
            jobs = scraper(slug, name)
        except Exception as e:
            err_count += 1
            err = str(e)[:120]
            print(f"  [{ats}/{slug}] {name}: ERROR {err}", file=sys.stderr)
            continue

        if not jobs:
            print(f"  [{ats}/{slug}] {name}: 0 jobs (404 or empty)")
            continue

        for j in jobs:
            j["scraped_at"] = scraped_at
        all_rows.extend(jobs)
        print(f"  [{ats}/{slug}] {name}: {len(jobs)} jobs")

    if not all_rows:
        return pd.DataFrame()
    df = pd.DataFrame(all_rows)
    print(
        f"\nTotals: {len(df)} jobs, "
        f"{skipped_unsupported} skipped (workday/custom/unknown), "
        f"{skipped_no_slug} skipped (no slug), "
        f"{err_count} errors"
    )
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile", required=True,
        help="Profile name (e.g., 'example'); reads profiles/{name}.yaml",
    )
    parser.add_argument(
        "--no-upload", action="store_true",
        help="Skip Google Drive upload step",
    )
    parser.add_argument(
        "--ats",
        help=("Comma-separated ATS filter, e.g. 'greenhouse,lever'. "
              "Default: run all supported ATSes."),
    )
    args = parser.parse_args()

    profile_path = PROJECT_ROOT / "profiles" / f"{args.profile}.yaml"
    profile = load_profile(profile_path)
    master_file = OUTPUT_DIR / profile.drive.master_filename
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ats_filter = None
    if args.ats:
        ats_filter = {a.strip().lower() for a in args.ats.split(",") if a.strip()}
        unknown = ats_filter - SUPPORTED_ATS
        if unknown:
            print(f"Warning: ignoring unknown ATS filter values: {unknown}")
        ats_filter = ats_filter & SUPPORTED_ATS
        if not ats_filter:
            print("ERROR: --ats produced empty filter. Supported: "
                  f"{sorted(SUPPORTED_ATS)}", file=sys.stderr)
            sys.exit(2)

    companies = load_companies_db()
    if not companies:
        print("ERROR: no companies in config/companies-db.json", file=sys.stderr)
        sys.exit(2)

    print(f"Profile: {profile.name}")
    print(f"Companies in DB: {len(companies)}")
    print(f"Master file: {master_file}")
    if ats_filter:
        print(f"ATS filter: {sorted(ats_filter)}")
    print()

    # NB: We pass companies through run_direct_scrape directly here (not via
    # run_direct) so we get the existing per-ATS filter UX. run_direct() is
    # the simpler entry-point used by tools.scrape when --source=direct.
    new_jobs = run_direct_scrape(companies, ats_filter=ats_filter)
    if new_jobs.empty:
        print("No jobs returned. Exiting.")
        return

    merged = merge_with_master(new_jobs, master_file)
    sort_keys = [k for k in ("scraped_at",) if k in merged.columns]
    if sort_keys:
        merged = merged.sort_values(by=sort_keys, ascending=False).reset_index(drop=True)
    merged = reorder_columns(merged)
    merged.to_excel(master_file, index=False)
    print(f"\nWrote {len(merged)} total jobs to {master_file}")

    if not args.no_upload:
        upload_to_drive(
            master_file, profile_name=args.profile,
            state_file=DRIVE_STATE_FILE,
        )


if __name__ == "__main__":
    main()
