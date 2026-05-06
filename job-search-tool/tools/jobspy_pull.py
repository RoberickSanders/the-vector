"""Standalone JobSpy puller for the Vector.

Wraps `python-jobspy` (https://github.com/speedyapply/JobSpy) as a focused
data source: takes a list of search terms + a location, sweeps LinkedIn /
Indeed / Glassdoor (configurable), normalizes the result frame to the
master Excel schema, optionally merges into the master file deduped by
`job_url`.

Differs from the existing tools/scrape.py JobSpy path in that it is
independent of profile YAML, takes its filters from CLI flags, and writes
a flat CSV by default. Useful for ad-hoc pulls outside the profile loop.

Install (already part of .venv):
    .venv/bin/pip install python-jobspy

CLI:
    .venv/bin/python -m tools.jobspy_pull \\
        --search-terms "GTM Engineer,Forward Deployed Engineer" \\
        --location "United States" \\
        --remote-only \\
        --hours-old 168 \\
        --output output/jobspy_pulled.csv

    .venv/bin/python -m tools.jobspy_pull \\
        --search-terms "Sales Engineer" \\
        --location "Tampa, FL" \\
        --merge-into-master

JobSpy authentication: JobSpy itself does NOT require API keys, but the
underlying boards (LinkedIn especially) rate-limit. We catch all
exceptions per term/site combo and continue, so a 429 on LinkedIn does
not kill the rest of the run.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from jobspy import scrape_jobs

from tools.master import merge_with_master, reorder_columns

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
DEFAULT_OUTPUT = OUTPUT_DIR / "jobspy_pulled.csv"

# Default site mix optimised for US tech jobs. ZipRecruiter / Google can be
# added with --sites if the caller wants them. Indeed + LinkedIn + Glassdoor
# are the highest-signal trio for the role mix the user targets (GTM / FDE / SE).
DEFAULT_SITES = ["indeed", "linkedin", "glassdoor"]
ALL_SITES = ["indeed", "linkedin", "glassdoor", "google", "zip_recruiter", "bayt", "bdjobs"]

# Master Excel canonical schema lives in tools/master.PREFERRED_COLUMN_ORDER.
# Pull the subset we know JobSpy emits so we can guarantee a well-shaped
# CSV regardless of which sites returned data this run.
MASTER_COLUMNS = [
    "company",
    "title",
    "location",
    "is_remote",
    "min_amount",
    "max_amount",
    "interval",
    "currency",
    "date_posted",
    "site",
    "job_url",
    "job_url_direct",
    "company_url",
    "description",
    "search_term",
    "search_location",
    "scraped_at",
]

# Arbitrary cool-down between (term, site) combos. JobSpy LinkedIn especially
# can 429 when hammered. 2-3 seconds is a comfortable middle-ground.
INTER_CALL_SLEEP = 2.0

logger = logging.getLogger(__name__)


def parse_search_terms(raw: str) -> list[str]:
    """Comma-separated string -> deduped list, trimmed, drops empties.

    "GTM Engineer, Forward Deployed Engineer ,Sales Engineer," ->
    ["GTM Engineer", "Forward Deployed Engineer", "Sales Engineer"]
    """
    if not raw:
        return []
    parts = [p.strip() for p in raw.split(",")]
    seen: set[str] = set()
    out: list[str] = []
    for p in parts:
        if not p:
            continue
        # Case-insensitive de-dup but keep original casing of first occurrence.
        key = p.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def normalize_to_master_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Make sure every MASTER_COLUMNS column exists; fill missing with NaN.

    JobSpy's column set varies slightly by site. By normalizing here we
    keep downstream merge logic (master.merge_with_master) happy.
    Returns a fresh DataFrame ordered with MASTER_COLUMNS first, then any
    extras JobSpy returned.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=MASTER_COLUMNS)
    out = df.copy()
    for col in MASTER_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    extras = [c for c in out.columns if c not in MASTER_COLUMNS]
    return out[MASTER_COLUMNS + extras]


def run_jobspy_pull(
    search_terms: list[str],
    location: Optional[str],
    sites: list[str],
    remote_only: bool,
    hours_old: int,
    results_wanted: int,
    min_salary: Optional[int],
    sleep_seconds: float = INTER_CALL_SLEEP,
    scrape_fn=scrape_jobs,
) -> pd.DataFrame:
    """Sweep the requested sites for each search term and concat the frames.

    `scrape_fn` is overridable for tests so we never touch the network
    in the suite.

    Salary filter: JobSpy itself doesn't accept a salary minimum, so we
    apply min_salary post-hoc against `min_amount` (rows with no salary
    info pass through).
    """
    frames: list[pd.DataFrame] = []
    for term in search_terms:
        try:
            kwargs = dict(
                site_name=sites,
                search_term=term,
                results_wanted=results_wanted,
                hours_old=hours_old,
                country_indeed="USA",
            )
            if location:
                kwargs["location"] = location
                # Mirror tools/scrape.py: feed Google Jobs a more natural query.
                kwargs["google_search_term"] = (
                    f"{term} jobs {location} since yesterday"
                )
            else:
                kwargs["google_search_term"] = f"{term} jobs since yesterday"
            if remote_only:
                kwargs["is_remote"] = True

            df = scrape_fn(**kwargs)
            if df is not None and len(df):
                df = df.copy()
                df["search_term"] = term
                df["search_location"] = location or "Remote (anywhere)"
                frames.append(df)
                logger.info("term=%r: %d jobs", term, len(df))
            else:
                logger.info("term=%r: 0 jobs", term)
        except Exception as e:
            # JobSpy can raise on rate-limit, captcha, or transient network
            # failures. We log and continue so one bad term doesn't kill the
            # whole pull.
            err = str(e)[:200]
            logger.warning("term=%r failed: %s", term, err)

        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    if not frames:
        return pd.DataFrame(columns=MASTER_COLUMNS)

    combined = pd.concat(frames, ignore_index=True)
    combined["scraped_at"] = datetime.now().isoformat(timespec="seconds")
    combined = normalize_to_master_schema(combined)

    # Salary floor: drop rows where min_amount is set and below threshold.
    # NaN min_amount rows pass through (no advertised salary != bad fit).
    if min_salary is not None and min_salary > 0 and "min_amount" in combined.columns:
        amt = pd.to_numeric(combined["min_amount"], errors="coerce")
        keep = amt.isna() | (amt >= min_salary)
        combined = combined[keep].reset_index(drop=True)

    return combined


def write_csv(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def merge_csv_into_master(new_df: pd.DataFrame, master_file: Path) -> int:
    """Merge new rows into the master Excel deduped by job_url.

    Returns the number of newly-added rows (i.e. master row count delta).
    """
    if new_df is None or new_df.empty:
        return 0
    before = 0
    if master_file.exists():
        try:
            existing = pd.read_excel(master_file)
            before = len(existing)
        except Exception as e:
            logger.warning(
                "could not read existing master at %s (%s) — treating as new",
                master_file,
                e,
            )
            before = 0
    merged = merge_with_master(new_df, master_file)
    merged = reorder_columns(merged)
    master_file.parent.mkdir(parents=True, exist_ok=True)
    merged.to_excel(master_file, index=False)
    return len(merged) - before


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Pull jobs via JobSpy across LinkedIn / Indeed / Glassdoor and "
            "write a Vector-schema CSV. Optionally merge into the master."
        ),
    )
    parser.add_argument(
        "--search-terms",
        required=True,
        help='Comma-separated list, e.g. "GTM Engineer,Forward Deployed Engineer".',
    )
    parser.add_argument(
        "--location",
        default=None,
        help='Optional location filter, e.g. "United States" or "Tampa, FL".',
    )
    parser.add_argument(
        "--sites",
        default=",".join(DEFAULT_SITES),
        help=(
            "Comma-separated JobSpy sites. Default: "
            f"{','.join(DEFAULT_SITES)}. All supported: {','.join(ALL_SITES)}."
        ),
    )
    parser.add_argument(
        "--remote-only",
        action="store_true",
        help="Filter to roles JobSpy flagged as remote.",
    )
    parser.add_argument(
        "--hours-old",
        type=int,
        default=168,
        help="Only pull jobs posted within this many hours. Default 168 (7 days).",
    )
    parser.add_argument(
        "--results-wanted",
        type=int,
        default=40,
        help="Per-site, per-term result cap passed to JobSpy. Default 40.",
    )
    parser.add_argument(
        "--min-salary",
        type=int,
        default=None,
        help="Drop rows where min_amount is set and below this threshold.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help=f"CSV output path. Default {DEFAULT_OUTPUT}.",
    )
    parser.add_argument(
        "--merge-into-master",
        action="store_true",
        help=(
            "Append new rows (deduped by job_url) into the master Excel. "
            "Use --master to override the path."
        ),
    )
    parser.add_argument(
        "--master",
        default=str(OUTPUT_DIR / "example-jobs-master.xlsx"),
        help="Master Excel path (used when --merge-into-master is set).",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=INTER_CALL_SLEEP,
        help=f"Seconds between term sweeps. Default {INTER_CALL_SLEEP}.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    terms = parse_search_terms(args.search_terms)
    if not terms:
        print("--search-terms is empty after parsing.", file=sys.stderr)
        return 2
    sites = parse_search_terms(args.sites) or DEFAULT_SITES
    bad_sites = [s for s in sites if s not in ALL_SITES]
    if bad_sites:
        print(
            f"Unknown --sites value(s): {bad_sites}. Valid: {ALL_SITES}.",
            file=sys.stderr,
        )
        return 2

    print(f"Search terms: {terms}")
    print(f"Sites: {sites}")
    print(f"Location: {args.location or '(any)'}")
    print(f"Remote only: {args.remote_only}")
    print(f"Hours old: {args.hours_old}")

    df = run_jobspy_pull(
        search_terms=terms,
        location=args.location,
        sites=sites,
        remote_only=args.remote_only,
        hours_old=args.hours_old,
        results_wanted=args.results_wanted,
        min_salary=args.min_salary,
        sleep_seconds=args.sleep,
    )

    output_path = Path(args.output)
    write_csv(df, output_path)
    print(f"Wrote {len(df)} jobs to {output_path}")

    if args.merge_into_master:
        master_file = Path(args.master)
        added = merge_csv_into_master(df, master_file)
        print(f"Master file: {master_file}")
        print(f"Newly-added rows (deduped by job_url): {added}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
