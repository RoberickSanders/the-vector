"""Master Excel I/O: read existing, dedup, merge, reorder columns."""
from pathlib import Path
import pandas as pd


PREFERRED_COLUMN_ORDER = [
    "remote_friendly_verified", "company", "title", "location", "is_remote",
    "min_amount", "max_amount", "interval", "currency",
    "date_posted", "site", "search_term", "search_location",
    "curated_remote_policy", "job_type", "job_level", "job_function",
    "company_industry", "company_num_employees", "company_revenue",
    "company_rating", "skills", "experience_range",
    "fit_score", "why_match", "recommended_action", "missing_skills",  # V1 score.py
    "manager_name", "manager_email", "manager_linkedin", "manager_source",  # V1 find_managers.py
    "job_url", "job_url_direct", "company_url", "description",
    "scraped_at", "id",
]


def dedup_within_batch(df: pd.DataFrame, key: str = "job_url") -> pd.DataFrame:
    if key not in df.columns:
        return df
    return df.drop_duplicates(subset=[key], keep="first").reset_index(drop=True)


def merge_with_master(new_df: pd.DataFrame, master_file: Path | str) -> pd.DataFrame:
    """Dedup new_df, then return concat of (existing master + truly-new rows)."""
    master_file = Path(master_file)
    new_df = dedup_within_batch(new_df)
    if master_file.exists():
        master = pd.read_excel(master_file)
        existing = set(master.get("job_url", pd.Series(dtype=str)).astype(str))
        truly_new = new_df[
            ~new_df.get("job_url", pd.Series(dtype=str)).astype(str).isin(existing)
        ]
        return pd.concat([master, truly_new], ignore_index=True)
    return new_df


def reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in PREFERRED_COLUMN_ORDER if c in df.columns]
    extras = [c for c in df.columns if c not in cols]
    return df[cols + extras]
