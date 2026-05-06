from pathlib import Path
import pandas as pd
import pytest

from tools.master import dedup_within_batch, merge_with_master, reorder_columns


def test_dedup_within_batch_removes_duplicate_urls():
    df = pd.DataFrame([
        {"job_url": "https://a.com/1", "title": "GTM Eng"},
        {"job_url": "https://a.com/1", "title": "GTM Eng"},  # dupe
        {"job_url": "https://a.com/2", "title": "RevOps"},
    ])
    result = dedup_within_batch(df)
    assert len(result) == 2


def test_merge_with_master_no_existing_file(tmp_path):
    new_df = pd.DataFrame([
        {"job_url": "https://a.com/1", "title": "Job A"},
    ])
    master_file = tmp_path / "nonexistent.xlsx"
    merged = merge_with_master(new_df, master_file)
    assert len(merged) == 1


def test_merge_with_master_finds_only_net_new(tmp_path):
    master_file = tmp_path / "master.xlsx"
    pd.DataFrame([
        {"job_url": "https://a.com/1", "title": "Existing"},
    ]).to_excel(master_file, index=False)

    new_df = pd.DataFrame([
        {"job_url": "https://a.com/1", "title": "Existing duplicate"},  # already in master
        {"job_url": "https://a.com/2", "title": "New job"},
    ])
    merged = merge_with_master(new_df, master_file)
    assert len(merged) == 2
    assert "New job" in merged["title"].values


def test_reorder_columns_puts_preferred_first():
    df = pd.DataFrame([{"id": 1, "company": "GitLab", "title": "GTM Eng",
                        "remote_friendly_verified": "Yes", "extra_field": "x"}])
    reordered = reorder_columns(df)
    cols = list(reordered.columns)
    assert cols[0] == "remote_friendly_verified"
    assert cols[1] == "company"
    assert cols[2] == "title"
    assert "extra_field" in cols
