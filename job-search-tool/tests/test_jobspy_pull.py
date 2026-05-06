"""Tests for tools/jobspy_pull.py — JobSpy + filesystem mocked, no network."""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from tools import jobspy_pull as jp
from tools.jobspy_pull import (
    MASTER_COLUMNS,
    build_arg_parser,
    main,
    merge_csv_into_master,
    normalize_to_master_schema,
    parse_search_terms,
    run_jobspy_pull,
    write_csv,
)


# ---------- parse_search_terms ----------

def test_parse_search_terms_splits_and_trims():
    assert parse_search_terms(
        " GTM Engineer , Forward Deployed Engineer ,Sales Engineer , "
    ) == ["GTM Engineer", "Forward Deployed Engineer", "Sales Engineer"]


def test_parse_search_terms_dedupes_case_insensitively():
    out = parse_search_terms("GTM Engineer,gtm engineer,GTM ENGINEER")
    assert out == ["GTM Engineer"]


def test_parse_search_terms_handles_empty():
    assert parse_search_terms("") == []
    assert parse_search_terms("   ,   ,   ") == []


# ---------- argparse ----------

def test_arg_parser_defaults_match_spec():
    parser = build_arg_parser()
    args = parser.parse_args([
        "--search-terms", "GTM Engineer,Solutions Engineer",
    ])
    assert args.search_terms == "GTM Engineer,Solutions Engineer"
    assert args.location is None
    assert args.sites == "indeed,linkedin,glassdoor"
    assert args.remote_only is False
    assert args.hours_old == 168
    assert args.results_wanted == 40
    assert args.min_salary is None
    assert args.merge_into_master is False
    # output should default to a path under output/
    assert args.output.endswith("jobspy_pulled.csv")


def test_arg_parser_full_flag_set_parses():
    parser = build_arg_parser()
    args = parser.parse_args([
        "--search-terms", "GTM Engineer,Forward Deployed Engineer,Solutions Engineer",
        "--location", "United States",
        "--remote-only",
        "--hours-old", "168",
        "--output", "output/jobspy_pulled.csv",
    ])
    assert args.search_terms == "GTM Engineer,Forward Deployed Engineer,Solutions Engineer"
    assert args.location == "United States"
    assert args.remote_only is True
    assert args.hours_old == 168
    assert args.output == "output/jobspy_pulled.csv"


def test_arg_parser_requires_search_terms():
    parser = build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


# ---------- normalize_to_master_schema ----------

def test_normalize_fills_missing_columns_with_na():
    df = pd.DataFrame([
        {"company": "Acme", "title": "GTM Engineer", "job_url": "https://a/1"},
    ])
    out = normalize_to_master_schema(df)
    # Every master column must be present
    for col in MASTER_COLUMNS:
        assert col in out.columns, f"missing column: {col}"
    # First-N columns must be in MASTER_COLUMNS order
    head = list(out.columns)[: len(MASTER_COLUMNS)]
    assert head == MASTER_COLUMNS


def test_normalize_handles_empty_input():
    out = normalize_to_master_schema(pd.DataFrame())
    assert list(out.columns) == MASTER_COLUMNS
    assert len(out) == 0


def test_normalize_preserves_extra_columns():
    df = pd.DataFrame([
        {"company": "Acme", "title": "X", "job_url": "u",
         "extra_col": "kept"},
    ])
    out = normalize_to_master_schema(df)
    assert "extra_col" in out.columns


# ---------- run_jobspy_pull (scrape_fn injected) ----------

def _fake_jobs_df(n: int, company_prefix: str = "Acme") -> pd.DataFrame:
    rows = []
    for i in range(n):
        rows.append({
            "company": f"{company_prefix}{i}",
            "title": "GTM Engineer",
            "location": "Remote",
            "is_remote": True,
            "min_amount": 150000,
            "max_amount": 180000,
            "interval": "yearly",
            "currency": "USD",
            "date_posted": "2026-05-01",
            "site": "linkedin",
            "job_url": f"https://example.com/job/{i}",
            "description": "JD body.",
        })
    return pd.DataFrame(rows)


def test_run_jobspy_pull_normalizes_and_tags_terms():
    fake_scrape = MagicMock(return_value=_fake_jobs_df(2, company_prefix="A"))
    df = run_jobspy_pull(
        search_terms=["GTM Engineer"],
        location="United States",
        sites=["indeed", "linkedin", "glassdoor"],
        remote_only=True,
        hours_old=168,
        results_wanted=10,
        min_salary=None,
        sleep_seconds=0,
        scrape_fn=fake_scrape,
    )
    assert len(df) == 2
    # Master schema
    for col in MASTER_COLUMNS:
        assert col in df.columns
    # Search term tagged
    assert (df["search_term"] == "GTM Engineer").all()
    assert (df["search_location"] == "United States").all()
    # JobSpy was called once per term with our flags
    fake_scrape.assert_called_once()
    kwargs = fake_scrape.call_args.kwargs
    assert kwargs["search_term"] == "GTM Engineer"
    assert kwargs["site_name"] == ["indeed", "linkedin", "glassdoor"]
    assert kwargs["location"] == "United States"
    assert kwargs["is_remote"] is True
    assert kwargs["hours_old"] == 168


def test_run_jobspy_pull_continues_on_per_term_failure():
    """If JobSpy raises on one term, other terms still run."""
    calls = []

    def flaky(**kwargs):
        calls.append(kwargs["search_term"])
        if kwargs["search_term"] == "GTM Engineer":
            raise RuntimeError("rate-limited")
        return _fake_jobs_df(1, company_prefix="OK")

    df = run_jobspy_pull(
        search_terms=["GTM Engineer", "Solutions Engineer"],
        location=None,
        sites=["linkedin"],
        remote_only=False,
        hours_old=72,
        results_wanted=5,
        min_salary=None,
        sleep_seconds=0,
        scrape_fn=flaky,
    )
    # One term failed, one succeeded.
    assert calls == ["GTM Engineer", "Solutions Engineer"]
    assert len(df) == 1
    assert df.iloc[0]["company"] == "OK0"


def test_run_jobspy_pull_returns_empty_master_schema_when_no_results():
    fake_scrape = MagicMock(return_value=pd.DataFrame())
    df = run_jobspy_pull(
        search_terms=["No Results Term"],
        location=None,
        sites=["linkedin"],
        remote_only=False,
        hours_old=72,
        results_wanted=5,
        min_salary=None,
        sleep_seconds=0,
        scrape_fn=fake_scrape,
    )
    assert df.empty
    assert list(df.columns) == MASTER_COLUMNS


def test_run_jobspy_pull_applies_min_salary_filter():
    """Rows below --min-salary drop; rows with NaN salary pass through."""
    df_in = _fake_jobs_df(3)
    df_in.loc[0, "min_amount"] = 90000   # below threshold -> drop
    df_in.loc[1, "min_amount"] = 200000  # above -> keep
    df_in.loc[2, "min_amount"] = pd.NA   # missing -> keep

    fake_scrape = MagicMock(return_value=df_in)
    out = run_jobspy_pull(
        search_terms=["GTM Engineer"],
        location=None,
        sites=["linkedin"],
        remote_only=False,
        hours_old=72,
        results_wanted=5,
        min_salary=140000,
        sleep_seconds=0,
        scrape_fn=fake_scrape,
    )
    # We dropped exactly the row whose min_amount was 90000.
    kept_amts = out["min_amount"].fillna(-1).tolist()
    assert 90000 not in kept_amts
    assert 200000 in kept_amts
    # One NaN row passed through
    assert (out["min_amount"].isna()).sum() == 1


# ---------- write_csv ----------

def test_write_csv_creates_directory_and_dumps(tmp_path):
    df = _fake_jobs_df(2)
    df = normalize_to_master_schema(df)
    out_path = tmp_path / "subdir" / "out.csv"
    write_csv(df, out_path)
    assert out_path.exists()
    reread = pd.read_csv(out_path)
    assert list(reread.columns)[: len(MASTER_COLUMNS)] == MASTER_COLUMNS
    assert len(reread) == 2


# ---------- merge_csv_into_master ----------

def test_merge_csv_into_master_dedupes_by_url(tmp_path):
    """Existing master with one URL; new pull has two rows including a dup."""
    master_file = tmp_path / "master.xlsx"

    existing = _fake_jobs_df(1)  # job_url=https://example.com/job/0
    existing = normalize_to_master_schema(existing)
    existing.to_excel(master_file, index=False)

    new = _fake_jobs_df(2)  # urls 0 and 1, 0 is the dup
    new = normalize_to_master_schema(new)

    added = merge_csv_into_master(new, master_file)
    assert added == 1, "only the new URL should be appended"

    final = pd.read_excel(master_file)
    assert len(final) == 2
    assert sorted(final["job_url"].tolist()) == [
        "https://example.com/job/0",
        "https://example.com/job/1",
    ]


def test_merge_csv_into_master_creates_when_missing(tmp_path):
    master_file = tmp_path / "fresh.xlsx"
    df = normalize_to_master_schema(_fake_jobs_df(3))
    added = merge_csv_into_master(df, master_file)
    assert added == 3
    assert master_file.exists()
    final = pd.read_excel(master_file)
    assert len(final) == 3


def test_merge_csv_into_master_no_op_on_empty_df(tmp_path):
    master_file = tmp_path / "master.xlsx"
    added = merge_csv_into_master(pd.DataFrame(columns=MASTER_COLUMNS), master_file)
    assert added == 0
    assert not master_file.exists()


# ---------- main() integration with all I/O mocked ----------

def test_main_writes_csv_with_master_schema(monkeypatch, tmp_path):
    """End-to-end CLI run: mock JobSpy, verify CSV has master schema header."""
    out_csv = tmp_path / "out.csv"

    fake_scrape = MagicMock(return_value=_fake_jobs_df(2))
    monkeypatch.setattr(jp, "scrape_jobs", fake_scrape)

    # Patch run_jobspy_pull to use the fake by default-arg, since the wrapper
    # will read jp.scrape_jobs at call time only via its default. We pass
    # scrape_fn directly in the focused test above; here we let main()
    # exercise the real plumbing minus the network.
    real_run = jp.run_jobspy_pull

    def runner(*args, **kwargs):
        kwargs.setdefault("scrape_fn", fake_scrape)
        return real_run(*args, **kwargs)

    monkeypatch.setattr(jp, "run_jobspy_pull", runner)

    rc = main([
        "--search-terms", "GTM Engineer",
        "--location", "United States",
        "--remote-only",
        "--hours-old", "168",
        "--output", str(out_csv),
        "--sleep", "0",
    ])
    assert rc == 0
    assert out_csv.exists()
    df = pd.read_csv(out_csv)
    assert list(df.columns)[: len(MASTER_COLUMNS)] == MASTER_COLUMNS


def test_main_merge_into_master_appends_deduped(monkeypatch, tmp_path):
    out_csv = tmp_path / "out.csv"
    master = tmp_path / "master.xlsx"

    # Seed master with one URL
    seed = normalize_to_master_schema(_fake_jobs_df(1))
    seed.to_excel(master, index=False)

    fake_scrape = MagicMock(return_value=_fake_jobs_df(2))
    monkeypatch.setattr(jp, "scrape_jobs", fake_scrape)

    real_run = jp.run_jobspy_pull
    monkeypatch.setattr(
        jp, "run_jobspy_pull",
        lambda *a, **kw: real_run(*a, **{**kw, "scrape_fn": fake_scrape}),
    )

    rc = main([
        "--search-terms", "GTM Engineer",
        "--output", str(out_csv),
        "--merge-into-master",
        "--master", str(master),
        "--sleep", "0",
    ])
    assert rc == 0
    final = pd.read_excel(master)
    # Started with 1 URL, added 1 new (the dup was deduped).
    assert len(final) == 2


def test_main_rejects_unknown_site(monkeypatch, capsys):
    rc = main([
        "--search-terms", "GTM Engineer",
        "--sites", "linkedin,reddit",  # reddit isn't supported
    ])
    assert rc == 2
    captured = capsys.readouterr()
    assert "reddit" in captured.err


def test_main_rejects_empty_search_terms(monkeypatch, capsys):
    rc = main([
        "--search-terms", "  ,  ",
    ])
    assert rc == 2
    captured = capsys.readouterr()
    assert "search-terms" in captured.err
