"""Tests for tools/scrape.py — verifies the --source flag routes correctly.

The actual scrape functions (JobSpy network, direct ATS) are mocked so the
suite never touches the network. We just verify dispatch behavior.
"""
import sys
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from tools import scrape as scrape_mod


def _fake_profile():
    """Minimal stand-in for a Profile object with the attributes scrape.main uses."""
    drive = MagicMock()
    drive.master_filename = "test-jobs-master.xlsx"
    p = MagicMock()
    p.name = "test"
    p.drive = drive
    return p


@pytest.fixture
def patched_io(monkeypatch, tmp_path):
    """Patches profile loading + filesystem side effects so main() can run end-to-end.

    Yields a dict with handles to the mocked scrape functions for assertions.
    """
    profile = _fake_profile()
    direct_mock = MagicMock(return_value=pd.DataFrame())
    jobspy_mock = MagicMock(return_value=pd.DataFrame())
    monkeypatch.setattr(scrape_mod, "load_profile", lambda p: profile)
    monkeypatch.setattr(scrape_mod, "run_direct_scrape", direct_mock)
    monkeypatch.setattr(scrape_mod, "run_jobspy_scrape", jobspy_mock)
    # Redirect output dir so we don't pollute real output/
    monkeypatch.setattr(scrape_mod, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(
        scrape_mod, "PROJECT_ROOT",
        tmp_path,
    )
    yield {"direct": direct_mock, "jobspy": jobspy_mock}


def _run_main(argv):
    with patch.object(sys, "argv", ["tools.scrape", *argv]):
        scrape_mod.main()


def test_source_default_runs_only_direct(patched_io):
    _run_main(["--profile", "example", "--no-upload"])
    assert patched_io["direct"].called, "direct scraper must run when --source is omitted"
    assert not patched_io["jobspy"].called, "jobspy must NOT run by default"


def test_source_jobspy_runs_only_jobspy(patched_io):
    _run_main(["--profile", "example", "--source", "jobspy", "--no-upload"])
    assert patched_io["jobspy"].called
    assert not patched_io["direct"].called


def test_source_both_runs_both(patched_io):
    _run_main(["--profile", "example", "--source", "both", "--no-upload"])
    assert patched_io["direct"].called
    assert patched_io["jobspy"].called
