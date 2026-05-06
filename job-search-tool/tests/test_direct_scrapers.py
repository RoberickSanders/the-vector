"""Unit tests for tools/direct_scrapers.py.

Each ATS scraper is tested with a mocked HTTP response built from a real
sample JSON payload (trimmed to 2 jobs) saved under tests/fixtures/.
The live API is never hit by the test suite.
"""
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from tools.direct_scrapers import (
    scrape_greenhouse,
    scrape_lever,
    scrape_ashby,
    scrape_workday,
    strip_html,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _load_fixture(name: str):
    return json.loads((FIXTURES / name).read_text())


def _mock_response(payload):
    mock = MagicMock()
    mock.status_code = 200
    mock.json.return_value = payload
    mock.raise_for_status = MagicMock()
    return mock


# ---------- Greenhouse ----------

def test_scrape_greenhouse_returns_list_of_dicts():
    payload = _load_fixture("greenhouse-sample.json")
    with patch("tools.direct_scrapers.requests.get",
               return_value=_mock_response(payload)):
        jobs = scrape_greenhouse("gitlab", "GitLab")
    assert isinstance(jobs, list)
    assert len(jobs) == 2
    for j in jobs:
        assert isinstance(j, dict)


def test_scrape_greenhouse_normalizes_required_fields():
    payload = _load_fixture("greenhouse-sample.json")
    with patch("tools.direct_scrapers.requests.get",
               return_value=_mock_response(payload)):
        jobs = scrape_greenhouse("gitlab", "GitLab")
    job = jobs[0]
    for k in ("company", "title", "location", "is_remote",
              "job_url", "description", "date_posted",
              "site", "search_term", "search_location"):
        assert k in job, f"missing key {k}"
    assert job["company"] == "GitLab"
    assert job["site"] == "direct_greenhouse"
    assert job["search_term"] == "direct_scrape"
    assert job["search_location"] == "direct_company"


def test_scrape_greenhouse_strips_html_from_description():
    payload = _load_fixture("greenhouse-sample.json")
    with patch("tools.direct_scrapers.requests.get",
               return_value=_mock_response(payload)):
        jobs = scrape_greenhouse("gitlab", "GitLab")
    desc = jobs[0]["description"]
    # No HTML tags, no &lt; entity leftovers
    assert "<p>" not in desc
    assert "<div" not in desc
    assert "&lt;" not in desc
    assert "&gt;" not in desc
    assert len(desc) > 50  # actually has content


# ---------- Lever ----------

def test_scrape_lever_returns_list_of_dicts():
    payload = _load_fixture("lever-sample.json")
    with patch("tools.direct_scrapers.requests.get",
               return_value=_mock_response(payload)):
        jobs = scrape_lever("plaid", "Plaid")
    assert isinstance(jobs, list)
    assert len(jobs) == 2


def test_scrape_lever_normalizes_required_fields():
    payload = _load_fixture("lever-sample.json")
    with patch("tools.direct_scrapers.requests.get",
               return_value=_mock_response(payload)):
        jobs = scrape_lever("plaid", "Plaid")
    job = jobs[0]
    for k in ("company", "title", "location", "is_remote",
              "job_url", "description", "date_posted",
              "site", "search_term", "search_location"):
        assert k in job
    assert job["company"] == "Plaid"
    assert job["site"] == "direct_lever"


def test_scrape_lever_uses_plain_description():
    payload = _load_fixture("lever-sample.json")
    with patch("tools.direct_scrapers.requests.get",
               return_value=_mock_response(payload)):
        jobs = scrape_lever("plaid", "Plaid")
    desc = jobs[0]["description"]
    assert "<div>" not in desc
    assert len(desc) > 50


# ---------- Ashby ----------

def test_scrape_ashby_returns_list_of_dicts():
    payload = _load_fixture("ashby-sample.json")
    with patch("tools.direct_scrapers.requests.get",
               return_value=_mock_response(payload)):
        jobs = scrape_ashby("supabase", "Supabase")
    assert isinstance(jobs, list)
    assert len(jobs) == 2


def test_scrape_ashby_normalizes_required_fields_and_remote_flag():
    payload = _load_fixture("ashby-sample.json")
    with patch("tools.direct_scrapers.requests.get",
               return_value=_mock_response(payload)):
        jobs = scrape_ashby("supabase", "Supabase")
    job = jobs[0]
    for k in ("company", "title", "location", "is_remote",
              "job_url", "description", "date_posted",
              "site", "search_term", "search_location"):
        assert k in job
    assert job["company"] == "Supabase"
    assert job["site"] == "direct_ashby"
    # Ashby supabase fixture has at least one Remote job
    any_remote = any(j["is_remote"] is True for j in jobs)
    assert any_remote


def test_scrape_ashby_strips_html():
    payload = _load_fixture("ashby-sample.json")
    with patch("tools.direct_scrapers.requests.get",
               return_value=_mock_response(payload)):
        jobs = scrape_ashby("supabase", "Supabase")
    desc = jobs[0]["description"]
    assert "<p" not in desc
    assert "</p>" not in desc
    assert len(desc) > 50


# ---------- Workday ----------

def _mock_workday_response(payload, status=200):
    """Build a mock for requests.post that mimics Workday's POST endpoint."""
    mock = MagicMock()
    mock.status_code = status
    mock.json.return_value = payload
    return mock


def test_scrape_workday_normalizes_required_fields():
    payload = _load_fixture("workday-sample.json")
    # Second pagination call returns empty so the loop exits
    empty = {"total": 0, "jobPostings": []}
    with patch(
        "tools.direct_scrapers.requests.post",
        side_effect=[
            _mock_workday_response(payload),
            _mock_workday_response(empty),
        ],
    ):
        jobs = scrape_workday("nvidia/wd5/NVIDIAExternalCareerSite", "NVIDIA")
    assert isinstance(jobs, list)
    assert len(jobs) == 2
    job = jobs[0]
    for k in ("company", "title", "location", "is_remote",
              "job_url", "description", "date_posted",
              "site", "search_term", "search_location"):
        assert k in job, f"missing key {k}"
    assert job["company"] == "NVIDIA"
    assert job["site"] == "direct_workday"
    assert job["search_term"] == "direct_scrape"
    assert job["search_location"] == "direct_company"
    # Workday URL composition: tenant.wd.myworkdayjobs.com/<site><externalPath>
    assert "nvidia.wd5.myworkdayjobs.com" in job["job_url"]
    assert "/NVIDIAExternalCareerSite/job/" in job["job_url"]
    # Second job is Remote
    assert jobs[1]["is_remote"] is True
    # First job has Tampa, FL, not Remote
    assert jobs[0]["is_remote"] is False


def test_scrape_workday_returns_empty_on_invalid_slug():
    """A malformed slug (missing pieces) should return empty without crashing."""
    assert scrape_workday("just-tenant", "Acme") == []
    assert scrape_workday("tenant/wd1", "Acme") == []
    assert scrape_workday("", "Acme") == []


def test_scrape_workday_handles_422_gracefully():
    """Workday returning 422 (wrong slug) should yield empty list, not raise."""
    err_resp = _mock_workday_response({"errorCode": "HTTP_422"}, status=422)
    with patch("tools.direct_scrapers.requests.post", return_value=err_resp):
        jobs = scrape_workday("badtenant/wd1/BadSite", "Bad")
    assert jobs == []


# ---------- Error handling ----------

def test_scrape_greenhouse_returns_empty_on_404():
    mock = MagicMock()
    mock.status_code = 404
    mock.raise_for_status.side_effect = Exception("404")
    with patch("tools.direct_scrapers.requests.get", return_value=mock):
        jobs = scrape_greenhouse("nonexistent-slug", "Nonexistent")
    assert jobs == []


# ---------- HTML stripping helper ----------

def test_strip_html_removes_tags_and_entities():
    raw = "&lt;p&gt;Hello &amp; goodbye&lt;/p&gt;"
    out = strip_html(raw)
    assert "<" not in out
    assert ">" not in out
    assert "&lt;" not in out
    assert "Hello" in out
    assert "goodbye" in out


def test_strip_html_handles_none_and_empty():
    assert strip_html(None) == ""
    assert strip_html("") == ""
