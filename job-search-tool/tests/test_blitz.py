"""Tests for tools/blitz.py — all HTTP mocked, no real API hits."""
from unittest.mock import MagicMock, patch

from tools.blitz import (
    domain_to_linkedin,
    find_employees_by_title,
    linkedin_to_email,
)


def _resp(status: int = 200, payload: dict | None = None):
    r = MagicMock()
    r.status_code = status
    r.json.return_value = payload or {}
    r.text = ""
    return r


# ---------- domain_to_linkedin ----------

def test_domain_to_linkedin_skips_when_no_api_key(monkeypatch):
    monkeypatch.delenv("BLITZ_API_KEY", raising=False)
    with patch("tools.blitz.requests.post") as p:
        result = domain_to_linkedin("acme.com")
    assert result is None
    p.assert_not_called()


def test_domain_to_linkedin_skips_when_domain_empty(monkeypatch):
    monkeypatch.setenv("BLITZ_API_KEY", "abc")
    with patch("tools.blitz.requests.post") as p:
        result = domain_to_linkedin("")
    assert result is None
    p.assert_not_called()


def test_domain_to_linkedin_returns_url_on_success(monkeypatch):
    monkeypatch.setenv("BLITZ_API_KEY", "abc")
    payload = {"company_linkedin_url": "https://www.linkedin.com/company/acme/"}
    with patch("tools.blitz.requests.post", return_value=_resp(200, payload)):
        result = domain_to_linkedin("acme.com")
    assert result == "https://www.linkedin.com/company/acme/"


def test_domain_to_linkedin_returns_none_on_4xx(monkeypatch):
    monkeypatch.setenv("BLITZ_API_KEY", "abc")
    with patch("tools.blitz.requests.post", return_value=_resp(404, {})):
        result = domain_to_linkedin("acme.com")
    assert result is None


def test_domain_to_linkedin_retries_on_429(monkeypatch):
    monkeypatch.setenv("BLITZ_API_KEY", "abc")
    rate_limited = _resp(429, {})
    success = _resp(200, {"company_linkedin_url": "https://linkedin.com/company/acme"})
    with patch("tools.blitz.requests.post", side_effect=[rate_limited, success]):
        with patch("tools.blitz.time.sleep"):
            result = domain_to_linkedin("acme.com")
    assert result == "https://linkedin.com/company/acme"


def test_domain_to_linkedin_returns_none_on_5xx(monkeypatch):
    monkeypatch.setenv("BLITZ_API_KEY", "abc")
    with patch("tools.blitz.requests.post", return_value=_resp(500, {})):
        result = domain_to_linkedin("acme.com")
    assert result is None


def test_domain_to_linkedin_handles_network_error(monkeypatch):
    monkeypatch.setenv("BLITZ_API_KEY", "abc")
    with patch("tools.blitz.requests.post", side_effect=Exception("conn refused")):
        result = domain_to_linkedin("acme.com")
    assert result is None


def test_domain_to_linkedin_returns_none_when_url_missing(monkeypatch):
    monkeypatch.setenv("BLITZ_API_KEY", "abc")
    with patch("tools.blitz.requests.post", return_value=_resp(200, {"foo": "bar"})):
        result = domain_to_linkedin("acme.com")
    assert result is None


# ---------- find_employees_by_title ----------

def test_find_employees_skips_when_no_api_key(monkeypatch):
    monkeypatch.delenv("BLITZ_API_KEY", raising=False)
    with patch("tools.blitz.requests.post") as p:
        result = find_employees_by_title(
            "https://linkedin.com/company/acme", ["VP Sales"]
        )
    assert result == []
    p.assert_not_called()


def test_find_employees_skips_when_url_or_titles_empty(monkeypatch):
    monkeypatch.setenv("BLITZ_API_KEY", "abc")
    with patch("tools.blitz.requests.post") as p:
        assert find_employees_by_title("", ["VP Sales"]) == []
        assert find_employees_by_title("https://x", []) == []
    p.assert_not_called()


def test_find_employees_returns_candidates_on_success(monkeypatch):
    monkeypatch.setenv("BLITZ_API_KEY", "abc")
    payload = {
        "results": [
            {
                "person": {
                    "first_name": "Sarah",
                    "last_name": "Doe",
                    "linkedin_url": "https://www.linkedin.com/in/sarahdoe",
                    "experiences": [
                        {"job_is_current": True, "job_title": "VP Sales"}
                    ],
                    "headline": "VP Sales at Acme",
                    "location": {"city": "NYC", "state_code": "NY"},
                },
                "icp": 1,
            },
            {
                "person": {
                    "first_name": "Tom",
                    "last_name": "Smith",
                    "linkedin_url": "https://www.linkedin.com/in/tomsmith",
                    "experiences": [],
                    "headline": "Head of GTM at Acme",
                    "location": {},
                },
                "icp": 2,
            },
        ]
    }
    with patch("tools.blitz.requests.post", return_value=_resp(200, payload)):
        result = find_employees_by_title(
            "https://linkedin.com/company/acme", ["VP Sales", "Head of GTM"]
        )
    assert len(result) == 2
    assert result[0]["first_name"] == "Sarah"
    assert result[0]["last_name"] == "Doe"
    assert result[0]["title"] == "VP Sales"
    assert result[0]["linkedin_url"] == "https://www.linkedin.com/in/sarahdoe"
    # Falls back to headline-derived title when no current experience
    assert result[1]["title"].startswith("Head of GTM")


def test_find_employees_returns_empty_on_4xx(monkeypatch):
    monkeypatch.setenv("BLITZ_API_KEY", "abc")
    with patch("tools.blitz.requests.post", return_value=_resp(400, {})):
        result = find_employees_by_title(
            "https://linkedin.com/company/acme", ["VP Sales"]
        )
    assert result == []


def test_find_employees_returns_empty_on_5xx(monkeypatch):
    monkeypatch.setenv("BLITZ_API_KEY", "abc")
    with patch("tools.blitz.requests.post", return_value=_resp(503, {})):
        result = find_employees_by_title(
            "https://linkedin.com/company/acme", ["VP Sales"]
        )
    assert result == []


def test_find_employees_handles_network_error(monkeypatch):
    monkeypatch.setenv("BLITZ_API_KEY", "abc")
    with patch("tools.blitz.requests.post", side_effect=Exception("timeout")):
        result = find_employees_by_title(
            "https://linkedin.com/company/acme", ["VP Sales"]
        )
    assert result == []


def test_find_employees_returns_empty_when_no_results(monkeypatch):
    monkeypatch.setenv("BLITZ_API_KEY", "abc")
    with patch(
        "tools.blitz.requests.post", return_value=_resp(200, {"results": []})
    ):
        result = find_employees_by_title(
            "https://linkedin.com/company/acme", ["VP Sales"]
        )
    assert result == []


# ---------- linkedin_to_email ----------

def test_linkedin_to_email_skips_when_no_api_key(monkeypatch):
    monkeypatch.delenv("BLITZ_API_KEY", raising=False)
    with patch("tools.blitz.requests.post") as p:
        result = linkedin_to_email("https://linkedin.com/in/sarah")
    assert result is None
    p.assert_not_called()


def test_linkedin_to_email_skips_when_url_empty(monkeypatch):
    monkeypatch.setenv("BLITZ_API_KEY", "abc")
    with patch("tools.blitz.requests.post") as p:
        assert linkedin_to_email("") is None
    p.assert_not_called()


def test_linkedin_to_email_returns_email_when_found(monkeypatch):
    monkeypatch.setenv("BLITZ_API_KEY", "abc")
    payload = {"found": True, "email": "sarah@acme.com"}
    with patch("tools.blitz.requests.post", return_value=_resp(200, payload)):
        result = linkedin_to_email("https://linkedin.com/in/sarah")
    assert result == "sarah@acme.com"


def test_linkedin_to_email_returns_none_when_not_found(monkeypatch):
    monkeypatch.setenv("BLITZ_API_KEY", "abc")
    payload = {"found": False, "email": ""}
    with patch("tools.blitz.requests.post", return_value=_resp(200, payload)):
        result = linkedin_to_email("https://linkedin.com/in/sarah")
    assert result is None


def test_linkedin_to_email_returns_none_on_4xx(monkeypatch):
    monkeypatch.setenv("BLITZ_API_KEY", "abc")
    with patch("tools.blitz.requests.post", return_value=_resp(404, {})):
        result = linkedin_to_email("https://linkedin.com/in/sarah")
    assert result is None


def test_linkedin_to_email_returns_none_on_5xx(monkeypatch):
    monkeypatch.setenv("BLITZ_API_KEY", "abc")
    with patch("tools.blitz.requests.post", return_value=_resp(500, {})):
        result = linkedin_to_email("https://linkedin.com/in/sarah")
    assert result is None


def test_linkedin_to_email_handles_network_error(monkeypatch):
    monkeypatch.setenv("BLITZ_API_KEY", "abc")
    with patch("tools.blitz.requests.post", side_effect=Exception("dns")):
        result = linkedin_to_email("https://linkedin.com/in/sarah")
    assert result is None
