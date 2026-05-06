"""Tests for tools/icypeas.py — all HTTP mocked, no real API hits."""
from unittest.mock import MagicMock, patch

import tools.icypeas as icypeas_module
from tools.icypeas import (
    _icypeas_headers,
    find_domain_emails_icypeas,
    find_email_icypeas,
)


def _resp(status: int = 200, payload: dict | None = None):
    r = MagicMock()
    r.status_code = status
    r.json.return_value = payload or {}
    return r


# ---------- auth ----------

def test_icypeas_headers_returns_none_without_key(monkeypatch):
    monkeypatch.delenv("ICYPEAS_API_KEY", raising=False)
    assert _icypeas_headers() is None


def test_icypeas_headers_includes_authorization(monkeypatch):
    monkeypatch.setenv("ICYPEAS_API_KEY", "abc123")
    h = _icypeas_headers()
    assert h is not None
    assert h["Authorization"] == "abc123"
    assert h["Content-Type"] == "application/json"


# ---------- find_email_icypeas ----------

def test_find_email_icypeas_skips_when_no_creds(monkeypatch):
    monkeypatch.delenv("ICYPEAS_API_KEY", raising=False)
    with patch("tools.icypeas.requests.post") as p:
        email, src = find_email_icypeas("Sarah", "Doe", "acme.com")
    p.assert_not_called()
    assert email == "" and src == ""


def test_find_email_icypeas_skips_when_missing_args(monkeypatch):
    monkeypatch.setenv("ICYPEAS_API_KEY", "abc")
    with patch("tools.icypeas.requests.post") as p:
        assert find_email_icypeas("", "Doe", "acme.com") == ("", "")
        assert find_email_icypeas("Sarah", "", "acme.com") == ("", "")
        assert find_email_icypeas("Sarah", "Doe", "") == ("", "")
    p.assert_not_called()


def test_find_email_icypeas_returns_email_on_success(monkeypatch):
    """Submit returns search_id, poll returns COMPLETED with one email."""
    monkeypatch.setenv("ICYPEAS_API_KEY", "abc")

    submit_resp = _resp(200, {"success": True, "item": {"_id": "sid-123"}})
    poll_resp = _resp(200, {
        "success": True,
        "items": [{
            "status": "COMPLETED",
            "results": {"emails": [{"email": "sarah.doe@acme.com"}]},
        }],
    })

    # First post = submit, all subsequent posts = poll. Use side_effect cycle.
    posts = [submit_resp, poll_resp]
    with patch("tools.icypeas.requests.post", side_effect=posts):
        with patch("tools.icypeas.time.sleep"):  # skip real polling delay
            email, src = find_email_icypeas("Sarah", "Doe", "acme.com")

    assert email == "sarah.doe@acme.com"
    assert src == "icypeas_email"


def test_find_email_icypeas_returns_empty_on_pending_then_timeout(monkeypatch):
    monkeypatch.setenv("ICYPEAS_API_KEY", "abc")
    # Lower poll budget so the test is fast.
    # NB: INTERVAL must be > 0 — `elapsed += INTERVAL` with INTERVAL=0
    # never grows elapsed, so the timeout never trips and the loop spins
    # forever consuming side_effect (whose StopIteration is caught by
    # the bare except in _icypeas_poll_result). With (1, 1) the loop
    # runs exactly one iteration and then exits cleanly.
    monkeypatch.setattr(icypeas_module, "ICYPEAS_POLL_INTERVAL", 1)
    monkeypatch.setattr(icypeas_module, "ICYPEAS_POLL_MAX_WAIT", 1)

    submit_resp = _resp(200, {"success": True, "item": {"_id": "sid"}})
    pending_resp = _resp(200, {
        "success": True,
        "items": [{"status": "IN_PROGRESS"}],
    })
    posts = [submit_resp] + [pending_resp] * 50
    with patch("tools.icypeas.requests.post", side_effect=posts):
        with patch("tools.icypeas.time.sleep"):
            email, src = find_email_icypeas("Sarah", "Doe", "acme.com")

    assert email == "" and src == ""


def test_find_email_icypeas_handles_submit_failure(monkeypatch):
    monkeypatch.setenv("ICYPEAS_API_KEY", "abc")
    with patch(
        "tools.icypeas.requests.post",
        return_value=_resp(200, {"success": False}),
    ):
        email, src = find_email_icypeas("Sarah", "Doe", "acme.com")
    assert (email, src) == ("", "")


def test_find_email_icypeas_handles_no_email_in_result(monkeypatch):
    monkeypatch.setenv("ICYPEAS_API_KEY", "abc")
    submit_resp = _resp(200, {"success": True, "item": {"_id": "sid"}})
    poll_resp = _resp(200, {
        "success": True,
        "items": [{"status": "COMPLETED", "results": {"emails": []}}],
    })
    with patch("tools.icypeas.requests.post", side_effect=[submit_resp, poll_resp]):
        with patch("tools.icypeas.time.sleep"):
            email, src = find_email_icypeas("Sarah", "Doe", "acme.com")
    assert (email, src) == ("", "")


# ---------- find_domain_emails_icypeas ----------

def test_find_domain_emails_icypeas_skips_when_no_creds(monkeypatch):
    monkeypatch.delenv("ICYPEAS_API_KEY", raising=False)
    with patch("tools.icypeas.requests.post") as p:
        assert find_domain_emails_icypeas("acme.com") == []
    p.assert_not_called()


def test_find_domain_emails_icypeas_returns_list_on_success(monkeypatch):
    monkeypatch.setenv("ICYPEAS_API_KEY", "abc")
    submit_resp = _resp(200, {"success": True, "item": {"_id": "sid"}})
    poll_resp = _resp(200, {
        "success": True,
        "items": [{
            "status": "COMPLETED",
            "results": {"emails": [
                {"email": "sarah@acme.com", "name": "Sarah Doe", "position": "VP HR"},
                {"email": "tom@acme.com"},
                {"email": ""},                    # filtered (empty)
                {"email": "no-at-sign-here"},     # filtered (no @)
            ]},
        }],
    })
    with patch("tools.icypeas.requests.post", side_effect=[submit_resp, poll_resp]):
        with patch("tools.icypeas.time.sleep"):
            results = find_domain_emails_icypeas("acme.com")

    assert len(results) == 2
    assert results[0]["email"] == "sarah@acme.com"
    assert results[0]["name"] == "Sarah Doe"
    assert results[0]["position"] == "VP HR"
    assert results[1]["email"] == "tom@acme.com"
    assert results[1]["name"] == ""


def test_find_domain_emails_icypeas_handles_network_error(monkeypatch):
    monkeypatch.setenv("ICYPEAS_API_KEY", "abc")
    with patch("tools.icypeas.requests.post", side_effect=Exception("conn refused")):
        assert find_domain_emails_icypeas("acme.com") == []
