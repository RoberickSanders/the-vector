import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import tools.email_tools as email_tools_module
from tools.email_tools import (
    extract_emails_from_text,
    has_mx_records,
    is_catch_all_domain,
    load_domain_memory,
    permutate_emails,
    remember_domain,
    save_domain_memory,
    scrape_website_emails,
    verify_email_mv,
)


def test_permutate_emails_first_last_dot():
    perms = permutate_emails("John", "Smith", "example.com")
    assert "john.smith@example.com" in perms
    assert "jsmith@example.com" in perms
    assert "john@example.com" in perms
    assert "j.smith@example.com" in perms


def test_permutate_emails_handles_compound_lastname():
    perms = permutate_emails("Maria", "Garcia Lopez", "example.com")
    # Should treat "Garcia Lopez" as a single last name with no spaces
    assert any("@example.com" in p for p in perms)
    assert all(" " not in p for p in perms)


def test_permutate_emails_lowercases():
    perms = permutate_emails("JOHN", "SMITH", "EXAMPLE.COM")
    assert all(p == p.lower() for p in perms)


def test_extract_emails_from_text_finds_basic():
    text = "Contact us at recruiter@acme.com or call 555-1234"
    found = extract_emails_from_text(text)
    assert "recruiter@acme.com" in found


def test_extract_emails_from_text_skips_obvious_noreply():
    text = "Email noreply@acme.com or apply@acme.com or sarah@acme.com"
    found = extract_emails_from_text(text)
    assert "sarah@acme.com" in found
    assert "noreply@acme.com" not in found
    assert "apply@acme.com" not in found


def test_extract_emails_from_text_returns_empty_when_none():
    assert extract_emails_from_text("No emails here") == []


# ---------- MillionVerifier ----------

def _make_mv_resp(status: int = 200, payload: dict | None = None):
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = payload or {}
    return resp


def test_verify_email_mv_returns_true_for_ok_result(monkeypatch):
    monkeypatch.setenv("MILLIONVERIFIER_API_KEY", "test-key")
    with patch(
        "tools.email_tools.requests.get",
        return_value=_make_mv_resp(200, {"result": "ok"}),
    ):
        assert verify_email_mv("john@acme.com") is True


def test_verify_email_mv_returns_true_for_catch_all_result(monkeypatch):
    monkeypatch.setenv("MILLIONVERIFIER_API_KEY", "test-key")
    with patch(
        "tools.email_tools.requests.get",
        return_value=_make_mv_resp(200, {"result": "catch_all"}),
    ):
        assert verify_email_mv("john@acme.com") is True


def test_verify_email_mv_returns_false_for_invalid(monkeypatch):
    monkeypatch.setenv("MILLIONVERIFIER_API_KEY", "test-key")
    with patch(
        "tools.email_tools.requests.get",
        return_value=_make_mv_resp(200, {"result": "invalid"}),
    ):
        assert verify_email_mv("nope@acme.com") is False


def test_verify_email_mv_returns_none_when_no_api_key(monkeypatch):
    monkeypatch.delenv("MILLIONVERIFIER_API_KEY", raising=False)
    with patch("tools.email_tools.requests.get") as gmock:
        assert verify_email_mv("john@acme.com") is None
    gmock.assert_not_called()


def test_verify_email_mv_returns_none_on_http_error(monkeypatch):
    monkeypatch.setenv("MILLIONVERIFIER_API_KEY", "test-key")
    with patch(
        "tools.email_tools.requests.get",
        return_value=_make_mv_resp(500, {}),
    ):
        assert verify_email_mv("john@acme.com") is None


def test_verify_email_mv_returns_none_for_unknown_result(monkeypatch):
    """Unrecognized results (e.g., 'unknown', 'role') -> None, not a guess."""
    monkeypatch.setenv("MILLIONVERIFIER_API_KEY", "test-key")
    with patch(
        "tools.email_tools.requests.get",
        return_value=_make_mv_resp(200, {"result": "unknown"}),
    ):
        assert verify_email_mv("maybe@acme.com") is None


def test_is_catch_all_domain_caches_per_domain(monkeypatch):
    """A second call for the same domain must not hit HTTP."""
    monkeypatch.setenv("MILLIONVERIFIER_API_KEY", "test-key")
    # Reset the in-memory cache between tests for determinism
    email_tools_module._MV_DOMAIN_CATCH_ALL_CACHE.clear()
    with patch(
        "tools.email_tools.requests.get",
        return_value=_make_mv_resp(200, {"result": "ok"}),
    ) as gmock:
        first = is_catch_all_domain("catchall-test.com")
        second = is_catch_all_domain("catchall-test.com")
    assert first is True
    assert second is True
    # One HTTP call total — second was served from cache
    assert gmock.call_count == 1


def test_is_catch_all_domain_detects_via_random_email(monkeypatch):
    """is_catch_all_domain sends a random nonexistent email; if MV says 'ok',
    the domain is catch-all.
    """
    monkeypatch.setenv("MILLIONVERIFIER_API_KEY", "test-key")
    email_tools_module._MV_DOMAIN_CATCH_ALL_CACHE.clear()
    captured = {}

    def fake_get(url, params=None, timeout=None, **kw):
        captured["params"] = params
        return _make_mv_resp(200, {"result": "ok"})

    with patch("tools.email_tools.requests.get", side_effect=fake_get):
        assert is_catch_all_domain("loose.com") is True
    # The probe email goes to the target domain and starts with the jstool prefix
    assert captured["params"]["email"].endswith("@loose.com")
    assert captured["params"]["email"].startswith("zz_jstool_")


def test_is_catch_all_domain_returns_false_when_not_catch_all(monkeypatch):
    monkeypatch.setenv("MILLIONVERIFIER_API_KEY", "test-key")
    email_tools_module._MV_DOMAIN_CATCH_ALL_CACHE.clear()
    with patch(
        "tools.email_tools.requests.get",
        return_value=_make_mv_resp(200, {"result": "invalid"}),
    ):
        assert is_catch_all_domain("strict.com") is False


# ---------- MX pre-check ----------

def test_has_mx_records_returns_false_for_empty():
    assert has_mx_records("") is False


def test_has_mx_records_returns_true_when_records_exist():
    """Mock dns.resolver.resolve to return a non-empty answer set."""
    with patch("dns.resolver.resolve", return_value=[MagicMock()]):
        assert has_mx_records("example.com") is True


def test_has_mx_records_returns_false_on_nxdomain():
    with patch("dns.resolver.resolve", side_effect=Exception("NXDOMAIN")):
        assert has_mx_records("does-not-exist-zzz.invalid") is False


# ---------- Domain memory ----------

def test_load_domain_memory_handles_missing_file(tmp_path, monkeypatch):
    monkeypatch.setattr(
        email_tools_module, "_DOMAIN_MEMORY_FILE", tmp_path / "missing.json"
    )
    assert load_domain_memory() == {}


def test_load_domain_memory_handles_corrupt_json(tmp_path, monkeypatch):
    f = tmp_path / "domain-memory.json"
    f.write_text("not valid json {{{")
    monkeypatch.setattr(email_tools_module, "_DOMAIN_MEMORY_FILE", f)
    assert load_domain_memory() == {}


def test_save_and_reload_domain_memory_roundtrip(tmp_path, monkeypatch):
    f = tmp_path / "domain-memory.json"
    monkeypatch.setattr(email_tools_module, "_DOMAIN_MEMORY_FILE", f)
    monkeypatch.setattr(email_tools_module, "_STATE_DIR", tmp_path)

    payload = {"acme.com": {"mx_ok": True, "pattern": "{first}.{last}"}}
    save_domain_memory(payload)
    assert f.exists()
    reloaded = load_domain_memory()
    assert reloaded["acme.com"]["mx_ok"] is True
    assert reloaded["acme.com"]["pattern"] == "{first}.{last}"


def test_remember_domain_merges_fields_and_stamps_time():
    memory = {"acme.com": {"mx_ok": True}}
    remember_domain(memory, "acme.com", last_recruiter_email="r@acme.com")
    entry = memory["acme.com"]
    assert entry["mx_ok"] is True
    assert entry["last_recruiter_email"] == "r@acme.com"
    assert "last_seen_iso" in entry


def test_remember_domain_creates_entry_when_missing():
    memory = {}
    remember_domain(memory, "newco.com", pattern="{first}@")
    assert "newco.com" in memory
    assert memory["newco.com"]["pattern"] == "{first}@"


def test_remember_domain_noops_on_empty_domain():
    memory = {}
    remember_domain(memory, "", pattern="{first}@")
    assert memory == {}


# ---------- Website scrape ----------

def test_scrape_website_emails_returns_personal_emails(monkeypatch):
    """Mock requests.get to return HTML containing emails."""
    email_tools_module._WEBSITE_CACHE.clear()
    html = "<html>Reach Sarah at sarah@acme.com or jobs@acme.com</html>"

    def fake_get(url, timeout=None, headers=None, **kw):
        resp = MagicMock()
        resp.status_code = 200
        resp.text = html
        return resp

    with patch("tools.email_tools.requests.get", side_effect=fake_get):
        emails = scrape_website_emails("acme.com")

    # sarah@ kept, jobs@ filtered (generic prefix)
    assert "sarah@acme.com" in emails
    assert "jobs@acme.com" not in emails


def test_scrape_website_emails_skips_off_domain(monkeypatch):
    """Vendor emails (different domain) should be filtered out."""
    email_tools_module._WEBSITE_CACHE.clear()
    html = "<html>support@vendor.io and sarah@acme.com</html>"

    def fake_get(url, timeout=None, headers=None, **kw):
        resp = MagicMock()
        resp.status_code = 200
        resp.text = html
        return resp

    with patch("tools.email_tools.requests.get", side_effect=fake_get):
        emails = scrape_website_emails("acme.com")

    assert "sarah@acme.com" in emails
    assert "support@vendor.io" not in emails


def test_scrape_website_emails_uses_cache(monkeypatch):
    """Second call for same domain must NOT issue HTTP."""
    email_tools_module._WEBSITE_CACHE.clear()
    html = "<html>sarah@acme.com</html>"
    call_count = {"n": 0}

    def fake_get(url, timeout=None, headers=None, **kw):
        call_count["n"] += 1
        resp = MagicMock()
        resp.status_code = 200
        resp.text = html
        return resp

    with patch("tools.email_tools.requests.get", side_effect=fake_get):
        first = scrape_website_emails("acme.com")
        second = scrape_website_emails("acme.com")

    assert first == second
    # First call hit at least one path; second was 100% cached.
    n1 = call_count["n"]
    assert n1 >= 1
    # Still n1 — no new calls — proves the cache worked.
    assert call_count["n"] == n1


def test_scrape_website_emails_returns_empty_on_all_failures(monkeypatch):
    """All paths timing out / 500ing -> empty list, no crash."""
    email_tools_module._WEBSITE_CACHE.clear()

    def fake_get(url, timeout=None, headers=None, **kw):
        raise Exception("connection refused")

    with patch("tools.email_tools.requests.get", side_effect=fake_get):
        emails = scrape_website_emails("dead.com")

    assert emails == []


def test_scrape_website_emails_returns_empty_for_no_domain():
    assert scrape_website_emails("") == []
