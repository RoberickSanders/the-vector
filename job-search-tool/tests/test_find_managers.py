"""Tests for tools/find_managers.py reliability fixes (Hunter API + summary)."""
from unittest.mock import MagicMock, patch

import pytest

import tools.find_managers as fm
from tools.find_managers import (
    _apex_domain,
    _domains_match,
    find_via_blitz,
    find_via_hunter,
    find_via_icypeas_domain,
    find_via_icypeas_email,
    format_summary,
)


@pytest.fixture(autouse=True)
def _reset_module_caches():
    """Each test starts with empty Blitz/Icypeas/Hunter caches.

    These caches are module-globals in tools.find_managers and would otherwise
    leak across tests in the same file (e.g., a cached '' for acme.com would
    short-circuit the next test's mocks).
    """
    fm._BLITZ_DOMAIN_CACHE.clear()
    fm._ICYPEAS_DOMAIN_CACHE.clear()
    fm._DOMAIN_CACHE.clear()
    yield
    fm._BLITZ_DOMAIN_CACHE.clear()
    fm._ICYPEAS_DOMAIN_CACHE.clear()
    fm._DOMAIN_CACHE.clear()


# ---------- Hunter API ----------

def test_find_via_hunter_skips_when_no_api_key(monkeypatch):
    """Without HUNTER_API_KEY in env, find_via_hunter must return empty
    without trying to hit the network."""
    monkeypatch.delenv("HUNTER_API_KEY", raising=False)
    with patch("tools.find_managers.requests.get") as gmock:
        result = find_via_hunter("John", "Smith", "acme.com")
    assert result == ("", "")
    gmock.assert_not_called()


def test_find_via_hunter_returns_email_when_high_confidence(monkeypatch):
    monkeypatch.setenv("HUNTER_API_KEY", "test-key")
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "data": {"email": "john.smith@acme.com", "score": 92}
    }
    with patch("tools.find_managers.requests.get", return_value=mock_resp):
        email, source = find_via_hunter("John", "Smith", "acme.com")
    assert email == "john.smith@acme.com"
    assert source.startswith("hunter")
    assert "92" in source


def test_find_via_hunter_drops_low_confidence(monkeypatch):
    """Hunter score below threshold (70) must be discarded."""
    monkeypatch.setenv("HUNTER_API_KEY", "test-key")
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "data": {"email": "guess@acme.com", "score": 35}
    }
    with patch("tools.find_managers.requests.get", return_value=mock_resp):
        email, source = find_via_hunter("John", "Smith", "acme.com")
    assert email == ""
    assert source == ""


def test_find_via_hunter_handles_error_response(monkeypatch):
    monkeypatch.setenv("HUNTER_API_KEY", "test-key")
    mock_resp = MagicMock()
    mock_resp.status_code = 401
    with patch("tools.find_managers.requests.get", return_value=mock_resp):
        email, source = find_via_hunter("John", "Smith", "acme.com")
    assert email == ""
    assert source == ""


def test_find_via_hunter_handles_exception(monkeypatch):
    monkeypatch.setenv("HUNTER_API_KEY", "test-key")
    with patch("tools.find_managers.requests.get",
               side_effect=Exception("network down")):
        email, source = find_via_hunter("John", "Smith", "acme.com")
    assert email == ""
    assert source == ""


# ---------- Blitz cascade ----------

def _candidate(first="Sarah", last="Doe", url="https://linkedin.com/in/sarah",
               title="VP Sales", icp_tier=1):
    """Helper: shape a Blitz find_employees_by_title row."""
    return {
        "first_name": first,
        "last_name": last,
        "title": title,
        "linkedin_url": url,
        "location_city": "NYC",
        "location_state": "NY",
        "icp_tier": icp_tier,
    }


# ---------- Domain-match helpers ----------

def test_apex_domain_extracts_last_two_components():
    assert _apex_domain("aws.amazon.com") == "amazon.com"
    assert _apex_domain("amazon.com") == "amazon.com"
    assert _apex_domain("mail.subdomain.langchain.com") == "langchain.com"
    assert _apex_domain("langchain.com.") == "langchain.com"  # trailing dot
    assert _apex_domain("") == ""


def test_apex_domain_lowercases():
    assert _apex_domain("Amazon.COM") == "amazon.com"


def test_domains_match_accepts_subdomain_to_apex():
    """AWS lookup with email at subdomain should be allowed."""
    assert _domains_match("amazon.com", "jen@aws.amazon.com") is True
    assert _domains_match("aws.amazon.com", "jen@amazon.com") is True


def test_domains_match_rejects_different_company():
    """The actual bug: AWS lookup returned jens@indeed.com -> reject."""
    assert _domains_match("amazon.com", "jens@indeed.com") is False
    assert _domains_match("aws.amazon.com", "jens@indeed.com") is False


def test_domains_match_handles_bad_input():
    assert _domains_match("", "x@acme.com") is False
    assert _domains_match("acme.com", "no-at-sign") is False
    assert _domains_match("acme.com", "") is False


# ---------- Blitz domain-mismatch guard ----------

def test_find_via_blitz_rejects_domain_mismatch():
    """Blitz returns email at a different company's domain -> reject and
    increment blitz_domain_mismatch."""
    counters = {}
    with patch("tools.find_managers.domain_to_linkedin",
               return_value="https://linkedin.com/company/aws"), \
         patch("tools.find_managers.find_employees_by_title",
               return_value=[_candidate(first="Jen", last="Sun")]), \
         patch("tools.find_managers.linkedin_to_email",
               return_value="jens@indeed.com"), \
         patch("tools.find_managers.verify_email_mv") as mv:
        email, name, linkedin, source = find_via_blitz("amazon.com", counters=counters)
    assert (email, name, linkedin, source) == ("", "", "", "")
    assert counters.get("blitz_domain_mismatch") == 1
    # MV not called — the guard short-circuits before MV verify
    mv.assert_not_called()


def test_find_via_blitz_skips_mismatch_then_accepts_match():
    """Two candidates: first has wrong-domain email, second has right-domain."""
    cands = [
        _candidate(first="Jen", last="Sun", url="https://linkedin.com/in/jen", icp_tier=1),
        _candidate(first="Real", last="Hire", url="https://linkedin.com/in/real", icp_tier=2),
    ]
    counters = {}
    with patch("tools.find_managers.domain_to_linkedin",
               return_value="https://linkedin.com/company/amazon"), \
         patch("tools.find_managers.find_employees_by_title", return_value=cands), \
         patch("tools.find_managers.linkedin_to_email",
               side_effect=["jens@indeed.com", "real@amazon.com"]), \
         patch("tools.find_managers.verify_email_mv", return_value=True):
        email, name, linkedin, source = find_via_blitz("amazon.com", counters=counters)
    assert email == "real@amazon.com"
    assert name == "Real Hire"
    assert source == "blitz+mv_ok"
    assert counters.get("blitz_domain_mismatch") == 1


def test_find_via_blitz_accepts_subdomain_email():
    """Email at subdomain of target should pass guard."""
    with patch("tools.find_managers.domain_to_linkedin",
               return_value="https://linkedin.com/company/amazon"), \
         patch("tools.find_managers.find_employees_by_title",
               return_value=[_candidate()]), \
         patch("tools.find_managers.linkedin_to_email",
               return_value="sarah@aws.amazon.com"), \
         patch("tools.find_managers.verify_email_mv", return_value=True):
        email, name, linkedin, source = find_via_blitz("amazon.com")
    assert email == "sarah@aws.amazon.com"
    assert source == "blitz+mv_ok"


def test_find_via_blitz_returns_email_when_mv_verified():
    with patch("tools.find_managers.domain_to_linkedin",
               return_value="https://linkedin.com/company/acme") as d2l, \
         patch("tools.find_managers.find_employees_by_title",
               return_value=[_candidate()]) as feb, \
         patch("tools.find_managers.linkedin_to_email",
               return_value="sarah.doe@acme.com") as l2e, \
         patch("tools.find_managers.verify_email_mv", return_value=True) as mv:
        email, name, linkedin, source = find_via_blitz("acme.com")
    assert email == "sarah.doe@acme.com"
    assert name == "Sarah Doe"
    assert linkedin == "https://linkedin.com/in/sarah"
    assert source == "blitz+mv_ok"
    d2l.assert_called_once_with("acme.com")
    feb.assert_called_once()
    l2e.assert_called_once_with("https://linkedin.com/in/sarah")
    mv.assert_called_once_with("sarah.doe@acme.com")


def test_find_via_blitz_skips_mv_rejected_candidate():
    """First candidate gets MV=False, second gets MV=True. We pick #2."""
    cands = [
        _candidate(first="Bad", last="One", url="https://linkedin.com/in/one", icp_tier=1),
        _candidate(first="Good", last="Two", url="https://linkedin.com/in/two", icp_tier=2),
    ]
    counters = {}
    with patch("tools.find_managers.domain_to_linkedin",
               return_value="https://linkedin.com/company/acme"), \
         patch("tools.find_managers.find_employees_by_title", return_value=cands), \
         patch("tools.find_managers.linkedin_to_email",
               side_effect=["bad@acme.com", "good@acme.com"]), \
         patch("tools.find_managers.verify_email_mv",
               side_effect=[False, True]):
        email, name, linkedin, source = find_via_blitz("acme.com", counters=counters)
    assert email == "good@acme.com"
    assert name == "Good Two"
    assert source == "blitz+mv_ok"
    assert counters.get("blitz_mv_rejected") == 1


def test_find_via_blitz_returns_uncertain_when_mv_none():
    with patch("tools.find_managers.domain_to_linkedin",
               return_value="https://linkedin.com/company/acme"), \
         patch("tools.find_managers.find_employees_by_title",
               return_value=[_candidate()]), \
         patch("tools.find_managers.linkedin_to_email",
               return_value="sarah.doe@acme.com"), \
         patch("tools.find_managers.verify_email_mv", return_value=None):
        email, name, linkedin, source = find_via_blitz("acme.com")
    assert email == "sarah.doe@acme.com"
    assert source == "blitz+mv_uncertain"


def test_find_via_blitz_returns_empty_when_no_employees():
    counters = {}
    with patch("tools.find_managers.domain_to_linkedin",
               return_value="https://linkedin.com/company/acme") as d2l, \
         patch("tools.find_managers.find_employees_by_title",
               return_value=[]) as feb, \
         patch("tools.find_managers.linkedin_to_email") as l2e:
        result = find_via_blitz("acme.com", counters=counters)
    assert result == ("", "", "", "")
    d2l.assert_called_once()
    feb.assert_called_once()
    l2e.assert_not_called()
    assert counters.get("blitz_no_candidates") == 1


def test_find_via_blitz_returns_empty_when_no_linkedin():
    counters = {}
    with patch("tools.find_managers.domain_to_linkedin",
               return_value=None) as d2l, \
         patch("tools.find_managers.find_employees_by_title") as feb, \
         patch("tools.find_managers.linkedin_to_email") as l2e:
        result = find_via_blitz("acme.com", counters=counters)
    assert result == ("", "", "", "")
    d2l.assert_called_once()
    feb.assert_not_called()
    l2e.assert_not_called()
    assert counters.get("blitz_no_candidates") == 1


def test_find_via_blitz_caches_per_domain():
    """Second call with same domain reuses the cached candidate list —
    no extra domain_to_linkedin or find_employees_by_title calls."""
    with patch("tools.find_managers.domain_to_linkedin",
               return_value="https://linkedin.com/company/acme") as d2l, \
         patch("tools.find_managers.find_employees_by_title",
               return_value=[_candidate()]) as feb, \
         patch("tools.find_managers.linkedin_to_email",
               return_value="sarah.doe@acme.com"), \
         patch("tools.find_managers.verify_email_mv", return_value=True):
        find_via_blitz("acme.com")
        find_via_blitz("acme.com")
    assert d2l.call_count == 1
    assert feb.call_count == 1


# ---------- Icypeas domain-search ----------

def test_find_via_icypeas_domain_returns_hiring_keyword_match():
    contacts = [
        {"email": "founder@acme.com", "name": "Joe Founder", "position": "Founder"},
        {"email": "talent@acme.com", "name": "Tina Talent", "position": "Head of Talent Acquisition"},
        {"email": "eng@acme.com", "name": "Eng Person", "position": "Software Engineer"},
    ]
    with patch("tools.find_managers.find_domain_emails_icypeas",
               return_value=contacts) as fde, \
         patch("tools.find_managers.verify_email_mv", return_value=True) as mv:
        email, name, source, position = find_via_icypeas_domain("acme.com")
    assert email == "talent@acme.com"
    assert name == "Tina Talent"
    assert source == "icypeas_domain+mv_ok"
    assert "Talent" in position
    fde.assert_called_once_with("acme.com")
    mv.assert_called_once_with("talent@acme.com")


def test_find_via_icypeas_domain_caches_per_domain():
    contacts = [
        {"email": "talent@acme.com", "name": "Tina", "position": "Head of Talent"},
    ]
    with patch("tools.find_managers.find_domain_emails_icypeas",
               return_value=contacts) as fde, \
         patch("tools.find_managers.verify_email_mv", return_value=True):
        find_via_icypeas_domain("acme.com")
        find_via_icypeas_domain("acme.com")
    assert fde.call_count == 1


def test_find_via_icypeas_domain_returns_empty_when_no_contacts():
    with patch("tools.find_managers.find_domain_emails_icypeas",
               return_value=[]) as fde, \
         patch("tools.find_managers.verify_email_mv") as mv:
        result = find_via_icypeas_domain("acme.com")
    assert result == ("", "", "", "")
    mv.assert_not_called()


# ---------- Icypeas email-finder ----------

def test_find_via_icypeas_email_returns_email_when_mv_verified():
    with patch("tools.find_managers.find_email_icypeas",
               return_value=("sarah.doe@acme.com", "icypeas_email")) as fei, \
         patch("tools.find_managers.verify_email_mv", return_value=True):
        email, source = find_via_icypeas_email("Sarah", "Doe", "acme.com")
    assert email == "sarah.doe@acme.com"
    assert source == "icypeas+mv_ok"
    fei.assert_called_once_with("Sarah", "Doe", "acme.com")


def test_find_via_icypeas_email_returns_uncertain_when_mv_none():
    with patch("tools.find_managers.find_email_icypeas",
               return_value=("sarah.doe@acme.com", "icypeas_email")), \
         patch("tools.find_managers.verify_email_mv", return_value=None):
        email, source = find_via_icypeas_email("Sarah", "Doe", "acme.com")
    assert source == "icypeas+mv_uncertain"


def test_find_via_icypeas_email_returns_empty_when_mv_rejected():
    with patch("tools.find_managers.find_email_icypeas",
               return_value=("sarah.doe@acme.com", "icypeas_email")), \
         patch("tools.find_managers.verify_email_mv", return_value=False):
        result = find_via_icypeas_email("Sarah", "Doe", "acme.com")
    assert result == ("", "")


def test_find_via_icypeas_email_skips_when_missing_args():
    with patch("tools.find_managers.find_email_icypeas") as fei:
        assert find_via_icypeas_email("", "Doe", "acme.com") == ("", "")
        assert find_via_icypeas_email("Sarah", "", "acme.com") == ("", "")
        assert find_via_icypeas_email("Sarah", "Doe", "") == ("", "")
    fei.assert_not_called()


# ---------- Summary ----------

def test_format_summary_counts_each_source():
    counters = {
        "jd_scrape": 3,
        "blitz_mv_ok": 6,
        "blitz_mv_uncertain": 2,
        "blitz_mv_rejected": 1,
        "blitz_domain_mismatch": 2,
        "blitz_irrelevant_titles": 1,
        "blitz_no_candidates": 3,
        "icypeas_domain": 1,
        "icypeas_email": 2,
        "hunter_domain_mv_ok": 4,
        "hunter_domain_mv_uncertain": 1,
        "hunter_domain_mv_rejected": 2,
        "hunter_finder": 1,
        "permutator_verified": 2,
        "permutator_catch_all": 1,
        "permutator_unverified": 4,
        "no_manager_found": 5,
    }
    out = format_summary(counters, total=23)
    assert "23" in out
    assert "JD scrape: 3" in out
    assert "Blitz (MV ok): 6" in out
    assert "Blitz (MV uncertain): 2" in out
    assert "Blitz (MV rejected): 1" in out
    assert "Blitz (domain mismatch): 2" in out
    assert "Blitz (irrelevant titles): 1" in out
    assert "Blitz (no candidates): 3" in out
    assert "Icypeas domain-search: 1" in out
    assert "Icypeas email-search: 2" in out
    assert "Hunter domain-search (MV ok): 4" in out
    assert "Hunter domain-search (MV uncertain): 1" in out
    assert "Hunter domain-search (MV rejected): 2" in out
    assert "Hunter email-finder: 1" in out
    assert "Permutator (MV verified): 2" in out
    assert "Permutator (catch-all): 1" in out
    assert "Permutator (unverified): 4" in out
    assert "No manager found: 5" in out


# ---------- Hunter domain-search: hiring-keyword filter (regression for 2026-04-29) ----------

def test_hunter_domain_returns_empty_when_no_hiring_keyword_match(monkeypatch):
    """Regression test for the silent-failure bug surfaced 2026-04-29.

    On the Webflow cascade run, no employee at webflow.com matched the
    hiring-position keyword filter. The OLD code fell back to "first
    high-confidence personal email" and surfaced a Senior Director of
    Product Marketing as a 'hiring manager' for a Forward Deployed
    Engineer search. New code returns empty in this case so the cascade
    can move to Icypeas / permutator instead of shipping a bad contact.
    """
    monkeypatch.setenv("HUNTER_API_KEY", "test-key")
    fm._DOMAIN_CACHE.clear()

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "data": {
            "emails": [
                # Senior PMM person — high confidence, personal, but NOT hiring-related
                {"value": "vikas.bhagat@webflow.com", "first_name": "Vikas",
                 "last_name": "Bhagat", "type": "personal", "confidence": 99,
                 "position": "Senior Director, Head of Product Marketing"},
                # Random sales person — high confidence, personal, NOT hiring-related
                {"value": "joe.sales@webflow.com", "first_name": "Joe",
                 "last_name": "Sales", "type": "personal", "confidence": 95,
                 "position": "Account Executive"},
            ]
        }
    }
    with patch("tools.find_managers.requests.get", return_value=mock_resp):
        result = fm.find_via_hunter_domain("webflow.com")

    assert result == ("", "", "", ""), (
        f"Expected empty result when no hiring-keyword match; "
        f"got {result!r} (the dangerous fallback came back)"
    )


def test_hunter_domain_still_returns_match_when_keyword_present(monkeypatch):
    """Happy path: a hiring-related title in the response IS returned.

    The fallback removal must not break the legitimate case where Hunter
    has a real talent / VP Eng / Director of Engineering record.
    """
    monkeypatch.setenv("HUNTER_API_KEY", "test-key")
    fm._DOMAIN_CACHE.clear()

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "data": {
            "emails": [
                # A non-hiring senior person FIRST in the list (would be
                # picked by the old fallback)
                {"value": "marketer@acme.com", "first_name": "M", "last_name": "P",
                 "type": "personal", "confidence": 99,
                 "position": "VP of Marketing"},
                # Then a real hiring person — should be picked
                {"value": "talent@acme.com", "first_name": "T", "last_name": "A",
                 "type": "personal", "confidence": 90,
                 "position": "Senior Talent Acquisition Partner"},
            ]
        }
    }
    with patch("tools.find_managers.requests.get", return_value=mock_resp):
        email, name, source, position = fm.find_via_hunter_domain("acme.com")

    assert email == "talent@acme.com"
    assert name == "T A"
    assert source.startswith("hunter_domain+conf90")
    assert "Talent" in position


def test_hunter_domain_returns_empty_on_zero_emails(monkeypatch):
    """If Hunter returns an empty emails array, return empty (not crash, not
    fall back to anything)."""
    monkeypatch.setenv("HUNTER_API_KEY", "test-key")
    fm._DOMAIN_CACHE.clear()

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"data": {"emails": []}}
    with patch("tools.find_managers.requests.get", return_value=mock_resp):
        result = fm.find_via_hunter_domain("nobody.com")
    assert result == ("", "", "", "")


# ---------- Blitz title-relevance filter (regression for 2026-04-29) ----------

def test_find_via_blitz_filters_irrelevant_titles():
    """Regression for the Yezi/Glean silent-failure 2026-04-29.

    Blitz returned 'VP of Strategy & Ops' for a Solutions Engineer role
    search at Glean, because Blitz's _bucket_titles used 'vp' as a
    prefix keyword and matched any 'VP of [anything]'. The new
    title-relevance filter drops these wrong-target candidates after
    the API returns them.
    """
    counters = {}
    cands = [
        # Wrong-target: VP of Strategy & Ops at the company
        _candidate(first="Yezi", last="Peng", title="VP of Strategy & Ops",
                   url="https://linkedin.com/in/yezi", icp_tier=1),
    ]
    with patch("tools.find_managers.domain_to_linkedin",
               return_value="https://linkedin.com/company/glean"), \
         patch("tools.find_managers.find_employees_by_title", return_value=cands), \
         patch("tools.find_managers.linkedin_to_email") as l2e, \
         patch("tools.find_managers.verify_email_mv") as mv:
        result = find_via_blitz("glean.com", counters=counters)

    assert result == ("", "", "", "")
    assert counters.get("blitz_irrelevant_titles") == 1
    # Short-circuited before linkedin_to_email or MV calls — no API spend
    l2e.assert_not_called()
    mv.assert_not_called()


def test_find_via_blitz_keeps_relevant_titles():
    """Happy path: candidates with role-relevant titles ARE kept."""
    cands = [
        _candidate(first="Real", last="Manager",
                   title="VP of Sales",  # contains 'sales' keyword
                   url="https://linkedin.com/in/real", icp_tier=1),
    ]
    with patch("tools.find_managers.domain_to_linkedin",
               return_value="https://linkedin.com/company/acme"), \
         patch("tools.find_managers.find_employees_by_title", return_value=cands), \
         patch("tools.find_managers.linkedin_to_email",
               return_value="real@acme.com"), \
         patch("tools.find_managers.verify_email_mv", return_value=True):
        email, name, _, source = find_via_blitz("acme.com")
    assert email == "real@acme.com"
    assert source == "blitz+mv_ok"


def test_find_via_blitz_filters_some_keeps_others():
    """Mixed list: 2 irrelevant + 1 relevant. The relevant one wins."""
    cands = [
        # Irrelevant
        _candidate(first="Bad1", last="A", title="VP of Finance", icp_tier=1,
                   url="https://linkedin.com/in/bad1"),
        # Relevant: contains "engineering"
        _candidate(first="Good", last="One", title="VP of Engineering",
                   icp_tier=2, url="https://linkedin.com/in/good"),
        # Irrelevant
        _candidate(first="Bad2", last="C", title="VP of Procurement",
                   icp_tier=1, url="https://linkedin.com/in/bad2"),
    ]
    with patch("tools.find_managers.domain_to_linkedin",
               return_value="https://linkedin.com/company/acme"), \
         patch("tools.find_managers.find_employees_by_title", return_value=cands), \
         patch("tools.find_managers.linkedin_to_email",
               return_value="good@acme.com"), \
         patch("tools.find_managers.verify_email_mv", return_value=True):
        email, name, _, _ = find_via_blitz("acme.com")
    assert email == "good@acme.com"
    assert name == "Good One"


def test_has_relevant_title_basic():
    """Whitelist coverage check. Common role titles pass / common
    wrong-org titles fail."""
    from tools.find_managers import _has_relevant_title

    # PASS — role-relevant
    assert _has_relevant_title("VP of Sales") is True
    assert _has_relevant_title("Head of Engineering") is True
    assert _has_relevant_title("Senior Director, Customer Engineering") is True
    assert _has_relevant_title("Co-Founder") is True
    assert _has_relevant_title("Solutions Architect") is True
    assert _has_relevant_title("VP, Forward Deployed") is True

    # FAIL — wrong-org
    assert _has_relevant_title("VP of Strategy & Ops") is False
    assert _has_relevant_title("VP of Finance") is False
    assert _has_relevant_title("VP of Procurement") is False
    assert _has_relevant_title("Director of Legal") is False
    assert _has_relevant_title("") is False
    assert _has_relevant_title(None) is False
    # Talent Acquisition is now an HR disqualifier (post-2026-04-29 fix).
    # Even if Blitz returns a TA person via the People/TA fallback search
    # bucket, the post-filter rejects them because TA contacts are
    # recruiters NOT engineering hiring managers.
    assert _has_relevant_title("Talent Acquisition Partner") is False


def test_has_relevant_title_disqualifies_ops_comp_strategy():
    """Regression for the Twilio/Alex Gousinov bug 2026-04-29.

    Before the disqualifier list was added, "Sr Manager, Sales
    Compensation" passed _has_relevant_title because the positive
    whitelist included "sales" as a substring. Comp ops, sales ops,
    GTM strategy, RevOps, and marketing ops people own quotas /
    dashboards / plans, NOT headcount for SE / GTM Engineer / FDE
    roles. They must be filtered.
    """
    from tools.find_managers import _has_relevant_title

    # FAIL: sales-adjacent ops / comp / strategy (the actual bug case)
    assert _has_relevant_title("Sr Manager, Sales Compensation") is False
    assert _has_relevant_title("Director of Sales Operations") is False
    assert _has_relevant_title("Sales Operations Lead") is False
    assert _has_relevant_title("Sales Strategy Director") is False
    assert _has_relevant_title("VP, Sales Enablement") is False
    assert _has_relevant_title("Head of Sales Planning") is False
    assert _has_relevant_title("Sales Analytics Manager") is False
    assert _has_relevant_title("Sales Finance Lead") is False

    # FAIL: GTM-adjacent ops / strategy
    assert _has_relevant_title("Director of GTM Strategy") is False
    assert _has_relevant_title("Head of GTM Operations") is False
    assert _has_relevant_title("VP, GTM Planning") is False

    # FAIL: Revenue-adjacent ops / strategy
    assert _has_relevant_title("VP Revenue Operations") is False
    assert _has_relevant_title("Director, Revenue Strategy") is False
    assert _has_relevant_title("Head of RevOps") is False

    # FAIL: Marketing-adjacent ops / planning / analytics
    assert _has_relevant_title("Director of Marketing Operations") is False
    assert _has_relevant_title("Head of Marketing Planning") is False
    assert _has_relevant_title("VP, Marketing Analytics") is False

    # PASS: true hiring managers must still survive
    assert _has_relevant_title("VP Sales") is True
    assert _has_relevant_title("Head of Solutions Engineering") is True
    assert _has_relevant_title("VP of Sales") is True
    assert _has_relevant_title("Director of Engineering") is True
    assert _has_relevant_title("CRO") is True
    assert _has_relevant_title("Head of GTM") is True


def test_has_relevant_title_disqualifies_hr_recruiting_people():
    """Regression for the Tara/Summer/Brooke wrong-target sample 2026-04-29.

    blitz+mv_ok shipped these three as 'hiring managers' for SE / FDE
    candidate searches because Blitz's HIRING_MANAGER_TITLES fallback
    bucket includes Head of Talent / Head of People as last-resort
    search targets, and the post-filter previously accepted any title
    containing 'talent' or 'people' as a positive keyword. Those
    contacts are HR / recruiting / people-ops generalists, NOT
    engineering hiring managers.
    """
    from tools.find_managers import _has_relevant_title

    # FAIL: the actual bug-sample candidates
    # Tara Murray (Webflow) — RevOps. Already covered by revenue-ops
    # disqualifier; included for the cohort completeness check.
    assert _has_relevant_title("VP Revenue Operations") is False
    # Summer Edstrom (Hex) — recruiter dressed as "Talent Lead"
    assert _has_relevant_title("Talent Lead") is False
    assert _has_relevant_title("Senior Talent Lead") is False
    # Brooke Worzalla (Drata) — HR generalist
    assert _has_relevant_title("Head of People") is False

    # FAIL: HR / People-Ops phrase variants
    assert _has_relevant_title("VP People") is False
    assert _has_relevant_title("VP of People") is False
    assert _has_relevant_title("VP, People") is False
    assert _has_relevant_title("Chief People Officer") is False
    assert _has_relevant_title("Director of People Operations") is False
    assert _has_relevant_title("People Operations Lead") is False
    assert _has_relevant_title("People Partner") is False

    # FAIL: HR-proper
    assert _has_relevant_title("Human Resources Director") is False
    assert _has_relevant_title("VP, Human Resources") is False
    assert _has_relevant_title("Chief Human Capital Officer") is False
    assert _has_relevant_title("Head of HR") is False
    assert _has_relevant_title("VP HR") is False
    assert _has_relevant_title("Director of HR") is False
    assert _has_relevant_title("HR Business Partner") is False
    assert _has_relevant_title("Senior HR Manager") is False
    assert _has_relevant_title("HR Operations Lead") is False

    # FAIL: Talent / Recruiting (bare-substring disqualifiers)
    assert _has_relevant_title("Talent Acquisition Partner") is False
    assert _has_relevant_title("Senior Talent Acquisition Manager") is False
    assert _has_relevant_title("Head of Talent") is False
    assert _has_relevant_title("Head of Talent Acquisition") is False
    assert _has_relevant_title("VP of Talent") is False
    assert _has_relevant_title("Senior Recruiter") is False
    assert _has_relevant_title("Technical Recruiter") is False
    assert _has_relevant_title("Recruiter") is False
    assert _has_relevant_title("Recruiting Manager") is False
    assert _has_relevant_title("Director of Recruiting") is False
    assert _has_relevant_title("Head of Recruiting") is False

    # PASS: a "people" mention attached to engineering must survive.
    # Eng leaders DO manage people — the disqualifier list uses
    # phrase-level matching for "people" so "Senior People Manager,
    # Engineering" is not vetoed by a bare-substring check.
    assert _has_relevant_title("Senior People Manager, Engineering") is True
    # Customer Engineering Manager — true target role
    assert _has_relevant_title("Customer Engineering Manager") is True
    # Engineering Manager — true target role
    assert _has_relevant_title("Engineering Manager") is True


def test_find_via_blitz_filters_hr_candidates():
    """End-to-end Blitz cascade test with the Tara/Summer/Brooke shape.

    Blitz returns 3 candidates: VP Revenue Operations, Talent Lead, and
    Head of People. All three must be filtered by the post-filter
    BEFORE linkedin_to_email or MV verify spend any API budget. Result
    is empty + blitz_irrelevant_titles=1.
    """
    counters = {}
    cands = [
        # Tara-shaped: RevOps
        _candidate(first="Tara", last="Murray", title="VP Revenue Operations",
                   url="https://linkedin.com/in/tara", icp_tier=1),
        # Summer-shaped: recruiter as "Talent Lead"
        _candidate(first="Summer", last="Edstrom", title="Talent Lead",
                   url="https://linkedin.com/in/summer", icp_tier=2),
        # Brooke-shaped: HR generalist
        _candidate(first="Brooke", last="Worzalla", title="Head of People",
                   url="https://linkedin.com/in/brooke", icp_tier=2),
    ]
    with patch("tools.find_managers.domain_to_linkedin",
               return_value="https://linkedin.com/company/acme"), \
         patch("tools.find_managers.find_employees_by_title", return_value=cands), \
         patch("tools.find_managers.linkedin_to_email") as l2e, \
         patch("tools.find_managers.verify_email_mv") as mv:
        result = find_via_blitz("acme.com", counters=counters)

    assert result == ("", "", "", "")
    assert counters.get("blitz_irrelevant_titles") == 1
    # No linkedin_to_email or MV calls — the post-filter short-circuited
    # before any paid API spend.
    l2e.assert_not_called()
    mv.assert_not_called()


def test_find_via_blitz_picks_engineer_over_hr_in_mixed_list():
    """Mixed list: 2 HR/recruiter candidates + 1 real engineering hire.
    The engineering candidate wins; HR candidates are silently filtered.
    """
    cands = [
        # HR — must be filtered
        _candidate(first="Brooke", last="HR", title="Head of People",
                   url="https://linkedin.com/in/brooke", icp_tier=1),
        # Recruiter — must be filtered
        _candidate(first="Summer", last="TA", title="Senior Recruiter",
                   url="https://linkedin.com/in/summer", icp_tier=1),
        # Real engineering hiring manager — must win
        _candidate(first="Anwesh", last="Rijal",
                   title="Head of Solutions Engineering",
                   url="https://linkedin.com/in/anwesh", icp_tier=2),
    ]
    with patch("tools.find_managers.domain_to_linkedin",
               return_value="https://linkedin.com/company/glean"), \
         patch("tools.find_managers.find_employees_by_title", return_value=cands), \
         patch("tools.find_managers.linkedin_to_email",
               return_value="anwesh@glean.com"), \
         patch("tools.find_managers.verify_email_mv", return_value=True):
        email, name, _, source = find_via_blitz("glean.com")
    assert email == "anwesh@glean.com"
    assert name == "Anwesh Rijal"
    assert source == "blitz+mv_ok"


# ---------- Regional variant detection (multi-posting bug, 2026-04-29) ----------

def test_detect_regional_variant_east_coast():
    """The original bug case: n8n's 'East Coast' SE posting must be detected."""
    assert fm._detect_regional_variant(
        "Senior Solutions Engineer | East Coast - Remote"
    ) == "east_coast"
    assert fm._detect_regional_variant("AE - EST") == "east_coast"
    assert fm._detect_regional_variant("GTM Engineer (Eastern)") == "east_coast"


def test_detect_regional_variant_west_coast():
    """The wrong-region paired posting from the same n8n bug."""
    assert fm._detect_regional_variant(
        "Senior Solutions Engineer | West Coast - Remote"
    ) == "west_coast"
    assert fm._detect_regional_variant("AE - PST") == "west_coast"
    assert fm._detect_regional_variant("Solutions Engineer, Pacific") == "west_coast"


def test_detect_regional_variant_global_regions():
    """EMEA / APAC / Americas / continent-level markers."""
    assert fm._detect_regional_variant("Sales Engineer EMEA") == "emea"
    assert fm._detect_regional_variant("Account Executive APAC") == "apac"
    assert fm._detect_regional_variant(
        "Forward Deployed Engineer (Europe)"
    ) == "emea"
    assert fm._detect_regional_variant("Senior SE - North America") == "north_america"
    assert fm._detect_regional_variant("Sales Director - Americas") == "americas"
    assert fm._detect_regional_variant("Asia Pacific Solutions Engineer") == "apac"


def test_detect_regional_variant_country_codes():
    """Country-level markers — UK / Germany / France / US."""
    assert fm._detect_regional_variant("Sales Lead - UK") == "uk"
    assert fm._detect_regional_variant("Senior Engineer Germany") == "germany"
    assert fm._detect_regional_variant("Solutions Architect, France") == "france"
    assert fm._detect_regional_variant("Senior Engineer - United States") == "us"
    assert fm._detect_regional_variant("VP, US Sales") == "us"


def test_detect_regional_variant_returns_none_for_plain_titles():
    """Titles without a regional marker return None — single-posting roles
    must keep the existing cascade behavior intact."""
    assert fm._detect_regional_variant("Senior Solutions Engineer") is None
    assert fm._detect_regional_variant("Forward Deployed Engineer") is None
    assert fm._detect_regional_variant("VP of Sales") is None
    assert fm._detect_regional_variant("") is None
    assert fm._detect_regional_variant(None) is None


def test_detect_regional_variant_no_false_positives():
    """Substring traps: 'Test' must not match 'EST', 'success' must not
    match 'us', 'easter[ly]' must not match 'east'."""
    # These titles contain letter sequences that look like region codes
    # but are part of unrelated words — must NOT match.
    assert fm._detect_regional_variant("Test Engineer") is None
    assert fm._detect_regional_variant("Customer Success Manager") is None
    assert fm._detect_regional_variant("Easterly Hiring Lead") is None
    assert fm._detect_regional_variant("Bus Driver") is None  # contains 'us' substring


# ---------- Multi-posting cascade integration tests ----------
#
# These exercise main() end-to-end with a tmp Excel file. The fixture writes
# a small DataFrame to disk, monkey-patches OUTPUT_DIR / load_profile, and
# stubs every external call (Blitz / Hunter / Icypeas / MV). We then assert
# the row's manager_email and manager_source after main() runs.

import tempfile
from pathlib import Path
from types import SimpleNamespace

import pandas as pd


def _make_master_df(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal master-shape DataFrame for main() to consume."""
    base_cols = {
        "company": "", "title": "", "description": "", "company_url": "",
        "job_url": "", "fit_score": 0.0,
        "manager_email": pd.NA, "manager_name": pd.NA,
        "manager_linkedin": pd.NA, "manager_source": pd.NA,
    }
    return pd.DataFrame([{**base_cols, **r} for r in rows])


@pytest.fixture
def _tmp_master_env(tmp_path, monkeypatch):
    """Wire main() to read/write a tmp Excel file and skip the Drive upload."""
    master_file = tmp_path / "master.xlsx"
    fake_profile = SimpleNamespace(
        drive=SimpleNamespace(master_filename="master.xlsx")
    )
    monkeypatch.setattr(fm, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(fm, "load_profile", lambda _path: fake_profile)
    monkeypatch.setattr(fm, "load_company_domains", dict)
    # Mute side effects we don't care about in unit-style runs.
    monkeypatch.setattr(fm, "upload_to_drive", lambda *a, **kw: None)
    monkeypatch.setattr(
        fm, "upload_vector_docs",
        lambda *a, **kw: {"uploaded": 0, "skipped": 0, "failed": 0},
    )
    monkeypatch.setattr(fm, "mv_credits_remaining", lambda: None)
    monkeypatch.setattr(fm, "reorder_columns", lambda df: df)
    # Force argparse to see only the args we want.
    monkeypatch.setattr(
        "sys.argv",
        ["find_managers", "--profile", "test", "--top", "10", "--no-upload"],
    )
    return master_file


def test_multi_posting_two_regions_each_gets_role_specific_blitz(
    _tmp_master_env, monkeypatch
):
    """Two postings, same company, different regions. Blitz returns a
    different role-specific manager for each — the cascade picks the
    Blitz result on both rows. This is the SUCCESS path: when Blitz
    is healthy, the regional flag is harmless, the role-aware match
    wins.
    """
    df = _make_master_df([
        {"company": "n8n", "title": "Senior Solutions Engineer | East Coast - Remote",
         "company_url": "https://n8n.io", "fit_score": 9.0},
        {"company": "n8n", "title": "Senior Solutions Engineer | West Coast - Remote",
         "company_url": "https://n8n.io", "fit_score": 8.5},
    ])
    df.to_excel(_tmp_master_env, index=False)

    # Blitz: same domain, but in this scenario it returns Hernan first — for
    # the test we just confirm Blitz wins both rows. (The cascade caches Blitz
    # candidates by domain, so both rows see the same result; what matters
    # is that the Hunter fallback is NEVER reached when Blitz hits.)
    monkeypatch.setattr(
        fm, "find_via_blitz",
        lambda domain, counters=None: (
            "hernan@n8n.io", "Hernan Echeverry",
            "https://linkedin.com/in/hernan", "blitz+mv_ok",
        ),
    )
    # Hunter / Icypeas must NOT be called (Blitz hit short-circuits).
    hunter_call_count = {"n": 0}
    def _hunter_should_not_run(*a, **kw):
        hunter_call_count["n"] += 1
        return ("", "", "", "")
    monkeypatch.setattr(fm, "find_via_hunter_domain", _hunter_should_not_run)
    monkeypatch.setattr(fm, "find_via_icypeas_domain",
                        lambda *a, **kw: ("", "", "", ""))
    monkeypatch.setattr(fm, "verify_email_mv", lambda email: True)

    fm.main()

    out = pd.read_excel(_tmp_master_env)
    assert len(out) == 2
    # Both rows pick Blitz; the regional flag never blocks because Blitz hit.
    assert all(out["manager_source"] == "blitz+mv_ok")
    assert all(out["manager_email"] == "hernan@n8n.io")
    assert hunter_call_count["n"] == 0


def test_single_posting_no_regional_variant_uses_hunter_fallback(
    _tmp_master_env, monkeypatch
):
    """Single posting, no regional marker. Blitz misses, so Hunter
    domain-search is allowed to run and seat its candidate. Existing
    cascade behavior preserved."""
    df = _make_master_df([
        {"company": "Acme", "title": "Senior Solutions Engineer",
         "company_url": "https://acme.com", "fit_score": 9.0},
    ])
    df.to_excel(_tmp_master_env, index=False)

    # Blitz returns nothing.
    monkeypatch.setattr(
        fm, "find_via_blitz",
        lambda domain, counters=None: ("", "", "", ""),
    )
    # Hunter returns a real hiring contact.
    monkeypatch.setattr(
        fm, "find_via_hunter_domain",
        lambda domain: (
            "talent@acme.com", "Tina Talent",
            "hunter_domain+conf90", "Head of Talent Acquisition",
        ),
    )
    monkeypatch.setattr(fm, "find_via_icypeas_domain",
                        lambda d: ("", "", "", ""))
    monkeypatch.setattr(fm, "verify_email_mv", lambda email: True)

    fm.main()

    out = pd.read_excel(_tmp_master_env)
    assert len(out) == 1
    assert out.iloc[0]["manager_email"] == "talent@acme.com"
    assert out.iloc[0]["manager_source"].startswith("hunter_domain+conf90")


def test_multi_posting_blocks_hunter_domain_fallback(
    _tmp_master_env, monkeypatch
):
    """The actual bug case: a regional posting where Blitz returns nothing.
    Hunter domain-search WOULD return a contact (a TA / hiring manager via
    domain fallback), but the cascade must REJECT it because the contact
    was found via domain-search not role-search and would route to the
    wrong-region manager. The row must end with manager_email NaN and
    manager_source 'multi_posting_blocked:<region>'.
    """
    df = _make_master_df([
        {"company": "n8n",
         "title": "Senior Solutions Engineer | West Coast - Remote",
         "company_url": "https://n8n.io", "fit_score": 8.5},
    ])
    df.to_excel(_tmp_master_env, index=False)

    # Blitz misses — would fall to Hunter in single-posting mode.
    monkeypatch.setattr(
        fm, "find_via_blitz",
        lambda domain, counters=None: ("", "", "", ""),
    )
    # Hunter has a domain-fallback contact (the wrong-region TA from the bug).
    hunter_calls = {"n": 0}
    def _hunter_must_not_run(domain):
        hunter_calls["n"] += 1
        return ("elena@n8n.io", "Elena Ayvazyan",
                "hunter_domain+conf99", "Talent Acquisition Manager")
    monkeypatch.setattr(fm, "find_via_hunter_domain", _hunter_must_not_run)
    icypeas_calls = {"n": 0}
    def _icypeas_must_not_run(domain):
        icypeas_calls["n"] += 1
        return ("ip@n8n.io", "I P", "icypeas_domain+mv_ok", "Recruiter")
    monkeypatch.setattr(fm, "find_via_icypeas_domain", _icypeas_must_not_run)
    monkeypatch.setattr(fm, "verify_email_mv", lambda email: True)

    fm.main()

    out = pd.read_excel(_tmp_master_env)
    assert len(out) == 1
    # No email (forces manual lookup).
    assert pd.isna(out.iloc[0]["manager_email"]), (
        f"Expected manager_email NaN; got {out.iloc[0]['manager_email']!r}. "
        "Hunter domain-fallback leaked into a multi-posting row."
    )
    # Source carries the multi_posting_blocked marker with region tag.
    assert out.iloc[0]["manager_source"] == "multi_posting_blocked:west_coast"
    # Hunter and Icypeas domain-search MUST NOT have been called.
    assert hunter_calls["n"] == 0, (
        "Hunter domain-search was called for a multi-posting row — guard failed"
    )
    assert icypeas_calls["n"] == 0, (
        "Icypeas domain-search was called for a multi-posting row — guard failed"
    )
