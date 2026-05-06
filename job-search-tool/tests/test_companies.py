import json
from pathlib import Path

import pytest

from tools.companies import normalize_company, find_match, load_remote_friendly_companies

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def registry():
    return {
        "amazon": {"title": "Amazon", "remote_policy": "fully-remote"},
        "gitlab": {"title": "GitLab", "remote_policy": "fully-remote"},
        "1password": {"title": "1Password", "remote_policy": "hybrid"},
    }


def test_normalize_strips_legal_suffixes():
    assert normalize_company("GitLab, Inc.") == "gitlab"
    assert normalize_company("Amazon Web Services") == "amazon web services"
    assert normalize_company("Foo Corp.") == "foo"


def test_normalize_handles_dotcom():
    assert normalize_company("Amazon.com") == "amazon"


def test_find_match_exact(registry):
    assert find_match("GitLab", registry)["title"] == "GitLab"


def test_find_match_prefix(registry):
    """Amazon should match Amazon Web Services and Amazon.com."""
    assert find_match("Amazon Web Services", registry)["title"] == "Amazon"
    assert find_match("Amazon.com", registry)["title"] == "Amazon"


def test_find_match_returns_none_when_no_match(registry):
    assert find_match("RandomStartup XYZ", registry) is None


def test_find_match_skips_short_curated_names():
    """Curated names <4 chars don't trigger prefix matches (avoids false positives)."""
    short_registry = {"hp": {"title": "HP"}}
    # "Hpcompany Inc" should NOT match "hp"
    assert find_match("Hpcompany Inc", short_registry) is None
