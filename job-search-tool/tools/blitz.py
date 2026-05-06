"""Blitz API integration for hiring-manager discovery.

Mirrored inline from the upstream `a separate codebase` —
DO NOT import from upstream. upstream code is reference-only (workspace policy).

Three public functions used by the find_managers cascade:
  - domain_to_linkedin(domain): company domain -> LinkedIn URL
  - find_employees_by_title(linkedin_url, titles): cascade ICP keyword search
  - linkedin_to_email(person_linkedin_url): person LinkedIn URL -> work email

All three skip silently (return None / []) when BLITZ_API_KEY is missing,
so the cascade can call them unconditionally. Failures log to stderr.
"""
import os
import sys
import time
from typing import Optional

import requests

BLITZ_BASE_URL = "https://api.blitz-api.ai/v2"

DOMAIN_TO_LINKEDIN_ENDPOINT = "/enrichment/domain-to-linkedin"
EMPLOYEE_FINDER_ENDPOINT = "/search/waterfall-icp-keyword"
EMAIL_ENRICHMENT_ENDPOINT = "/enrichment/email"

DEFAULT_TIMEOUT = 30


def _blitz_headers() -> Optional[dict]:
    """Auth headers for Blitz API. None when key missing.

    Uses os.environ.get(...) (vs `in os.environ`) to handle the
    `ANTHROPIC_API_KEY=""` empty-string quirk noted upstream.
    """
    api_key = os.environ.get("BLITZ_API_KEY")
    if not api_key:
        return None
    return {"x-api-key": api_key, "Content-Type": "application/json"}


def _blitz_post(
    endpoint: str, payload: dict, timeout: int = DEFAULT_TIMEOUT
) -> Optional[dict]:
    """POST to Blitz with a single 429-retry. Returns parsed JSON or None.

    Logs failure shapes to stderr at the same level as the upstream logger.debug
    so a tail -f on the run shows what's going on.
    """
    headers = _blitz_headers()
    if headers is None:
        return None
    url = f"{BLITZ_BASE_URL}{endpoint}"
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
    except Exception as e:
        print(f"    blitz POST {endpoint} error: {str(e)[:200]}", file=sys.stderr)
        return None

    if resp.status_code == 200:
        try:
            return resp.json()
        except (ValueError, AttributeError):
            return None

    if resp.status_code == 429:
        # Rate limited — back off briefly and retry once
        time.sleep(1)
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
        except Exception as e:
            print(
                f"    blitz POST {endpoint} retry error: {str(e)[:200]}",
                file=sys.stderr,
            )
            return None
        if resp.status_code == 200:
            try:
                return resp.json()
            except (ValueError, AttributeError):
                return None
        print(
            f"    blitz {endpoint} retry -> {resp.status_code}",
            file=sys.stderr,
        )
        return None

    # Any other 4xx/5xx
    body = (resp.text or "")[:200]
    print(f"    blitz {endpoint} -> {resp.status_code}: {body}", file=sys.stderr)
    return None


def domain_to_linkedin(domain: str) -> Optional[str]:
    """Resolve a company domain to its LinkedIn company URL.

    Returns the LinkedIn URL string on success, or None when the key is
    missing, the domain is empty, the API errors, or no URL is returned.
    """
    if not domain:
        return None
    data = _blitz_post(DOMAIN_TO_LINKEDIN_ENDPOINT, {"domain": domain})
    if not data:
        return None
    url = data.get("company_linkedin_url")
    return url if url else None


# Title keyword buckets used to build the Blitz ICP cascade. Tier 1 is the
# tightest match ("VP Sales", "CRO"), tier 2 broadens to director-level,
# tier 3 is a catch-all for whatever's left. Mirrors the bucketing logic
# in the upstream blitz.py without importing it.
_OWNER_KEYWORDS = {
    "founder", "co-founder", "cofounder", "ceo", "cro",
    "chief revenue officer", "chief executive officer", "president",
    "head", "vp", "vice president",
}
_EXEC_KEYWORDS = {
    "director", "senior director", "principal", "lead", "manager",
}


def _bucket_titles(titles: list[str]) -> tuple[list[str], list[str], list[str]]:
    """Split a flat list of target titles into 3 cascade tiers."""
    tier1: list[str] = []
    tier2: list[str] = []
    tier3: list[str] = []
    for t in titles:
        low = (t or "").lower()
        if not low:
            continue
        if any(k in low for k in _OWNER_KEYWORDS):
            tier1.append(t)
        elif any(k in low for k in _EXEC_KEYWORDS):
            tier2.append(t)
        else:
            tier3.append(t)
    if not tier1:
        tier1 = titles[:1] if titles else ["VP Sales"]
    if not tier2:
        tier2 = ["Director", "Senior Director"]
    if not tier3:
        tier3 = ["Manager", "Lead"]
    return tier1, tier2, tier3


def find_employees_by_title(
    linkedin_url: str,
    titles: list[str],
    max_results: int = 3,
) -> list[dict]:
    """Find employees at a company that match any of the given titles.

    Uses the Blitz waterfall-ICP-keyword endpoint with a 3-tier cascade
    (tightest titles first). Returns a list of dicts:
        {first_name, last_name, title, linkedin_url, location_city,
         location_state, icp_tier}
    """
    if not linkedin_url or not titles:
        return []

    tier1, tier2, tier3 = _bucket_titles(titles)
    cascade = [
        {
            "include_title": tier1,
            "exclude_title": ["assistant", "intern", "junior", "associate"],
            "location": ["US"],
            "include_headline_search": False,
        },
        {
            "include_title": tier2,
            "exclude_title": ["assistant", "intern", "junior"],
            "location": ["US"],
            "include_headline_search": True,
        },
        {
            "include_title": tier3,
            "exclude_title": ["assistant", "intern", "junior"],
            "location": ["US"],
            "include_headline_search": True,
        },
    ]
    payload = {
        "company_linkedin_url": linkedin_url,
        "cascade": cascade,
        "max_results": max_results,
    }
    data = _blitz_post(EMPLOYEE_FINDER_ENDPOINT, payload, timeout=DEFAULT_TIMEOUT)
    if not data or not data.get("results"):
        return []

    out: list[dict] = []
    for result in data["results"]:
        person = result.get("person", {}) or {}
        location = person.get("location", {}) or {}
        title = ""
        for exp in person.get("experiences", []) or []:
            if exp.get("job_is_current"):
                title = exp.get("job_title", "") or ""
                break
        if not title:
            headline = person.get("headline", "") or ""
            if headline:
                title = headline.split(" at ")[0].split(" @ ")[0].strip()
        out.append(
            {
                "first_name": person.get("first_name", "") or "",
                "last_name": person.get("last_name", "") or "",
                "title": title,
                "linkedin_url": person.get("linkedin_url", "") or "",
                "location_city": location.get("city", "") or "",
                "location_state": location.get("state_code", "") or "",
                "icp_tier": result.get("icp", 0),
            }
        )
    return out


def linkedin_to_email(person_linkedin_url: str) -> Optional[str]:
    """Get a verified work email from a person's LinkedIn profile URL.

    Returns the email string when Blitz reports `found=True`; otherwise
    None (missing key, empty URL, API error, or simply not found).
    """
    if not person_linkedin_url:
        return None
    data = _blitz_post(
        EMAIL_ENRICHMENT_ENDPOINT, {"person_linkedin_url": person_linkedin_url}
    )
    if not data:
        return None
    if data.get("found") and data.get("email"):
        return data["email"]
    return None
