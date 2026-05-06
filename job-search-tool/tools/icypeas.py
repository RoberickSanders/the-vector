"""Icypeas email enrichment.

Mirrored inline from the upstream `a separate codebase` —
DO NOT import from upstream. upstream code is reference-only.

Auth: simple `Authorization: <API_KEY>` header. No HMAC. The API_SECRET
and USER_ID env vars are kept around for forward-compat (Icypeas's other
endpoints reference them) but the search/submit/poll endpoints we use
only need the key.

Pattern (async):
  1. POST /email-search or /domain-search with payload -> get search_id
  2. Poll POST /bulk-single-searchs/read until status leaves
     {NONE, SCHEDULED, IN_PROGRESS}
  3. Read result.results.emails[] (or empty)

Both wrappers skip silently (return empty) when ICYPEAS_API_KEY is missing,
so the cascade in find_managers can call them unconditionally.
"""
import os
import time
from typing import Optional

import requests

ICYPEAS_BASE = "https://app.icypeas.com/api"
ICYPEAS_EMAIL_ENDPOINT = f"{ICYPEAS_BASE}/email-search"
ICYPEAS_DOMAIN_ENDPOINT = f"{ICYPEAS_BASE}/domain-search"
ICYPEAS_POLL_ENDPOINT = f"{ICYPEAS_BASE}/bulk-single-searchs/read"

# Polling tunables (matched to the upstream defaults).
ICYPEAS_POLL_INTERVAL = 2  # seconds between polls
ICYPEAS_POLL_MAX_WAIT = 30  # seconds total before giving up

# Status values that mean "still working" — we keep polling on these.
_PENDING_STATUSES = ("NONE", "SCHEDULED", "IN_PROGRESS")


def _icypeas_headers() -> Optional[dict]:
    """Build Icypeas auth headers. Returns None when API key is missing.

    The check uses os.environ.get(...) for compatibility with the
    `ANTHROPIC_API_KEY=""` quirk noted in the upstream CLAUDE.md.
    """
    api_key = os.environ.get("ICYPEAS_API_KEY")
    if not api_key:
        return None
    return {
        "Content-Type": "application/json",
        "Authorization": api_key,
    }


def _icypeas_poll_result(search_id: str, headers: dict) -> Optional[dict]:
    """Poll the bulk-single-searchs/read endpoint until result is ready.

    Returns the first non-pending item, or None on timeout/error.
    """
    elapsed = 0
    while elapsed < ICYPEAS_POLL_MAX_WAIT:
        time.sleep(ICYPEAS_POLL_INTERVAL)
        elapsed += ICYPEAS_POLL_INTERVAL
        try:
            r = requests.post(
                ICYPEAS_POLL_ENDPOINT,
                headers=headers,
                json={"id": search_id},
                timeout=15,
            )
        except Exception:
            continue
        if r.status_code != 200:
            continue
        try:
            data = r.json()
        except (ValueError, AttributeError):
            continue
        if not data.get("success"):
            continue
        items = data.get("items", []) or []
        if not items:
            continue
        item = items[0]
        if item.get("status", "") in _PENDING_STATUSES:
            continue
        return item
    return None


def find_email_icypeas(first: str, last: str, domain: str) -> tuple[str, str]:
    """Find an email by first + last + domain via Icypeas /email-search.

    Returns (email, source_tag) on success; ('', '') on missing creds,
    no result, or any error. Costs ~$0.015 per call when it runs.
    """
    headers = _icypeas_headers()
    if not headers or not first or not last or not domain:
        return ("", "")

    payload = {
        "firstname": first,
        "lastname": last,
        "domainOrCompany": domain,
    }
    try:
        r = requests.post(
            ICYPEAS_EMAIL_ENDPOINT, headers=headers, json=payload, timeout=15
        )
    except Exception:
        return ("", "")
    if r.status_code != 200:
        return ("", "")
    try:
        submit_data = r.json()
    except (ValueError, AttributeError):
        return ("", "")
    if not submit_data.get("success"):
        return ("", "")
    search_id = (submit_data.get("item") or {}).get("_id", "")
    if not search_id:
        return ("", "")

    result = _icypeas_poll_result(search_id, headers)
    if not result:
        return ("", "")

    results_data = result.get("results", {}) or {}
    emails_list = results_data.get("emails", []) or []
    if not emails_list:
        return ("", "")
    email = (emails_list[0] or {}).get("email", "")
    if not email:
        return ("", "")
    return (email, "icypeas_email")


def find_domain_emails_icypeas(domain: str) -> list[dict]:
    """Find emails at a domain via Icypeas /domain-search.

    Returns a list of dicts: [{"email": "...", "name": "", "position": ""}].
    Empty list on missing creds, no result, or error. ~$0.015 per call.
    """
    headers = _icypeas_headers()
    if not headers or not domain:
        return []

    payload = {"domainOrCompany": domain}
    try:
        r = requests.post(
            ICYPEAS_DOMAIN_ENDPOINT, headers=headers, json=payload, timeout=15
        )
    except Exception:
        return []
    if r.status_code != 200:
        return []
    try:
        submit_data = r.json()
    except (ValueError, AttributeError):
        return []
    if not submit_data.get("success"):
        return []
    search_id = (submit_data.get("item") or {}).get("_id", "")
    if not search_id:
        return []

    result = _icypeas_poll_result(search_id, headers)
    if not result:
        return []

    contacts: list[dict] = []
    results_data = result.get("results", {}) or {}
    for item in (results_data.get("emails", []) or []):
        email = (item or {}).get("email", "")
        if email and "@" in email:
            contacts.append({
                "email": email,
                "name": item.get("name", "") or "",
                "position": item.get("position", "") or "",
            })
    return contacts
