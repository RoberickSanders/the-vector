"""Email utilities: permutator, JD-text extractor, SMTP verifier, MillionVerifier.

upstream-style cascade extensions (added April 2026):
- has_mx_records:        MX pre-check (upstream step 1)
- load/save_domain_memory + remember_domain: persistent cache (upstream step 2)
- scrape_website_emails: contact-page email extraction (upstream step 7)

All Icypeas/upstream logic is duplicated inline here — NO imports from
a separate codebase: . upstream code is reference-only.
"""
import json
import os
import random
import re
import smtplib
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests

EMAIL_REGEX = re.compile(
    r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
)

# --- MillionVerifier (mirror of upstream pattern; duplicated, NOT imported) ---
MV_ENDPOINT = "https://api.millionverifier.com/api/v3/"
_MV_LAST_CALL = 0.0
_MV_RATE_DELAY = 0.2
_MV_DOMAIN_CATCH_ALL_CACHE: dict[str, bool] = {}

GENERIC_PREFIXES = {
    # Auto-reply / no-reply
    "noreply", "no-reply", "donotreply", "do-not-reply",
    # Application portals (won't reach a hiring manager)
    "apply", "applications", "jobs", "careers", "hiring",
    # Generic inboxes
    "info", "hello", "contact", "support", "admin",
    # Department aliases
    "marketing", "sales", "press",
    # Compliance / accommodations / legal — NOT hiring managers, easy to
    # mistake. Buildkite's JD literally lists `accommodations@buildkite.com`
    # as the ADA contact, which the JD-scrape regex grabbed once before
    # this filter caught it.
    "accommodation", "accommodations", "ada", "compliance",
    "legal", "privacy", "dpo", "security",
}


def permutate_emails(first: str, last: str, domain: str) -> list[str]:
    """Generate common email patterns for a name + domain."""
    f = first.lower().strip()
    l = last.lower().strip().replace(" ", "")  # compound last names join
    d = domain.lower().strip().lstrip("@")
    if not f or not d:
        return []
    patterns = [f"{f}.{l}@{d}"] if l else []
    patterns += [
        f"{f}@{d}",
        f"{f}{l}@{d}" if l else "",
        f"{f[0]}{l}@{d}" if l else "",
        f"{f[0]}.{l}@{d}" if l else "",
        f"{l}.{f}@{d}" if l else "",
        f"{f}_{l}@{d}" if l else "",
    ]
    # Dedup, drop empties, preserve order
    seen = set()
    out = []
    for p in patterns:
        if p and p not in seen:
            seen.add(p)
            out.append(p)
    return out


def extract_emails_from_text(text: str) -> list[str]:
    """Extract personal-looking emails from a job description.

    Filters out generic prefixes (noreply, apply, careers, etc.) since
    those won't reach a hiring manager.
    """
    if not isinstance(text, str):
        return []
    found = EMAIL_REGEX.findall(text)
    out = []
    for email in found:
        local = email.split("@")[0].lower()
        if local in GENERIC_PREFIXES:
            continue
        if email not in out:
            out.append(email)
    return out


def smtp_verify(email: str, mail_from: str = "verifier@example.com",
                timeout: int = 5) -> Optional[bool]:
    """Best-effort SMTP RCPT verify.

    Returns True if mailbox accepted, False if 5xx rejection, None on
    inconclusive (timeout, greylisted, server hostile to verification).
    """
    domain = email.split("@", 1)[1] if "@" in email else ""
    if not domain:
        return False
    try:
        import dns.resolver
        mx_records = dns.resolver.resolve(domain, "MX")
        mx_host = sorted(mx_records, key=lambda r: r.preference)[0].exchange.to_text().rstrip(".")
    except Exception:
        return None
    try:
        with smtplib.SMTP(mx_host, 25, timeout=timeout) as smtp:
            smtp.helo("verifier.local")
            smtp.mail(mail_from)
            code, _ = smtp.rcpt(email)
            if code in (250, 251):
                return True
            if 500 <= code < 600:
                return False
            return None
    except Exception:
        return None


def _mv_throttle():
    """Simple throttle to avoid hammering MV."""
    global _MV_LAST_CALL
    elapsed = time.time() - _MV_LAST_CALL
    if elapsed < _MV_RATE_DELAY:
        time.sleep(_MV_RATE_DELAY - elapsed)
    _MV_LAST_CALL = time.time()


def verify_email_mv(email: str) -> Optional[bool]:
    """Verify a single email via MillionVerifier. Returns True/False/None.

    Returns:
        True  - MV says "ok" or "catch_all" (deliverable)
        False - MV says "invalid" or "disposable"
        None  - error, unknown, or no API key (skip silently)

    Mirrors the upstream pattern; no shared code.
    """
    api_key = os.environ.get("MILLIONVERIFIER_API_KEY")
    if not api_key or not email or "@" not in email:
        return None
    _mv_throttle()
    try:
        resp = requests.get(
            MV_ENDPOINT,
            params={"api": api_key, "email": email},
            timeout=15,
        )
    except Exception:
        return None
    if resp.status_code != 200:
        return None
    try:
        data = resp.json()
    except (ValueError, AttributeError):
        return None
    result = (data.get("result") or "unknown").lower()
    if result in ("ok", "catch_all"):
        return True
    if result in ("invalid", "disposable"):
        return False
    return None


def is_catch_all_domain(domain: str) -> bool:
    """Check whether a domain accepts all emails (catch-all).

    Caches per-domain in memory. Sends a random nonexistent email to MV;
    if MV returns 'ok' or 'catch_all', the domain is catch-all.

    Returns False on errors, missing API key, or non-catch-all.
    """
    if not domain:
        return False
    if domain in _MV_DOMAIN_CATCH_ALL_CACHE:
        return _MV_DOMAIN_CATCH_ALL_CACHE[domain]
    api_key = os.environ.get("MILLIONVERIFIER_API_KEY")
    if not api_key:
        _MV_DOMAIN_CATCH_ALL_CACHE[domain] = False
        return False
    test_email = f"zz_jstool_{random.randint(10000, 99999)}@{domain}"
    result = verify_email_mv(test_email)
    is_ca = result is True  # MV said the random email is "deliverable" -> catch-all
    _MV_DOMAIN_CATCH_ALL_CACHE[domain] = is_ca
    return is_ca


# --- upstream-style cascade extensions ---

# Project state directory for persistent caches.
# domain-memory.json schema:
#   {domain: {pattern, accept_all, mx_ok, last_verified_email,
#             last_seen_iso, source, last_recruiter_name,
#             last_recruiter_email, last_recruiter_position}}
_STATE_DIR = Path(__file__).parent.parent / "state"
_DOMAIN_MEMORY_FILE = _STATE_DIR / "domain-memory.json"


def has_mx_records(domain: str) -> bool:
    """Check if a domain publishes any MX records.

    FREE — purely DNS, ~50ms. Used as the first cascade step to short-circuit
    enrichment on dead domains (parking pages, defunct companies).
    Returns False on any DNS error (NXDOMAIN, timeout, no MX, etc.).
    """
    if not domain:
        return False
    try:
        import dns.resolver
        answers = dns.resolver.resolve(domain, "MX", lifetime=5)
        return len(list(answers)) > 0
    except Exception:
        return False


def load_domain_memory() -> dict:
    """Load the persistent domain pattern cache from state/domain-memory.json.

    Returns an empty dict on missing file or corrupt JSON. Never raises.
    """
    if not _DOMAIN_MEMORY_FILE.exists():
        return {}
    try:
        return json.loads(_DOMAIN_MEMORY_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def save_domain_memory(memory: dict) -> None:
    """Persist domain memory to state/domain-memory.json (creates dir as needed)."""
    _STATE_DIR.mkdir(parents=True, exist_ok=True)
    _DOMAIN_MEMORY_FILE.write_text(json.dumps(memory, indent=2, sort_keys=True))


def remember_domain(memory: dict, domain: str, **fields) -> None:
    """Update memory entry for a domain.

    Provided fields are merged into the existing entry; an ISO timestamp
    is always refreshed so we can age-out stale entries later. No-op when
    domain is empty.
    """
    if not domain:
        return
    entry = memory.get(domain, {})
    entry.update(fields)
    entry["last_seen_iso"] = datetime.now().isoformat(timespec="seconds")
    memory[domain] = entry


# --- Website scrape (upstream step 7) ---

WEBSITE_SCRAPE_PATHS = ("", "/careers", "/about", "/contact", "/team")
WEBSITE_SCRAPE_TIMEOUT = 4

# Per-process cache so multiple jobs at the same domain only hit HTTP once.
_WEBSITE_CACHE: dict[str, list[str]] = {}


def scrape_website_emails(domain: str) -> list[str]:
    """Hit a few common pages and return personal-looking emails.

    Caches per-domain in-process. Filters generic prefixes through
    extract_emails_from_text. FREE except for our HTTP egress.
    Stops early once 5 emails are found to keep runs fast.
    """
    if not domain:
        return []
    cached = _WEBSITE_CACHE.get(domain)
    if cached is not None:
        return cached

    found: list[str] = []
    seen = set()
    for path in WEBSITE_SCRAPE_PATHS:
        url = f"https://{domain}{path}"
        try:
            resp = requests.get(
                url,
                timeout=WEBSITE_SCRAPE_TIMEOUT,
                headers={"User-Agent": "Mozilla/5.0"},
            )
        except Exception:
            continue
        if resp.status_code != 200:
            continue
        text = resp.text or ""
        for email in extract_emails_from_text(text):
            if email in seen:
                continue
            # Only keep emails on the same domain — avoids vendor footers.
            try:
                email_domain = email.split("@", 1)[1].lower()
            except IndexError:
                continue
            if email_domain != domain.lower():
                continue
            seen.add(email)
            found.append(email)
        if len(found) >= 5:
            break
    _WEBSITE_CACHE[domain] = found
    return found


def mv_credits_remaining() -> Optional[int]:
    """Returns the MV credit count from a tiny check call. None on failure/no key."""
    api_key = os.environ.get("MILLIONVERIFIER_API_KEY")
    if not api_key:
        return None
    try:
        resp = requests.get(
            MV_ENDPOINT,
            params={"api": api_key, "email": "test@test.com"},
            timeout=10,
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        return data.get("credits")
    except Exception:
        return None
