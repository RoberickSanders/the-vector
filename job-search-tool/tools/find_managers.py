"""Find hiring-manager contacts for top-scored jobs.

Strategy per job (in cascade order):
1. JD scrape (free) — direct contact emails in the description.
2. Blitz cascade (NEW PRIMARY) — domain -> LinkedIn -> employees -> email,
   gated by MV verify.
3. Hunter domain-search (demoted) — falls through if Blitz returned nothing.
4. Icypeas domain-search (NEW backup) — only if Hunter returned nothing.
5. If we have name + domain but no email yet:
   Hunter email-finder -> Icypeas email-search -> Permutator + MV.

A run summary at the end shows how many enrichments came from each source.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests
from dotenv import load_dotenv

from tools.profile import load_profile
from tools.master import reorder_columns
from tools.drive import upload_to_drive, upload_vector_docs
from tools.blitz import (
    domain_to_linkedin,
    find_employees_by_title,
    linkedin_to_email,
)
from tools.icypeas import (
    find_domain_emails_icypeas,
    find_email_icypeas,
)
from tools.email_tools import (
    extract_emails_from_text,
    is_catch_all_domain,
    mv_credits_remaining,
    permutate_emails,
    smtp_verify,
    verify_email_mv,
)

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
DRIVE_STATE_FILE = PROJECT_ROOT / "config" / "drive-state.json"
COMPANIES_DB_FILE = PROJECT_ROOT / "config" / "companies-db.json"


def load_company_domains() -> dict[str, str]:
    """Return {company_name_lower: domain} from companies-db.json."""
    if not COMPANIES_DB_FILE.exists():
        return {}
    try:
        db = json.loads(COMPANIES_DB_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    out = {}
    for c in db.get("companies", []):
        name = (c.get("name") or "").strip().lower()
        domain = (c.get("domain") or "").strip().lower()
        if name and domain:
            out[name] = domain
    return out

# Auto-load workspace .env so HUNTER_API_KEY (etc.) is available.
# Mirrors the pattern in tools/score.py.
WORKSPACE_ENV = PROJECT_ROOT.parent.parent.parent / ".env"
if WORKSPACE_ENV.exists():
    load_dotenv(WORKSPACE_ENV, override=True)  # override empty strings exported by Claude Code

HUNTER_URL = "https://api.hunter.io/v2/email-finder"
HUNTER_DOMAIN_URL = "https://api.hunter.io/v2/domain-search"
HUNTER_MIN_CONFIDENCE = 70

# Position keywords that indicate a real hiring-related contact (vs. random employee).
# Used to filter Hunter's domain-search results to people likely to respond to
# a job-seeker outreach.
HIRING_POSITION_KEYWORDS = (
    "talent", "recruit", "people", "hr", "human resources",
    "hiring", "staffing", "head of engineering", "vp engineering",
    "director of engineering", "head of revenue", "vp revenue",
    "vp sales", "head of sales", "head of growth", "head of gtm",
)

# Title cascade for the Blitz waterfall-ICP-keyword endpoint. Bucketed by
# blitz._bucket_titles into (owner / exec / manager) tiers — broadest set we'd
# accept as a hiring decision-maker for a GTM Engineer / Forward Deployed role.
HIRING_MANAGER_TITLES = [
    # GTM / Revenue leadership
    "VP Sales", "Head of Sales", "VP Revenue", "Head of Revenue",
    "VP GTM", "Head of GTM", "Chief Revenue Officer", "CRO",
    "Director of Sales", "Director of Revenue Operations",
    "Director of GTM", "Senior Director GTM",
    # Engineering leadership (for Forward Deployed / GTM Engineer roles)
    "VP Engineering", "Head of Engineering",
    # People / TA (fallback)
    "Head of Talent", "Head of Talent Acquisition",
    "VP People", "Head of People", "Head of Recruiting",
]

# Role-relevant keyword whitelist applied to Blitz response titles AFTER
# the API returns candidates. Prevents Blitz's loose matching from leaking
# wrong-org seniors through (e.g., on 2026-04-29 Blitz returned a "VP of
# Strategy & Ops" at Glean as a hit for "VP Sales / VP Revenue / VP GTM /
# VP Engineering" search because the bucketing keyword "vp" matched any
# "VP of [anything]"). A candidate's actual title MUST contain at least
# one of these keywords; otherwise drop them as a wrong-target match.
_BLITZ_TITLE_RELEVANT_KEYWORDS = (
    # Founder / C-suite (always hiring decision-makers at small cos)
    "founder", "co-founder", "cofounder", "ceo", "cro", "cmo", "coo", "cpo",
    # GTM / Revenue / Sales
    "sales", "revenue", "gtm", "go-to-market", "go to market",
    "growth", "outbound", "inbound", "marketing", "demand",
    # Engineering / FDE / Solutions
    "engineering", "engineer", "platform", "infrastructure",
    "solutions", "solution architect", "customer engineering",
    "forward deployed",
    # Customer success / Solutions Eng
    "customer success", "solutions engineer", "solutions architect",
    # People / Talent / Recruiting
    "talent", "people", "recruit", "hiring", "human resources",
    "human capital",
)

# Negative-keyword filter applied BEFORE the positive whitelist. The
# positive list accepts substrings like "sales" / "marketing" / "revenue"
# which leak in non-hiring-manager ops/comp/strategy roles such as
# "Sr Manager, Sales Compensation" or "Director of GTM Strategy". These
# people own quotas, plans, dashboards, or comp models, NOT headcount
# for SE / GTM Engineer / FDE roles. Match is case-insensitive substring
# on the full disqualifier phrase, so "Sales Operations Lead" trips
# "sales operations" but a true "VP Sales" stays clean.
_BLITZ_TITLE_DISQUALIFIERS = (
    # Sales-adjacent ops / strategy / comp (NOT hiring managers)
    "sales compensation", "sales operations", "sales strategy",
    "sales enablement", "sales planning", "sales analytics",
    "sales finance", "sales ops",
    # GTM-adjacent ops / strategy (NOT hiring managers)
    "gtm strategy", "gtm planning", "gtm operations", "gtm ops",
    # Revenue-adjacent ops / strategy (NOT hiring managers)
    "revenue operations", "revenue strategy", "revops", "rev ops",
    # Marketing-adjacent ops / planning / analytics (NOT hiring managers)
    "marketing operations", "marketing planning", "marketing analytics",
    "marketing ops",
    # HR / Recruiting / People-Ops (NOT GTM hiring managers).
    # Surfaced 2026-04-29 by the Tara/Summer/Brooke wrong-target sample:
    # blitz+mv_ok shipped a Talent Lead at Hex and a Head of People at
    # Drata as "hiring managers" for SE / FDE candidate searches because
    # Blitz's HIRING_MANAGER_TITLES fallback bucket includes Head of
    # Talent / Head of People as last-resort search targets. Those
    # contacts are recruiters/HR generalists, NOT engineering hiring
    # managers, so the post-filter must veto them.
    #
    # "talent" matches as a bare substring — there is no positive use of
    # "talent" in tech titles that isn't HR (Talent Lead / Talent
    # Acquisition Partner / Talent Engineering = all HR functions).
    "talent",
    # "recruiter" / "recruiting" — unambiguously HR.
    "recruiter", "recruiting",
    # "people" — phrase-level only. "Senior People Manager, Engineering"
    # is a real engineering title (eng leaders DO manage people), so we
    # cannot use bare "people" here. Match the HR-specific phrases
    # instead.
    "head of people", "vp people", "vp of people", "vp, people",
    "chief people", "director of people", "director, people",
    "people operations", "people ops", "people partner",
    # HR-proper. "human resources" is unambiguous; "hr " / " hr,"
    # protect against false positives on words that happen to contain
    # "hr" (Christopher / Lehrer).
    "human resources", "human capital",
    "head of hr", "vp hr", "vp, hr", "vp of hr", "chief hr",
    "director of hr", "director, hr", "hr business partner",
    "hr operations", "hr ops", "hr director", "hr manager",
)


def _has_relevant_title(title: str) -> bool:
    """True if a candidate's title contains a role-relevant keyword AND
    no disqualifier substring.

    Filters out candidates Blitz returned via loose 'VP/Head/Director'
    bucketing whose actual specialty is unrelated to hiring for our
    target roles. Two passes:

    1. Disqualifier check (negative): rejects ops/comp/strategy titles
       that would otherwise pass the positive whitelist on substrings
       like "sales" or "marketing" (e.g., "Sr Manager, Sales
       Compensation" at Twilio is comp ops, NOT the SE hiring manager).
    2. Positive keyword check: must contain at least one role-relevant
       keyword (e.g., "VP of Strategy & Ops" still fails this).
    """
    low = (title or "").lower()
    if not low:
        return False
    # Negative filter first: disqualifiers veto even when a positive
    # keyword matches.
    if any(bad in low for bad in _BLITZ_TITLE_DISQUALIFIERS):
        return False
    return any(kw in low for kw in _BLITZ_TITLE_RELEVANT_KEYWORDS)


# Regional variant patterns. Tuples of (compiled_regex, canonical_label).
# Match order matters: more specific phrases come before their bare
# counterparts so "east coast" doesn't get short-circuited by "east"
# matching a substring of "easter" / "easterly". Word-boundary regex
# protects single-token markers (EST, EU, US) from false hits in titles
# like "Test Engineer" or "Customer Success".
_REGIONAL_VARIANT_PATTERNS = [
    # Compound phrases (highest specificity first)
    (re.compile(r"\beast\s*coast\b", re.I), "east_coast"),
    (re.compile(r"\bwest\s*coast\b", re.I), "west_coast"),
    (re.compile(r"\bnorth\s*america\b", re.I), "north_america"),
    (re.compile(r"\basia\s*pacific\b", re.I), "apac"),
    # Single-token regional codes
    (re.compile(r"\bemea\b", re.I), "emea"),
    (re.compile(r"\bapac\b", re.I), "apac"),
    (re.compile(r"\bamericas\b", re.I), "americas"),
    (re.compile(r"\beurope\b", re.I), "emea"),
    (re.compile(r"\beuropean\b", re.I), "emea"),
    (re.compile(r"\beu\b", re.I), "emea"),
    (re.compile(r"\buk\b", re.I), "uk"),
    (re.compile(r"\bunited\s+kingdom\b", re.I), "uk"),
    (re.compile(r"\bgermany\b", re.I), "germany"),
    (re.compile(r"\bdeutschland\b", re.I), "germany"),
    (re.compile(r"\bfrance\b", re.I), "france"),
    (re.compile(r"\bfrench\b", re.I), "france"),
    # Time-zone / coast aliases (must follow compound matches above)
    (re.compile(r"\beastern\b", re.I), "east_coast"),
    (re.compile(r"\bpacific\b", re.I), "west_coast"),
    (re.compile(r"\beast\b", re.I), "east_coast"),
    (re.compile(r"\bwest\b", re.I), "west_coast"),
    (re.compile(r"\best\b", re.I), "east_coast"),
    (re.compile(r"\bpst\b", re.I), "west_coast"),
    # Country codes - keep US last so it doesn't shadow "United Kingdom"
    (re.compile(r"\bunited\s+states\b", re.I), "us"),
    (re.compile(r"\busa\b", re.I), "us"),
    (re.compile(r"\bus\b", re.I), "us"),
]


def _detect_regional_variant(title: str) -> str | None:
    """Return canonical region tag if the title contains a regional marker.

    Returns one of: 'east_coast', 'west_coast', 'emea', 'apac',
    'north_america', 'europe', 'americas', 'us', 'uk', 'germany',
    'france', or None.

    Used by the cascade to detect multi-posting roles (e.g., n8n posts
    "Senior Solutions Engineer | East Coast" and "...| West Coast" with
    different regional managers per posting). When present, the cascade
    blocks domain-search fallbacks (Hunter / Icypeas domain) because
    those fall back to the same domain-keyed contact regardless of the
    posting's region, and that contact may be the wrong-region manager.
    """
    if not title:
        return None
    for pattern, label in _REGIONAL_VARIANT_PATTERNS:
        if pattern.search(title):
            return label
    return None


# In-process cache so multiple jobs at the same domain only cost 1 Hunter call.
_DOMAIN_CACHE: dict[str, tuple[str, str, str, str]] = {}

# Per-domain caches for the new Blitz / Icypeas cascade steps. Blitz cache
# stores the post-employee-finder candidate list (so a second job at the same
# domain reuses both the domain->LinkedIn lookup AND the employee-finder call).
_BLITZ_DOMAIN_CACHE: dict[str, list[dict]] = {}
_ICYPEAS_DOMAIN_CACHE: dict[str, list[dict]] = {}


def _apex_domain(d: str) -> str:
    """Return the last 2 dot-separated components of a hostname.

    Naive but adequate for US-tech orgs (amazon.com, langchain.com,
    aws.amazon.com -> amazon.com). Doesn't handle .co.uk / .com.au;
    swap to tldextract if we ever need to. Empty input -> empty string.
    """
    if not d:
        return ""
    parts = d.lower().strip().rstrip(".").split(".")
    if len(parts) >= 2:
        return ".".join(parts[-2:])
    return d.lower()


def _domains_match(target: str, email: str) -> bool:
    """True when `email`'s apex domain == `target`'s apex domain.

    Used by the Blitz cascade to reject "Blitz returned a candidate's email
    at a different company" cases (the AWS->jens@indeed.com bug).
    """
    if not target or not email or "@" not in email:
        return False
    email_domain = email.rsplit("@", 1)[1]
    return _apex_domain(target) == _apex_domain(email_domain)


def domain_from_company_url(url: str) -> str:
    if not isinstance(url, str) or not url:
        return ""
    try:
        host = urlparse(url).netloc.lower()
        # removeprefix avoids the lstrip("www.") footgun where lstrip strips
        # any chars in the set {w, .} (e.g., would mangle "wayfair.com").
        if host.startswith("www."):
            host = host.removeprefix("www.")
        return host
    except Exception:
        return ""


def find_via_jd(description: str) -> tuple[str, str]:
    """Returns (email, source) — source = 'jd_scrape' if found, else ('','')."""
    emails = extract_emails_from_text(description)
    return (emails[0], "jd_scrape") if emails else ("", "")


def find_via_gooseworks(company: str, role_keyword: str) -> tuple[str, str, str]:
    """Returns (manager_name, manager_linkedin, source) or ('','','').

    Uses `gooseworks search` CLI for "[role] at [company]" queries. Logs
    failure shapes (rc != 0, non-JSON output) to stderr so we can tune
    the parser based on real responses.
    """
    if not shutil.which("gooseworks") and not shutil.which("npx"):
        return ("", "", "")
    query = f"hiring manager {role_keyword} at {company}"
    cmd = ["npx", "gooseworks", "search", query, "--limit", "1"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if proc.returncode != 0:
            print(
                f"    gooseworks rc={proc.returncode} for '{company}': "
                f"{(proc.stderr or '')[:200]}",
                file=sys.stderr,
            )
            return ("", "", "")
        if not proc.stdout:
            return ("", "", "")
        try:
            data = json.loads(proc.stdout)
        except json.JSONDecodeError:
            print(
                f"    gooseworks non-JSON for '{company}' (q={query!r}): "
                f"{proc.stdout[:200]}",
                file=sys.stderr,
            )
            return ("", "", "")
        if isinstance(data, list) and data:
            top = data[0]
            return (top.get("name", ""), top.get("linkedin", ""), "gooseworks")
        return ("", "", "")
    except Exception as e:
        print(
            f"    gooseworks subprocess error for '{company}': {str(e)[:200]}",
            file=sys.stderr,
        )
        return ("", "", "")


def find_via_hunter_domain(domain: str) -> tuple[str, str, str, str]:
    """Look up a hiring-relevant contact via Hunter's domain-search endpoint.

    Returns (email, name, source, position). One call per domain, then cached
    per process. Filters returned emails to those whose position contains a
    hiring-related keyword (talent, recruit, people, hr, hiring, etc.).

    Skips silently if HUNTER_API_KEY missing. Never crashes.
    """
    if not domain:
        return ("", "", "", "")
    if domain in _DOMAIN_CACHE:
        return _DOMAIN_CACHE[domain]

    api_key = os.environ.get("HUNTER_API_KEY")
    if not api_key:
        result = ("", "", "", "")
        _DOMAIN_CACHE[domain] = result
        return result

    try:
        # NOTE: Hunter Free tier returns 0 emails when limit > ~10. Use limit=10
        # to maximize signal without tripping the silent throttle.
        resp = requests.get(
            HUNTER_DOMAIN_URL,
            params={"domain": domain, "api_key": api_key, "limit": 10},
            timeout=15,
        )
    except Exception:
        result = ("", "", "", "")
        _DOMAIN_CACHE[domain] = result
        return result
    if resp.status_code != 200:
        _DOMAIN_CACHE[domain] = ("", "", "", "")
        return _DOMAIN_CACHE[domain]
    try:
        data = resp.json().get("data", {}) or {}
    except (ValueError, AttributeError):
        _DOMAIN_CACHE[domain] = ("", "", "", "")
        return _DOMAIN_CACHE[domain]

    emails = data.get("emails", []) or []
    # Find first email whose position matches a hiring-related keyword.
    # If no match, return empty — DO NOT fall back to "any high-confidence
    # personal email" because that path silently surfaces wrong-org contacts
    # (e.g., Webflow returned a Senior Director of Product Marketing for a
    # Forward Deployed Engineer role search on 2026-04-29). Better to report
    # "no manager found" and let the cascade move to Icypeas / permutator
    # than to ship a wrong-audience contact downstream.
    for e in emails:
        position = (e.get("position") or "").lower()
        if not position:
            continue
        if any(kw in position for kw in HIRING_POSITION_KEYWORDS):
            email = e.get("value", "") or ""
            first = e.get("first_name", "") or ""
            last = e.get("last_name", "") or ""
            name = f"{first} {last}".strip()
            confidence = e.get("confidence") or 0
            if email and confidence >= HUNTER_MIN_CONFIDENCE:
                result = (email, name, f"hunter_domain+conf{confidence}", e.get("position", ""))
                _DOMAIN_CACHE[domain] = result
                return result

    # No hiring-keyword match — return empty. The dangerous fallback to
    # "first high-confidence personal email" was removed 2026-04-29 after
    # surfacing a PMM Senior Director as a "hiring manager" for the Webflow
    # FDE role.
    _DOMAIN_CACHE[domain] = ("", "", "", "")
    return _DOMAIN_CACHE[domain]


def find_via_hunter(first: str, last: str, domain: str) -> tuple[str, str]:
    """Look up an email via Hunter's email-finder endpoint.

    Returns (email, source) on confidence >= HUNTER_MIN_CONFIDENCE, else ('','').
    Skips silently if HUNTER_API_KEY is not in env. Never crashes — every
    failure path returns empty.
    """
    api_key = os.environ.get("HUNTER_API_KEY")
    if not api_key:
        return ("", "")
    if not first or not last or not domain:
        return ("", "")
    try:
        resp = requests.get(
            HUNTER_URL,
            params={
                "domain": domain,
                "first_name": first,
                "last_name": last,
                "api_key": api_key,
            },
            timeout=10,
        )
    except Exception:
        return ("", "")
    if resp.status_code != 200:
        return ("", "")
    try:
        data = resp.json().get("data", {}) or {}
    except (ValueError, AttributeError):
        return ("", "")
    email = data.get("email") or ""
    confidence = data.get("score") or 0
    if email and confidence >= HUNTER_MIN_CONFIDENCE:
        return (email, f"hunter+conf{confidence}")
    return ("", "")


def find_via_blitz(
    domain: str, counters: dict | None = None
) -> tuple[str, str, str, str]:
    """Blitz cascade: domain -> LinkedIn -> employees -> verified email.

    Returns (email, name, linkedin_url, source) where source is one of
    'blitz+mv_ok' / 'blitz+mv_uncertain'. Returns ('','','','') on miss.

    Uses a per-domain cache so a second job at the same domain reuses the
    candidate list (domain->LinkedIn + employee-finder are both cached).
    Each call still re-runs MV verification on candidates in icp_tier order
    — that's the cheap step.

    Optional `counters` dict: when provided, increments 'blitz_mv_rejected'
    for each candidate we tossed for MV=False, and 'blitz_no_candidates' if
    Blitz didn't surface any employees for the domain. Lets main() track the
    inner cascade state without changing the public return shape.

    Skips silently when BLITZ_API_KEY is missing (the underlying tools.blitz
    helpers all return empty in that case, so the cascade short-circuits).
    """
    if not domain:
        return ("", "", "", "")

    candidates = _BLITZ_DOMAIN_CACHE.get(domain)
    if candidates is None:
        company_linkedin = domain_to_linkedin(domain)
        if not company_linkedin:
            _BLITZ_DOMAIN_CACHE[domain] = []
            if counters is not None:
                counters["blitz_no_candidates"] = (
                    counters.get("blitz_no_candidates", 0) + 1
                )
            return ("", "", "", "")
        candidates = find_employees_by_title(
            company_linkedin, HIRING_MANAGER_TITLES, max_results=3
        )
        _BLITZ_DOMAIN_CACHE[domain] = candidates

    if not candidates:
        if counters is not None:
            counters["blitz_no_candidates"] = (
                counters.get("blitz_no_candidates", 0) + 1
            )
        return ("", "", "", "")

    # Title-relevance filter (added 2026-04-29). Blitz's loose keyword bucketing
    # surfaces wrong-org seniors (e.g., "VP of Strategy & Ops" returned as a
    # match for our "VP Sales / VP Revenue / VP GTM / VP Engineering" search
    # because the substring "vp" matched). Drop any candidate whose actual
    # title doesn't contain a role-relevant keyword.
    relevant = [c for c in candidates if _has_relevant_title(c.get("title", ""))]
    if not relevant:
        if counters is not None:
            counters["blitz_irrelevant_titles"] = (
                counters.get("blitz_irrelevant_titles", 0) + 1
            )
        return ("", "", "", "")
    candidates = relevant

    # Walk candidates in icp_tier ascending order (tier 1 = tightest match).
    ordered = sorted(candidates, key=lambda c: c.get("icp_tier", 99))
    for cand in ordered:
        person_url = cand.get("linkedin_url", "") or ""
        if not person_url:
            continue
        email = linkedin_to_email(person_url)
        if not email:
            continue
        # Domain-match guard: Blitz sometimes returns a candidate's secondary
        # email at a different employer (e.g., AWS lookup -> jens@indeed.com
        # because Jen Sun used to work at Indeed). Reject when the email's apex
        # doesn't match the target domain's apex. amazon.com / aws.amazon.com
        # share apex 'amazon.com' so subsidiaries still pass.
        if not _domains_match(domain, email):
            if counters is not None:
                counters["blitz_domain_mismatch"] = (
                    counters.get("blitz_domain_mismatch", 0) + 1
                )
            continue
        name = f"{cand.get('first_name', '')} {cand.get('last_name', '')}".strip()
        mv_verdict = verify_email_mv(email)
        if mv_verdict is False:
            if counters is not None:
                counters["blitz_mv_rejected"] = (
                    counters.get("blitz_mv_rejected", 0) + 1
                )
            continue
        suffix = "+mv_ok" if mv_verdict is True else "+mv_uncertain"
        return (email, name, person_url, f"blitz{suffix}")

    return ("", "", "", "")


def find_via_icypeas_domain(domain: str) -> tuple[str, str, str, str]:
    """Icypeas domain-search backup. Same return shape as find_via_hunter_domain.

    Returns (email, name, source, position) — source like 'icypeas_domain+mv_ok'
    or 'icypeas_domain+mv_uncertain'. Returns ('','','','') on miss.

    Filters returned contacts by HIRING_POSITION_KEYWORDS (same filter as
    Hunter domain-search) and gates on MV verify.
    """
    if not domain:
        return ("", "", "", "")

    contacts = _ICYPEAS_DOMAIN_CACHE.get(domain)
    if contacts is None:
        contacts = find_domain_emails_icypeas(domain)
        _ICYPEAS_DOMAIN_CACHE[domain] = contacts

    if not contacts:
        return ("", "", "", "")

    for contact in contacts:
        position = (contact.get("position") or "").lower()
        if not position:
            continue
        if not any(kw in position for kw in HIRING_POSITION_KEYWORDS):
            continue
        email = contact.get("email", "") or ""
        name = contact.get("name", "") or ""
        if not email:
            continue
        mv_verdict = verify_email_mv(email)
        if mv_verdict is False:
            continue
        suffix = "+mv_ok" if mv_verdict is True else "+mv_uncertain"
        return (email, name, f"icypeas_domain{suffix}", contact.get("position", ""))

    return ("", "", "", "")


def find_via_icypeas_email(first: str, last: str, domain: str) -> tuple[str, str]:
    """Icypeas /email-search backup for a known name + domain.

    Returns (email, source) where source is 'icypeas+mv_ok' /
    'icypeas+mv_uncertain'. Returns ('','') on miss or MV reject.
    """
    if not first or not last or not domain:
        return ("", "")
    email, _icy_src = find_email_icypeas(first, last, domain)
    if not email:
        return ("", "")
    mv_verdict = verify_email_mv(email)
    if mv_verdict is False:
        return ("", "")
    suffix = "+mv_ok" if mv_verdict is True else "+mv_uncertain"
    return (email, f"icypeas{suffix}")


def format_summary(counters: dict, total: int) -> str:
    """Render a per-run summary table for end-of-run reporting."""
    lines = [f"Enriched {total} candidates:"]
    rows = [
        ("JD scrape", counters.get("jd_scrape", 0)),
        ("Blitz (MV ok)", counters.get("blitz_mv_ok", 0)),
        ("Blitz (MV uncertain)", counters.get("blitz_mv_uncertain", 0)),
        ("Blitz (MV rejected)", counters.get("blitz_mv_rejected", 0)),
        ("Blitz (domain mismatch)", counters.get("blitz_domain_mismatch", 0)),
        ("Blitz (irrelevant titles)", counters.get("blitz_irrelevant_titles", 0)),
        ("Blitz (no candidates)", counters.get("blitz_no_candidates", 0)),
        ("Icypeas domain-search", counters.get("icypeas_domain", 0)),
        ("Icypeas email-search", counters.get("icypeas_email", 0)),
        ("Hunter domain-search (MV ok)", counters.get("hunter_domain_mv_ok", 0)),
        ("Hunter domain-search (MV uncertain)", counters.get("hunter_domain_mv_uncertain", 0)),
        ("Hunter domain-search (MV rejected)", counters.get("hunter_domain_mv_rejected", 0)),
        ("Hunter email-finder", counters.get("hunter_finder", 0)),
        ("Permutator (MV verified)", counters.get("permutator_verified", 0)),
        ("Permutator (catch-all)", counters.get("permutator_catch_all", 0)),
        ("Permutator (unverified)", counters.get("permutator_unverified", 0)),
        ("Multi-posting blocked (region-variant)", counters.get("multi_posting_blocked", 0)),
        ("No manager found", counters.get("no_manager_found", 0)),
    ]
    for label, n in rows:
        prefix = "x" if "No manager" in label or "rejected" in label.lower() else "*"
        lines.append(f"  {prefix} {label}: {n}")
    return "\n".join(lines)


def find_via_permutator(name: str, domain: str) -> tuple[str, str]:
    """Permutate name+domain combos and verify each with MillionVerifier.

    SMTP RCPT verification (smtp_verify) is fragile — many MX servers reject
    or greylist verification probes. MV is the reliable option.

    Returns first MV-verified email; if domain is catch-all (every probe
    looks valid), returns the most-common pattern with a 'permutator_catch_all'
    tag. If nothing verifies and the domain isn't catch-all, returns
    'permutator_unverified' on the most-likely pattern.
    """
    if not name or not domain:
        return ("", "")
    parts = name.strip().split(maxsplit=1)
    if len(parts) < 2:
        return ("", "")
    first, last = parts[0], parts[1]
    candidates = permutate_emails(first, last, domain)
    if not candidates:
        return ("", "")

    # If domain is catch-all, MV will say every pattern is "valid" — useless.
    # Just return the most-common pattern with a "catch_all" tag.
    if is_catch_all_domain(domain):
        return (candidates[0], "permutator_catch_all")

    for email in candidates:
        if verify_email_mv(email) is True:
            return (email, "permutator+mv_verified")
    # No MV-verified candidate — fall back to most-likely pattern
    return (candidates[0], "permutator_unverified")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    parser.add_argument("--top", type=int, default=20,
                        help="How many top-scored jobs to enrich")
    parser.add_argument("--no-upload", action="store_true")
    args = parser.parse_args()

    profile = load_profile(PROJECT_ROOT / "profiles" / f"{args.profile}.yaml")
    master_file = OUTPUT_DIR / profile.drive.master_filename

    df = pd.read_excel(master_file)
    if "fit_score" not in df.columns:
        print("No fit_score column. Run tools/score.py first.")
        return

    # New columns if missing
    for col in ("manager_name", "manager_email", "manager_linkedin", "manager_source"):
        if col not in df.columns:
            df[col] = pd.NA

    # Pick top-N jobs by fit_score that don't yet have a manager_email
    candidates = df[
        (df["fit_score"].notna())
        & (df["manager_email"].isna())
    ].sort_values("fit_score", ascending=False).head(args.top)

    company_domains = load_company_domains()
    print(f"Enriching {len(candidates)} candidate jobs (top by fit_score). "
          f"{len(company_domains)} companies in domain DB.")
    counters = {
        "jd_scrape": 0,
        "blitz_mv_ok": 0,
        "blitz_mv_uncertain": 0,
        "blitz_mv_rejected": 0,
        "blitz_domain_mismatch": 0,
        "blitz_irrelevant_titles": 0,
        "blitz_no_candidates": 0,
        "icypeas_domain": 0,
        "icypeas_email": 0,
        "hunter_domain_mv_ok": 0,
        "hunter_domain_mv_uncertain": 0,
        "hunter_domain_mv_rejected": 0,
        "hunter_finder": 0,
        "permutator_verified": 0,
        "permutator_catch_all": 0,
        "permutator_unverified": 0,
        "multi_posting_blocked": 0,
        "no_manager_found": 0,
    }

    for idx, row in candidates.iterrows():
        company = str(row.get("company") or "")
        description = str(row.get("description") or "")
        company_url = str(row.get("company_url") or "")
        title = str(row.get("title") or "")
        # Multi-posting detection: when a title carries a regional marker
        # (e.g., "Senior Solutions Engineer | East Coast - Remote"), the
        # company likely posts a sister role for a different region with
        # a different hiring manager. Domain-search fallbacks (Hunter /
        # Icypeas) cache per-domain and would surface the SAME contact for
        # both postings, which silently routes the wrong-region manager.
        # When this flag is set the cascade only accepts Blitz (role-aware)
        # results and rejects domain-search fallbacks.
        region_variant = _detect_regional_variant(title)
        print(f"  [{idx}] {company}")

        # Step 1: JD scrape (free)
        email, source = find_via_jd(description)
        if email:
            df.at[idx, "manager_email"] = email
            df.at[idx, "manager_source"] = source
            print(f"    -> JD scrape: {email}")
            counters["jd_scrape"] += 1
            continue

        # Domain lookup priority:
        #  1. companies-db.json (curated, real company domain — handles
        #     Workday/Ashby ATS hosts that aren't real domains)
        #  2. company_url field if present
        #  3. job_url's host as last resort
        domain = company_domains.get(company.lower(), "")
        if not domain:
            domain = domain_from_company_url(company_url)
        if not domain:
            domain = domain_from_company_url(str(row.get("job_url") or ""))

        # Step 2: Blitz cascade (NEW PRIMARY).
        # domain -> LinkedIn -> employee-finder -> linkedin_to_email,
        # gated on MV verify. Per-domain cached. find_via_blitz updates
        # blitz_no_candidates + blitz_mv_rejected internally.
        bz_email, bz_name, bz_linkedin, bz_source = find_via_blitz(
            domain, counters=counters
        )
        if bz_email:
            df.at[idx, "manager_email"] = bz_email
            df.at[idx, "manager_name"] = bz_name
            df.at[idx, "manager_linkedin"] = bz_linkedin
            df.at[idx, "manager_source"] = bz_source
            label = f"{bz_name} | {bz_email} ({bz_source})"
            print(f"    -> {label}")
            if bz_source.endswith("+mv_ok"):
                counters["blitz_mv_ok"] += 1
            else:
                counters["blitz_mv_uncertain"] += 1
            continue

        # Multi-posting guard: when the title carries a regional marker, skip
        # the domain-search fallbacks entirely. Hunter and Icypeas both cache
        # per-domain and would hand back the SAME contact for "East Coast"
        # and "West Coast" postings of the same role, silently routing the
        # wrong-region manager. Force the row to surface as
        # multi_posting_blocked so it gets a manual lookup. Bug surfaced
        # 2026-04-29: n8n's West Coast SE posting matched to Elena (TA via
        # hunter_domain) when the user (in EST) should have been routed to
        # Hernan via the East Coast posting.
        if region_variant:
            df.at[idx, "manager_source"] = f"multi_posting_blocked:{region_variant}"
            print(
                f"    -> multi-posting role ({region_variant}); blocking "
                f"domain fallbacks (manual lookup required)"
            )
            counters["multi_posting_blocked"] += 1
            continue

        # Step 3: Hunter domain-search (1 paid call per unique domain, cached).
        # Demoted to backup behind Blitz. Returns a hiring-related contact
        # (talent/recruit/people/hr/etc.) if one exists at the domain, else
        # first high-confidence personal email. MV-gate the result.
        hd_email, hd_name, hd_source, hd_position = find_via_hunter_domain(domain)
        if hd_email:
            mv_verdict = verify_email_mv(hd_email)
            if mv_verdict is False:
                print(f"    -> MV rejected Hunter result: {hd_email}")
                counters["hunter_domain_mv_rejected"] += 1
                # fall through — try Icypeas domain-search next
            else:
                # mv_verdict True or None (uncertain). Both acceptable; tag accordingly.
                suffix = "+mv_ok" if mv_verdict is True else "+mv_uncertain"
                df.at[idx, "manager_email"] = hd_email
                df.at[idx, "manager_name"] = hd_name
                df.at[idx, "manager_source"] = hd_source + suffix
                label = f"{hd_name} ({hd_position})" if hd_name else hd_email
                print(f"    -> {label} | {hd_email} ({suffix.lstrip('+')})")
                if mv_verdict is True:
                    counters["hunter_domain_mv_ok"] += 1
                else:
                    counters["hunter_domain_mv_uncertain"] += 1
                continue

        # Step 4: Icypeas domain-search (NEW backup) — only if Hunter returned nothing.
        # Same shape as find_via_hunter_domain, with hiring-keyword filter applied.
        ip_email, ip_name, ip_source, ip_position = find_via_icypeas_domain(domain)
        if ip_email:
            df.at[idx, "manager_email"] = ip_email
            df.at[idx, "manager_name"] = ip_name
            df.at[idx, "manager_source"] = ip_source
            label = f"{ip_name} ({ip_position})" if ip_name else ip_email
            print(f"    -> {label} | {ip_email} ({ip_source})")
            counters["icypeas_domain"] += 1
            continue

        # Steps 5+: if we somehow obtained a name (Blitz partial result, JD text,
        # or pre-existing manager_name column), try Hunter email-finder, then
        # Icypeas email-search, then permutator+MV.
        name = str(row.get("manager_name") or "").strip()
        if name and domain:
            parts = name.split(maxsplit=1)
            if len(parts) >= 2:
                first, last = parts[0], parts[1]
                hunter_email, hunter_src = find_via_hunter(first, last, domain)
                if hunter_email:
                    df.at[idx, "manager_email"] = hunter_email
                    df.at[idx, "manager_source"] = hunter_src
                    print(f"    -> {name} | {hunter_email} ({hunter_src})")
                    counters["hunter_finder"] += 1
                    continue

                # Icypeas /email-search backup (MV-gated).
                icy_email, icy_src = find_via_icypeas_email(first, last, domain)
                if icy_email:
                    df.at[idx, "manager_email"] = icy_email
                    df.at[idx, "manager_source"] = icy_src
                    print(f"    -> {name} | {icy_email} ({icy_src})")
                    counters["icypeas_email"] += 1
                    continue

            email, perm_source = find_via_permutator(name, domain)
            if email:
                df.at[idx, "manager_email"] = email
                df.at[idx, "manager_source"] = perm_source
                print(f"    -> {name} | {email} ({perm_source})")
                if perm_source == "permutator+mv_verified":
                    counters["permutator_verified"] += 1
                elif perm_source == "permutator_catch_all":
                    counters["permutator_catch_all"] += 1
                else:
                    counters["permutator_unverified"] += 1
                continue

        print(f"    -> no manager found")
        counters["no_manager_found"] += 1

    df = reorder_columns(df)
    df.to_excel(master_file, index=False)
    print(f"Updated {master_file}")
    print()
    print(format_summary(counters, total=len(candidates)))

    mv_left = mv_credits_remaining()
    if mv_left is not None:
        print(f"\nMV credits remaining: {mv_left:,}")

    if not args.no_upload:
        upload_to_drive(master_file, profile_name=args.profile,
                        state_file=DRIVE_STATE_FILE)
        # Mirror the umbrella docs (STATUS, BRIEF, resume, d100, etc.) too.
        # mtime guard means unchanged docs skip — fast on repeat runs.
        docs_dir = PROJECT_ROOT.parent  # the-vector/ — one level above job-search-tool/
        doc_counts = upload_vector_docs(
            profile_name=args.profile,
            state_file=DRIVE_STATE_FILE,
            docs_dir=docs_dir,
        )
        if any(doc_counts.values()):
            print(f"Docs sync: {doc_counts['uploaded']} uploaded, "
                  f"{doc_counts['skipped']} skipped, "
                  f"{doc_counts['failed']} failed")


if __name__ == "__main__":
    main()
