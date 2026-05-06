"""Direct ATS careers-page scrapers.

Four pure-function scrapers (Greenhouse, Lever, Ashby, Workday) that hit each
ATS's public job-board JSON API and return a list of normalized job dicts.
The output schema is a subset of the JobSpy column schema so results can be
merged into the master Excel via tools.master.merge_with_master without
any column mapping.

Workday note: Workday's `/wday/cxs/{tenant}/{site}/jobs` endpoint takes a
POST with an empty `appliedFacets` filter. The slug field in companies-db
is encoded as `"tenant/wdN/site"` (e.g., `nvidia/wd5/NVIDIAExternalCareerSite`)
and parsed at runtime. Different tenants live on different cluster numbers
(wd1, wd3, wd5, ...). Many tenants reject anonymous POSTs with 401/422 — we
handle those gracefully by returning [] and continuing.
"""
from __future__ import annotations

import html
import re
from typing import Optional

import requests

# Public ATS job-board endpoints
GREENHOUSE_URL = "https://boards-api.greenhouse.io/v1/boards/{slug}/jobs?content=true"
LEVER_URL = "https://api.lever.co/v0/postings/{slug}?mode=json"
ASHBY_URL = "https://api.ashbyhq.com/posting-api/job-board/{slug}"
WORKDAY_URL = "https://{tenant}.{wd}.myworkdayjobs.com/wday/cxs/{tenant}/{site}/jobs"
WORKDAY_PAGE_LIMIT = 20  # max pages * 20 = 400 jobs/company (sane V1 cap)

# A polite UA — some endpoints rate-limit blank UAs
USER_AGENT = "the-vector-tool/2.0 (+https://github.com/8OzOfMilk)"
DEFAULT_TIMEOUT = 15

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"[ \t]+")
_NL_RE = re.compile(r"\n{3,}")


def strip_html(s: Optional[str]) -> str:
    """Strip HTML tags and decode entities into plain text.

    Greenhouse double-encodes (`&lt;p&gt;...`); we unescape twice to be safe.
    Lever and Ashby are already entity-clean — extra unescape is a no-op.
    """
    if not s:
        return ""
    # Decode entities; second pass handles double-encoded Greenhouse content
    decoded = html.unescape(html.unescape(s))
    no_tags = _TAG_RE.sub(" ", decoded)
    cleaned = _WS_RE.sub(" ", no_tags)
    cleaned = _NL_RE.sub("\n\n", cleaned)
    return cleaned.strip()


def _is_remote_text(text: str) -> bool:
    if not text:
        return False
    t = text.lower()
    return any(kw in t for kw in ("remote", "anywhere", "worldwide", "distributed"))


def _empty_optional() -> dict:
    """Optional fields kept consistent across scrapers (None where unknown)."""
    return {
        "min_amount": None,
        "max_amount": None,
        "interval": None,
        "currency": None,
        "job_type": None,
        "job_level": None,
        "job_function": None,
        "company_industry": None,
        "company_url": None,
        "job_url_direct": None,
    }


def _fetch_json(url: str) -> Optional[dict | list]:
    """GET json or return None on any error (caller decides what to do)."""
    try:
        r = requests.get(
            url,
            headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
            timeout=DEFAULT_TIMEOUT,
        )
    except requests.RequestException:
        return None
    if r.status_code != 200:
        return None
    try:
        return r.json()
    except ValueError:
        return None


# ---------- Greenhouse ----------

def scrape_greenhouse(slug: str, company_name: str) -> list[dict]:
    """GET https://boards-api.greenhouse.io/v1/boards/{slug}/jobs?content=true.

    Returns a list of normalized job dicts. Returns [] on 404 or transport error.
    """
    data = _fetch_json(GREENHOUSE_URL.format(slug=slug))
    if not data or not isinstance(data, dict):
        return []
    jobs = []
    for j in data.get("jobs", []) or []:
        loc_name = ((j.get("location") or {}).get("name") or "")
        url = j.get("absolute_url") or ""
        normalized = {
            "company": company_name,
            "title": j.get("title") or "",
            "location": loc_name,
            "is_remote": _is_remote_text(loc_name),
            "job_url": url,
            "description": strip_html(j.get("content")),
            "date_posted": j.get("updated_at") or j.get("first_published") or "",
            "site": "direct_greenhouse",
            "search_term": "direct_scrape",
            "search_location": "direct_company",
            "id": str(j.get("id") or ""),
        }
        normalized.update(_empty_optional())
        jobs.append(normalized)
    return jobs


# ---------- Lever ----------

def scrape_lever(slug: str, company_name: str) -> list[dict]:
    """GET https://api.lever.co/v0/postings/{slug}?mode=json.

    Returns a list of normalized job dicts. Returns [] on 404 or transport error.
    """
    data = _fetch_json(LEVER_URL.format(slug=slug))
    if not data or not isinstance(data, list):
        return []
    jobs = []
    for j in data:
        cats = j.get("categories") or {}
        loc = cats.get("location") or ""
        # Lever provides plain-text bodies natively
        desc_plain = j.get("descriptionPlain") or strip_html(j.get("description"))
        addl_plain = j.get("additionalPlain") or strip_html(j.get("additional"))
        full_desc = "\n\n".join([p for p in (desc_plain, addl_plain) if p]).strip()
        # createdAt is a unix-ms timestamp
        created = j.get("createdAt")
        if isinstance(created, (int, float)):
            from datetime import datetime, timezone
            try:
                date_str = datetime.fromtimestamp(created / 1000, tz=timezone.utc) \
                    .date().isoformat()
            except (ValueError, OSError):
                date_str = ""
        else:
            date_str = ""
        url = j.get("hostedUrl") or j.get("applyUrl") or ""
        workplace = (j.get("workplaceType") or "").lower()
        is_remote = workplace == "remote" or _is_remote_text(loc)
        normalized = {
            "company": company_name,
            "title": j.get("text") or "",
            "location": loc,
            "is_remote": is_remote,
            "job_url": url,
            "description": full_desc,
            "date_posted": date_str,
            "site": "direct_lever",
            "search_term": "direct_scrape",
            "search_location": "direct_company",
            "id": str(j.get("id") or ""),
        }
        normalized.update(_empty_optional())
        jobs.append(normalized)
    return jobs


# ---------- Workday ----------

# Workday checks UA on some tenants — use a real browser-shape value
WORKDAY_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


def scrape_workday(slug: str, company_name: str) -> list[dict]:
    """POST to a Workday public job-board API and return normalized jobs.

    `slug` is encoded as ``"tenant/wdN/site"`` (e.g., ``"nvidia/wd5/NVIDIAExternalCareerSite"``).
    Returns [] on any error (404, 401, 422 wrong-slug, transport error). V1 leaves
    `description` empty — Workday requires a second per-job fetch for full JD,
    which we'll add as a follow-up. Untranslated jobs still appear in the master
    and get picked up later (Kimi will skip them per the no-JD path).
    """
    if not slug or "/" not in slug:
        return []
    parts = slug.split("/")
    if len(parts) != 3:
        return []
    tenant, wd, site = parts
    url = WORKDAY_URL.format(tenant=tenant, wd=wd, site=site)
    headers = {
        "Content-Type": "application/json",
        "User-Agent": WORKDAY_USER_AGENT,
        "Accept": "application/json",
    }
    out: list[dict] = []
    offset = 0
    limit = 20
    base_host = f"https://{tenant}.{wd}.myworkdayjobs.com"
    for _ in range(WORKDAY_PAGE_LIMIT):
        body = {
            "appliedFacets": {},
            "limit": limit,
            "offset": offset,
            "searchText": "",
        }
        try:
            resp = requests.post(url, json=body, headers=headers, timeout=DEFAULT_TIMEOUT)
        except requests.RequestException:
            break
        if resp.status_code != 200:
            break
        try:
            data = resp.json()
        except (ValueError, AttributeError):
            break
        postings = data.get("jobPostings", []) if isinstance(data, dict) else []
        if not postings:
            break
        for p in postings:
            ext_path = p.get("externalPath", "") or ""
            location_text = p.get("locationsText", "") or ""
            job_url = f"{base_host}/{site}{ext_path}" if ext_path else ""
            normalized = {
                "company": company_name,
                "title": p.get("title", "") or "",
                "location": location_text,
                "is_remote": _is_remote_text(location_text),
                "job_url": job_url,
                # V1: description requires a second fetch; leave empty.
                "description": "",
                "date_posted": p.get("postedOn", "") or "",
                "site": "direct_workday",
                "search_term": "direct_scrape",
                "search_location": "direct_company",
                "id": "/".join(p.get("bulletFields", []) or []),
            }
            normalized.update(_empty_optional())
            out.append(normalized)
        if len(postings) < limit:
            break
        offset += limit
    return out


# ---------- Ashby ----------

def scrape_ashby(slug: str, company_name: str) -> list[dict]:
    """GET https://api.ashbyhq.com/posting-api/job-board/{slug}.

    Returns a list of normalized job dicts. Returns [] on 404 or transport error.
    """
    data = _fetch_json(ASHBY_URL.format(slug=slug))
    if not data or not isinstance(data, dict):
        return []
    jobs = []
    for j in data.get("jobs", []) or []:
        loc = j.get("location") or ""
        # Ashby gives an explicit isRemote flag — use it, fall back to text
        is_remote = bool(j.get("isRemote")) if "isRemote" in j else _is_remote_text(loc)
        # Prefer plain-text description if Ashby provides it
        desc = j.get("descriptionPlain") or strip_html(j.get("descriptionHtml"))
        url = j.get("jobUrl") or j.get("applyUrl") or ""
        normalized = {
            "company": company_name,
            "title": j.get("title") or "",
            "location": loc,
            "is_remote": is_remote,
            "job_url": url,
            "description": desc,
            "date_posted": j.get("publishedAt") or "",
            "site": "direct_ashby",
            "search_term": "direct_scrape",
            "search_location": "direct_company",
            "id": str(j.get("id") or ""),
            "job_type": j.get("employmentType"),
        }
        # Optional defaults — but preserve job_type set above
        opt = _empty_optional()
        opt.pop("job_type", None)
        normalized.update(opt)
        jobs.append(normalized)
    return jobs
