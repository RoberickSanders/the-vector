"""Agentic auto-apply for the Vector job-search pipeline (proof-of-concept).

Wraps browser-use (https://github.com/browser-use/browser-use) with
Vector-specific guardrails:

  1. Loads an application profile (profiles/<name>.yaml -> ApplicationProfile)
  2. Detects the ATS (Greenhouse / Lever / Ashby / Workday) from the apply URL
  3. Hands a structured task prompt to a browser-use Agent driven by Claude
     (ChatAnthropic) so the model fills standard fields autonomously
  4. HALTS BEFORE SUBMIT — the agent is instructed to stop after every field
     is filled and explicitly NOT click any submit button. The human reviews
     in the live browser and types "yes submit" on stdin to proceed.
  5. After human-confirmed submit (or skip), captures a confirmation
     screenshot and logs the run to output/auto_apply_log.csv.

CLI:
    .venv/bin/python -m tools.auto_apply \\
        --url https://job-boards.greenhouse.io/tailscale/jobs/4527064005 \\
        --resume applications/gtm-engineer-resume.pdf \\
        --profile profiles/example-applications.yaml

Out of scope for this initial version:
    - Workday (most fragile, save for later)
    - LinkedIn Easy Apply (different mechanism)
    - Cover-letter customization (existing tools.tailor handles that)
    - Auto-clicking submit (NEVER, by design)

Note: this module is intentionally separate from tools.batch_apply. It does
not modify upstream code (per "the upstream is hands-off"). Test harness mocks the
browser-use Agent so unit tests never open a real browser.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import datetime
import logging
import os
import sys
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
LOG_PATH = OUTPUT_DIR / "auto_apply_log.csv"

# Auto-load workspace .env so ANTHROPIC_API_KEY is available.
# Same pattern as tools.tailor and tools.batch_apply.
WORKSPACE_ENV = PROJECT_ROOT.parent.parent.parent / ".env"
if WORKSPACE_ENV.exists():
    load_dotenv(WORKSPACE_ENV, override=True)

CLAUDE_MODEL = "claude-sonnet-4-6"

logger = logging.getLogger(__name__)


# ---------- Profile schema ----------


class Identity(BaseModel):
    full_name: str
    email: str
    phone: str
    location: str
    linkedin: str = ""
    github: str = ""
    website: str = ""


class WorkAuthorization(BaseModel):
    authorized_to_work_us: bool = True
    requires_sponsorship: bool = False


class Salary(BaseModel):
    default_expected: int = 0
    default_acceptable_floor: int = 0


class VeteranStatus(BaseModel):
    is_veteran: bool = False
    has_disability: bool = False


class Demographics(BaseModel):
    voluntary_disclosure: str = "decline_to_self_identify"


class CustomAnswers(BaseModel):
    why_this_company: str = ""
    why_this_role: str = ""
    notice_period_days: int = 14
    earliest_start: str = "2 weeks after offer"


class ApplicationProfile(BaseModel):
    identity: Identity
    work_authorization: WorkAuthorization = Field(default_factory=WorkAuthorization)
    salary: Salary = Field(default_factory=Salary)
    veteran_status: VeteranStatus = Field(default_factory=VeteranStatus)
    demographics: Demographics = Field(default_factory=Demographics)
    custom_answers: CustomAnswers = Field(default_factory=CustomAnswers)


def load_application_profile(path: Path | str) -> ApplicationProfile:
    """Load and validate a profile YAML file. Raises pydantic ValidationError."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text()) or {}
    return ApplicationProfile(**raw)


# ---------- ATS detection ----------


def detect_ats(url: str) -> str:
    """Return one of: 'greenhouse', 'lever', 'ashby', 'workday', 'unknown'.

    Detection is host-based so we tolerate query strings, fragments, and
    company-specific subpaths. This is a pure function — easy to unit-test
    without hitting the network.
    """
    if not url:
        return "unknown"
    host = (urlparse(url).hostname or "").lower()
    if not host:
        return "unknown"

    if "greenhouse.io" in host:
        return "greenhouse"
    if "lever.co" in host:
        return "lever"
    if "ashbyhq.com" in host:
        return "ashby"
    # Workday host pattern, e.g. mycompany.wd1.myworkdayjobs.com
    if "myworkdayjobs.com" in host or "workday.com" in host:
        return "workday"
    return "unknown"


# ---------- Field-name mapping ----------

# A small canonical map: the agent doesn't need this (it can read the page
# via vision), but the unit tests do — and surfacing the map makes it easy
# to extend the prompt with explicit field hints later.
FIELD_NAME_MAP: dict[str, list[str]] = {
    "full_name": ["full name", "name", "your name", "first name + last name"],
    "first_name": ["first name", "given name"],
    "last_name": ["last name", "family name", "surname"],
    "email": ["email", "email address", "e-mail"],
    "phone": ["phone", "phone number", "mobile", "telephone"],
    "location": ["location", "city", "current location", "where do you live"],
    "linkedin": ["linkedin", "linkedin url", "linkedin profile"],
    "github": ["github", "github url", "github profile"],
    "website": ["website", "portfolio", "personal site"],
    "resume": ["resume", "cv", "upload resume"],
    "work_auth": [
        "authorized to work",
        "legally authorized",
        "right to work",
        "work authorization",
    ],
    "sponsorship": [
        "require sponsorship",
        "need sponsorship",
        "visa sponsorship",
    ],
    "salary": [
        "salary expectation",
        "expected salary",
        "compensation expectation",
    ],
}


def map_field_name(label: str) -> Optional[str]:
    """Map a free-form form-label string to a canonical profile key.

    Matching is alias-by-alias, longest-alias first across all canonicals,
    so "first name" beats the shorter "name" alias on "full name". Returns
    None if no match. Used by tests; the live agent does its own matching
    via vision/text — this stays the single source of truth for canonicals.
    """
    if not label:
        return None
    norm = label.lower().strip()
    # Build (alias, canonical) pairs sorted by descending alias length
    # so "first name" wins over "name", "github url" wins over "github".
    candidates = [
        (alias, canonical)
        for canonical, aliases in FIELD_NAME_MAP.items()
        for alias in aliases
    ]
    candidates.sort(key=lambda pair: -len(pair[0]))
    for alias, canonical in candidates:
        if alias in norm:
            return canonical
    return None


# ---------- Task prompt construction ----------


HALT_INSTRUCTION = (
    "CRITICAL HUMAN-IN-LOOP RULE: After all standard fields are filled and "
    "the resume is uploaded, STOP. Do NOT click Submit, Apply, Send, or any "
    "button that finalizes the application. Take a screenshot of the filled "
    "form and end your run with the message READY_FOR_HUMAN_REVIEW. The "
    "human will inspect the form in the live browser and either type "
    "'yes submit' to proceed or 'skip' to abort."
)


def build_task_prompt(
    apply_url: str,
    resume_path: Path,
    profile: ApplicationProfile,
    ats: str,
) -> str:
    """Compose the natural-language instructions for the browser-use Agent.

    Includes (a) the apply URL, (b) explicit field-by-field instructions
    drawn from the profile, (c) the resume upload path, and (d) the halt
    rule. Custom questions with empty answers are flagged as 'PAUSE FOR
    HUMAN' so the agent surfaces them rather than hallucinating responses.
    """
    why_co = profile.custom_answers.why_this_company.strip() or "PAUSE FOR HUMAN"
    why_role = profile.custom_answers.why_this_role.strip() or "PAUSE FOR HUMAN"

    sponsorship_str = (
        "No, I do not require sponsorship"
        if not profile.work_authorization.requires_sponsorship
        else "Yes, I will require sponsorship"
    )
    work_auth_str = (
        "Yes, I am authorized to work in the United States"
        if profile.work_authorization.authorized_to_work_us
        else "No, I am not currently authorized to work in the United States"
    )

    parts = [
        f"You are filling out a job application form on this page: {apply_url}",
        f"ATS: {ats}",
        "",
        "Use the following profile values to fill standard fields. Match "
        "form labels to these values by their meaning, not exact text:",
        f"  - Full name: {profile.identity.full_name}",
        f"  - Email: {profile.identity.email}",
        f"  - Phone: {profile.identity.phone}",
        f"  - Current location: {profile.identity.location}",
        f"  - LinkedIn URL: {profile.identity.linkedin}",
        f"  - GitHub URL: {profile.identity.github}",
        f"  - Personal website: {profile.identity.website or '(none)'}",
        f"  - Work authorization: {work_auth_str}",
        f"  - Sponsorship: {sponsorship_str}",
        f"  - Salary expectation (USD/year): {profile.salary.default_expected}",
        f"  - Veteran status: {'Yes, protected veteran' if profile.veteran_status.is_veteran else 'No'}",
        f"  - Disability status: {'Yes, I have a disability' if profile.veteran_status.has_disability else profile.demographics.voluntary_disclosure}",
        f"  - Earliest start: {profile.custom_answers.earliest_start}",
        "",
        f"Upload this resume file when the form requests it: {resume_path.resolve()}",
        "",
        "For these custom questions, use the answers below. If an answer is "
        "literally 'PAUSE FOR HUMAN', leave the field blank and add it to a "
        "'NEEDS HUMAN INPUT' list in your final summary.",
        f"  - Why this company: {why_co}",
        f"  - Why this role: {why_role}",
        "",
        "If you encounter an unexpected field that does not map to any "
        "profile value (e.g. a coding question, a free-form essay, a "
        "demographic question without a clear default), DO NOT GUESS. "
        "Leave it blank, log it as needing human review, and continue.",
        "",
        HALT_INSTRUCTION,
    ]
    return "\n".join(parts)


# ---------- Logging ----------


def append_log_row(
    timestamp: str,
    url: str,
    ats: str,
    action: str,
    outcome: str,
    note: str = "",
    log_path: Path = LOG_PATH,
) -> None:
    """Append a single row to output/auto_apply_log.csv.

    Creates the file with a header if it does not exist. Caller passes a
    custom log_path in tests to avoid polluting the real log.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not log_path.exists()
    with log_path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if is_new:
            writer.writerow(["timestamp", "url", "ats", "action", "outcome", "note"])
        writer.writerow([timestamp, url, ats, action, outcome, note])


# ---------- Human-in-loop confirmation ----------


def wait_for_human_confirmation(
    prompt: str = "Form filled. Review the browser. Type 'yes submit' to submit, or 'skip' to abort: ",
    input_fn=input,
) -> bool:
    """Block on stdin until the human types an explicit submit-or-skip.

    Returns True only on the literal string 'yes submit' (case-insensitive,
    whitespace-stripped). Anything else aborts. The injectable input_fn
    parameter exists so tests can drive this with a stub.
    """
    while True:
        try:
            answer = input_fn(prompt)
        except EOFError:
            return False
        norm = (answer or "").strip().lower()
        if norm == "yes submit":
            return True
        if norm in {"skip", "no", "abort", "n", "cancel"}:
            return False
        # Reject anything ambiguous; force the human to type one of the
        # two explicit tokens. This is the non-negotiable safety gate.


# ---------- Browser-use orchestration ----------


async def run_auto_apply(
    apply_url: str,
    resume_path: Path,
    profile: ApplicationProfile,
    *,
    log_path: Path = LOG_PATH,
    confirm_fn=wait_for_human_confirmation,
    agent_factory=None,
) -> dict:
    """Drive the browser-use Agent through the apply form, halting before
    submit, then waiting for human confirmation.

    Returns a dict with keys: ats, status (in {'submitted', 'skipped',
    'fill_failed', 'unknown_ats'}), confirmation_screenshot (path or None),
    needs_human_review (list[str]).

    agent_factory is injected for testability — production code defaults
    to constructing a real browser_use.Agent. Tests pass a stub.
    """
    ts = datetime.datetime.now().isoformat(timespec="seconds")
    ats = detect_ats(apply_url)
    result: dict = {
        "ats": ats,
        "status": "fill_failed",
        "confirmation_screenshot": None,
        "needs_human_review": [],
    }

    if ats == "unknown":
        append_log_row(ts, apply_url, ats, "detect_ats", "unknown_ats", log_path=log_path)
        result["status"] = "unknown_ats"
        return result

    if ats == "workday":
        append_log_row(
            ts, apply_url, ats, "detect_ats", "out_of_scope",
            note="Workday support is out of scope for v1",
            log_path=log_path,
        )
        result["status"] = "out_of_scope"
        return result

    task = build_task_prompt(apply_url, resume_path, profile, ats)

    if agent_factory is None:
        # Local import so the rest of the module is importable without
        # browser-use installed (e.g. for unit tests that mock it out).
        from browser_use import Agent, ChatAnthropic  # type: ignore

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            append_log_row(
                ts, apply_url, ats, "init_agent", "missing_api_key",
                note="ANTHROPIC_API_KEY is unset; aborting",
                log_path=log_path,
            )
            result["status"] = "missing_api_key"
            return result

        llm = ChatAnthropic(model=CLAUDE_MODEL, api_key=api_key)

        def _default_factory(t: str):
            return Agent(task=t, llm=llm)

        agent_factory = _default_factory

    append_log_row(ts, apply_url, ats, "navigate", "starting", log_path=log_path)

    agent = agent_factory(task)
    try:
        agent_result = await agent.run()
    except Exception as exc:  # noqa: BLE001 — the agent can raise anything
        logger.exception("browser-use agent crashed")
        append_log_row(
            datetime.datetime.now().isoformat(timespec="seconds"),
            apply_url, ats, "fill_form", "agent_error",
            note=str(exc)[:200],
            log_path=log_path,
        )
        result["status"] = "fill_failed"
        return result

    append_log_row(
        datetime.datetime.now().isoformat(timespec="seconds"),
        apply_url, ats, "fill_form", "ready_for_review",
        log_path=log_path,
    )

    # The browser-use Agent returns a structured history; the caller can
    # inspect agent_result for the final screenshot path or message body.
    # We surface anything the agent said about needs-human-review.
    final_text = ""
    try:
        final_text = str(agent_result.final_result()) if hasattr(agent_result, "final_result") else str(agent_result)
    except Exception:  # noqa: BLE001
        final_text = ""
    if "NEEDS HUMAN INPUT" in final_text:
        # Keep this list informational — populated by the agent's own log.
        result["needs_human_review"].append(final_text)

    confirmed = confirm_fn()
    ts2 = datetime.datetime.now().isoformat(timespec="seconds")
    if not confirmed:
        append_log_row(ts2, apply_url, ats, "submit", "skipped", log_path=log_path)
        result["status"] = "skipped"
        return result

    # Hand control back to the agent ONLY for the final submit click. We
    # use a second, scoped task so the model can't be confused into
    # submitting on the first pass.
    submit_task = (
        "The human has approved the application. Click the Submit / Apply "
        "button now. After the page transitions to a confirmation state, "
        "take a screenshot and report SUBMITTED."
    )
    submit_agent = agent_factory(submit_task)
    try:
        submit_result = await submit_agent.run()
    except Exception as exc:  # noqa: BLE001
        logger.exception("submit agent crashed")
        append_log_row(
            datetime.datetime.now().isoformat(timespec="seconds"),
            apply_url, ats, "submit", "agent_error",
            note=str(exc)[:200],
            log_path=log_path,
        )
        result["status"] = "submit_failed"
        return result

    # Best-effort confirmation-screenshot extraction. browser-use exposes
    # the latest screenshot through its history; we tolerate either a
    # `screenshots` attr or a final message that names a path.
    shot_path = None
    try:
        if hasattr(submit_result, "screenshots") and submit_result.screenshots:
            shot_path = str(submit_result.screenshots[-1])
    except Exception:  # noqa: BLE001
        shot_path = None
    result["confirmation_screenshot"] = shot_path
    result["status"] = "submitted"

    append_log_row(
        datetime.datetime.now().isoformat(timespec="seconds"),
        apply_url, ats, "submit", "submitted",
        note=str(shot_path or ""),
        log_path=log_path,
    )
    return result


# ---------- CLI ----------


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="tools.auto_apply",
        description="Agentic auto-apply (proof-of-concept) for the Vector pipeline.",
    )
    p.add_argument("--url", required=True, help="Apply URL (Greenhouse/Lever/Ashby)")
    p.add_argument("--resume", required=True, type=Path, help="Path to a resume PDF")
    p.add_argument("--profile", required=True, type=Path, help="Path to applications profile YAML")
    p.add_argument("--log", type=Path, default=LOG_PATH, help="Override log CSV path")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)

    if not args.resume.exists():
        print(f"ERROR: resume file not found: {args.resume}", file=sys.stderr)
        return 2
    if not args.profile.exists():
        print(f"ERROR: profile file not found: {args.profile}", file=sys.stderr)
        return 2

    profile = load_application_profile(args.profile)
    print(f"Loaded profile for: {profile.identity.full_name}")
    print(f"Detected ATS: {detect_ats(args.url)}")
    print(f"Resume: {args.resume.resolve()}")
    print()
    print("Starting browser-use agent. The agent will pause before submit.")
    print()

    result = asyncio.run(
        run_auto_apply(
            args.url,
            args.resume,
            profile,
            log_path=args.log,
        )
    )
    print()
    print(f"Status: {result['status']}")
    if result.get("confirmation_screenshot"):
        print(f"Confirmation screenshot: {result['confirmation_screenshot']}")
    if result.get("needs_human_review"):
        print("Needs human review:")
        for item in result["needs_human_review"]:
            print(f"  - {item}")

    return 0 if result["status"] in {"submitted", "skipped"} else 1


if __name__ == "__main__":
    sys.exit(main())
