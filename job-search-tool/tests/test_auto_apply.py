"""Tests for tools.auto_apply.

The browser-use Agent is mocked out via the agent_factory injection point
so no test actually opens a browser. Tests cover:
    - profile loading + schema validation
    - URL -> ATS detection (Greenhouse / Lever / Ashby / Workday / unknown)
    - Field-name canonical mapping
    - Confirmation halt logic (only the literal "yes submit" submits)
    - The full run_auto_apply orchestration with mocked agent
"""
from __future__ import annotations

import asyncio
import csv
import textwrap
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from tools.auto_apply import (
    HALT_INSTRUCTION,
    ApplicationProfile,
    append_log_row,
    build_task_prompt,
    detect_ats,
    load_application_profile,
    map_field_name,
    run_auto_apply,
    wait_for_human_confirmation,
)


# ---------- Helpers ----------


SAMPLE_PROFILE_YAML = textwrap.dedent(
    """\
    identity:
      full_name: "Example User"
      email: "your-email@example.com"
      phone: "+1-555-555-5555"
      location: "Your City, ST"
      linkedin: "https://www.linkedin.com/in/your-profile"
      github: "https://github.com/your-github"
      website: ""

    work_authorization:
      authorized_to_work_us: true
      requires_sponsorship: false

    salary:
      default_expected: 175000
      default_acceptable_floor: 150000

    veteran_status:
      is_veteran: true
      has_disability: true

    demographics:
      voluntary_disclosure: "decline_to_self_identify"

    custom_answers:
      why_this_company: ""
      why_this_role: ""
      notice_period_days: 14
      earliest_start: "2 weeks after offer"
    """
)


@pytest.fixture()
def profile_path(tmp_path: Path) -> Path:
    p = tmp_path / "example-applications.yaml"
    p.write_text(SAMPLE_PROFILE_YAML)
    return p


@pytest.fixture()
def profile(profile_path: Path) -> ApplicationProfile:
    return load_application_profile(profile_path)


@pytest.fixture()
def resume_path(tmp_path: Path) -> Path:
    p = tmp_path / "fake-resume.pdf"
    p.write_bytes(b"%PDF-1.4 stub")
    return p


# ---------- Profile loading ----------


def test_load_profile_returns_validated_model(profile_path: Path):
    p = load_application_profile(profile_path)
    assert isinstance(p, ApplicationProfile)
    assert p.identity.full_name == "Example User"
    assert p.work_authorization.authorized_to_work_us is True
    assert p.salary.default_expected == 175000


def test_load_profile_raises_on_missing_identity(tmp_path: Path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("salary:\n  default_expected: 100000\n")
    with pytest.raises(ValidationError):
        load_application_profile(bad)


def test_load_profile_uses_defaults_for_optional_sections(tmp_path: Path):
    minimal = tmp_path / "minimal.yaml"
    minimal.write_text(
        textwrap.dedent(
            """\
            identity:
              full_name: "Test User"
              email: "test@example.com"
              phone: "+1-555-555-5555"
              location: "Tampa, FL"
            """
        )
    )
    p = load_application_profile(minimal)
    # Defaults should fill in
    assert p.work_authorization.authorized_to_work_us is True
    assert p.work_authorization.requires_sponsorship is False
    assert p.demographics.voluntary_disclosure == "decline_to_self_identify"


def test_real_committed_profile_loads():
    """The committed profiles/example-applications.yaml must round-trip."""
    project_root = Path(__file__).parent.parent
    profile_path = project_root / "profiles" / "example-applications.yaml"
    p = load_application_profile(profile_path)
    assert p.identity.full_name == "Example User"


# ---------- ATS detection ----------


@pytest.mark.parametrize(
    "url,expected",
    [
        ("https://job-boards.greenhouse.io/tailscale/jobs/4527064005", "greenhouse"),
        ("https://boards.greenhouse.io/airbnb/jobs/123", "greenhouse"),
        ("https://jobs.lever.co/plaid/abc123/apply", "lever"),
        ("https://jobs.ashbyhq.com/supabase/d5573afa/application", "ashby"),
        ("https://workday.wd1.myworkdayjobs.com/External/job/123", "workday"),
        ("https://example.com/careers/apply/foo", "unknown"),
        ("", "unknown"),
        ("not-a-url", "unknown"),
    ],
)
def test_detect_ats(url: str, expected: str):
    assert detect_ats(url) == expected


# ---------- Field-name mapping ----------


@pytest.mark.parametrize(
    "label,expected",
    [
        ("First Name", "first_name"),
        ("Last Name", "last_name"),
        ("Full Name", "full_name"),
        ("Email Address", "email"),
        ("Phone Number", "phone"),
        ("Current Location", "location"),
        ("LinkedIn URL", "linkedin"),
        ("GitHub Profile", "github"),
        ("Personal Website", "website"),
        ("Upload Resume", "resume"),
        ("Are you legally authorized to work in the U.S.?", "work_auth"),
        ("Do you require sponsorship?", "sponsorship"),
        ("Salary Expectation (USD)", "salary"),
        ("Why are you the best candidate?", None),
        ("", None),
    ],
)
def test_map_field_name(label: str, expected):
    assert map_field_name(label) == expected


# ---------- Halt instruction baked into the prompt ----------


def test_build_task_prompt_includes_halt_rule(profile: ApplicationProfile, resume_path: Path):
    task = build_task_prompt(
        "https://job-boards.greenhouse.io/tailscale/jobs/4527064005",
        resume_path,
        profile,
        ats="greenhouse",
    )
    assert "READY_FOR_HUMAN_REVIEW" in task
    assert "Do NOT click Submit" in task
    assert HALT_INSTRUCTION in task


def test_build_task_prompt_flags_blank_custom_answers(profile: ApplicationProfile, resume_path: Path):
    task = build_task_prompt(
        "https://job-boards.greenhouse.io/x/jobs/1", resume_path, profile, "greenhouse"
    )
    assert "PAUSE FOR HUMAN" in task


def test_build_task_prompt_includes_resume_absolute_path(profile: ApplicationProfile, resume_path: Path):
    task = build_task_prompt(
        "https://job-boards.greenhouse.io/x/jobs/1", resume_path, profile, "greenhouse"
    )
    assert str(resume_path.resolve()) in task


# ---------- Confirmation halt logic ----------


def test_confirmation_only_on_exact_yes_submit():
    answers = iter(["yes submit"])
    assert wait_for_human_confirmation(input_fn=lambda _: next(answers)) is True


def test_confirmation_yes_submit_is_case_insensitive():
    answers = iter(["YES SUBMIT"])
    assert wait_for_human_confirmation(input_fn=lambda _: next(answers)) is True


def test_confirmation_skip_aborts():
    answers = iter(["skip"])
    assert wait_for_human_confirmation(input_fn=lambda _: next(answers)) is False


def test_confirmation_repeats_on_ambiguous_input():
    """Plain 'yes' is too dangerous — must require the literal 'yes submit'."""
    answers = iter(["yes", "yes please", "yes submit"])
    assert wait_for_human_confirmation(input_fn=lambda _: next(answers)) is True


def test_confirmation_eof_aborts():
    def raise_eof(_):
        raise EOFError()
    assert wait_for_human_confirmation(input_fn=raise_eof) is False


# ---------- Logging ----------


def test_append_log_row_creates_file_with_header(tmp_path: Path):
    log = tmp_path / "log.csv"
    append_log_row("2026-05-06T12:00:00", "https://x", "greenhouse", "navigate", "ok", log_path=log)
    rows = list(csv.reader(log.open()))
    assert rows[0] == ["timestamp", "url", "ats", "action", "outcome", "note"]
    assert rows[1][3] == "navigate"


def test_append_log_row_appends_without_duplicating_header(tmp_path: Path):
    log = tmp_path / "log.csv"
    append_log_row("t1", "u", "greenhouse", "a", "o", log_path=log)
    append_log_row("t2", "u", "greenhouse", "a", "o", log_path=log)
    rows = list(csv.reader(log.open()))
    assert len(rows) == 3  # header + 2 data rows


# ---------- Full run_auto_apply orchestration with mocked agent ----------


class _FakeAgentRun:
    def __init__(self, final: str):
        self._final = final

    def final_result(self) -> str:
        return self._final


class _FakeAgent:
    """Minimal stand-in for browser_use.Agent. No browser, no LLM."""

    last_task: str = ""

    def __init__(self, task: str):
        _FakeAgent.last_task = task
        self.task = task

    async def run(self) -> _FakeAgentRun:
        return _FakeAgentRun("READY_FOR_HUMAN_REVIEW")


def _factory(task: str):
    return _FakeAgent(task)


def test_run_auto_apply_unknown_ats_short_circuits(tmp_path, profile, resume_path):
    log = tmp_path / "log.csv"
    result = asyncio.run(
        run_auto_apply(
            "https://example.com/careers/apply",
            resume_path,
            profile,
            log_path=log,
            confirm_fn=lambda *a, **k: False,
            agent_factory=_factory,
        )
    )
    assert result["status"] == "unknown_ats"
    # No agent should have been built — last_task is whatever it was before
    # this test (or empty). We at least know nothing went wrong downstream.
    rows = list(csv.reader(log.open()))
    assert any(r[4] == "unknown_ats" for r in rows[1:])


def test_run_auto_apply_workday_is_out_of_scope(tmp_path, profile, resume_path):
    log = tmp_path / "log.csv"
    result = asyncio.run(
        run_auto_apply(
            "https://acme.wd1.myworkdayjobs.com/External/job/123",
            resume_path,
            profile,
            log_path=log,
            confirm_fn=lambda *a, **k: True,
            agent_factory=_factory,
        )
    )
    assert result["status"] == "out_of_scope"


def test_run_auto_apply_pauses_for_confirmation_and_skips(tmp_path, profile, resume_path):
    log = tmp_path / "log.csv"
    result = asyncio.run(
        run_auto_apply(
            "https://job-boards.greenhouse.io/tailscale/jobs/4527064005",
            resume_path,
            profile,
            log_path=log,
            confirm_fn=lambda *a, **k: False,  # human says skip
            agent_factory=_factory,
        )
    )
    assert result["status"] == "skipped"
    assert result["ats"] == "greenhouse"
    # The fill-form task ran (so the agent saw the halt rule)
    assert "READY_FOR_HUMAN_REVIEW" in _FakeAgent.last_task
    rows = list(csv.reader(log.open()))
    outcomes = [r[4] for r in rows[1:]]
    assert "skipped" in outcomes
    # And submit was NEVER attempted
    assert "submitted" not in outcomes


def test_run_auto_apply_submits_only_after_explicit_confirmation(tmp_path, profile, resume_path):
    """The two-stage agent flow: stage 1 fills + halts, stage 2 submits.

    The fill-stage task must contain the halt instruction; the submit-stage
    task must explicitly invoke 'Submit'. This locks in that we never auto-
    submit on the first agent invocation.
    """
    log = tmp_path / "log.csv"
    captured_tasks: list[str] = []

    class _CaptureAgent(_FakeAgent):
        def __init__(self, task: str):
            super().__init__(task)
            captured_tasks.append(task)

    def _capture_factory(task: str):
        return _CaptureAgent(task)

    result = asyncio.run(
        run_auto_apply(
            "https://job-boards.greenhouse.io/tailscale/jobs/4527064005",
            resume_path,
            profile,
            log_path=log,
            confirm_fn=lambda *a, **k: True,  # human approves
            agent_factory=_capture_factory,
        )
    )
    assert result["status"] == "submitted"
    assert len(captured_tasks) == 2
    assert "Do NOT click Submit" in captured_tasks[0]
    assert "Click the Submit" in captured_tasks[1]


def test_run_auto_apply_logs_every_action(tmp_path, profile, resume_path):
    log = tmp_path / "log.csv"
    asyncio.run(
        run_auto_apply(
            "https://job-boards.greenhouse.io/x/jobs/1",
            resume_path,
            profile,
            log_path=log,
            confirm_fn=lambda *a, **k: True,
            agent_factory=_factory,
        )
    )
    rows = list(csv.reader(log.open()))
    actions = [r[3] for r in rows[1:]]
    assert "navigate" in actions
    assert "fill_form" in actions
    assert "submit" in actions
