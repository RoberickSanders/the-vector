"""Tests for tools/batch_apply.py — Anthropic SDK + subprocess + pandas mocked, no real I/O."""
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from tools.batch_apply import (
    EMAIL_WORDS_MAX,
    EMAIL_WORDS_MIN,
    build_package_prompt,
    call_claude_for_package,
    parse_package_response,
    parse_score_from_output,
    process_candidate,
    render_application_md,
    render_batch_summary_md,
    run_score_resume,
    run_tailor,
    select_candidates,
)


# ---------- select_candidates ----------


def _candidates_df(rows: list[dict]) -> pd.DataFrame:
    """Helper to build a DataFrame; pandas keeps NaN for missing fields."""
    return pd.DataFrame(rows)


def test_select_candidates_skips_rows_without_manager_email(tmp_path: Path):
    """Row with fit_score but no manager_email -> warning, not selected."""
    df = _candidates_df([
        {"company": "LangChain", "title": "GTM Eng", "fit_score": 9,
         "manager_email": None},
        {"company": "Notion", "title": "Sales Eng", "fit_score": 8,
         "manager_email": "hm@notion.so"},
    ])
    selected, warnings = select_candidates(
        df, top_n=2, applications_dir=tmp_path, regenerate=False
    )
    assert len(selected) == 1
    assert selected[0]["company"] == "Notion"
    assert any("LangChain" in w and "no manager_email" in w for w in warnings)


def test_select_candidates_skips_rows_without_fit_score(tmp_path: Path):
    """Rows where fit_score is null are silently dropped from contention."""
    df = _candidates_df([
        {"company": "Foo", "title": "X", "fit_score": None,
         "manager_email": "x@foo.com"},
        {"company": "Bar", "title": "Y", "fit_score": 7,
         "manager_email": "y@bar.com"},
    ])
    selected, _warnings = select_candidates(
        df, top_n=5, applications_dir=tmp_path, regenerate=False
    )
    assert len(selected) == 1
    assert selected[0]["company"] == "Bar"


def test_select_candidates_skips_existing_md_unless_regenerate(tmp_path: Path):
    """Existing applications/<slug>.md skips by default; --regenerate takes."""
    # Create the .md that would block the Notion row.
    (tmp_path / "notion-sales-eng.md").write_text("existing")

    df = _candidates_df([
        {"company": "Notion", "title": "Sales Eng", "fit_score": 9,
         "manager_email": "y@notion.so"},
    ])

    # Without --regenerate: skipped.
    selected, warnings = select_candidates(
        df, top_n=5, applications_dir=tmp_path, regenerate=False
    )
    assert selected == []
    assert any("Notion" in w and "exists" in w for w in warnings)

    # With --regenerate: picked.
    selected2, _warns2 = select_candidates(
        df, top_n=5, applications_dir=tmp_path, regenerate=True
    )
    assert len(selected2) == 1


def test_select_candidates_returns_top_n_sorted_by_fit_desc(tmp_path: Path):
    """N=2 from a pool of 4, ordered by fit_score desc."""
    df = _candidates_df([
        {"company": "C1", "title": "T", "fit_score": 5, "manager_email": "a@c1.com"},
        {"company": "C2", "title": "T", "fit_score": 9, "manager_email": "a@c2.com"},
        {"company": "C3", "title": "T", "fit_score": 7, "manager_email": "a@c3.com"},
        {"company": "C4", "title": "T", "fit_score": 8, "manager_email": "a@c4.com"},
    ])
    selected, _ = select_candidates(
        df, top_n=2, applications_dir=tmp_path, regenerate=False
    )
    assert [r["company"] for r in selected] == ["C2", "C4"]


def test_select_candidates_single_company_mode(tmp_path: Path):
    """--company filters down to one company before applying top_n."""
    df = _candidates_df([
        {"company": "LangChain", "title": "GTM Eng", "fit_score": 7,
         "manager_email": "x@langchain.com"},
        {"company": "Notion", "title": "Sales Eng", "fit_score": 9,
         "manager_email": "y@notion.so"},
    ])
    selected, _ = select_candidates(
        df, top_n=5, applications_dir=tmp_path, regenerate=False,
        company_filter="LangChain",
    )
    assert len(selected) == 1
    assert selected[0]["company"] == "LangChain"


# ---------- run_tailor / run_score_resume ----------


def test_calls_tailor_subprocess_per_candidate(tmp_path: Path):
    """run_tailor invokes the tailor module via subprocess.run."""
    fake_proc = MagicMock(returncode=0, stdout="ok", stderr="")
    with patch("tools.batch_apply.subprocess.run", return_value=fake_proc) as mock_sp:
        ok, out = run_tailor("example", "LangChain", "GTM Engineer")
    assert ok is True
    assert mock_sp.call_count == 1
    cmd = mock_sp.call_args.args[0]
    # Expect: [python, -m, tools.tailor, --profile, example, --company, LangChain, --role, GTM Engineer]
    assert "tools.tailor" in cmd
    assert "--profile" in cmd and "example" in cmd
    assert "--company" in cmd and "LangChain" in cmd
    assert "--role" in cmd and "GTM Engineer" in cmd


def test_skips_tailor_when_skip_tailor_flag(tmp_path: Path):
    """process_candidate with skip_tailor=True must NOT call run_tailor."""
    row = pd.Series({
        "company": "LangChain",
        "title": "GTM Engineer",
        "fit_score": 9,
        "location": "Remote",
        "manager_email": "hm@langchain.com",
        "manager_name": "Jane Doe",
        "manager_linkedin": "",
        "manager_source": "blitz",
        "job_url_direct": "https://langchain.com/jobs/1",
        "job_url": "",
        "company_url": "",
        "description": "We're hiring a GTM Engineer.",
    })

    fake_client = MagicMock()
    fake_resp = MagicMock()
    fake_resp.content = [MagicMock(text=json.dumps({
        "subject": "Just submitted: GTM Engineer — Example User",
        "email_body": "Hi Jane,\n" + "x " * 130,
        "email_word_count": 130,
        "linkedin_dm": "x " * 35,
        "linkedin_dm_word_count": 35,
        "evaluation": {
            "executive_summary": "Solid fit.",
            "background_match": [{"jd_requirement": "X", "candidate_evidence": "Y"}],
            "positioning_strategy": "Lead with the agentic platform.",
            "tailoring_plan": ["Edit 1"],
            "star_stories": [{"jd_bullet": "X", "story": {"S": "s", "T": "t", "A": "a", "R": "r"}}],
        },
    }))]
    fake_client.messages.create.return_value = fake_resp

    with patch("tools.batch_apply.run_tailor") as mock_tailor, \
         patch("tools.batch_apply.run_score_resume", return_value=(75, "Score: 75/100")):
        result = process_candidate(
            row=row,
            profile_name="example",
            applications_dir=tmp_path,
            resume_yaml="cv: {}",
            positioning_md="## Proof-point library\n",
            client=fake_client,
            skip_tailor=True,
            submitted_on="2026-04-27",
        )
    mock_tailor.assert_not_called()
    assert result is not None
    assert result["company"] == "LangChain"


def test_calls_score_resume_subprocess_per_candidate():
    """run_score_resume invokes score_resume.py via subprocess.run."""
    fake_proc = MagicMock(
        returncode=0,
        stdout="Score: 67/100  (extractor: llm, match: token-level)\n",
        stderr="",
    )
    with patch("tools.batch_apply.subprocess.run", return_value=fake_proc) as mock_sp:
        score, out = run_score_resume(
            Path("/tmp/foo.pdf"), "LangChain", profile="example"
        )
    assert score == 67
    cmd = mock_sp.call_args.args[0]
    assert "tools.score_resume" in cmd
    assert "--pdf" in cmd
    assert "--company" in cmd and "LangChain" in cmd


def test_parses_score_from_score_resume_output():
    """Score: NN/NN parsing extracts the integer."""
    s1 = "Score: 67/100  (extractor: llm, match: token-level)\nMatched (10): foo"
    assert parse_score_from_output(s1) == 67
    s2 = "Score: 100/100  (extractor: heuristic, match: strict)\n"
    assert parse_score_from_output(s2) == 100
    # Garbled output -> None
    assert parse_score_from_output("no score here") is None
    assert parse_score_from_output("") is None


# ---------- LLM package call ----------


def test_drafts_email_and_dm_via_anthropic_with_positioning_context():
    """Verify the prompt sent to Claude includes positioning.md content,
    proof-point references, and the email length rules."""
    fake_client = MagicMock()
    fake_resp = MagicMock()
    fake_resp.content = [MagicMock(text=json.dumps({
        "subject": "Just submitted: GTM Engineer — Example User",
        "email_body": "Hi Jane,\nbody here\n",
        "email_word_count": 142,
        "linkedin_dm": "Short DM.",
        "linkedin_dm_word_count": 32,
        "evaluation": {
            "executive_summary": "Fit.",
            "background_match": [],
            "positioning_strategy": "",
            "tailoring_plan": [],
            "star_stories": [],
        },
    }))]
    fake_client.messages.create.return_value = fake_resp

    positioning_md = (
        "## Proof-point library\n"
        "| Claim | Evidence |\n"
        "| Top performer at AcmeCo | AcmeCo 2024 |\n\n"
        "## Anti-patterns\n"
        "Don't name Plascencia in copy.\n"
    )

    result = call_claude_for_package(
        client=fake_client,
        company="LangChain",
        role="GTM Engineer",
        jd="We're hiring a GTM Engineer who builds agentic workflows.",
        jd_url="https://langchain.com/jobs/1",
        manager_name="Jane Doe",
        manager_title="VP Engineering",
        manager_email="jane@langchain.com",
        resume_yaml="cv:\n  name: Example\n",
        positioning_md=positioning_md,
    )

    assert result is not None
    assert result["email_word_count"] == 142
    assert result["linkedin_dm_word_count"] == 32

    # Inspect the prompt content sent to the LLM.
    fake_client.messages.create.assert_called_once()
    kwargs = fake_client.messages.create.call_args.kwargs
    user_msg = kwargs["messages"][0]["content"]

    # Positioning.md content must be embedded.
    assert "Top performer at AcmeCo" in user_msg
    assert "Plascencia" in user_msg  # the anti-pattern itself appears in positioning.md content
    # Email-length context (the 100-175 target) must be in the prompt.
    assert "100-175" in user_msg or f"{EMAIL_WORDS_MIN}-{EMAIL_WORDS_MAX}" in user_msg
    # Manager identity in prompt.
    assert "Jane Doe" in user_msg
    assert "jane@langchain.com" in user_msg


def test_drafts_email_word_count_in_target_range(tmp_path: Path, caplog):
    """When LLM returns 142 words: accept silently. When 60 words: warn."""
    row = pd.Series({
        "company": "LangChain", "title": "GTM Engineer", "fit_score": 9,
        "location": "Remote", "manager_email": "hm@langchain.com",
        "manager_name": "Jane", "manager_linkedin": "", "manager_source": "",
        "job_url_direct": "", "job_url": "", "company_url": "",
        "description": "We hire GTM Engineers.",
    })

    def _make_client(email_wc: int):
        c = MagicMock()
        c.messages.create.return_value = MagicMock(content=[MagicMock(text=json.dumps({
            "subject": "Just submitted: GTM Engineer — Example User",
            "email_body": "body",
            "email_word_count": email_wc,
            "linkedin_dm": "dm",
            "linkedin_dm_word_count": 35,
            "evaluation": {
                "executive_summary": "", "background_match": [],
                "positioning_strategy": "", "tailoring_plan": [],
                "star_stories": [],
            },
        }))])
        return c

    # Acceptable count (142) — no word-count warning logged.
    with patch("tools.batch_apply.run_score_resume", return_value=(70, "Score: 70/100")):
        with caplog.at_level("WARNING", logger="tools.batch_apply"):
            caplog.clear()
            result_ok = process_candidate(
                row=row, profile_name="example", applications_dir=tmp_path,
                resume_yaml="cv: {}", positioning_md="## Proof-point library\n",
                client=_make_client(142), skip_tailor=True,
                submitted_on="2026-04-27",
            )
        assert result_ok is not None
        assert not any("word count" in r.message.lower() for r in caplog.records)

    # Below target (60) — warning logged.
    # Use a different slug to avoid clashing with the 142-word run's md.
    row2 = row.copy()
    row2["company"] = "Other"
    with patch("tools.batch_apply.run_score_resume", return_value=(70, "Score: 70/100")):
        with caplog.at_level("WARNING", logger="tools.batch_apply"):
            caplog.clear()
            result_short = process_candidate(
                row=row2, profile_name="example", applications_dir=tmp_path,
                resume_yaml="cv: {}", positioning_md="## Proof-point library\n",
                client=_make_client(60), skip_tailor=True,
                submitted_on="2026-04-27",
            )
        assert result_short is not None
        assert any("word count" in r.message.lower() for r in caplog.records)


# ---------- Application MD writer ----------


def test_writes_application_md_with_template_sections(tmp_path: Path):
    """Verify render_application_md outputs the Buildkite-mirrored sections."""
    package = {
        "subject": "Just submitted: GTM Engineer — Example User",
        "email_body": "Hi Jane,\nWords here.",
        "email_word_count": 142,
        "linkedin_dm": "Short DM with stage plays callback.",
        "linkedin_dm_word_count": 35,
        "evaluation": {
            "executive_summary": "Solid fit. Apply with tailored resume.",
            "background_match": [
                {"jd_requirement": "Engineering background",
                 "candidate_evidence": "Backend stack: Python, SQLite, Anthropic SDK"},
            ],
            "positioning_strategy": "Lead with the agentic platform. De-emphasize TypeScript gap.",
            "tailoring_plan": ["Move backend stack up", "Drop Spanish from skills"],
            "star_stories": [
                {"jd_bullet": "Build segmentation framework",
                 "story": {"S": "Company needed scoring.", "T": "Rank niches.",
                           "A": "Built segmentation framework.", "R": "3.63% reply."}},
            ],
        },
    }
    md = render_application_md(
        company="LangChain",
        role="GTM Engineer",
        fit_score=9,
        keyword_score=67,
        location="Remote",
        apply_url="https://langchain.com/jobs/1",
        submitted_on="2026-04-27",
        manager_name="Jane Doe",
        manager_title="VP Engineering",
        manager_email="jane@langchain.com",
        manager_linkedin="",
        manager_source="blitz+mv_ok",
        package=package,
    )

    # Buildkite-template section headings must all appear.
    assert "# LangChain — GTM Engineer" in md
    assert "**Apply via:**" in md
    assert "**Fit:**" in md
    assert "## Hiring manager" in md
    assert "## A–F Evaluation" in md
    assert "### A. Executive Summary" in md
    assert "### B. Background match" in md
    assert "### C. Positioning strategy" in md
    assert "### D. Compensation" in md
    assert "### E. Tailoring plan" in md
    assert "### F. STAR stories" in md
    assert "## Outreach — cold email" in md
    assert "## Outreach — LinkedIn DM" in md
    assert "## Status log" in md

    # Content checks
    assert "Just submitted: GTM Engineer — Example User" in md
    assert "67/100" in md  # keyword score in header
    assert "blitz+mv_ok" in md  # manager source
    assert "segmentation framework" in md  # STAR story content


def test_writes_application_md_warns_on_out_of_range_word_count():
    """When email_word_count < 100, the rendered md must include a warning."""
    package = {
        "subject": "Subj",
        "email_body": "body",
        "email_word_count": 50,  # below range
        "linkedin_dm": "dm",
        "linkedin_dm_word_count": 35,
        "evaluation": {"executive_summary": "", "background_match": [],
                       "positioning_strategy": "", "tailoring_plan": [],
                       "star_stories": []},
    }
    md = render_application_md(
        company="C", role="R", fit_score=8, keyword_score=70, location="",
        apply_url="", submitted_on="2026-04-27",
        manager_name="N", manager_title="", manager_email="e@c.com",
        manager_linkedin="", manager_source="",
        package=package,
    )
    assert "Warning" in md
    assert "100-175" in md or f"{EMAIL_WORDS_MIN}-{EMAIL_WORDS_MAX}" in md


# ---------- Batch summary ----------


def test_writes_batch_summary_md():
    """Verify the batch-<date>.md is generated with a row per candidate."""
    rows = [
        {"company": "LangChain", "role": "GTM Engineer", "fit": 9, "score": 67,
         "manager": "Jane <jane@langchain.com>",
         "apply_url": "https://langchain.com/j/1",
         "package_path": "applications/langchain-gtm-engineer.md"},
        {"company": "Notion", "role": "Sales Engineer", "fit": 8, "score": 72,
         "manager": "John <john@notion.so>",
         "apply_url": "https://notion.so/careers/2",
         "package_path": "applications/notion-sales-engineer.md"},
    ]
    md = render_batch_summary_md(rows, "2026-04-27")
    assert "# Batch run — 2026-04-27" in md
    assert "Generated 2 application package(s)" in md
    # Both companies present in the table
    assert "LangChain" in md
    assert "Notion" in md
    assert "67/100" in md
    assert "72/100" in md
    # Next-actions reminders
    assert "Next actions" in md
    assert "Spot-check" in md or "spot-check" in md.lower()


# ---------- Anthropic error handling ----------


def test_handles_anthropic_error_gracefully(tmp_path: Path, caplog):
    """LLM raises -> log warning, return None, don't crash."""
    row = pd.Series({
        "company": "LangChain", "title": "GTM Engineer", "fit_score": 9,
        "location": "Remote", "manager_email": "hm@langchain.com",
        "manager_name": "Jane", "manager_linkedin": "", "manager_source": "",
        "job_url_direct": "", "job_url": "", "company_url": "",
        "description": "We hire GTM Engineers.",
    })
    bad_client = MagicMock()
    bad_client.messages.create.side_effect = RuntimeError("API down")

    with patch("tools.batch_apply.run_score_resume", return_value=(None, "")):
        with caplog.at_level("WARNING", logger="tools.batch_apply"):
            result = process_candidate(
                row=row, profile_name="example", applications_dir=tmp_path,
                resume_yaml="cv: {}", positioning_md="## Proof-point library\n",
                client=bad_client, skip_tailor=True,
                submitted_on="2026-04-27",
            )
    assert result is None
    assert any("anthropic" in r.message.lower() or "api down" in r.message.lower()
               for r in caplog.records)


# ---------- skip-md when exists ----------


def test_skips_md_when_already_exists_without_regenerate(tmp_path: Path):
    """If applications/<slug>.md exists and --regenerate not set, the row is skipped."""
    # Pre-create the .md.
    (tmp_path / "langchain-gtm-engineer.md").write_text("existing content")
    df = _candidates_df([
        {"company": "LangChain", "title": "GTM Engineer", "fit_score": 9,
         "manager_email": "hm@langchain.com"},
    ])
    selected, warnings = select_candidates(
        df, top_n=5, applications_dir=tmp_path, regenerate=False
    )
    assert selected == []
    assert any("LangChain" in w for w in warnings)


# ---------- parse_package_response sanity ----------


def test_parse_package_response_strips_markdown_fence():
    raw = (
        "```json\n"
        '{"subject": "x", "email_body": "y", "email_word_count": 100,'
        ' "linkedin_dm": "d", "linkedin_dm_word_count": 30,'
        ' "evaluation": {"executive_summary": "s"}}\n'
        "```"
    )
    parsed = parse_package_response(raw)
    assert parsed is not None
    assert parsed["subject"] == "x"
    assert parsed["evaluation"]["executive_summary"] == "s"
    # Defaulted-in keys
    assert parsed["evaluation"]["background_match"] == []
    assert parsed["evaluation"]["star_stories"] == []


def test_parse_package_response_returns_none_on_garbage():
    assert parse_package_response("not json") is None
    assert parse_package_response("") is None


# ---------- prompt construction ----------


def test_build_package_prompt_includes_required_constraints():
    prompt = build_package_prompt(
        company="LangChain", role="GTM Engineer",
        jd="agentic workflows JD",
        jd_url="https://langchain.com/j/1",
        manager_name="Jane", manager_title="VP Eng",
        manager_email="jane@langchain.com",
        resume_yaml="cv:\n  name: Example\n",
        positioning_md="## Proof-point library\n| Claim | Evidence |\n",
    )
    # Hiring-manager email rule referenced in prompt:
    assert "100-175" in prompt or f"{EMAIL_WORDS_MIN}-{EMAIL_WORDS_MAX}" in prompt
    # JD content embedded
    assert "agentic workflows JD" in prompt
    # Resume embedded
    assert "Example" in prompt
    # No-fabrication / anti-pattern guidance
    assert ("Plascencia" in prompt) or ("modern outbound" in prompt.lower())
