"""Tests for tools/tailor.py — Anthropic SDK + pandas mocked, no real I/O."""
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from tools.tailor import (
    CATEGORY_MUST_HAVE,
    CATEGORY_NICE_TO_HAVE,
    CATEGORY_RED_HERRING,
    PRIORITY_HIGH,
    PRIORITY_LOW,
    PRIORITY_MEDIUM,
    RankedRequirement,
    apply_rewrites_to_yaml,
    build_tailor_prompt,
    call_claude_for_tailor,
    extract_proof_point_section,
    find_company_jd,
    format_ranked_requirements,
    parse_jd_requirements,
    parse_tailor_response,
    rank_requirements,
    slugify,
)


# ---------- slugify ----------

def test_slugify_basic_company():
    assert slugify("Buildkite", None) == "buildkite"


def test_slugify_with_role():
    assert slugify("Buildkite", "Staff GTM Engineer") == "buildkite-staff-gtm-engineer"


def test_slugify_handles_special_chars():
    # AT&T has '&' and Sales Engineer is the role — we expect a-z0-9-/-only.
    assert slugify("AT&T", "Sales Engineer") == "at-t-sales-engineer"


def test_slugify_collapses_repeats_and_strips():
    # Multiple weird chars should collapse to a single hyphen.
    assert slugify("  Hello!!!  ", "  World  ") == "hello-world"


def test_slugify_lowercases():
    assert slugify("MongoDB", "VP Sales") == "mongodb-vp-sales"


# ---------- find_company_jd ----------

def _master_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_find_company_jd_in_master(tmp_path: Path):
    master = tmp_path / "master.xlsx"
    master.write_bytes(b"x")  # for .exists() check; pd.read_excel is mocked
    df = _master_df([
        {"company": "Buildkite", "title": "Staff GTM Engineer",
         "description": "Build agentic workflows.", "fit_score": 9},
        {"company": "Notion", "title": "Sales Engineer",
         "description": "Sales stuff.", "fit_score": 7},
    ])
    with patch("tools.tailor.pd.read_excel", return_value=df):
        jd = find_company_jd(master, "Buildkite")
    assert jd == "Build agentic workflows."


def test_find_company_jd_filters_by_role(tmp_path: Path):
    """Two Buildkite rows; --role 'Staff GTM' must pick the GTM one."""
    master = tmp_path / "master.xlsx"
    master.write_bytes(b"x")  # for .exists() check
    df = _master_df([
        {"company": "Buildkite", "title": "Sales Engineer",
         "description": "Sales JD.", "fit_score": 6},
        {"company": "Buildkite", "title": "Staff GTM Engineer",
         "description": "GTM JD with stage plays.", "fit_score": 9},
    ])
    with patch("tools.tailor.pd.read_excel", return_value=df):
        jd = find_company_jd(master, "Buildkite", role="Staff GTM")
    assert jd == "GTM JD with stage plays."


def test_find_company_jd_picks_highest_fit_score(tmp_path: Path):
    """Multiple matches without role filter — highest fit_score wins."""
    master = tmp_path / "master.xlsx"
    master.write_bytes(b"x")
    df = _master_df([
        {"company": "Buildkite", "title": "Junior",
         "description": "low-fit JD.", "fit_score": 4},
        {"company": "Buildkite", "title": "Senior",
         "description": "high-fit JD.", "fit_score": 9},
    ])
    with patch("tools.tailor.pd.read_excel", return_value=df):
        jd = find_company_jd(master, "Buildkite")
    assert jd == "high-fit JD."


def test_find_company_jd_returns_none_when_no_match(tmp_path: Path):
    master = tmp_path / "master.xlsx"
    master.write_bytes(b"x")
    df = _master_df([
        {"company": "Notion", "title": "Sales Engineer",
         "description": "irrelevant.", "fit_score": 5},
    ])
    with patch("tools.tailor.pd.read_excel", return_value=df):
        jd = find_company_jd(master, "Buildkite")
    assert jd is None


def test_find_company_jd_returns_none_when_master_missing(tmp_path: Path):
    """File doesn't exist -> None, no read attempt."""
    missing = tmp_path / "does-not-exist.xlsx"
    with patch("tools.tailor.pd.read_excel") as rx:
        jd = find_company_jd(missing, "Buildkite")
    assert jd is None
    rx.assert_not_called()


def test_find_company_jd_case_insensitive(tmp_path: Path):
    master = tmp_path / "master.xlsx"
    master.write_bytes(b"x")
    df = _master_df([
        {"company": "buildkite", "title": "Engineer",
         "description": "lowercased company.", "fit_score": 8},
    ])
    with patch("tools.tailor.pd.read_excel", return_value=df):
        jd = find_company_jd(master, "Buildkite")
    assert jd == "lowercased company."


# ---------- proof-point extraction ----------

def test_extract_proof_point_section_slices_between_headings():
    md = (
        "# Top\n\n"
        "intro\n\n"
        "## Vocabulary translation table\n\n"
        "table content\n\n"
        "## Proof-point library\n\n"
        "claim | evidence\n"
        "top performer | AcmeCo 2024\n\n"
        "## The 4 phrases\n\n"
        "post-content here that should NOT appear.\n"
    )
    out = extract_proof_point_section(md)
    assert "## Proof-point library" in out
    assert "top performer" in out
    assert "post-content here" not in out


def test_extract_proof_point_section_returns_full_when_heading_missing():
    md = "# Just a title\n\nbody without the heading\n"
    out = extract_proof_point_section(md)
    assert out == md


# ---------- apply_rewrites_to_yaml ----------

def _sample_resume_dict() -> dict:
    return {
        "cv": {
            "name": "Example User",
            "sections": {
                "summary": [
                    "Revenue Operations & GTM Engineer with 10+ years building "
                    "and scaling revenue systems.",
                ],
                "experience": [
                    {
                        "company": "your company",
                        "highlights": [
                            "Built end-to-end agentic GTM orchestration system in Python",
                            "Designed customer segmentation framework",
                        ],
                    },
                ],
            },
        }
    }


def test_apply_rewrites_to_yaml_replaces_first_match():
    resume = _sample_resume_dict()
    rewrites = [
        {
            "old_bullet_substring": "Designed customer segmentation framework",
            "new_bullet": "Designed customer segmentation framework with stage plays mapped to lifecycle triggers",
        },
    ]
    out_yaml = apply_rewrites_to_yaml(resume, rewrites)
    assert "stage plays mapped to lifecycle triggers" in out_yaml
    # The substring approach: bullet was wholly replaced — the old form
    # should no longer appear as a standalone line.
    bullets = resume["cv"]["sections"]["experience"][0]["highlights"]
    assert any("stage plays" in b for b in bullets)


def test_apply_rewrites_warns_when_old_text_not_found(caplog):
    """Rewrite for a substring that doesn't exist anywhere -> log warning,
    don't crash, don't mutate the resume."""
    resume = _sample_resume_dict()
    bullets_before = list(resume["cv"]["sections"]["experience"][0]["highlights"])
    rewrites = [
        {
            "old_bullet_substring": "this string does not appear anywhere in the resume",
            "new_bullet": "should not be inserted",
        },
    ]
    with caplog.at_level("WARNING", logger="tools.tailor"):
        out_yaml = apply_rewrites_to_yaml(resume, rewrites)

    bullets_after = resume["cv"]["sections"]["experience"][0]["highlights"]
    assert bullets_after == bullets_before
    assert "should not be inserted" not in out_yaml
    # We at least logged something at WARNING level.
    assert any("not found" in r.message.lower() for r in caplog.records)


def test_apply_rewrites_handles_empty_rewrite_list():
    """Empty list returns the YAML unchanged (round-tripped)."""
    resume = _sample_resume_dict()
    out_yaml = apply_rewrites_to_yaml(resume, [])
    assert "Example User" in out_yaml
    assert "your company" in out_yaml


def test_apply_rewrites_skips_blank_rewrite_entries():
    """Empty old/new are quietly skipped, not crashed on."""
    resume = _sample_resume_dict()
    rewrites = [
        {"old_bullet_substring": "", "new_bullet": "doesn't matter"},
        {"old_bullet_substring": "customer segmentation framework", "new_bullet": ""},
    ]
    # Should not crash; resume should be unchanged because both entries are invalid.
    out_yaml = apply_rewrites_to_yaml(resume, rewrites)
    assert "customer segmentation framework" in out_yaml


# ---------- prompt construction ----------

def test_build_tailor_prompt_includes_jd_resume_and_proof_points():
    jd = "We're hiring a GTM Engineer who builds agentic workflows."
    resume_yaml = "cv:\n  name: Test\n"
    proof = "## Proof-point library\n| Claim | Evidence |\n|---|---|\n"
    prompt = build_tailor_prompt(jd, resume_yaml, proof)
    assert jd in prompt
    assert resume_yaml in prompt
    assert proof in prompt
    # Must call out the no-fabrication rule somewhere in the prompt.
    assert "DO NOT fabricate" in prompt or "fabricate" in prompt.lower()


# ---------- parse_tailor_response ----------

def test_parse_tailor_response_strips_markdown_fence():
    raw = (
        "```json\n"
        "{\"added\": [\"a\"], \"skipped\": [], \"rewrites\": []}\n"
        "```"
    )
    parsed = parse_tailor_response(raw)
    assert parsed == {"added": ["a"], "skipped": [], "rewrites": []}


def test_parse_tailor_response_returns_none_on_garbage():
    assert parse_tailor_response("not json at all") is None
    assert parse_tailor_response("") is None


def test_parse_tailor_response_fills_missing_keys():
    raw = '{"added": ["agentic workflows"]}'
    parsed = parse_tailor_response(raw)
    assert parsed == {"added": ["agentic workflows"], "skipped": [], "rewrites": []}


# ---------- call_claude_for_tailor ----------

def test_call_claude_with_proof_points_passes_all_inputs_to_prompt():
    """Mock Anthropic client; assert the prompt sent includes the JD,
    resume YAML, and proof-point section. Verify parsed JSON is returned."""
    fake_client = MagicMock()
    fake_resp = MagicMock()
    fake_resp.content = [MagicMock(text=json.dumps({
        "added": ["stage plays"],
        "skipped": [{"keyword": "metaflow", "reason": "no proof-point"}],
        "rewrites": [
            {
                "old_bullet_substring": "old bullet text",
                "new_bullet": "new bullet text with stage plays",
            }
        ],
    }))]
    fake_client.messages.create.return_value = fake_resp

    jd = "JD says: ship segmentation framework and stage plays."
    resume_yaml = "cv:\n  name: Example\n  sections:\n    summary:\n      - foo\n"
    proof = "## Proof-point library\n| stage plays | Buildkite term |\n"

    result = call_claude_for_tailor(
        client=fake_client,
        jd=jd,
        resume_yaml=resume_yaml,
        proof_points=proof,
    )

    assert result == {
        "added": ["stage plays"],
        "skipped": [{"keyword": "metaflow", "reason": "no proof-point"}],
        "rewrites": [
            {
                "old_bullet_substring": "old bullet text",
                "new_bullet": "new bullet text with stage plays",
            }
        ],
    }
    # Inspect the prompt text we sent
    fake_client.messages.create.assert_called_once()
    kwargs = fake_client.messages.create.call_args.kwargs
    user_msg = kwargs["messages"][0]["content"]
    assert jd in user_msg
    assert resume_yaml in user_msg
    assert "stage plays" in user_msg  # from proof points


def test_call_claude_returns_none_on_unparseable_response():
    fake_client = MagicMock()
    fake_resp = MagicMock()
    fake_resp.content = [MagicMock(text="totally not json")]
    fake_client.messages.create.return_value = fake_resp

    result = call_claude_for_tailor(
        client=fake_client,
        jd="x" * 60,
        resume_yaml="cv: {}",
        proof_points="## Proof-point library",
    )
    assert result is None


# ---------- parse_jd_requirements (Task B) ----------

_SAMPLE_JD = """\
Senior Sales Engineer

Acme Co is hiring a Senior Sales Engineer to drive technical wins.

Required Qualifications:
- 5+ years of experience as a Sales Engineer or Solutions Engineer
- Strong Python proficiency is required
- Hands-on experience with REST APIs

Preferred Qualifications:
- Experience with Salesforce
- Background in cybersecurity is a plus
- Familiarity with Terraform would be helpful

About You:
- You are a self-starter and team player
- Passion for solving customer problems
"""


def test_parse_jd_requirements_extracts_bullets():
    reqs = parse_jd_requirements(_SAMPLE_JD)
    texts = [r.text.lower() for r in reqs]
    assert any("python proficiency" in t for t in texts)
    assert any("rest apis" in t for t in texts)
    assert any("salesforce" in t for t in texts)
    assert any("self-starter" in t for t in texts)
    # We pulled at least 8 bullet items.
    assert len(reqs) >= 8


def test_parse_jd_requirements_categorizes_must_have():
    reqs = parse_jd_requirements(_SAMPLE_JD)
    by_text = {r.text.lower(): r for r in reqs}
    py = next(r for k, r in by_text.items() if "python" in k)
    assert py.category == CATEGORY_MUST_HAVE
    rest = next(r for k, r in by_text.items() if "rest apis" in k)
    assert rest.category == CATEGORY_MUST_HAVE


def test_parse_jd_requirements_categorizes_nice_to_have():
    reqs = parse_jd_requirements(_SAMPLE_JD)
    by_text = {r.text.lower(): r for r in reqs}
    sf = next(r for k, r in by_text.items() if "salesforce" in k)
    assert sf.category == CATEGORY_NICE_TO_HAVE
    cyber = next(r for k, r in by_text.items() if "cybersecurity" in k)
    assert cyber.category == CATEGORY_NICE_TO_HAVE
    tf = next(r for k, r in by_text.items() if "terraform" in k)
    assert tf.category == CATEGORY_NICE_TO_HAVE


def test_parse_jd_requirements_flags_red_herrings():
    reqs = parse_jd_requirements(_SAMPLE_JD)
    by_text = {r.text.lower(): r for r in reqs}
    starter = next(r for k, r in by_text.items() if "self-starter" in k)
    assert starter.category == CATEGORY_RED_HERRING
    passion = next(r for k, r in by_text.items() if "passion" in k)
    assert passion.category == CATEGORY_RED_HERRING


def test_parse_jd_requirements_handles_empty_input():
    assert parse_jd_requirements("") == []
    assert parse_jd_requirements("    ") == []


def test_parse_jd_requirements_dedupes_repeats_and_bumps_frequency():
    """Repeating a bullet bumps frequency and gives it more weight."""
    jd = (
        "Required:\n"
        "- Strong Python proficiency\n"
        "- Strong Python proficiency\n"
        "- Strong Python proficiency\n"
        "- Background in cybersecurity\n"
    )
    reqs = parse_jd_requirements(jd)
    texts = {_ for _ in (r.text.lower() for r in reqs)}
    # Three identical bullets dedup to one entry with freq=3.
    py = next(r for r in reqs if "python" in r.text.lower())
    assert py.frequency == 3
    # The other bullet is freq=1.
    cyber = next(r for r in reqs if "cybersecurity" in r.text.lower())
    assert cyber.frequency == 1


# ---------- rank_requirements (Task B) ----------

def test_rank_requirements_orders_by_priority_on_sample_jd():
    reqs = parse_jd_requirements(_SAMPLE_JD)
    ranked = rank_requirements(reqs, _SAMPLE_JD)
    # Sorted descending by score
    scores = [r.score for r in ranked]
    assert scores == sorted(scores, reverse=True)
    # Must-have requirements outrank nice-to-haves
    must_idx = [i for i, r in enumerate(ranked) if r.category == CATEGORY_MUST_HAVE]
    nice_idx = [i for i, r in enumerate(ranked) if r.category == CATEGORY_NICE_TO_HAVE]
    if must_idx and nice_idx:
        assert min(must_idx) < min(nice_idx), (
            "must-have should come before nice-to-have"
        )
    # Red-herrings are always priority=low
    red = [r for r in ranked if r.category == CATEGORY_RED_HERRING]
    assert red, "sample JD should produce at least one red-herring"
    assert all(r.priority == PRIORITY_LOW for r in red)


def test_rank_requirements_assigns_high_priority_to_top_must_haves():
    reqs = parse_jd_requirements(_SAMPLE_JD)
    ranked = rank_requirements(reqs, _SAMPLE_JD)
    # The first ranked item should be HIGH priority.
    assert ranked[0].priority == PRIORITY_HIGH
    # That first item should be a must-have (not a nice-to-have or red-herring).
    assert ranked[0].category == CATEGORY_MUST_HAVE


def test_rank_requirements_frequency_bump_promotes_repeated_keyword():
    """A keyword that appears 3x should outrank a singleton must-have."""
    jd = (
        "Required:\n"
        "- Strong Kubernetes operator experience\n"
        "- Strong Kubernetes operator experience\n"
        "- Strong Kubernetes operator experience\n"
        "- Background in compliance frameworks\n"
    )
    reqs = parse_jd_requirements(jd)
    ranked = rank_requirements(reqs, jd)
    # K8s appeared 3x so should outrank the singleton.
    top = ranked[0]
    assert "kubernetes" in top.text.lower()
    assert top.frequency == 3
    # And its score should be strictly higher than the other.
    other = next(r for r in ranked if "compliance" in r.text.lower())
    assert top.score > other.score


def test_rank_requirements_signals_are_recorded_for_debug_log():
    reqs = parse_jd_requirements(_SAMPLE_JD)
    ranked = rank_requirements(reqs, _SAMPLE_JD)
    # Every ranked item should have at least one signal explaining its score.
    for r in ranked:
        assert r.signals, f"{r.text!r} has no signals (debug log will be empty)"
    # And we should see category signals in the trail.
    assert any(
        any("category:" in s for s in r.signals) for r in ranked
    )


def test_rank_requirements_handles_empty_list():
    assert rank_requirements([], "") == []


# ---------- format_ranked_requirements (Task B) ----------

def test_format_ranked_requirements_renders_header_and_signals():
    reqs = parse_jd_requirements(_SAMPLE_JD)
    ranked = rank_requirements(reqs, _SAMPLE_JD)
    out = format_ranked_requirements(ranked)
    assert "[HIGH" in out or "[high" in out.lower()
    assert "must-have" in out.lower()
    assert "signals:" in out
    # Red-herring entries should be visible too (they're still in the list).
    assert "red-herring" in out.lower()


def test_format_ranked_requirements_handles_empty():
    out = format_ranked_requirements([])
    assert "no structured" in out.lower()


# ---------- prompt + proof-point gate integration (Task B) ----------

def test_build_tailor_prompt_includes_ranked_requirements_block():
    jd = "We need a Senior Sales Engineer."
    resume_yaml = "cv:\n  name: Test\n"
    proof = "## Proof-point library\n"
    ranked = [
        RankedRequirement(
            text="Strong Python proficiency",
            category=CATEGORY_MUST_HAVE,
            priority=PRIORITY_HIGH,
            score=5.5,
            frequency=2,
            signals=["category:must-have+3", "frequency:2+1"],
        ),
    ]
    prompt = build_tailor_prompt(jd, resume_yaml, proof, ranked_requirements=ranked)
    assert "Strong Python proficiency" in prompt
    assert "RANKED JD REQUIREMENTS" in prompt
    # The proof-point gate language must remain — that's the safety net
    # the spec asks us to preserve for high-priority requirements.
    assert "fabricate" in prompt.lower()


def test_build_tailor_prompt_works_without_ranked_requirements():
    """Backwards compat: existing callers that don't pass ranked= still work."""
    prompt = build_tailor_prompt("jd", "cv: {}", "## Proof")
    assert "no ranked requirements" in prompt.lower()


def test_call_claude_for_tailor_passes_ranked_block_to_prompt():
    fake_client = MagicMock()
    fake_resp = MagicMock()
    fake_resp.content = [MagicMock(text='{"added": [], "skipped": [], "rewrites": []}')]
    fake_client.messages.create.return_value = fake_resp

    ranked = [
        RankedRequirement(
            text="Hands-on experience with REST APIs",
            category=CATEGORY_MUST_HAVE,
            priority=PRIORITY_HIGH,
            score=4.0,
            frequency=1,
            signals=["category:must-have+3"],
        ),
    ]
    call_claude_for_tailor(
        client=fake_client,
        jd="JD body",
        resume_yaml="cv: {}",
        proof_points="## Proof",
        ranked_requirements=ranked,
    )
    fake_client.messages.create.assert_called_once()
    user_msg = fake_client.messages.create.call_args.kwargs["messages"][0]["content"]
    assert "REST APIs" in user_msg
    # The proof-point gate (no fabrication, even for HIGH priority)
    # must still appear in the prompt sent to the model.
    assert "fabricate" in user_msg.lower()
