"""Tests for career_translator: prompt construction, JSON parsing, markdown report assembly."""
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from tools.career_translator import (
    build_skills_prompt,
    build_adjacent_titles_prompt,
    build_car_bullets_prompt,
    build_gap_analysis_prompt,
    parse_kimi_json,
    render_markdown_report,
    cross_reference_companies,
)


# ============================================================================
# PROMPT CONSTRUCTION
# ============================================================================

def test_build_skills_prompt_includes_resume():
    resume = "Worked at AcmeCo as SDR, top performer."
    rubric = "Strong fit: GTM Engineer, RevOps roles."
    prompt = build_skills_prompt(resume, rubric)
    assert "AcmeCo" in prompt
    assert "top performer" in prompt


def test_build_skills_prompt_requests_structured_output():
    prompt = build_skills_prompt("resume", "rubric")
    # Asks for STRICT JSON with the 3 skill categories
    assert "JSON" in prompt
    assert "hard_skills" in prompt
    assert "soft_skills" in prompt
    assert "tools" in prompt


def test_build_adjacent_titles_prompt_asks_5_to_8():
    prompt = build_adjacent_titles_prompt("resume", "rubric")
    # Quality over quantity: ask for 5-8 titles
    assert "5" in prompt and "8" in prompt
    assert "adjacent" in prompt.lower() or "title" in prompt.lower()
    # Asks for structured fields
    assert "title" in prompt
    assert "why_qualified" in prompt
    assert "gap_to_close" in prompt


def test_build_adjacent_titles_prompt_includes_rubric():
    rubric = "Strong fit signals include AI tooling companies."
    prompt = build_adjacent_titles_prompt("resume body", rubric)
    assert "AI tooling companies" in prompt


def test_build_car_bullets_prompt_includes_target_titles():
    target_titles = ["GTM Engineer", "Forward Deployed Engineer"]
    prompt = build_car_bullets_prompt("resume body", "rubric", target_titles)
    assert "GTM Engineer" in prompt
    assert "Forward Deployed Engineer" in prompt


def test_build_car_bullets_prompt_explains_car_format():
    prompt = build_car_bullets_prompt("resume", "rubric", ["X"])
    # CAR = Challenge / Action / Result
    assert "Challenge" in prompt
    assert "Action" in prompt
    assert "Result" in prompt


def test_build_gap_analysis_prompt_includes_jd_text():
    jd_text = "Looking for a Staff GTM Engineer with deep Postgres + Python skills."
    prompt = build_gap_analysis_prompt("resume body", jd_text, "rubric")
    assert "Staff GTM Engineer" in prompt
    assert "Postgres" in prompt
    # asks for the right structured fields
    assert "jd_required_skills" in prompt
    assert "candidate_has" in prompt
    assert "gap" in prompt
    assert "close_gap_actions" in prompt


# ============================================================================
# RESPONSE PARSING
# ============================================================================

def test_parse_kimi_json_handles_markdown_fences():
    raw = '```json\n{"hard_skills": ["Python", "SQL"]}\n```'
    parsed = parse_kimi_json(raw)
    assert parsed == {"hard_skills": ["Python", "SQL"]}


def test_parse_kimi_json_handles_bare_fences():
    raw = '```\n{"a": 1}\n```'
    parsed = parse_kimi_json(raw)
    assert parsed == {"a": 1}


def test_parse_kimi_json_returns_none_on_invalid():
    assert parse_kimi_json("not JSON") is None
    assert parse_kimi_json("") is None
    assert parse_kimi_json(None) is None


def test_parse_kimi_json_accepts_list():
    raw = '[{"title": "GTM Engineer"}, {"title": "Solutions Engineer"}]'
    parsed = parse_kimi_json(raw)
    assert isinstance(parsed, list)
    assert parsed[0]["title"] == "GTM Engineer"


# ============================================================================
# COMPANIES CROSS-REFERENCE
# ============================================================================

def test_cross_reference_companies_returns_tier1_d100(tmp_path):
    db_file = tmp_path / "companies-db.json"
    db_file.write_text(json.dumps({
        "companies": [
            {"name": "GitLab", "tier": 1, "ats": "greenhouse"},
            {"name": "Vanta", "tier": 1, "ats": "ashby"},
            {"name": "MysteryCo", "tier": 2, "ats": "greenhouse"},
        ]
    }))
    matches = cross_reference_companies(
        adjacent_titles=[{"title": "GTM Engineer"}],
        db_path=db_file,
        limit=5,
    )
    # Tier-1 d100 surfaced; results should include GitLab and Vanta (and tier 2 MysteryCo if room)
    names = {m["name"] for m in matches}
    assert "GitLab" in names
    assert "Vanta" in names


def test_cross_reference_companies_handles_missing_db(tmp_path):
    matches = cross_reference_companies(
        adjacent_titles=[{"title": "GTM Engineer"}],
        db_path=tmp_path / "nonexistent.json",
        limit=5,
    )
    assert matches == []


# ============================================================================
# REPORT ASSEMBLY
# ============================================================================

def _fake_profile():
    """Lightweight stand-in mimicking the Profile object's attribute access."""
    profile = MagicMock()
    profile.name = "Example User"
    return profile


def test_render_markdown_report_includes_all_sections():
    skills = {
        "hard_skills": ["Python", "SQL"],
        "soft_skills": ["Leadership"],
        "tools": ["Salesforce", "Smartlead"],
    }
    adjacent = [
        {"title": "GTM Engineer", "why_qualified": "7yr revops", "gap_to_close": "deeper Python"},
    ]
    bullets = [
        {"role": "GTM Engineer", "bullet": "Challenge: ... Action: ... Result: ..."},
    ]
    md = render_markdown_report(
        profile=_fake_profile(),
        skills=skills,
        adjacent=adjacent,
        bullets=bullets,
        gap=None,
        jd_text=None,
        companies_hiring=[],
    )
    # Header w/ name
    assert "Example User" in md
    # Section 1: Skills
    assert "Skills" in md
    assert "Python" in md
    assert "Leadership" in md
    assert "Salesforce" in md
    # Section 2: Adjacent titles
    assert "Adjacent" in md
    assert "GTM Engineer" in md
    assert "7yr revops" in md
    # Section 3: CAR bullets
    assert "CAR" in md or "Resume Bullet" in md
    assert "Challenge:" in md
    # Section V2 hooks
    assert "O*NET" in md or "ONET" in md or "CareerOneStop" in md


def test_render_markdown_report_includes_gap_section_when_jd_provided():
    skills = {"hard_skills": [], "soft_skills": [], "tools": []}
    gap = {
        "jd_required_skills": ["Postgres", "Python"],
        "candidate_has": ["Python"],
        "gap": ["Postgres"],
        "close_gap_actions": ["Build a Postgres demo project"],
    }
    md = render_markdown_report(
        profile=_fake_profile(),
        skills=skills,
        adjacent=[],
        bullets=[],
        gap=gap,
        jd_text="Looking for Staff GTM Engineer",
        companies_hiring=[],
    )
    assert "Gap Analysis" in md
    assert "Postgres" in md
    assert "Build a Postgres demo project" in md


def test_render_markdown_report_includes_companies_section():
    skills = {"hard_skills": [], "soft_skills": [], "tools": []}
    md = render_markdown_report(
        profile=_fake_profile(),
        skills=skills,
        adjacent=[],
        bullets=[],
        gap=None,
        jd_text=None,
        companies_hiring=[{"name": "GitLab", "ats": "greenhouse", "tier": 1}],
    )
    assert "Companies" in md
    assert "GitLab" in md


def test_render_markdown_report_skips_gap_when_no_jd():
    skills = {"hard_skills": [], "soft_skills": [], "tools": []}
    md = render_markdown_report(
        profile=_fake_profile(),
        skills=skills,
        adjacent=[],
        bullets=[],
        gap=None,
        jd_text=None,
        companies_hiring=[],
    )
    # No JD, no gap section
    assert "Gap Analysis" not in md
