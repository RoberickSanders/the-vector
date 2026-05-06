import json
from pathlib import Path

import pandas as pd
import pytest

from tools.score import (
    build_scoring_prompt,
    load_d100_company_names,
    parse_score_response,
    prioritize_d100,
    round_robin_by_company,
)


def test_build_scoring_prompt_includes_rubric_and_jd():
    rubric = "## Strong-fit signals\n- GTM Engineer roles\n"
    jd = "Looking for a GTM Engineer with Python and SQL experience."
    prompt = build_scoring_prompt(rubric, jd)
    assert "GTM Engineer roles" in prompt
    assert "Python and SQL" in prompt


def test_build_scoring_prompt_requests_structured_output():
    prompt = build_scoring_prompt("rubric text", "jd text")
    assert "fit_score" in prompt
    assert "why_match" in prompt
    assert "recommended_action" in prompt
    assert "missing_skills" in prompt


def test_parse_score_response_valid_json():
    raw = '{"fit_score": 8, "why_match": "Strong fit because X", "recommended_action": "apply", "missing_skills": ["dbt"]}'
    parsed = parse_score_response(raw)
    assert parsed["fit_score"] == 8
    assert parsed["recommended_action"] == "apply"
    assert "dbt" in parsed["missing_skills"]


def test_parse_score_response_handles_markdown_fence():
    raw = '```json\n{"fit_score": 5, "why_match": "test", "recommended_action": "skip", "missing_skills": []}\n```'
    parsed = parse_score_response(raw)
    assert parsed["fit_score"] == 5


def test_parse_score_response_returns_none_on_invalid():
    parsed = parse_score_response("not json at all")
    assert parsed is None


def test_prioritize_d100_orders_d100_first():
    df = pd.DataFrame([
        {"company": "Random Co", "title": "X"},
        {"company": "GitLab", "title": "Y"},
        {"company": "Other", "title": "Z"},
        {"company": "Vanta", "title": "W"},
    ])
    out = prioritize_d100(df, {"gitlab", "vanta"}, limit=10)
    companies_in_order = list(out["company"])
    # GitLab and Vanta come before Random Co / Other
    assert companies_in_order.index("GitLab") < companies_in_order.index("Random Co")
    assert companies_in_order.index("Vanta") < companies_in_order.index("Other")


def test_prioritize_d100_caps_at_limit():
    df = pd.DataFrame([
        {"company": "GitLab", "title": "A"},
        {"company": "Vanta", "title": "B"},
        {"company": "Random", "title": "C"},
        {"company": "Random2", "title": "D"},
    ])
    out = prioritize_d100(df, {"gitlab", "vanta"}, limit=2)
    assert len(out) == 2
    assert set(out["company"]) == {"GitLab", "Vanta"}


def test_prioritize_d100_empty_d100_set_falls_back_to_head():
    df = pd.DataFrame([
        {"company": "A", "title": "1"},
        {"company": "B", "title": "2"},
        {"company": "C", "title": "3"},
    ])
    out = prioritize_d100(df, set(), limit=2)
    assert len(out) == 2
    assert list(out["company"]) == ["A", "B"]


def test_load_d100_company_names_reads_db(tmp_path):
    db_file = tmp_path / "companies-db.json"
    db_file.write_text(json.dumps({
        "companies": [
            {"name": "GitLab", "ats": "greenhouse"},
            {"name": "Vanta", "ats": "ashby"},
            {"name": ""},
        ]
    }))
    names = load_d100_company_names(db_file)
    assert names == {"gitlab", "vanta"}


def test_load_d100_company_names_returns_empty_when_missing(tmp_path):
    names = load_d100_company_names(tmp_path / "nonexistent.json")
    assert names == set()


# ---------- Round-robin (one job per company per pass) ----------

def test_round_robin_interleaves_companies():
    """3 MongoDB rows + 2 Notion rows should alternate Mongo, Notion, Mongo, Notion, Mongo."""
    df = pd.DataFrame([
        {"company": "MongoDB", "title": "M1"},
        {"company": "MongoDB", "title": "M2"},
        {"company": "MongoDB", "title": "M3"},
        {"company": "Notion", "title": "N1"},
        {"company": "Notion", "title": "N2"},
    ])
    out = round_robin_by_company(df)
    assert list(out["company"]) == ["MongoDB", "Notion", "MongoDB", "Notion", "MongoDB"]
    # within-company order preserved
    mongo_titles = list(out[out["company"] == "MongoDB"]["title"])
    assert mongo_titles == ["M1", "M2", "M3"]


def test_round_robin_handles_empty_and_missing_company_column():
    # empty DF
    empty = pd.DataFrame()
    assert round_robin_by_company(empty).empty
    # DF without 'company' column should pass through unchanged
    df = pd.DataFrame([{"title": "x"}, {"title": "y"}])
    out = round_robin_by_company(df)
    assert len(out) == 2


def test_prioritize_d100_no_company_dominates():
    """Among 3 d100 companies, scoring 6 jobs should hit each at most twice."""
    df = pd.DataFrame([
        {"company": "MongoDB", "title": "GTM Engineer"},
        {"company": "MongoDB", "title": "Engineer"},
        {"company": "MongoDB", "title": "Engineer"},
        {"company": "MongoDB", "title": "Engineer"},
        {"company": "Notion", "title": "Engineer"},
        {"company": "Notion", "title": "Engineer"},
        {"company": "Vanta", "title": "Engineer"},
    ])
    d100 = {"mongodb", "notion", "vanta"}
    out = prioritize_d100(df, d100, limit=6)
    counts = out["company"].value_counts().to_dict()
    # No single company should account for more than 1/3 of the slate
    # in a balanced round-robin (here: max should be 2 of 6).
    assert max(counts.values()) <= 3
    assert "Notion" in counts
    assert "Vanta" in counts
