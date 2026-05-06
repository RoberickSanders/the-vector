"""Tests for tools/score_resume.py — pypdf + Anthropic SDK mocked, no real I/O."""
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tools.score_resume import (
    compute_score,
    extract_keywords,
    extract_keywords_via_llm,
    extract_pdf_text,
    make_anthropic_client,
    match_keywords,
)


def _llm_response(payload: dict) -> MagicMock:
    """Mock an Anthropic messages.create return shape."""
    resp = MagicMock()
    msg = MagicMock()
    msg.text = json.dumps(payload)
    resp.content = [msg]
    return resp


def _llm_response_text(text: str) -> MagicMock:
    resp = MagicMock()
    msg = MagicMock()
    msg.text = text
    resp.content = [msg]
    return resp


# ---------- extract_keywords ----------

def test_extract_keywords_from_jd_drops_stopwords():
    """Common stopwords like 'the', 'and', 'with', 'we' must not appear."""
    jd = (
        "We are looking for a GTM engineer with the experience to build "
        "agentic workflows. The team needs orchestration."
    )
    kws = extract_keywords(jd, max_keywords=30)
    lowered = [k.lower() for k in kws]
    for stop in ("the", "and", "with", "we", "are", "to", "a", "for"):
        assert stop not in lowered, f"stopword '{stop}' leaked into keywords"


def test_extract_keywords_drops_common_jd_noise():
    """Words like 'looking', 'experience', 'role', 'team', 'company' get dropped."""
    jd = "We are looking for a candidate with experience for this role on the team at our company."
    kws = extract_keywords(jd)
    lowered = [k.lower() for k in kws]
    for noise in ("looking", "experience", "role", "team", "company", "candidate"):
        assert noise not in lowered, f"JD noise '{noise}' leaked"


def test_extract_keywords_includes_bigrams():
    """A bigram repeated >=2 times in the JD must appear in keywords."""
    jd = (
        "We need segmentation framework expertise. Our segmentation framework maps triggers "
        "to plays, and segmentation framework work is the core of the role. "
        "Build agentic workflows; ship reliable agentic workflows daily."
    )
    kws = extract_keywords(jd)
    # segmentation framework appears 3x; agentic workflows 2x
    assert "segmentation framework" in kws
    assert "agentic workflows" in kws


def test_extract_keywords_includes_trigrams():
    """A trigram repeated >=2 times beats unigrams of similar count."""
    jd = (
        "Build cost waterfall enrichment. Our cost waterfall enrichment "
        "cascade saves money. Cost waterfall enrichment is core."
    )
    kws = extract_keywords(jd)
    assert "cost waterfall enrichment" in kws


def test_extract_keywords_caps_at_30():
    """Long, varied JD with many candidates should top out at 30."""
    # 60 distinct technical words, each repeated 3 times
    distinct = [f"distinctword{i:02d}" for i in range(60)]
    jd = " ".join(distinct * 3)
    kws = extract_keywords(jd, max_keywords=30)
    assert len(kws) == 30


def test_extract_keywords_empty_jd_returns_empty():
    assert extract_keywords("") == []
    assert extract_keywords("   \n\n  ") == []


def test_extract_keywords_dedupes_subphrases():
    """A unigram ('agentic') already covered by a multi-word phrase
    ('agentic workflows') should not occupy a second slot."""
    jd = (
        "agentic agentic agentic agentic workflows agentic workflows "
        "agentic workflows agentic workflows"
    )
    kws = extract_keywords(jd)
    # "agentic workflows" should win; "agentic" alone should be deduped out.
    assert "agentic workflows" in kws
    assert "agentic" not in kws


# ---------- match_keywords (token-level default + strict opt-in) ----------

def test_match_keywords_case_insensitive_match():
    keywords = ["Python", "Agentic Workflows", "TypeScript"]
    pdf_text = "I write python code and ship agentic workflows daily."
    matched, missing = match_keywords(keywords, pdf_text)
    assert "Python" in matched
    assert "Agentic Workflows" in matched
    assert "TypeScript" in missing


def test_match_keywords_token_level_handles_phrase_variation():
    """LLM 'intent systems' should match resume's 'intent/ABM systems'."""
    keywords = ["intent systems", "crm architecture", "systems thinking"]
    pdf_text = (
        "I designed intent/ABM systems for outbound. "
        "CRM data architecture across Salesforce + HubSpot. "
        "Systems thinking under constraints."
    )
    matched, missing = match_keywords(keywords, pdf_text)
    assert "intent systems" in matched
    assert "crm architecture" in matched
    assert "systems thinking" in matched
    assert missing == []


def test_match_keywords_token_level_rejects_partial_brand():
    """'Customer.io' must NOT match if only 'customer' is in haystack."""
    keywords = ["Customer.io", "Metaflow"]
    pdf_text = "I focus on the customer journey at every stage."
    matched, missing = match_keywords(keywords, pdf_text)
    # 'customer' alone is not enough — 'io' is also required
    assert "Customer.io" in missing
    assert "Metaflow" in missing


def test_match_keywords_token_level_handles_slashes():
    """'outbound/inbound' tokenizes to ['outbound','inbound']; resume has both."""
    keywords = ["outbound/inbound", "AI/ML"]
    pdf_text = "Built outbound systems with inbound channels and AI/ML pipelines."
    matched, missing = match_keywords(keywords, pdf_text)
    assert "outbound/inbound" in matched
    assert "AI/ML" in matched


def test_match_keywords_strict_mode_requires_contiguous():
    """--strict-match requires the entire phrase as substring."""
    keywords = ["intent systems", "intent/ABM systems"]
    pdf_text = "I designed intent/ABM systems."
    matched, missing = match_keywords(keywords, pdf_text, strict=True)
    # strict: 'intent systems' (with space) is NOT in the text;
    # 'intent/ABM systems' IS.
    assert "intent systems" in missing
    assert "intent/ABM systems" in matched


def test_match_keywords_strict_mode_more_conservative_than_default():
    """Same keywords + same PDF: strict produces fewer matches when phrasing varies."""
    keywords = ["intent systems", "crm architecture", "agentic tooling"]
    pdf_text = (
        "intent/ABM systems, CRM data architecture, "
        "AI-led tooling and agentic workflows"
    )
    default_matched, _ = match_keywords(keywords, pdf_text)
    strict_matched, _ = match_keywords(keywords, pdf_text, strict=True)
    # Token-level catches all 3 (tokens present even though phrasing varies);
    # strict catches none (no contiguous substring matches the JD's wording).
    assert len(default_matched) == 3
    assert len(strict_matched) == 0
    assert len(default_matched) > len(strict_matched)


def test_match_keywords_brand_keyword_requires_strict_match():
    """Customer.io must NOT match a resume that has Gong.io (sharing 'io' token).

    This is the false-positive that pure token-level matching would create.
    Brand-style keywords (containing '.', '+', '#') get strict treatment.
    """
    keywords = ["Customer.io", "reo.dev", "C++", "C#"]
    # Haystack has 'customer', 'io' (from Gong.io), 'reo' (made up), 'dev', 'c'
    pdf_text = (
        "Customer journey work. Daily stack: Gong.io, Outreach. "
        "Reo customer focus. Dev tools background. C language exposure."
    )
    matched, missing = match_keywords(keywords, pdf_text)
    # All four brand-style keywords should be MISSING because they don't
    # appear as contiguous substrings, even though their tokens overlap.
    assert "Customer.io" in missing
    assert "reo.dev" in missing
    assert "C++" in missing
    assert "C#" in missing


def test_match_keywords_brand_keyword_matches_when_full_name_present():
    """Customer.io appears verbatim -> matched."""
    keywords = ["Customer.io"]
    pdf_text = "Daily stack: Customer.io for sequencing."
    matched, missing = match_keywords(keywords, pdf_text)
    assert "Customer.io" in matched
    assert missing == []


def test_match_keywords_default_still_token_level_for_non_brand():
    """Plain phrases (no brand punctuation) keep token-level treatment."""
    keywords = ["intent systems", "crm architecture"]
    pdf_text = "intent/ABM systems and CRM data architecture across the stack."
    matched, missing = match_keywords(keywords, pdf_text)
    assert "intent systems" in matched
    assert "crm architecture" in matched


def test_match_keywords_substring_strict_match():
    """Strict mode preserves the legacy stem-substring behavior."""
    keywords = ["orchestrat"]  # stem
    pdf_text = "I orchestrate workflows."
    matched, missing = match_keywords(keywords, pdf_text, strict=True)
    assert "orchestrat" in matched


def test_lists_missing_keywords():
    """All missing keywords must appear in the missing list, none in matched."""
    keywords = ["python", "rust", "go", "haskell"]
    pdf_text = "I know python."
    matched, missing = match_keywords(keywords, pdf_text)
    assert matched == ["python"]
    assert set(missing) == {"rust", "go", "haskell"}


def test_match_keywords_empty_keyword_skipped():
    """A keyword with no extractable tokens (e.g., '!@#') goes to missing."""
    keywords = ["!@#", "python"]
    pdf_text = "I know python."
    matched, missing = match_keywords(keywords, pdf_text)
    assert matched == ["python"]
    assert "!@#" in missing


# ---------- compute_score ----------

def test_score_high_when_most_match():
    """JD has 10 keywords, PDF contains 8 -> score = 80 >= 70."""
    keywords = [f"kw{i}" for i in range(10)]
    pdf_text = " ".join(keywords[:8])  # 8 of 10 match
    matched, missing = match_keywords(keywords, pdf_text)
    score = compute_score(len(matched), len(keywords))
    assert score == 80
    assert score >= 70


def test_score_low_when_few_match():
    """JD has 10 keywords, PDF contains 2 -> score = 20 < 30."""
    keywords = [f"kw{i}" for i in range(10)]
    pdf_text = " ".join(keywords[:2])
    matched, missing = match_keywords(keywords, pdf_text)
    score = compute_score(len(matched), len(keywords))
    assert score == 20
    assert score < 30


def test_compute_score_handles_zero_total():
    """No keywords extracted shouldn't divide-by-zero."""
    assert compute_score(0, 0) == 0


def test_compute_score_perfect():
    assert compute_score(10, 10) == 100


# ---------- extract_pdf_text ----------

def test_extract_pdf_text_returns_string(tmp_path: Path):
    """Mock pypdf.PdfReader; verify text is concatenated across pages."""
    fake_path = tmp_path / "fake.pdf"
    fake_path.write_bytes(b"%PDF-1.4 dummy")  # not a real PDF; we mock the reader

    page1 = MagicMock()
    page1.extract_text.return_value = "Page one text. agentic workflows."
    page2 = MagicMock()
    page2.extract_text.return_value = "Page two text. segmentation framework."

    fake_reader = MagicMock()
    fake_reader.pages = [page1, page2]

    with patch("tools.score_resume.PdfReader", return_value=fake_reader):
        text = extract_pdf_text(fake_path)

    assert isinstance(text, str)
    assert "agentic workflows" in text
    assert "segmentation framework" in text


def test_extract_pdf_text_returns_empty_on_unreadable_pdf(tmp_path: Path):
    """If pypdf raises, return '' instead of crashing — caller will score 0."""
    fake_path = tmp_path / "bad.pdf"
    fake_path.write_bytes(b"not a pdf")
    with patch("tools.score_resume.PdfReader", side_effect=Exception("corrupt")):
        text = extract_pdf_text(fake_path)
    assert text == ""


def test_extract_pdf_text_skips_pages_that_throw(tmp_path: Path):
    """One bad page shouldn't kill the whole extraction."""
    fake_path = tmp_path / "fake.pdf"
    fake_path.write_bytes(b"%PDF-1.4")

    good_page = MagicMock()
    good_page.extract_text.return_value = "good content."
    bad_page = MagicMock()
    bad_page.extract_text.side_effect = Exception("page broken")

    fake_reader = MagicMock()
    fake_reader.pages = [good_page, bad_page]

    with patch("tools.score_resume.PdfReader", return_value=fake_reader):
        text = extract_pdf_text(fake_path)

    assert "good content." in text


# ---------- extract_keywords_via_llm ----------

def test_extract_keywords_via_llm_returns_list_from_json():
    client = MagicMock()
    client.messages.create.return_value = _llm_response({
        "keywords": ["agentic workflows", "segmentation framework", "quality gate"]
    })
    keywords = extract_keywords_via_llm("a long enough jd " * 20, client)
    assert keywords == ["agentic workflows", "segmentation framework", "quality gate"]
    client.messages.create.assert_called_once()


def test_extract_keywords_via_llm_lowercases_and_dedupes():
    client = MagicMock()
    client.messages.create.return_value = _llm_response({
        "keywords": ["Agentic Workflows", "agentic workflows", "Signal Taxonomy"]
    })
    keywords = extract_keywords_via_llm("a long enough jd " * 20, client)
    # lowercased + deduped
    assert keywords == ["agentic workflows", "segmentation framework"]


def test_extract_keywords_via_llm_strips_markdown_fences():
    client = MagicMock()
    raw = '```json\n{"keywords": ["python", "sql"]}\n```'
    client.messages.create.return_value = _llm_response_text(raw)
    keywords = extract_keywords_via_llm("a long enough jd " * 20, client)
    assert keywords == ["python", "sql"]


def test_extract_keywords_via_llm_returns_empty_on_non_json():
    client = MagicMock()
    client.messages.create.return_value = _llm_response_text("not json at all")
    keywords = extract_keywords_via_llm("a long enough jd " * 20, client)
    assert keywords == []


def test_extract_keywords_via_llm_returns_empty_on_api_error():
    client = MagicMock()
    client.messages.create.side_effect = Exception("api down")
    keywords = extract_keywords_via_llm("a long enough jd " * 20, client)
    assert keywords == []


def test_extract_keywords_via_llm_caps_to_max():
    client = MagicMock()
    client.messages.create.return_value = _llm_response({
        "keywords": [f"kw{i}" for i in range(50)]
    })
    keywords = extract_keywords_via_llm("a long enough jd " * 20, client, max_keywords=10)
    assert len(keywords) == 10


def test_extract_keywords_via_llm_handles_non_string_entries():
    """If LLM returns malformed entries (numbers, nulls), skip them."""
    client = MagicMock()
    client.messages.create.return_value = _llm_response({
        "keywords": ["python", None, 42, "sql", ""]
    })
    keywords = extract_keywords_via_llm("a long enough jd " * 20, client)
    assert keywords == ["python", "sql"]


def test_extract_keywords_via_llm_handles_missing_keywords_field():
    client = MagicMock()
    client.messages.create.return_value = _llm_response({"foo": "bar"})
    keywords = extract_keywords_via_llm("a long enough jd " * 20, client)
    assert keywords == []


# ---------- make_anthropic_client ----------

def test_make_anthropic_client_returns_none_without_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert make_anthropic_client() is None


def test_make_anthropic_client_returns_none_when_empty(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    assert make_anthropic_client() is None


def test_make_anthropic_client_builds_when_key_present(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    client = make_anthropic_client()
    assert client is not None
