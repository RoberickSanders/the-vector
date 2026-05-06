"""Tests for tools/drive.py — Composio CLI mocked, no real uploads."""
import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import tools.drive as drive_module
from tools.drive import (
    load_state,
    save_state,
    upload_doc_to_drive,
    upload_vector_docs,
)


# ---------- helpers ----------

def _composio_response(file_id: str = "new-fid", display_url: str = "https://drive/x"):
    """Mock a successful composio CLI stdout response."""
    payload = {"data": {"file": {"id": file_id}, "display_url": display_url}}
    proc = MagicMock()
    proc.returncode = 0
    proc.stdout = "Some Composio chatter before JSON\n" + json.dumps(payload)
    proc.stderr = ""
    return proc


def _composio_failure(rc: int = 1, stderr: str = "boom"):
    proc = MagicMock()
    proc.returncode = rc
    proc.stdout = ""
    proc.stderr = stderr
    return proc


@pytest.fixture
def state_file(tmp_path: Path) -> Path:
    p = tmp_path / "drive-state.json"
    p.write_text(json.dumps({
        "example": {
            "folder_id": "folder-123",
            "folder_url": "https://drive/folder",
            "file_id": "excel-fid",
            "file_url": "https://drive/excel",
        }
    }, indent=2))
    return p


@pytest.fixture
def md_file(tmp_path: Path) -> Path:
    p = tmp_path / "STATUS.md"
    p.write_text("# Status\n\nHello, Vector.\n")
    return p


# ---------- upload_doc_to_drive ----------

def test_upload_doc_skips_when_composio_missing(md_file, state_file):
    with patch("tools.drive.shutil.which", return_value=None):
        result = upload_doc_to_drive(md_file, "example", state_file)
    assert result == "failed"


def test_upload_doc_returns_no_state_when_no_folder_id(tmp_path, md_file):
    # State has no folder_id for the profile
    sf = tmp_path / "empty.json"
    sf.write_text(json.dumps({"example": {}}))
    with patch("tools.drive.shutil.which", return_value="/usr/local/bin/composio"), \
         patch("tools.drive.subprocess.run") as run:
        result = upload_doc_to_drive(md_file, "example", sf)
    assert result == "no-state"
    run.assert_not_called()


def test_upload_doc_uploads_first_time(md_file, state_file):
    with patch("tools.drive.shutil.which", return_value="/usr/local/bin/composio"), \
         patch("tools.drive.subprocess.run", return_value=_composio_response("fid-1")) as run:
        result = upload_doc_to_drive(md_file, "example", state_file)
    assert result == "uploaded"
    run.assert_called_once()

    # Verify state was persisted with file_id + uploaded_at
    state = load_state(state_file)
    docs = state["example"]["docs"]
    assert "STATUS.md" in docs
    assert docs["STATUS.md"]["file_id"] == "fid-1"
    assert docs["STATUS.md"]["uploaded_at"] > 0


def test_upload_doc_skips_when_unchanged(md_file, state_file):
    """Second call with same mtime should skip the composio subprocess."""
    # Seed state with an uploaded_at >= local mtime
    seeded_at = md_file.stat().st_mtime + 100  # well in the future
    state = load_state(state_file)
    state["example"]["docs"] = {
        "STATUS.md": {
            "file_id": "existing-fid",
            "file_url": "https://drive/x",
            "uploaded_at": seeded_at,
        }
    }
    save_state(state_file, state)

    with patch("tools.drive.shutil.which", return_value="/usr/local/bin/composio"), \
         patch("tools.drive.subprocess.run") as run:
        result = upload_doc_to_drive(md_file, "example", state_file)
    assert result == "skipped"
    run.assert_not_called()


def test_upload_doc_force_bypasses_mtime_guard(md_file, state_file):
    seeded_at = md_file.stat().st_mtime + 100
    state = load_state(state_file)
    state["example"]["docs"] = {
        "STATUS.md": {
            "file_id": "existing-fid",
            "file_url": "https://drive/x",
            "uploaded_at": seeded_at,
        }
    }
    save_state(state_file, state)

    with patch("tools.drive.shutil.which", return_value="/usr/local/bin/composio"), \
         patch("tools.drive.subprocess.run", return_value=_composio_response()) as run:
        result = upload_doc_to_drive(md_file, "example", state_file, force=True)
    assert result == "uploaded"
    run.assert_called_once()


def test_upload_doc_uploads_when_local_newer(md_file, state_file):
    """uploaded_at older than local mtime => upload."""
    seeded_at = md_file.stat().st_mtime - 100  # older than the file
    state = load_state(state_file)
    state["example"]["docs"] = {
        "STATUS.md": {
            "file_id": "old-fid",
            "file_url": "https://drive/x",
            "uploaded_at": seeded_at,
        }
    }
    save_state(state_file, state)

    with patch("tools.drive.shutil.which", return_value="/usr/local/bin/composio"), \
         patch("tools.drive.subprocess.run", return_value=_composio_response("fresh-fid")) as run:
        result = upload_doc_to_drive(md_file, "example", state_file)
    assert result == "uploaded"
    run.assert_called_once()

    # And the persisted file_id was updated
    new_state = load_state(state_file)
    assert new_state["example"]["docs"]["STATUS.md"]["file_id"] == "fresh-fid"


def test_upload_doc_returns_failed_on_composio_error(md_file, state_file):
    with patch("tools.drive.shutil.which", return_value="/usr/local/bin/composio"), \
         patch("tools.drive.subprocess.run", return_value=_composio_failure(2, "auth")):
        result = upload_doc_to_drive(md_file, "example", state_file)
    assert result == "failed"
    # State should not have been written with success markers
    state = load_state(state_file)
    assert "STATUS.md" not in state["example"].get("docs", {})


def test_upload_doc_handles_unparseable_composio_output(md_file, state_file):
    """If composio prints garbage but rc=0, treat as success and stamp uploaded_at
    so future runs don't re-upload forever."""
    proc = MagicMock()
    proc.returncode = 0
    proc.stdout = "no json here at all"
    proc.stderr = ""
    with patch("tools.drive.shutil.which", return_value="/usr/local/bin/composio"), \
         patch("tools.drive.subprocess.run", return_value=proc):
        result = upload_doc_to_drive(md_file, "example", state_file)
    assert result == "uploaded"
    state = load_state(state_file)
    assert state["example"]["docs"]["STATUS.md"]["uploaded_at"] > 0


def test_upload_doc_returns_failed_when_local_missing(state_file, tmp_path):
    missing = tmp_path / "does-not-exist.md"
    with patch("tools.drive.shutil.which", return_value="/usr/local/bin/composio"), \
         patch("tools.drive.subprocess.run") as run:
        result = upload_doc_to_drive(missing, "example", state_file)
    assert result == "failed"
    run.assert_not_called()


# ---------- upload_vector_docs ----------

def test_upload_vector_docs_iterates_md_files(tmp_path, state_file):
    docs = tmp_path / "vector-docs"
    docs.mkdir()
    (docs / "STATUS.md").write_text("# A")
    (docs / "BRIEF.md").write_text("# B")
    (docs / "resume.md").write_text("# C")
    # Non-md should be ignored
    (docs / "ignore.txt").write_text("not md")
    # Subdir should be ignored
    (docs / "specs").mkdir()
    (docs / "specs" / "deep.md").write_text("# nested")

    with patch("tools.drive.shutil.which", return_value="/usr/local/bin/composio"), \
         patch("tools.drive.subprocess.run", return_value=_composio_response()):
        counts = upload_vector_docs("example", state_file, docs)

    assert counts["uploaded"] == 3
    assert counts["skipped"] == 0
    assert counts["failed"] == 0


def test_upload_vector_docs_returns_zero_when_dir_missing(tmp_path, state_file):
    counts = upload_vector_docs("example", state_file, tmp_path / "nope")
    assert counts == {"uploaded": 0, "skipped": 0, "failed": 0, "no-state": 0}


def test_upload_vector_docs_returns_zero_when_no_md(tmp_path, state_file):
    docs = tmp_path / "empty-docs"
    docs.mkdir()
    (docs / "x.txt").write_text("nope")
    counts = upload_vector_docs("example", state_file, docs)
    assert sum(counts.values()) == 0


def test_upload_vector_docs_short_circuits_on_no_state(tmp_path):
    """If the first doc returns no-state, don't bother trying the rest."""
    sf = tmp_path / "empty-state.json"
    sf.write_text(json.dumps({"example": {}}))
    docs = tmp_path / "vector-docs"
    docs.mkdir()
    (docs / "STATUS.md").write_text("# A")
    (docs / "BRIEF.md").write_text("# B")
    (docs / "resume.md").write_text("# C")

    with patch("tools.drive.shutil.which", return_value="/usr/local/bin/composio"), \
         patch("tools.drive.subprocess.run") as run:
        counts = upload_vector_docs("example", sf, docs)

    assert counts["no-state"] == 1
    # Did not iterate further
    run.assert_not_called()


def test_upload_vector_docs_mixes_uploaded_and_skipped(tmp_path, state_file):
    docs = tmp_path / "vector-docs"
    docs.mkdir()
    f_new = docs / "STATUS.md"
    f_new.write_text("# A")
    f_old = docs / "BRIEF.md"
    f_old.write_text("# B")

    # Pre-seed BRIEF as already-uploaded with future timestamp -> should skip
    state = load_state(state_file)
    state["example"]["docs"] = {
        "BRIEF.md": {
            "file_id": "brief-fid",
            "file_url": "https://drive/brief",
            "uploaded_at": f_old.stat().st_mtime + 100,
        }
    }
    save_state(state_file, state)

    with patch("tools.drive.shutil.which", return_value="/usr/local/bin/composio"), \
         patch("tools.drive.subprocess.run", return_value=_composio_response()) as run:
        counts = upload_vector_docs("example", state_file, docs)

    assert counts["uploaded"] == 1   # STATUS.md
    assert counts["skipped"] == 1    # BRIEF.md
    assert counts["failed"] == 0
    # Composio called only once (for STATUS.md)
    assert run.call_count == 1
