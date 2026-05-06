"""Google Drive sync via Composio CLI. Per-profile state.

Two flavors:
  - upload_to_drive(...)     — the master Excel (one per profile, replaces in place)
  - upload_doc_to_drive(...) — individual markdown docs (umbrella docs of the Vector)
  - upload_vector_docs(...)  — convenience: glob the-vector/*.md and sync each, skipping unchanged

mtime guard: state[profile]["docs"][filename]["uploaded_at"] = epoch float of last
successful upload. Subsequent calls compare to local mtime and skip if nothing changed —
keeps the doc-sync near-instant on every run after the first.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path


def load_state(state_file: Path | str) -> dict:
    state_file = Path(state_file)
    if not state_file.exists():
        return {}
    return json.loads(state_file.read_text())


def save_state(state_file: Path | str, state: dict) -> None:
    Path(state_file).write_text(json.dumps(state, indent=2))


def upload_to_drive(local_path: Path, profile_name: str, state_file: Path | str) -> bool:
    """Upload (or update in place) the master Excel via the Composio CLI.

    Per-profile: each profile has its own folder_id and file_id under
    state[profile_name]. If file_id exists, UPDATE that file; else if
    folder_id exists, CREATE in that folder; else skip.
    """
    if not shutil.which("composio"):
        print("Composio CLI not on PATH; skipping Drive upload")
        return False

    state = load_state(state_file)
    profile_state = state.get(profile_name, {})
    folder_id = profile_state.get("folder_id")
    file_id = profile_state.get("file_id")

    args = {"metadata": {"name": local_path.name}}
    if file_id:
        args["file_id"] = file_id
    elif folder_id:
        args["folder_to_upload_to"] = folder_id
    else:
        print(f"No drive state for profile '{profile_name}'; skipping")
        return False

    cmd = [
        "composio", "execute", "GOOGLEDRIVE_RESUMABLE_UPLOAD",
        "--file", str(local_path),
        "-d", json.dumps(args),
    ]
    print(f"Uploading {local_path.name} to Google Drive (profile: {profile_name})...")
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if proc.returncode != 0:
        print(f"Upload failed (returncode={proc.returncode})")
        print(proc.stderr[-500:])
        return False

    # Save returned file_id on first-time upload
    try:
        out = proc.stdout
        idx = out.find("{")
        if idx >= 0:
            payload = json.loads(out[idx:])
            file_data = payload.get("data", {}).get("file", {})
            new_id = file_data.get("id")
            display_url = payload.get("data", {}).get("display_url")
            if new_id and not file_id:
                profile_state["file_id"] = new_id
                profile_state["file_url"] = display_url
                state[profile_name] = profile_state
                save_state(state_file, state)
                print(f"Saved new file_id for profile '{profile_name}': {new_id}")
            print(f"Drive: {display_url}")
    except Exception as e:
        print(f"Couldn't parse upload response (file is probably uploaded anyway): {e}")
    return True


# ---------- markdown doc sync (umbrella docs of the Vector) ----------

# These are the umbrella files at the-vector/ that get mirrored to Drive
# alongside the master Excel. Mobile-readable strategy + state docs.
# Top-level *.md only — specs/, applications/, and the inner job-search-tool/
# tree are dev-facing and stay local.

def upload_doc_to_drive(
    local_path: Path,
    profile_name: str,
    state_file: Path | str,
    force: bool = False,
) -> str:
    """Upload one markdown doc to the profile's Drive folder.

    Returns 'uploaded', 'skipped' (mtime-guard hit), 'failed' (composio error),
    or 'no-state' (no folder_id for this profile yet — run upload_to_drive first
    to seed it).

    State layout:
      state[profile]['docs'][filename] = {file_id, file_url, uploaded_at}

    The mtime guard compares the local file's mtime against the recorded
    uploaded_at. Local newer => upload, else skip. `force=True` bypasses.
    """
    local_path = Path(local_path)
    if not local_path.exists():
        return "failed"

    if not shutil.which("composio"):
        print("Composio CLI not on PATH; skipping doc upload", file=sys.stderr)
        return "failed"

    state = load_state(state_file)
    profile_state = state.get(profile_name, {})
    folder_id = profile_state.get("folder_id")
    docs = profile_state.get("docs", {})
    doc_state = docs.get(local_path.name, {})
    file_id = doc_state.get("file_id")
    uploaded_at = doc_state.get("uploaded_at", 0)

    # mtime guard — local file unchanged since last upload
    if not force:
        try:
            local_mtime = local_path.stat().st_mtime
        except OSError:
            return "failed"
        if local_mtime <= uploaded_at and file_id:
            return "skipped"

    args = {
        "metadata": {
            "name": local_path.name,
            "mime_type": "text/markdown",
        }
    }
    if file_id:
        args["file_id"] = file_id
    elif folder_id:
        args["folder_to_upload_to"] = folder_id
    else:
        return "no-state"

    cmd = [
        "composio", "execute", "GOOGLEDRIVE_RESUMABLE_UPLOAD",
        "--file", str(local_path),
        "-d", json.dumps(args),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if proc.returncode != 0:
        print(f"  doc upload failed: {local_path.name} (rc={proc.returncode})",
              file=sys.stderr)
        return "failed"

    # Update state with file_id (first time) and timestamp
    new_id = file_id
    new_url = doc_state.get("file_url", "")
    try:
        out = proc.stdout
        idx = out.find("{")
        if idx >= 0:
            payload = json.loads(out[idx:])
            file_data = payload.get("data", {}).get("file", {})
            returned_id = file_data.get("id")
            returned_url = payload.get("data", {}).get("display_url")
            if returned_id:
                new_id = returned_id
            if returned_url:
                new_url = returned_url
    except Exception:
        pass

    # Always update uploaded_at on a successful upload, even if we couldn't
    # parse the file_id from the response — re-parsing the local file's mtime
    # rather than time.time() so the next mtime guard compares apples to apples.
    try:
        new_uploaded_at = local_path.stat().st_mtime
    except OSError:
        new_uploaded_at = uploaded_at

    docs[local_path.name] = {
        "file_id": new_id,
        "file_url": new_url,
        "uploaded_at": new_uploaded_at,
    }
    profile_state["docs"] = docs
    state[profile_name] = profile_state
    save_state(state_file, state)
    return "uploaded"


def upload_vector_docs(
    profile_name: str,
    state_file: Path | str,
    docs_dir: Path | str,
    force: bool = False,
) -> dict:
    """Mirror every *.md in `docs_dir` (top-level only, no recursion) to Drive.

    Returns counts: {'uploaded': int, 'skipped': int, 'failed': int, 'no-state': int}.

    Sequential so we don't fan out subprocesses. The mtime guard makes
    repeat calls near-instant — the slow path is the first run only.
    """
    docs_dir = Path(docs_dir)
    counts = {"uploaded": 0, "skipped": 0, "failed": 0, "no-state": 0}
    if not docs_dir.exists():
        return counts

    md_files = sorted(p for p in docs_dir.glob("*.md") if p.is_file())
    if not md_files:
        return counts

    # First file gates the rest — if there's no folder_id seeded, nothing
    # else will land either. Bail fast with a clear message.
    first_status = upload_doc_to_drive(
        md_files[0], profile_name, state_file, force=force
    )
    counts[first_status] = counts.get(first_status, 0) + 1
    if first_status == "no-state":
        print(
            f"  No drive folder_id for profile '{profile_name}'; "
            "run upload_to_drive (e.g. via scrape/score/find_managers) first.",
            file=sys.stderr,
        )
        return counts

    for path in md_files[1:]:
        status = upload_doc_to_drive(path, profile_name, state_file, force=force)
        counts[status] = counts.get(status, 0) + 1

    return counts
