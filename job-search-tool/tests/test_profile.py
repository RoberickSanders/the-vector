from pathlib import Path
import pytest
from tools.profile import load_profile, Profile

FIXTURES = Path(__file__).parent / "fixtures"


def test_load_sample_profile_returns_profile_object():
    profile = load_profile(FIXTURES / "sample-profile.yaml")
    assert isinstance(profile, Profile)
    assert profile.name == "Test User"


def test_profile_has_required_search_fields():
    profile = load_profile(FIXTURES / "sample-profile.yaml")
    assert profile.search.terms == ["GTM Engineer", "RevOps Engineer"]
    assert len(profile.search.locations) == 2


def test_profile_locations_have_correct_shape():
    profile = load_profile(FIXTURES / "sample-profile.yaml")
    loc = profile.search.locations[0]
    assert loc.location is None
    assert loc.is_remote is True


def test_profile_drive_filename_uses_profile_name():
    profile = load_profile(FIXTURES / "sample-profile.yaml")
    assert profile.drive.master_filename == "test-jobs-master.xlsx"


def test_load_profile_raises_on_missing_required_field():
    bad_path = FIXTURES / "incomplete.yaml"
    bad_path.write_text("name: only_name\n")
    with pytest.raises(Exception):  # pydantic ValidationError
        load_profile(bad_path)
    bad_path.unlink()
