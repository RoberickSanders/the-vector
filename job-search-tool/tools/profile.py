"""Profile loader and schema validation for job-search-tool."""
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field


class Identity(BaseModel):
    email: str
    signature_block: str
    resume_path: str
    cover_letter_path: str
    linkedin: str


class LocationConfig(BaseModel):
    location: Optional[str] = None
    is_remote: Optional[bool] = None


class SearchConfig(BaseModel):
    terms: list[str]
    locations: list[LocationConfig]
    exclude_terms: list[str] = Field(default_factory=list)
    min_salary: int = 0
    hours_old: int = 72


class ScoringConfig(BaseModel):
    llm_model: str = "claude-sonnet-4-6"
    rubric_path: str
    threshold_for_outreach: int = 7


class DriveConfig(BaseModel):
    folder_name: str
    master_filename: str


class OutreachConfig(BaseModel):
    mailbox_pool: list[str] = Field(default_factory=list)
    daily_send_limit: int = 5
    signature_path: str = ""


class Profile(BaseModel):
    name: str
    identity: Identity
    search: SearchConfig
    scoring: ScoringConfig
    drive: DriveConfig
    outreach: OutreachConfig = Field(default_factory=OutreachConfig)


def load_profile(path: Path | str) -> Profile:
    """Load a profile YAML file and validate against the schema.

    Raises pydantic.ValidationError on bad input.
    """
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    return Profile(**raw)
