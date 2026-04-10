from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import IntEnum, StrEnum
from pathlib import Path
from typing import Literal

RepoType = Literal["model", "dataset", "space"]


class NodeCheckState(IntEnum):
    UNCHECKED = 0
    PARTIAL = 1
    CHECKED = 2


class JobStatus(StrEnum):
    PENDING = "Pending"
    RUNNING = "Running"
    PAUSED = "Paused"
    FAILED = "Failed"
    COMPLETED = "Completed"
    CANCELLED = "Cancelled"


class TaskStatus(StrEnum):
    PENDING = "Pending"
    RUNNING = "Running"
    PAUSED = "Paused"
    FAILED = "Failed"
    COMPLETED = "Completed"
    SKIPPED = "Skipped"
    CANCELLED = "Cancelled"


@dataclass(slots=True)
class RepoRef:
    repo_type: RepoType
    repo_id: str
    requested_revision: str | None = None
    pinned_sha: str | None = None
    deep_path: str | None = None

    @property
    def effective_revision(self) -> str | None:
        return self.pinned_sha or self.requested_revision

    def with_pinned_sha(self, sha: str | None) -> "RepoRef":
        return RepoRef(
            repo_type=self.repo_type,
            repo_id=self.repo_id,
            requested_revision=self.requested_revision,
            pinned_sha=sha,
            deep_path=self.deep_path,
        )


@dataclass(slots=True)
class SearchResult:
    repo_id: str
    repo_type: RepoType
    sha: str | None = None
    summary: str = ""
    likes: int | None = None
    downloads: int | None = None
    last_modified: datetime | None = None


@dataclass(slots=True)
class RevisionOption:
    label: str
    value: str | None
    kind: str


@dataclass(slots=True)
class RepoClassification:
    primary_label: str = "Repository"
    package_kind: str = "repo"
    modality_label: str = "General"
    detected_labels: list[str] = field(default_factory=list)
    summary: str = ""
    license_name: str = ""
    has_readme: bool = False


@dataclass(slots=True)
class RepoTreeNode:
    name: str
    path: str
    is_dir: bool
    size: int = 0
    file_count: int = 0
    check_state: NodeCheckState = NodeCheckState.UNCHECKED
    children: list["RepoTreeNode"] = field(default_factory=list)
    parent: "RepoTreeNode | None" = None

    def add_child(self, child: "RepoTreeNode") -> None:
        child.parent = self
        self.children.append(child)


@dataclass(slots=True)
class RepoDetails:
    repo: RepoRef
    title: str
    revision: str | None
    description: str
    root: RepoTreeNode
    total_size: int
    total_files: int
    preselected_paths: set[str] = field(default_factory=set)
    available_revisions: list[RevisionOption] = field(default_factory=list)
    classification: RepoClassification = field(default_factory=RepoClassification)


@dataclass(slots=True)
class DownloadTask:
    filename: str
    size: int
    local_path: Path
    will_download: bool
    status: TaskStatus = TaskStatus.PENDING
    downloaded_bytes: int = 0
    attempts: int = 0
    error: str = ""


@dataclass(slots=True)
class DownloadJob:
    job_id: str
    repo: RepoRef
    title: str
    destination: Path
    allow_patterns: list[str]
    tasks: list[DownloadTask]
    total_selected_bytes: int
    bytes_to_download: int
    cached_bytes: int
    worker_count: int
    retry_count: int
    status: JobStatus = JobStatus.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_error: str = ""
    runtime_token: str | bool | None = None


@dataclass(slots=True)
class ProgressSnapshot:
    job_id: str
    status: JobStatus
    total_bytes: int
    downloaded_bytes: int
    percent: float
    speed_bps: float
    eta_seconds: float | None
    completed_files: int
    failed_files: int
    skipped_files: int
    queued_files: int
    active_files: int
    cancelled_files: int


@dataclass(slots=True)
class AppSettings:
    default_download_root: str
    worker_count: int = 4
    retry_count: int = 3
    recent_repos: list[dict[str, str]] = field(default_factory=list)
    queue_history: list[dict[str, str]] = field(default_factory=list)
