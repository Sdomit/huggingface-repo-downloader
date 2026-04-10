from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from shutil import _ntuple_diskusage

import pytest

from hf_downloader.auth import AuthResolver
from hf_downloader.hf_service import DiskSpaceError, HuggingFaceService
from hf_downloader.models import RepoRef


@dataclass
class FakeInfo:
    id: str
    sha: str
    description: str = ""


@dataclass
class FakeFile:
    path: str
    size: int


@dataclass
class FakeDryRun:
    filename: str
    file_size: int
    local_path: str
    is_cached: bool
    will_download: bool
    commit_hash: str = "sha"


class FakeApi:
    def auth_check(self, repo_id, repo_type=None, token=None, write=False):
        return None

    def repo_info(self, repo_id, repo_type=None, revision=None, token=None):
        return FakeInfo(id=repo_id, sha="resolved-sha", description="test repo")

    def list_repo_tree(self, repo_id, recursive=False, expand=False, revision=None, repo_type=None, token=None):
        return [FakeFile(path="folder/a.bin", size=10), FakeFile(path="folder/b.json", size=5)]

    def list_models(self, search=None, limit=None, full=False, token=None):
        return []

    def list_datasets(self, search=None, limit=None, full=False, token=None):
        return []

    def list_spaces(self, search=None, limit=None, full=False, token=None):
        return []


def test_load_repo_details_resolves_sha_and_preselects_deep_path(tmp_path: Path) -> None:
    service = HuggingFaceService(api=FakeApi(), auth_resolver=AuthResolver())
    details = service.load_repo_details(
        RepoRef(repo_type="model", repo_id="user/repo", requested_revision="main", deep_path="folder"),
        None,
    )

    assert details.repo.pinned_sha == "resolved-sha"
    assert details.total_files == 2
    assert details.preselected_paths == {"folder/a.bin", "folder/b.json"}
    assert details.classification.primary_label == "Single-File Checkpoint"


def test_plan_download_builds_tasks_and_checks_disk_space(tmp_path: Path) -> None:
    service = HuggingFaceService(
        api=FakeApi(),
        auth_resolver=AuthResolver(),
        snapshot_func=lambda **kwargs: [
            FakeDryRun("folder/a.bin", 10, str(tmp_path / "folder" / "a.bin"), False, True),
            FakeDryRun("folder/b.json", 5, str(tmp_path / "folder" / "b.json"), True, False),
        ],
    )

    job = service.plan_download(
        repo=RepoRef(repo_type="model", repo_id="user/repo", pinned_sha="resolved-sha"),
        title="user/repo",
        selected_files=["folder/a.bin", "folder/b.json"],
        destination=tmp_path,
        session_token=None,
        worker_count=4,
        retry_count=3,
    )

    assert len(job.tasks) == 2
    assert job.total_selected_bytes == 15
    assert job.bytes_to_download == 10
    assert job.cached_bytes == 5


def test_plan_download_raises_on_insufficient_disk(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    service = HuggingFaceService(
        api=FakeApi(),
        auth_resolver=AuthResolver(),
        snapshot_func=lambda **kwargs: [FakeDryRun("folder/a.bin", 1024, str(tmp_path / "folder" / "a.bin"), False, True)],
    )
    monkeypatch.setattr("hf_downloader.hf_service.shutil.disk_usage", lambda path: _ntuple_diskusage(0, 0, 1))

    with pytest.raises(DiskSpaceError):
        service.plan_download(
            repo=RepoRef(repo_type="model", repo_id="user/repo", pinned_sha="resolved-sha"),
            title="user/repo",
            selected_files=["folder/a.bin"],
            destination=tmp_path,
            session_token=None,
            worker_count=4,
            retry_count=3,
        )
