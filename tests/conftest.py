from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class TempSettingsStore:
    def __init__(self, root: Path) -> None:
        self.data_dir = root / "data"
        self.log_dir = root / "logs"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def save(self, settings) -> None:
        return

    def record_recent_repo(self, settings, repo) -> None:
        settings.recent_repos.append({"repo_type": repo.repo_type, "repo_id": repo.repo_id})

    def record_job_history(self, settings, job) -> None:
        settings.queue_history.append({"job_id": job.job_id, "status": job.status})


@pytest.fixture
def temp_settings_store(tmp_path: Path) -> TempSettingsStore:
    return TempSettingsStore(tmp_path)
