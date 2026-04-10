from __future__ import annotations

import json
from pathlib import Path

from platformdirs import user_data_dir, user_log_dir

from .models import AppSettings, DownloadJob, RepoRef

APP_NAME = "HF Repo Downloader"
APP_AUTHOR = "Codex"


class SettingsStore:
    def __init__(self) -> None:
        self.data_dir = Path(user_data_dir(APP_NAME, APP_AUTHOR))
        self.log_dir = Path(user_log_dir(APP_NAME, APP_AUTHOR))
        self.settings_path = self.data_dir / "settings.json"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def load(self) -> AppSettings:
        if not self.settings_path.exists():
            return self.default_settings()

        payload = json.loads(self.settings_path.read_text(encoding="utf-8"))
        return AppSettings(
            default_download_root=payload.get("default_download_root", str(Path.home() / "Downloads" / "hf-repos")),
            worker_count=int(payload.get("worker_count", 4)),
            retry_count=int(payload.get("retry_count", 3)),
            recent_repos=list(payload.get("recent_repos", [])),
            queue_history=list(payload.get("queue_history", [])),
        )

    def save(self, settings: AppSettings) -> None:
        payload = {
            "default_download_root": settings.default_download_root,
            "worker_count": settings.worker_count,
            "retry_count": settings.retry_count,
            "recent_repos": settings.recent_repos[-20:],
            "queue_history": settings.queue_history[-50:],
        }
        self.settings_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def record_recent_repo(self, settings: AppSettings, repo: RepoRef) -> None:
        entry = {
            "repo_type": repo.repo_type,
            "repo_id": repo.repo_id,
            "revision": repo.requested_revision or "",
        }
        settings.recent_repos = [item for item in settings.recent_repos if item != entry]
        settings.recent_repos.append(entry)
        self.save(settings)

    def record_job_history(self, settings: AppSettings, job: DownloadJob) -> None:
        settings.queue_history.append(
            {
                "job_id": job.job_id,
                "repo_id": job.repo.repo_id,
                "repo_type": job.repo.repo_type,
                "destination": str(job.destination),
                "status": job.status,
                "created_at": job.created_at.isoformat(),
            }
        )
        self.save(settings)

    def default_settings(self) -> AppSettings:
        return AppSettings(default_download_root=str(Path.home() / "Downloads" / "hf-repos"))

