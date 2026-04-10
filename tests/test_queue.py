from __future__ import annotations

import time
from pathlib import Path

from PyQt6 import QtCore

from hf_downloader.models import AppSettings, DownloadJob, DownloadTask, JobStatus, RepoRef, TaskStatus
from hf_downloader.queue_manager import QueueManager


def make_job(job_id: str, destination: Path) -> DownloadJob:
    return DownloadJob(
        job_id=job_id,
        repo=RepoRef(repo_type="model", repo_id=f"user/{job_id}", pinned_sha="sha"),
        title=f"user/{job_id}",
        destination=destination,
        allow_patterns=["a.bin"],
        tasks=[DownloadTask(filename="a.bin", size=10, local_path=destination / "a.bin", will_download=True)],
        total_selected_bytes=10,
        bytes_to_download=10,
        cached_bytes=0,
        worker_count=1,
        retry_count=1,
    )


def wait_for(condition, timeout: float = 5.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        QtCore.QCoreApplication.processEvents()
        if condition():
            return
        time.sleep(0.05)
    raise AssertionError("Timed out waiting for condition")


def test_queue_manager_runs_jobs_and_auto_starts_next(temp_settings_store, tmp_path: Path) -> None:
    calls: list[str] = []

    def fake_download(**kwargs):
        calls.append(kwargs["repo_id"])
        progress = kwargs["tqdm_class"](total=10, initial=0)
        progress.update(10)
        return str(tmp_path / kwargs["filename"])

    manager = QueueManager(fake_download, settings_store=temp_settings_store, settings=AppSettings(default_download_root=str(tmp_path)))
    job1 = make_job("job1", tmp_path / "job1")
    job2 = make_job("job2", tmp_path / "job2")
    manager.add_job(job1)
    manager.add_job(job2)

    manager.start_job()
    wait_for(lambda: job1.status == "Completed" and job2.status == "Completed")

    assert calls == ["user/job1", "user/job2"]


def test_queue_manager_retries_timeout_once(temp_settings_store, tmp_path: Path) -> None:
    attempts = {"count": 0}

    def flaky_download(**kwargs):
        attempts["count"] += 1
        progress = kwargs["tqdm_class"](total=10, initial=0)
        if attempts["count"] == 1:
            raise TimeoutError("temporary")
        progress.update(10)
        return str(tmp_path / kwargs["filename"])

    manager = QueueManager(flaky_download, settings_store=temp_settings_store, settings=AppSettings(default_download_root=str(tmp_path)))
    job = make_job("job1", tmp_path / "job1")
    manager.add_job(job)
    manager.start_job()

    wait_for(lambda: job.status in {"Completed", "Failed"})
    assert job.status == "Completed"
    assert attempts["count"] == 2
    assert job.tasks[0].status == TaskStatus.COMPLETED


def test_queue_manager_moves_jobs_to_change_priority(temp_settings_store, tmp_path: Path) -> None:
    manager = QueueManager(lambda **kwargs: None, settings_store=temp_settings_store, settings=AppSettings(default_download_root=str(tmp_path)))
    job1 = make_job("job1", tmp_path / "job1")
    job2 = make_job("job2", tmp_path / "job2")
    job3 = make_job("job3", tmp_path / "job3")
    manager.add_job(job1)
    manager.add_job(job2)
    manager.add_job(job3)

    moved = manager.move_job(2, 0)

    assert moved is True
    assert [job.job_id for job in manager.jobs] == ["job3", "job1", "job2"]


def test_queue_manager_deletes_pending_job(temp_settings_store, tmp_path: Path) -> None:
    manager = QueueManager(lambda **kwargs: None, settings_store=temp_settings_store, settings=AppSettings(default_download_root=str(tmp_path)))
    job1 = make_job("job1", tmp_path / "job1")
    job2 = make_job("job2", tmp_path / "job2")
    manager.add_job(job1)
    manager.add_job(job2)

    deleted = manager.delete_job(job1.job_id)

    assert deleted is True
    assert manager.get_job(job1.job_id) is None
    assert [job.job_id for job in manager.jobs] == ["job2"]


def test_queue_manager_deletes_running_job_after_cancel(temp_settings_store, tmp_path: Path) -> None:
    def slow_download(**kwargs):
        progress = kwargs["tqdm_class"](total=50, initial=0)
        for _ in range(50):
            progress.update(1)
            time.sleep(0.02)
        return str(tmp_path / kwargs["filename"])

    manager = QueueManager(slow_download, settings_store=temp_settings_store, settings=AppSettings(default_download_root=str(tmp_path)))
    job = make_job("job1", tmp_path / "job1")
    manager.add_job(job)
    manager.start_job()

    wait_for(lambda: job.status == JobStatus.RUNNING)
    deleted = manager.delete_job(job.job_id)

    assert deleted is True
    wait_for(lambda: manager.get_job(job.job_id) is None)


def test_queue_manager_clears_pending_queue(temp_settings_store, tmp_path: Path) -> None:
    manager = QueueManager(lambda **kwargs: None, settings_store=temp_settings_store, settings=AppSettings(default_download_root=str(tmp_path)))
    manager.add_job(make_job("job1", tmp_path / "job1"))
    manager.add_job(make_job("job2", tmp_path / "job2"))

    cleared = manager.clear_queue()

    assert cleared is True
    assert manager.jobs == []


def test_queue_manager_clears_active_queue_after_cancel(temp_settings_store, tmp_path: Path) -> None:
    def slow_download(**kwargs):
        progress = kwargs["tqdm_class"](total=50, initial=0)
        for _ in range(50):
            progress.update(1)
            time.sleep(0.02)
        return str(tmp_path / kwargs["filename"])

    manager = QueueManager(slow_download, settings_store=temp_settings_store, settings=AppSettings(default_download_root=str(tmp_path)))
    active_job = make_job("job1", tmp_path / "job1")
    queued_job = make_job("job2", tmp_path / "job2")
    manager.add_job(active_job)
    manager.add_job(queued_job)
    manager.start_job()

    wait_for(lambda: active_job.status == JobStatus.RUNNING)
    cleared = manager.clear_queue()

    assert cleared is True
    wait_for(lambda: manager.jobs == [])
