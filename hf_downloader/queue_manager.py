from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx
from PyQt6 import QtCore
from huggingface_hub.errors import HfHubHTTPError

from .models import DownloadJob, DownloadTask, JobStatus, ProgressSnapshot, TaskStatus
from .progress import CancelledDownload, JobGate, ProgressTracker
from .settings import SettingsStore

logger = logging.getLogger("hf_downloader")


def _is_retryable(error: Exception) -> bool:
    if isinstance(error, TimeoutError):
        return True
    if isinstance(error, httpx.TimeoutException):
        return True
    if isinstance(error, HfHubHTTPError):
        status_code = getattr(getattr(error, "response", None), "status_code", None)
        return status_code == 429 or (status_code is not None and status_code >= 500)
    return isinstance(error, httpx.TransportError)


def make_qt_tqdm_bridge(tracker: ProgressTracker, gate: JobGate, task: DownloadTask):
    class QtTqdmBridge:
        def __init__(self, *args, **kwargs) -> None:
            total = kwargs.pop("total", None)
            initial = kwargs.pop("initial", 0)
            gate.checkpoint()
            tracker.start_task(task, total, initial)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback) -> bool:
            return False

        def update(self, amount: int | float | None = 1) -> None:
            gate.checkpoint()
            tracker.advance(task, amount)

        def refresh(self) -> None:
            return

        def close(self) -> None:
            return

        def set_description(self, *_args, **_kwargs) -> None:
            return

        def set_postfix(self, *_args, **_kwargs) -> None:
            return

    return QtTqdmBridge


class QueueManager(QtCore.QObject):
    job_updated = QtCore.pyqtSignal(object, object)
    queue_changed = QtCore.pyqtSignal(object)
    message_emitted = QtCore.pyqtSignal(str)

    def __init__(self, download_func, settings_store: SettingsStore, settings, parent=None) -> None:
        super().__init__(parent)
        self.download_func = download_func
        self.settings_store = settings_store
        self.settings = settings
        self.jobs: list[DownloadJob] = []
        self._pending_delete_job_ids: set[str] = set()
        self._active_job_id: str | None = None
        self._runner_thread: threading.Thread | None = None
        self._gate: JobGate | None = None
        self._lock = threading.Lock()

    def add_job(self, job: DownloadJob) -> None:
        self.jobs.append(job)
        self.queue_changed.emit(self.jobs)
        self._emit_job_update(job)

    def start_job(self, job_id: str | None = None) -> None:
        with self._lock:
            if self._runner_thread and self._runner_thread.is_alive():
                return
            job = self._select_job_to_start(job_id)
            if job is None:
                return
            self._runner_thread = threading.Thread(target=self._run_job, args=(job,), daemon=True)
            self._runner_thread.start()

    def pause_active(self) -> None:
        gate = self._gate
        job = self.active_job
        if gate is None or job is None or job.status != JobStatus.RUNNING:
            return
        gate.pause()
        job.status = JobStatus.PAUSED
        for task in job.tasks:
            if task.status == TaskStatus.RUNNING:
                task.status = TaskStatus.PAUSED
        self._emit_job_update(job)

    def resume_job(self, job_id: str) -> None:
        job = self.get_job(job_id)
        if job is None:
            return

        if job.job_id == self._active_job_id and self._gate is not None and job.status == JobStatus.PAUSED:
            self._gate.resume()
            job.status = JobStatus.RUNNING
            for task in job.tasks:
                if task.status == TaskStatus.PAUSED:
                    task.status = TaskStatus.RUNNING
            self._emit_job_update(job)
            return

        if job.status in {JobStatus.FAILED, JobStatus.CANCELLED, JobStatus.PAUSED}:
            for task in job.tasks:
                if task.status in {TaskStatus.FAILED, TaskStatus.CANCELLED, TaskStatus.PAUSED, TaskStatus.RUNNING}:
                    if task.downloaded_bytes >= task.size:
                        task.status = TaskStatus.COMPLETED
                    else:
                        task.status = TaskStatus.PENDING
                        task.error = ""
            job.status = JobStatus.PENDING
            job.last_error = ""
            self._emit_job_update(job)
            self.start_job(job.job_id)

    def cancel_job(self, job_id: str) -> None:
        job = self.get_job(job_id)
        if job is None:
            return
        if job.job_id == self._active_job_id and self._gate is not None:
            self._gate.cancel()
            job.status = JobStatus.CANCELLED
            self._emit_job_update(job)
            return
        if job.status == JobStatus.PENDING:
            job.status = JobStatus.CANCELLED
            for task in job.tasks:
                if task.status == TaskStatus.PENDING:
                    task.status = TaskStatus.CANCELLED
            self._emit_job_update(job)

    def delete_job(self, job_id: str) -> bool:
        job = self.get_job(job_id)
        if job is None:
            return False

        if job.job_id == self._active_job_id and self._gate is not None:
            self._pending_delete_job_ids.add(job_id)
            self._gate.cancel()
            job.status = JobStatus.CANCELLED
            self._emit_job_update(job)
            self.message_emitted.emit(f"Deleting {job.title} after it stops.")
            return True

        self._remove_job(job_id)
        self.message_emitted.emit(f"Deleted {job.title} from the queue.")
        return True

    def move_job(self, source_index: int, target_index: int) -> bool:
        if not (0 <= source_index < len(self.jobs)):
            return False
        if not (0 <= target_index <= len(self.jobs)):
            return False

        source_job = self.jobs[source_index]
        if source_job.job_id == self._active_job_id:
            self.message_emitted.emit("The active job cannot be moved.")
            return False

        moved_job = self.jobs.pop(source_index)
        if source_index < target_index:
            target_index -= 1
        if target_index < 0:
            target_index = 0
        if target_index > len(self.jobs):
            target_index = len(self.jobs)
        self.jobs.insert(target_index, moved_job)
        self.queue_changed.emit(self.jobs)
        return True

    def clear_queue(self) -> bool:
        if not self.jobs:
            return False

        active_job_id = self._active_job_id
        active_job = self.active_job
        changed = False

        if active_job is not None and self._gate is not None:
            self._pending_delete_job_ids.add(active_job_id)
            self._gate.cancel()
            active_job.status = JobStatus.CANCELLED
            self._emit_job_update(active_job)
            changed = True

        if active_job_id is not None:
            remaining_jobs = [job for job in self.jobs if job.job_id == active_job_id]
        else:
            remaining_jobs = []

        if len(remaining_jobs) != len(self.jobs):
            self.jobs = remaining_jobs
            self.queue_changed.emit(self.jobs)
            changed = True

        if changed:
            if active_job is not None:
                self.message_emitted.emit("Clearing queue. The active job will be removed after it stops.")
            else:
                self.message_emitted.emit("Cleared the queue.")
        return changed

    def retry_failed(self, job_id: str) -> None:
        job = self.get_job(job_id)
        if job is None:
            return
        for task in job.tasks:
            if task.status in {TaskStatus.FAILED, TaskStatus.CANCELLED} and task.downloaded_bytes < task.size:
                task.status = TaskStatus.PENDING
                task.error = ""
        job.status = JobStatus.PENDING
        job.last_error = ""
        self._emit_job_update(job)

    def get_job(self, job_id: str) -> DownloadJob | None:
        return next((job for job in self.jobs if job.job_id == job_id), None)

    @property
    def active_job(self) -> DownloadJob | None:
        if self._active_job_id is None:
            return None
        return self.get_job(self._active_job_id)

    def _select_job_to_start(self, requested_id: str | None) -> DownloadJob | None:
        if requested_id:
            job = self.get_job(requested_id)
            if job is not None and job.status == JobStatus.PENDING:
                return job
        return next((job for job in self.jobs if job.status == JobStatus.PENDING), None)

    def _run_job(self, job: DownloadJob) -> None:
        self._active_job_id = job.job_id
        self._gate = JobGate()
        tracker = ProgressTracker(job, self._emit_job_update)
        tracker.set_job_status(JobStatus.RUNNING)

        pending_tasks = [task for task in job.tasks if task.status == TaskStatus.PENDING]
        if not pending_tasks:
            tracker.set_job_status(JobStatus.COMPLETED)
            self._finish_job(job)
            return

        with ThreadPoolExecutor(max_workers=job.worker_count) as executor:
            futures = [executor.submit(self._download_task, job, task, tracker, self._gate) for task in pending_tasks]
            for future in as_completed(futures):
                try:
                    future.result()
                except CancelledDownload:
                    continue
                except Exception as exc:  # pragma: no cover - safety net
                    logger.exception("Unexpected task failure: %s", exc)
                    job.last_error = str(exc)

        if self._gate.cancelled:
            job.status = JobStatus.CANCELLED
        elif any(task.status == TaskStatus.FAILED for task in job.tasks):
            job.status = JobStatus.FAILED
            job.last_error = next((task.error for task in job.tasks if task.error), job.last_error)
        else:
            job.status = JobStatus.COMPLETED

        self._emit_job_update(job)
        self._finish_job(job)

    def _download_task(self, job: DownloadJob, task: DownloadTask, tracker: ProgressTracker, gate: JobGate) -> None:
        for attempt in range(job.retry_count + 1):
            try:
                gate.checkpoint()
                task.attempts = attempt + 1
                bridge = make_qt_tqdm_bridge(tracker, gate, task)
                self.download_func(
                    repo_id=job.repo.repo_id,
                    filename=task.filename,
                    repo_type=job.repo.repo_type,
                    revision=job.repo.effective_revision,
                    local_dir=job.destination,
                    token=job.runtime_token,
                    tqdm_class=bridge,
                )
                tracker.finish_task(task)
                return
            except CancelledDownload:
                tracker.cancel_task(task)
                raise
            except Exception as exc:
                if gate.cancelled:
                    tracker.cancel_task(task)
                    raise CancelledDownload() from exc
                if attempt < job.retry_count and _is_retryable(exc):
                    tracker.mark_pending(task)
                    gate.sleep(2**attempt)
                    continue
                task.error = str(exc)
                tracker.fail_task(task, str(exc))
                return

    def _finish_job(self, job: DownloadJob) -> None:
        delete_after_finish = job.job_id in self._pending_delete_job_ids
        if delete_after_finish:
            self._pending_delete_job_ids.discard(job.job_id)

        self.settings_store.record_job_history(self.settings, job)
        self._active_job_id = None
        self._gate = None
        self._runner_thread = None
        if delete_after_finish:
            self._remove_job(job.job_id)
        else:
            self.queue_changed.emit(self.jobs)
        next_job = next((candidate for candidate in self.jobs if candidate.status == JobStatus.PENDING), None)
        if next_job is not None:
            self.start_job(next_job.job_id)

    def _emit_job_update(self, job: DownloadJob, snapshot: ProgressSnapshot | None = None) -> None:
        if snapshot is None:
            total = sum(task.size for task in job.tasks)
            downloaded = sum(task.downloaded_bytes for task in job.tasks)
            percent = 100.0 if total == 0 else (downloaded / total) * 100
            snapshot = ProgressSnapshot(
                job_id=job.job_id,
                status=job.status,
                total_bytes=total,
                downloaded_bytes=downloaded,
                percent=percent,
                speed_bps=0.0,
                eta_seconds=None,
                completed_files=sum(task.status == TaskStatus.COMPLETED for task in job.tasks),
                failed_files=sum(task.status == TaskStatus.FAILED for task in job.tasks),
                skipped_files=sum(task.status == TaskStatus.SKIPPED for task in job.tasks),
                queued_files=sum(task.status == TaskStatus.PENDING for task in job.tasks),
                active_files=sum(task.status in {TaskStatus.RUNNING, TaskStatus.PAUSED} for task in job.tasks),
                cancelled_files=sum(task.status == TaskStatus.CANCELLED for task in job.tasks),
            )
        self.job_updated.emit(job, snapshot)

    def _remove_job(self, job_id: str) -> None:
        self.jobs = [job for job in self.jobs if job.job_id != job_id]
        self.queue_changed.emit(self.jobs)
