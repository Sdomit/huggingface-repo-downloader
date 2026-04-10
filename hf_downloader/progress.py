from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable

from .models import DownloadJob, DownloadTask, JobStatus, ProgressSnapshot, TaskStatus


class CancelledDownload(RuntimeError):
    """Raised when a job is cancelled."""


class JobGate:
    def __init__(self) -> None:
        self._resume_event = threading.Event()
        self._resume_event.set()
        self._cancel_event = threading.Event()

    def pause(self) -> None:
        self._resume_event.clear()

    def resume(self) -> None:
        self._resume_event.set()

    def cancel(self) -> None:
        self._cancel_event.set()
        self._resume_event.set()

    def checkpoint(self) -> None:
        while not self._resume_event.wait(timeout=0.1):
            if self._cancel_event.is_set():
                raise CancelledDownload()
        if self._cancel_event.is_set():
            raise CancelledDownload()

    def sleep(self, seconds: float) -> None:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            self.checkpoint()
            time.sleep(min(0.1, deadline - time.monotonic()))

    @property
    def cancelled(self) -> bool:
        return self._cancel_event.is_set()

    @property
    def paused(self) -> bool:
        return not self._resume_event.is_set()


@dataclass(slots=True)
class _SpeedEvent:
    timestamp: float
    delta: int


class ProgressTracker:
    def __init__(self, job: DownloadJob, on_update: Callable[[DownloadJob, ProgressSnapshot], None]) -> None:
        self.job = job
        self.on_update = on_update
        self._lock = threading.Lock()
        self._speed_events: deque[_SpeedEvent] = deque()

        for task in self.job.tasks:
            if task.will_download:
                task.status = TaskStatus.PENDING
                task.downloaded_bytes = 0
            else:
                task.status = TaskStatus.SKIPPED
                task.downloaded_bytes = task.size

        self._emit()

    def set_job_status(self, status: JobStatus) -> None:
        with self._lock:
            self.job.status = status
        self._emit()

    def start_task(self, task: DownloadTask, total: int | None, initial: int = 0) -> None:
        with self._lock:
            task.status = TaskStatus.RUNNING
            if total is not None and total > 0:
                task.size = max(task.size, int(total))
            initial_value = max(0, min(int(initial), task.size))
            if initial_value > task.downloaded_bytes:
                delta = initial_value - task.downloaded_bytes
                task.downloaded_bytes = initial_value
                self._record_speed(delta)
        self._emit()

    def advance(self, task: DownloadTask, delta: int | float | None) -> None:
        increment = int(delta or 0)
        if increment <= 0:
            return
        with self._lock:
            new_value = min(task.size, task.downloaded_bytes + increment)
            actual_increment = new_value - task.downloaded_bytes
            task.downloaded_bytes = new_value
            if actual_increment > 0:
                self._record_speed(actual_increment)
        self._emit()

    def finish_task(self, task: DownloadTask) -> None:
        with self._lock:
            if task.downloaded_bytes < task.size:
                delta = task.size - task.downloaded_bytes
                task.downloaded_bytes = task.size
                self._record_speed(delta)
            task.status = TaskStatus.COMPLETED
            task.error = ""
        self._emit()

    def fail_task(self, task: DownloadTask, message: str) -> None:
        with self._lock:
            task.status = TaskStatus.FAILED
            task.error = message
        self._emit()

    def cancel_task(self, task: DownloadTask) -> None:
        with self._lock:
            if task.downloaded_bytes >= task.size:
                task.status = TaskStatus.COMPLETED
            else:
                task.status = TaskStatus.CANCELLED
        self._emit()

    def mark_pending(self, task: DownloadTask) -> None:
        with self._lock:
            if task.downloaded_bytes >= task.size:
                task.status = TaskStatus.COMPLETED
            else:
                task.status = TaskStatus.PENDING
        self._emit()

    def snapshot(self) -> ProgressSnapshot:
        with self._lock:
            total_bytes = sum(task.size for task in self.job.tasks)
            downloaded_bytes = sum(task.downloaded_bytes for task in self.job.tasks)
            completed_files = sum(task.status == TaskStatus.COMPLETED for task in self.job.tasks)
            failed_files = sum(task.status == TaskStatus.FAILED for task in self.job.tasks)
            skipped_files = sum(task.status == TaskStatus.SKIPPED for task in self.job.tasks)
            queued_files = sum(task.status == TaskStatus.PENDING for task in self.job.tasks)
            active_files = sum(task.status == TaskStatus.RUNNING for task in self.job.tasks)
            cancelled_files = sum(task.status == TaskStatus.CANCELLED for task in self.job.tasks)
            speed_bps = self._speed()
            percent = 100.0 if total_bytes == 0 else (downloaded_bytes / total_bytes) * 100
            remaining = max(0, total_bytes - downloaded_bytes)
            eta_seconds = None if speed_bps <= 0 or remaining == 0 else remaining / speed_bps
            return ProgressSnapshot(
                job_id=self.job.job_id,
                status=self.job.status,
                total_bytes=total_bytes,
                downloaded_bytes=downloaded_bytes,
                percent=percent,
                speed_bps=speed_bps,
                eta_seconds=eta_seconds,
                completed_files=completed_files,
                failed_files=failed_files,
                skipped_files=skipped_files,
                queued_files=queued_files,
                active_files=active_files,
                cancelled_files=cancelled_files,
            )

    def _emit(self) -> None:
        self.on_update(self.job, self.snapshot())

    def _record_speed(self, delta: int) -> None:
        now = time.monotonic()
        self._speed_events.append(_SpeedEvent(timestamp=now, delta=delta))
        self._trim_speed_events(now)

    def _speed(self) -> float:
        now = time.monotonic()
        self._trim_speed_events(now)
        if not self._speed_events:
            return 0.0
        window = max(1.0, now - self._speed_events[0].timestamp)
        transferred = sum(event.delta for event in self._speed_events)
        return transferred / window

    def _trim_speed_events(self, now: float) -> None:
        while self._speed_events and now - self._speed_events[0].timestamp > 5.0:
            self._speed_events.popleft()
