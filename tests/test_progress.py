from hf_downloader.models import DownloadJob, DownloadTask, JobStatus, RepoRef
from hf_downloader.progress import ProgressTracker


def make_job() -> DownloadJob:
    return DownloadJob(
        job_id="job-1",
        repo=RepoRef(repo_type="model", repo_id="user/repo", pinned_sha="abc123"),
        title="user/repo",
        destination="unused",  # type: ignore[arg-type]
        allow_patterns=["weights.bin", "config.json"],
        tasks=[
            DownloadTask(filename="weights.bin", size=100, local_path="unused", will_download=True),  # type: ignore[arg-type]
            DownloadTask(filename="config.json", size=20, local_path="unused", will_download=False),  # type: ignore[arg-type]
        ],
        total_selected_bytes=120,
        bytes_to_download=100,
        cached_bytes=20,
        worker_count=4,
        retry_count=3,
    )


def test_progress_tracker_counts_cached_and_resumed_bytes() -> None:
    events = []
    job = make_job()
    tracker = ProgressTracker(job, lambda current_job, snapshot: events.append((current_job.job_id, snapshot)))
    tracker.set_job_status(JobStatus.RUNNING)

    task = job.tasks[0]
    tracker.start_task(task, total=100, initial=25)
    tracker.advance(task, 25)
    tracker.finish_task(task)
    snapshot = tracker.snapshot()

    assert snapshot.total_bytes == 120
    assert snapshot.downloaded_bytes == 120
    assert snapshot.completed_files == 1
    assert snapshot.skipped_files == 1
    assert snapshot.percent == 100.0
    assert events
