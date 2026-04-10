from __future__ import annotations

import logging
import math
import shutil
from pathlib import Path

import httpx
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError

from .auth import AuthResolver, ResolvedToken
from .models import DownloadJob, DownloadTask, RepoDetails, RepoRef, RepoTreeNode, RevisionOption, SearchResult
from .repo_analysis import classify_repo
from .tree_ops import build_tree_from_paths

logger = logging.getLogger("hf_downloader")

SAFETY_MARGIN_BYTES = 512 * 1024 * 1024


class HubServiceError(RuntimeError):
    """User-facing error raised by the service layer."""


class DiskSpaceError(HubServiceError):
    """Raised when the target disk does not have enough space."""


class HuggingFaceService:
    def __init__(
        self,
        api: HfApi | None = None,
        auth_resolver: AuthResolver | None = None,
        download_func=hf_hub_download,
        snapshot_func=snapshot_download,
    ) -> None:
        self.api = api or HfApi()
        self.auth_resolver = auth_resolver or AuthResolver()
        self.download_func = download_func
        self.snapshot_func = snapshot_func

    def resolve_token(self, session_token: str | None = None) -> ResolvedToken:
        return self.auth_resolver.resolve(session_token)

    def search_repos(self, query: str, scope: str, session_token: str | None = None, limit: int = 25) -> list[SearchResult]:
        query = query.strip()
        if not query:
            return []

        resolved = self.resolve_token(session_token)
        results: list[SearchResult] = []
        include_models = scope in {"all", "model"}
        include_datasets = scope in {"all", "dataset"}
        include_spaces = scope in {"all", "space"}
        per_scope_limit = max(1, limit if scope != "all" else math.ceil(limit / 3))

        if include_models:
            for item in self.api.list_models(search=query, limit=per_scope_limit, full=False, token=resolved.token):
                results.append(
                    SearchResult(
                        repo_id=item.id,
                        repo_type="model",
                        sha=getattr(item, "sha", None),
                        summary=(getattr(item, "cardData", None) or {}).get("summary", "") if getattr(item, "cardData", None) else "",
                        likes=getattr(item, "likes", None),
                        downloads=getattr(item, "downloads", None),
                        last_modified=getattr(item, "lastModified", None),
                    )
                )

        if include_datasets:
            for item in self.api.list_datasets(search=query, limit=per_scope_limit, full=False, token=resolved.token):
                results.append(
                    SearchResult(
                        repo_id=item.id,
                        repo_type="dataset",
                        sha=getattr(item, "sha", None),
                        summary=getattr(item, "description", "") or "",
                        likes=getattr(item, "likes", None),
                        downloads=getattr(item, "downloads", None),
                        last_modified=getattr(item, "lastModified", None),
                    )
                )

        if include_spaces:
            for item in self.api.list_spaces(search=query, limit=per_scope_limit, full=False, token=resolved.token):
                results.append(
                    SearchResult(
                        repo_id=item.id,
                        repo_type="space",
                        sha=getattr(item, "sha", None),
                        summary=getattr(item, "sdk", "") or "",
                        likes=getattr(item, "likes", None),
                        downloads=None,
                        last_modified=getattr(item, "lastModified", None),
                    )
                )

        return sorted(results, key=lambda result: (result.repo_type, result.repo_id.lower()))[:limit]

    def load_repo_details(self, repo: RepoRef, session_token: str | None = None) -> RepoDetails:
        resolved = self.resolve_token(session_token)
        try:
            self.api.auth_check(repo.repo_id, repo_type=repo.repo_type, token=resolved.token)
            info = self.api.repo_info(
                repo.repo_id,
                repo_type=repo.repo_type,
                revision=repo.requested_revision,
                token=resolved.token,
            )
        except GatedRepoError as exc:
            raise HubServiceError("This repository is gated. Add a valid token to continue.") from exc
        except RepositoryNotFoundError as exc:
            raise HubServiceError("Repository not found or access denied.") from exc
        except HfHubHTTPError as exc:
            raise HubServiceError(self._format_http_error(exc)) from exc
        except httpx.HTTPError as exc:
            raise HubServiceError(f"Network error while opening repository: {exc}") from exc

        try:
            entries = list(
                self.api.list_repo_tree(
                    repo.repo_id,
                    recursive=True,
                    expand=True,
                    revision=info.sha or repo.requested_revision,
                    repo_type=repo.repo_type,
                    token=resolved.token,
                )
            )
        except HfHubHTTPError as exc:
            raise HubServiceError(self._format_http_error(exc)) from exc
        except httpx.HTTPError as exc:
            raise HubServiceError(f"Network error while loading file tree: {exc}") from exc

        files: list[tuple[str, int]] = []
        for entry in entries:
            path = getattr(entry, "path", "")
            size = getattr(entry, "size", None)
            if path and size is not None:
                files.append((path, size))

        root = build_tree_from_paths(files)
        pinned = repo.with_pinned_sha(info.sha)
        preselected_paths = self._expand_preselected_paths(root, repo.deep_path)
        available_revisions = self._load_revision_options(repo, resolved)
        classification = classify_repo(pinned, info, [path for path, _size in files])
        description = getattr(info, "description", "") or getattr(info, "sha", "") or ""
        return RepoDetails(
            repo=pinned,
            title=repo.repo_id,
            revision=info.sha or repo.requested_revision,
            description=description,
            root=root,
            total_size=root.size,
            total_files=root.file_count,
            preselected_paths=preselected_paths,
            available_revisions=available_revisions,
            classification=classification,
        )

    def plan_download(
        self,
        repo: RepoRef,
        title: str,
        selected_files: list[str],
        destination: Path,
        session_token: str | None,
        worker_count: int,
        retry_count: int,
    ) -> DownloadJob:
        if not selected_files:
            raise HubServiceError("Select at least one file or folder before adding a job.")

        resolved = self.resolve_token(session_token)
        destination.mkdir(parents=True, exist_ok=True)
        try:
            dry_run_results = self.snapshot_func(
                repo_id=repo.repo_id,
                repo_type=repo.repo_type,
                revision=repo.effective_revision,
                local_dir=destination,
                allow_patterns=sorted(set(selected_files)),
                max_workers=worker_count,
                token=resolved.token,
                dry_run=True,
            )
        except GatedRepoError as exc:
            raise HubServiceError("This repository is gated. Add a valid token to download it.") from exc
        except RepositoryNotFoundError as exc:
            raise HubServiceError("Repository not found or access denied.") from exc
        except HfHubHTTPError as exc:
            raise HubServiceError(self._format_http_error(exc)) from exc
        except httpx.HTTPError as exc:
            raise HubServiceError(f"Network error while planning download: {exc}") from exc

        tasks: list[DownloadTask] = []
        total_selected_bytes = 0
        cached_bytes = 0
        bytes_to_download = 0

        for item in dry_run_results:
            local_path = Path(item.local_path) if item.local_path else destination / item.filename
            task = DownloadTask(
                filename=item.filename,
                size=int(item.file_size or 0),
                local_path=local_path,
                will_download=bool(item.will_download),
            )
            total_selected_bytes += task.size
            if task.will_download:
                bytes_to_download += task.size
            else:
                cached_bytes += task.size
            tasks.append(task)

        self.ensure_disk_capacity(destination, bytes_to_download)

        from uuid import uuid4

        return DownloadJob(
            job_id=str(uuid4()),
            repo=repo,
            title=title,
            destination=destination,
            allow_patterns=sorted(set(selected_files)),
            tasks=tasks,
            total_selected_bytes=total_selected_bytes,
            bytes_to_download=bytes_to_download,
            cached_bytes=cached_bytes,
            worker_count=worker_count,
            retry_count=retry_count,
            runtime_token=resolved.token,
        )

    def ensure_disk_capacity(self, destination: Path, bytes_to_download: int) -> None:
        target = destination if destination.exists() else destination.parent
        usage = shutil.disk_usage(target)
        margin = max(SAFETY_MARGIN_BYTES, int(bytes_to_download * 0.05))
        required = bytes_to_download + margin
        if usage.free < required:
            raise DiskSpaceError(
                "Not enough free disk space. "
                f"Need about {required:,} bytes including safety margin, have {usage.free:,} bytes."
            )

    def _expand_preselected_paths(self, root: RepoTreeNode, deep_path: str | None) -> set[str]:
        if not deep_path:
            return set()

        normalized = deep_path.strip("/")
        if not normalized:
            return set()

        node = self._find_node(root, normalized)
        if node is None:
            return set()

        if not node.is_dir:
            return {node.path}

        selected = set()
        self._collect_descendant_files(node, selected)
        return selected

    def _collect_descendant_files(self, node: RepoTreeNode, output: set[str]) -> None:
        if not node.is_dir:
            output.add(node.path)
            return
        for child in node.children:
            self._collect_descendant_files(child, output)

    def _find_node(self, node: RepoTreeNode, path: str) -> RepoTreeNode | None:
        if node.path == path:
            return node
        for child in node.children:
            found = self._find_node(child, path)
            if found is not None:
                return found
        return None

    def _format_http_error(self, exc: HfHubHTTPError) -> str:
        status_code = getattr(getattr(exc, "response", None), "status_code", None)
        if status_code == 401:
            return "Authentication failed. Check the Hugging Face token."
        if status_code == 403:
            return "Access denied for this repository."
        if status_code == 404:
            return "Repository not found."
        if status_code == 429:
            return "Hugging Face rate limit reached. Try again shortly."
        if status_code and status_code >= 500:
            return "Hugging Face is currently unavailable. Try again later."
        return f"Hub request failed: {exc}"

    def _load_revision_options(self, repo: RepoRef, resolved: ResolvedToken) -> list[RevisionOption]:
        options: list[RevisionOption] = [RevisionOption(label="Latest / Default", value=None, kind="default")]
        seen: set[str | None] = {None}

        try:
            refs = self.api.list_repo_refs(repo.repo_id, repo_type=repo.repo_type, token=resolved.token)
            for branch in getattr(refs, "branches", []):
                if branch.name not in seen:
                    options.append(RevisionOption(label=f"Branch: {branch.name}", value=branch.name, kind="branch"))
                    seen.add(branch.name)
            for tag in getattr(refs, "tags", []):
                if tag.name not in seen:
                    options.append(RevisionOption(label=f"Tag: {tag.name}", value=tag.name, kind="tag"))
                    seen.add(tag.name)
        except Exception as exc:  # pragma: no cover - non-critical metadata fetch
            logger.info("Failed to load repo refs for %s: %s", repo.repo_id, exc)

        try:
            commits = self.api.list_repo_commits(repo.repo_id, repo_type=repo.repo_type, token=resolved.token)
            for commit in commits[:5]:
                commit_id = getattr(commit, "commit_id", None)
                if commit_id and commit_id not in seen:
                    options.append(RevisionOption(label=f"Commit: {commit_id[:12]}", value=commit_id, kind="commit"))
                    seen.add(commit_id)
        except Exception as exc:  # pragma: no cover - non-critical metadata fetch
            logger.info("Failed to load recent commits for %s: %s", repo.repo_id, exc)

        requested = repo.requested_revision
        if requested not in seen:
            options.insert(1, RevisionOption(label=f"Current: {requested}", value=requested, kind="current"))

        return options
