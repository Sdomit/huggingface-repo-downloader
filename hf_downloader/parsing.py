from __future__ import annotations

from urllib.parse import urlparse

from .models import RepoRef

_KNOWN_ACTIONS = {"tree", "blob", "resolve"}
_KNOWN_PREFIXES = {"datasets": "dataset", "spaces": "space"}


def parse_repo_input(text: str) -> RepoRef | None:
    value = text.strip()
    if not value:
        return None

    if value.startswith("hf://"):
        return _parse_hf_uri(value)

    parsed = urlparse(value)
    if parsed.scheme in {"http", "https"} and parsed.netloc in {"huggingface.co", "www.huggingface.co"}:
        return _parse_hf_url(parsed)

    if "/" not in value:
        return None

    parts = [part for part in value.strip("/").split("/") if part]
    if not parts:
        return None

    if parts[0] in _KNOWN_PREFIXES and len(parts) >= 3:
        repo_type = _KNOWN_PREFIXES[parts[0]]
        repo_id = "/".join(parts[1:3])
        deep_path = "/".join(parts[3:]) or None
        return RepoRef(repo_type=repo_type, repo_id=repo_id, deep_path=deep_path)

    repo_id = "/".join(parts[:2])
    deep_path = "/".join(parts[2:]) or None
    return RepoRef(repo_type="model", repo_id=repo_id, deep_path=deep_path)


def build_repo_url(repo: RepoRef) -> str:
    if repo.repo_type == "model":
        return f"https://huggingface.co/{repo.repo_id}"
    return f"https://huggingface.co/{repo.repo_type}s/{repo.repo_id}"


def _parse_hf_uri(value: str) -> RepoRef | None:
    rest = value.removeprefix("hf://").strip("/")
    parts = [part for part in rest.split("/") if part]
    if not parts:
        return None

    if parts[0] == "datasets" and len(parts) >= 3:
        return RepoRef(repo_type="dataset", repo_id="/".join(parts[1:3]), deep_path="/".join(parts[3:]) or None)
    if parts[0] == "spaces" and len(parts) >= 3:
        return RepoRef(repo_type="space", repo_id="/".join(parts[1:3]), deep_path="/".join(parts[3:]) or None)
    if len(parts) >= 2:
        return RepoRef(repo_type="model", repo_id="/".join(parts[:2]), deep_path="/".join(parts[2:]) or None)
    return None


def _parse_hf_url(parsed) -> RepoRef | None:
    parts = [part for part in parsed.path.strip("/").split("/") if part]
    if not parts:
        return None

    repo_type = "model"
    if parts[0] in _KNOWN_PREFIXES:
        repo_type = _KNOWN_PREFIXES[parts[0]]
        parts = parts[1:]

    if len(parts) < 2:
        return None

    repo_id = "/".join(parts[:2])
    action_index = next((index for index, part in enumerate(parts[2:], start=2) if part in _KNOWN_ACTIONS), None)
    if action_index is None:
        deep_path = "/".join(parts[2:]) or None
        return RepoRef(repo_type=repo_type, repo_id=repo_id, deep_path=deep_path)

    action = parts[action_index]
    revision = parts[action_index + 1] if len(parts) > action_index + 1 else None
    deep_path = "/".join(parts[action_index + 2 :]) or None
    if action == "resolve" and deep_path is None:
        deep_path = revision
        revision = None
    return RepoRef(
        repo_type=repo_type,
        repo_id=repo_id,
        requested_revision=revision,
        deep_path=deep_path,
    )

