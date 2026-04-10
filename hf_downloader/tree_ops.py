from __future__ import annotations

from collections.abc import Iterable

from .models import NodeCheckState, RepoTreeNode


def build_tree_from_paths(paths: Iterable[tuple[str, int]]) -> RepoTreeNode:
    root = RepoTreeNode(name="", path="", is_dir=True)
    for path, size in sorted(paths):
        _insert_path(root, path, size)
    _recompute_tree(root)
    return root


def set_node_check_state(node: RepoTreeNode, state: NodeCheckState) -> None:
    node.check_state = state
    if state != NodeCheckState.PARTIAL:
        for child in node.children:
            set_node_check_state(child, state)
    _refresh_ancestors(node.parent)


def find_node(root: RepoTreeNode, path: str) -> RepoTreeNode | None:
    if root.path == path:
        return root
    for child in root.children:
        found = find_node(child, path)
        if found is not None:
            return found
    return None


def iter_selected_files(root: RepoTreeNode) -> list[RepoTreeNode]:
    selected: list[RepoTreeNode] = []
    _collect_selected_files(root, selected)
    return selected


def selected_allow_patterns(root: RepoTreeNode) -> list[str]:
    return sorted(node.path for node in iter_selected_files(root))


def summarize_selection(root: RepoTreeNode) -> tuple[int, int]:
    files = iter_selected_files(root)
    return len(files), sum(node.size for node in files)


def _insert_path(root: RepoTreeNode, path: str, size: int) -> None:
    parts = [part for part in path.split("/") if part]
    current = root
    accumulated: list[str] = []
    for index, part in enumerate(parts):
        accumulated.append(part)
        is_last = index == len(parts) - 1
        child_path = "/".join(accumulated)
        child = next((candidate for candidate in current.children if candidate.name == part), None)
        if child is None:
            child = RepoTreeNode(
                name=part,
                path=child_path,
                is_dir=not is_last,
                size=0 if not is_last else size,
            )
            current.add_child(child)
        current = child
        if is_last:
            current.is_dir = False
            current.size = size


def _recompute_tree(node: RepoTreeNode) -> tuple[int, int]:
    if not node.children:
        node.file_count = 0 if node.is_dir else 1
        return node.file_count, node.size

    total_files = 0
    total_size = 0
    node.children.sort(key=lambda child: (not child.is_dir, child.name.lower()))
    for child in node.children:
        child_files, child_size = _recompute_tree(child)
        total_files += child_files
        total_size += child_size
    node.file_count = total_files
    node.size = total_size
    return node.file_count, node.size


def _refresh_ancestors(node: RepoTreeNode | None) -> None:
    while node is not None:
        child_states = {child.check_state for child in node.children}
        if child_states == {NodeCheckState.CHECKED}:
            node.check_state = NodeCheckState.CHECKED
        elif child_states == {NodeCheckState.UNCHECKED}:
            node.check_state = NodeCheckState.UNCHECKED
        else:
            node.check_state = NodeCheckState.PARTIAL
        node = node.parent


def _collect_selected_files(node: RepoTreeNode, selected: list[RepoTreeNode]) -> None:
    if not node.is_dir:
        if node.check_state == NodeCheckState.CHECKED:
            selected.append(node)
        return

    for child in node.children:
        _collect_selected_files(child, selected)
