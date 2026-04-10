from hf_downloader.models import NodeCheckState
from hf_downloader.tree_ops import build_tree_from_paths, find_node, selected_allow_patterns, set_node_check_state, summarize_selection


def test_tree_selection_propagates_and_summarizes() -> None:
    root = build_tree_from_paths(
        [
            ("folder/a.txt", 10),
            ("folder/b.txt", 20),
            ("folder/nested/c.txt", 30),
            ("top.bin", 5),
        ]
    )

    folder = find_node(root, "folder")
    assert folder is not None

    set_node_check_state(folder, NodeCheckState.CHECKED)
    count, size = summarize_selection(root)
    assert count == 3
    assert size == 60
    assert selected_allow_patterns(root) == ["folder/a.txt", "folder/b.txt", "folder/nested/c.txt"]


def test_partial_parent_state_after_single_file_selection() -> None:
    root = build_tree_from_paths([("folder/a.txt", 10), ("folder/b.txt", 20)])
    file_a = find_node(root, "folder/a.txt")
    folder = find_node(root, "folder")
    assert file_a is not None
    assert folder is not None

    set_node_check_state(file_a, NodeCheckState.CHECKED)
    assert folder.check_state == NodeCheckState.PARTIAL


def test_partial_parent_update_does_not_overwrite_child_selection() -> None:
    root = build_tree_from_paths([("folder/a.txt", 10), ("folder/b.txt", 20)])
    folder = find_node(root, "folder")
    file_a = find_node(root, "folder/a.txt")
    file_b = find_node(root, "folder/b.txt")
    assert folder is not None
    assert file_a is not None
    assert file_b is not None

    set_node_check_state(folder, NodeCheckState.CHECKED)
    set_node_check_state(file_a, NodeCheckState.UNCHECKED)

    assert folder.check_state == NodeCheckState.PARTIAL
    assert file_a.check_state == NodeCheckState.UNCHECKED
    assert file_b.check_state == NodeCheckState.CHECKED

    set_node_check_state(folder, NodeCheckState.PARTIAL)

    assert folder.check_state == NodeCheckState.PARTIAL
    assert file_a.check_state == NodeCheckState.UNCHECKED
    assert file_b.check_state == NodeCheckState.CHECKED
