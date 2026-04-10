from __future__ import annotations

from pathlib import Path

from PyQt6 import QtCore
from PyQt6 import QtWidgets

from hf_downloader.auth import AuthResolver
from hf_downloader.hf_service import HuggingFaceService
from hf_downloader.models import AppSettings, DownloadJob, DownloadTask, RepoDetails, RepoRef, RevisionOption, SearchResult
from hf_downloader.queue_manager import QueueManager
from hf_downloader.settings import SettingsStore
from hf_downloader.tree_ops import build_tree_from_paths, selected_allow_patterns
from hf_downloader.ui import MainWindow


class DummySettingsStore(SettingsStore):
    def __init__(self, root: Path) -> None:
        self.data_dir = root / "data"
        self.log_dir = root / "logs"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def save(self, settings) -> None:
        return

    def record_recent_repo(self, settings, repo) -> None:
        return

    def record_job_history(self, settings, job) -> None:
        return


class DummyService(HuggingFaceService):
    def __init__(self, details: RepoDetails) -> None:
        super().__init__(auth_resolver=AuthResolver())
        self.details = details
        self.loaded_inputs: list[RepoRef] = []

    def load_repo_details(self, repo: RepoRef, session_token: str | None = None) -> RepoDetails:
        self.loaded_inputs.append(repo)
        return self.details


def test_main_window_updates_selection_summary(qtbot, tmp_path: Path) -> None:
    settings_store = DummySettingsStore(tmp_path)
    settings = AppSettings(default_download_root=str(tmp_path))
    service = HuggingFaceService(auth_resolver=AuthResolver())
    queue_manager = QueueManager(lambda **kwargs: None, settings_store=settings_store, settings=settings)
    window = MainWindow(service=service, queue_manager=queue_manager, settings_store=settings_store, settings=settings, logger=None)
    qtbot.addWidget(window)

    details = RepoDetails(
        repo=RepoRef(repo_type="model", repo_id="user/repo", pinned_sha="sha"),
        title="user/repo",
        revision="sha",
        description="desc",
        root=build_tree_from_paths([("folder/a.bin", 10), ("folder/b.bin", 20)]),
        total_size=30,
        total_files=2,
    )
    window._on_repo_loaded(details)
    window._select_all()

    assert "2 files" in window.selection_summary_label.text()
    assert "30 B" in window.selection_summary_label.text()

    window._clear_selection()
    assert "0 files" in window.selection_summary_label.text()


def test_initial_ui_shows_empty_states_and_disables_repo_actions(qtbot, tmp_path: Path) -> None:
    settings_store = DummySettingsStore(tmp_path)
    settings = AppSettings(default_download_root=str(tmp_path))
    service = HuggingFaceService(auth_resolver=AuthResolver())
    queue_manager = QueueManager(lambda **kwargs: None, settings_store=settings_store, settings=settings)
    window = MainWindow(service=service, queue_manager=queue_manager, settings_store=settings_store, settings=settings, logger=None)
    qtbot.addWidget(window)

    assert window.search_button.text() == "Search / Open"
    assert not window.search_button.isEnabled()
    assert not window.open_result_button.isEnabled()
    assert window.results_stack.currentWidget() is window.results_empty_label
    assert window.repo_tree_stack.currentWidget() is window.repo_tree_empty_label
    assert not window.add_to_queue_button.isEnabled()
    assert not window.expand_all_button.isEnabled()


def test_search_actions_and_results_panel_react_to_input_and_selection(qtbot, tmp_path: Path) -> None:
    settings_store = DummySettingsStore(tmp_path)
    settings = AppSettings(default_download_root=str(tmp_path))
    service = HuggingFaceService(auth_resolver=AuthResolver())
    queue_manager = QueueManager(lambda **kwargs: None, settings_store=settings_store, settings=settings)
    window = MainWindow(service=service, queue_manager=queue_manager, settings_store=settings_store, settings=settings, logger=None)
    qtbot.addWidget(window)

    window.input_edit.setText("Qwen")
    assert window.search_button.isEnabled()
    assert window.search_button.text() == "Search Hugging Face"

    window.input_edit.setText("https://huggingface.co/user/repo")
    assert window.search_button.text() == "Open Repo"

    window._on_search_results(
        [
            SearchResult(repo_id="user/repo", repo_type="model", downloads=123, likes=4, summary="Summary"),
        ]
    )

    assert window.results_stack.currentWidget() is window.results_list
    assert window.results_status_label.text() == "1 result"

    window.results_list.setCurrentRow(0)
    assert window.open_result_button.isEnabled()


def test_pasted_hf_link_auto_opens_repo(qtbot, tmp_path: Path) -> None:
    settings_store = DummySettingsStore(tmp_path)
    settings = AppSettings(default_download_root=str(tmp_path))
    details = RepoDetails(
        repo=RepoRef(repo_type="model", repo_id="user/repo", pinned_sha="sha"),
        title="user/repo",
        revision="sha",
        description="desc",
        root=build_tree_from_paths([("folder/a.bin", 10)]),
        total_size=10,
        total_files=1,
    )
    service = DummyService(details)
    queue_manager = QueueManager(lambda **kwargs: None, settings_store=settings_store, settings=settings)
    window = MainWindow(service=service, queue_manager=queue_manager, settings_store=settings_store, settings=settings, logger=None)
    qtbot.addWidget(window)

    window.input_edit.setFocus()
    qtbot.keyClicks(window.input_edit, "https://huggingface.co/user/repo/tree/main/folder")

    qtbot.waitUntil(lambda: len(service.loaded_inputs) == 1, timeout=2000)
    assert service.loaded_inputs[0].repo_id == "user/repo"
    assert service.loaded_inputs[0].requested_revision == "main"
    assert service.loaded_inputs[0].deep_path == "folder"


def test_revision_picker_loads_selected_revision(qtbot, tmp_path: Path) -> None:
    settings_store = DummySettingsStore(tmp_path)
    settings = AppSettings(default_download_root=str(tmp_path))
    details = RepoDetails(
        repo=RepoRef(repo_type="model", repo_id="user/repo", pinned_sha="sha"),
        title="user/repo",
        revision="sha",
        description="desc",
        root=build_tree_from_paths([("folder/a.bin", 10)]),
        total_size=10,
        total_files=1,
        available_revisions=[
            RevisionOption(label="Latest / Default", value=None, kind="default"),
            RevisionOption(label="Branch: main", value="main", kind="branch"),
            RevisionOption(label="Branch: dev", value="dev", kind="branch"),
        ],
    )
    service = DummyService(details)
    queue_manager = QueueManager(lambda **kwargs: None, settings_store=settings_store, settings=settings)
    window = MainWindow(service=service, queue_manager=queue_manager, settings_store=settings_store, settings=settings, logger=None)
    qtbot.addWidget(window)

    window._on_repo_loaded(details)
    window.revision_combo.setCurrentText("Branch: dev")
    qtbot.mouseClick(window.load_revision_button, QtCore.Qt.MouseButton.LeftButton)

    qtbot.waitUntil(lambda: len(service.loaded_inputs) == 1, timeout=2000)
    assert service.loaded_inputs[0].requested_revision == "dev"


def test_expand_and_collapse_all_buttons_and_wide_name_column(qtbot, tmp_path: Path) -> None:
    settings_store = DummySettingsStore(tmp_path)
    settings = AppSettings(default_download_root=str(tmp_path))
    service = HuggingFaceService(auth_resolver=AuthResolver())
    queue_manager = QueueManager(lambda **kwargs: None, settings_store=settings_store, settings=settings)
    window = MainWindow(service=service, queue_manager=queue_manager, settings_store=settings_store, settings=settings, logger=None)
    qtbot.addWidget(window)

    details = RepoDetails(
        repo=RepoRef(repo_type="model", repo_id="user/repo", pinned_sha="sha"),
        title="user/repo",
        revision="sha",
        description="desc",
        root=build_tree_from_paths(
            [
                ("very-long-folder-name-for-model-assets/tokenizer/special_tokens_map.json", 10),
                ("very-long-folder-name-for-model-assets/tokenizer/tokenizer.json", 20),
            ]
        ),
        total_size=30,
        total_files=2,
    )
    window._on_repo_loaded(details)

    top_item = window.tree_widget.topLevelItem(0)
    assert top_item is not None

    qtbot.mouseClick(window.expand_all_button, QtCore.Qt.MouseButton.LeftButton)
    assert top_item.isExpanded()

    qtbot.mouseClick(window.collapse_all_button, QtCore.Qt.MouseButton.LeftButton)
    assert not top_item.isExpanded()

    assert window.tree_widget.columnWidth(0) >= window.tree_widget.sizeHintForColumn(0)


def test_safetensors_preset_selects_only_matching_files(qtbot, tmp_path: Path) -> None:
    settings_store = DummySettingsStore(tmp_path)
    settings = AppSettings(default_download_root=str(tmp_path))
    service = HuggingFaceService(auth_resolver=AuthResolver())
    queue_manager = QueueManager(lambda **kwargs: None, settings_store=settings_store, settings=settings)
    window = MainWindow(service=service, queue_manager=queue_manager, settings_store=settings_store, settings=settings, logger=None)
    qtbot.addWidget(window)

    details = RepoDetails(
        repo=RepoRef(repo_type="model", repo_id="user/repo", pinned_sha="sha"),
        title="user/repo",
        revision="sha",
        description="desc",
        root=build_tree_from_paths(
            [
                ("weights/model.safetensors", 10),
                ("weights/model.bin", 20),
                ("tokenizer/tokenizer.json", 5),
            ]
        ),
        total_size=35,
        total_files=3,
    )
    window._on_repo_loaded(details)
    window.selection_preset_combo.setCurrentIndex(window.selection_preset_combo.findData("safetensors"))
    qtbot.mouseClick(window.apply_preset_button, QtCore.Qt.MouseButton.LeftButton)

    assert selected_allow_patterns(window.current_repo_details.root) == ["weights/model.safetensors"]
    assert "1 files" in window.selection_summary_label.text()


def test_guided_mode_selects_recommended_runtime_files(qtbot, tmp_path: Path) -> None:
    settings_store = DummySettingsStore(tmp_path)
    settings = AppSettings(default_download_root=str(tmp_path))
    service = HuggingFaceService(auth_resolver=AuthResolver())
    queue_manager = QueueManager(lambda **kwargs: None, settings_store=settings_store, settings=settings)
    window = MainWindow(service=service, queue_manager=queue_manager, settings_store=settings_store, settings=settings, logger=None)
    qtbot.addWidget(window)

    details = RepoDetails(
        repo=RepoRef(repo_type="model", repo_id="user/repo", pinned_sha="sha"),
        title="user/repo",
        revision="sha",
        description="desc",
        root=build_tree_from_paths(
            [
                ("model.safetensors", 10),
                ("config.json", 2),
                ("preview.png", 1),
            ]
        ),
        total_size=13,
        total_files=3,
    )
    details.classification = details.classification.__class__(
        primary_label="Single-File Checkpoint",
        package_kind="single_checkpoint",
        modality_label="Image Model",
        detected_labels=["Single-File Checkpoint", "Image Model"],
        summary="Detected as Single-File Checkpoint | Image Model | heuristic",
    )
    window._on_repo_loaded(details)
    window.guided_mode_combo.setCurrentIndex(window.guided_mode_combo.findData("recommended"))
    qtbot.mouseClick(window.apply_guided_mode_button, QtCore.Qt.MouseButton.LeftButton)

    assert selected_allow_patterns(window.current_repo_details.root) == ["config.json", "model.safetensors"]
    assert "Recommended" in window.guided_mode_description_label.text()


def test_queue_drag_reorder_updates_visible_priority(qtbot, tmp_path: Path) -> None:
    settings_store = DummySettingsStore(tmp_path)
    settings = AppSettings(default_download_root=str(tmp_path))
    service = HuggingFaceService(auth_resolver=AuthResolver())
    queue_manager = QueueManager(lambda **kwargs: None, settings_store=settings_store, settings=settings)
    window = MainWindow(service=service, queue_manager=queue_manager, settings_store=settings_store, settings=settings, logger=None)
    qtbot.addWidget(window)

    job1 = DownloadJob(
        job_id="job1",
        repo=RepoRef(repo_type="model", repo_id="user/job1", pinned_sha="sha"),
        title="user/job1",
        destination=tmp_path / "job1",
        allow_patterns=["a.bin"],
        tasks=[DownloadTask(filename="a.bin", size=10, local_path=tmp_path / "job1" / "a.bin", will_download=True)],
        total_selected_bytes=10,
        bytes_to_download=10,
        cached_bytes=0,
        worker_count=1,
        retry_count=1,
    )
    job2 = DownloadJob(
        job_id="job2",
        repo=RepoRef(repo_type="model", repo_id="user/job2", pinned_sha="sha"),
        title="user/job2",
        destination=tmp_path / "job2",
        allow_patterns=["a.bin"],
        tasks=[DownloadTask(filename="a.bin", size=10, local_path=tmp_path / "job2" / "a.bin", will_download=True)],
        total_selected_bytes=10,
        bytes_to_download=10,
        cached_bytes=0,
        worker_count=1,
        retry_count=1,
    )
    queue_manager.add_job(job1)
    queue_manager.add_job(job2)
    window._refresh_queue_table()

    window.queue_table.selectRow(1)
    window.queue_table.rows_reordered.emit(1, 0)

    assert queue_manager.jobs[0].job_id == "job2"
    assert window.queue_table.item(0, 0).text() == "user/job2"


def test_queue_row_has_progress_bar_and_inline_actions(qtbot, tmp_path: Path) -> None:
    settings_store = DummySettingsStore(tmp_path)
    settings = AppSettings(default_download_root=str(tmp_path))
    service = HuggingFaceService(auth_resolver=AuthResolver())
    queue_manager = QueueManager(lambda **kwargs: None, settings_store=settings_store, settings=settings)
    window = MainWindow(service=service, queue_manager=queue_manager, settings_store=settings_store, settings=settings, logger=None)
    qtbot.addWidget(window)

    job = DownloadJob(
        job_id="job1",
        repo=RepoRef(repo_type="model", repo_id="user/job1", pinned_sha="sha"),
        title="user/job1",
        destination=tmp_path / "job1",
        allow_patterns=["a.bin"],
        tasks=[DownloadTask(filename="a.bin", size=10, local_path=tmp_path / "job1" / "a.bin", will_download=True)],
        total_selected_bytes=10,
        bytes_to_download=10,
        cached_bytes=0,
        worker_count=1,
        retry_count=1,
    )
    queue_manager.add_job(job)
    window._refresh_queue_table()

    assert isinstance(window.queue_table.cellWidget(0, 3), QtWidgets.QProgressBar)
    actions_widget = window.queue_table.cellWidget(0, 8)
    assert actions_widget is not None
    action_texts = {button.text() for button in actions_widget.findChildren(QtWidgets.QToolButton)}
    assert {"Start", "Delete", "Folder"} <= action_texts


def test_repo_tree_file_search_finds_safetensors_match(qtbot, tmp_path: Path) -> None:
    settings_store = DummySettingsStore(tmp_path)
    settings = AppSettings(default_download_root=str(tmp_path))
    service = HuggingFaceService(auth_resolver=AuthResolver())
    queue_manager = QueueManager(lambda **kwargs: None, settings_store=settings_store, settings=settings)
    window = MainWindow(service=service, queue_manager=queue_manager, settings_store=settings_store, settings=settings, logger=None)
    qtbot.addWidget(window)

    details = RepoDetails(
        repo=RepoRef(repo_type="model", repo_id="user/repo", pinned_sha="sha"),
        title="user/repo",
        revision="sha",
        description="desc",
        root=build_tree_from_paths(
            [
                ("weights/Qwen-Image-Edit-2511-bf16.safetensors", 10),
                ("weights/Qwen-Image-Edit-2511-fp8.safetensors", 20),
                ("tokenizer/tokenizer.json", 5),
            ]
        ),
        total_size=35,
        total_files=3,
    )
    window._on_repo_loaded(details)

    qtbot.keyClicks(window.file_search_edit, "Qwen-Image-Edit-2511-bf16.safetensors")

    qtbot.waitUntil(lambda: window.tree_widget.currentItem() is not None, timeout=1000)
    current_item = window.tree_widget.currentItem()
    assert current_item is not None
    assert current_item.text(0) == "Qwen-Image-Edit-2511-bf16.safetensors"
    assert window.file_search_status_label.text() == "1 / 1"

    parent_item = current_item.parent()
    assert parent_item is not None
    assert parent_item.isExpanded()


def test_queue_area_is_larger_and_has_clear_button(qtbot, tmp_path: Path) -> None:
    settings_store = DummySettingsStore(tmp_path)
    settings = AppSettings(default_download_root=str(tmp_path))
    service = HuggingFaceService(auth_resolver=AuthResolver())
    queue_manager = QueueManager(lambda **kwargs: None, settings_store=settings_store, settings=settings)
    window = MainWindow(service=service, queue_manager=queue_manager, settings_store=settings_store, settings=settings, logger=None)
    qtbot.addWidget(window)
    window.show()
    qtbot.wait(50)

    assert window.clear_queue_button.text() == "Clear Queue"
    assert window.main_splitter.orientation() == QtCore.Qt.Orientation.Vertical
    sizes = window.main_splitter.sizes()
    assert len(sizes) == 2
    assert sizes[1] >= sizes[0]


def test_queue_and_selected_job_areas_are_boxed(qtbot, tmp_path: Path) -> None:
    settings_store = DummySettingsStore(tmp_path)
    settings = AppSettings(default_download_root=str(tmp_path))
    service = HuggingFaceService(auth_resolver=AuthResolver())
    queue_manager = QueueManager(lambda **kwargs: None, settings_store=settings_store, settings=settings)
    window = MainWindow(service=service, queue_manager=queue_manager, settings_store=settings_store, settings=settings, logger=None)
    qtbot.addWidget(window)

    assert isinstance(window.queue_group, QtWidgets.QGroupBox)
    assert window.queue_group.title() == "Download Queue"
    assert window.queue_stack.parentWidget() is window.queue_group

    assert isinstance(window.task_group, QtWidgets.QGroupBox)
    assert window.task_group.title() == "Selected Queue Job"
    assert window.task_stack.parentWidget() is window.task_group


def test_repo_action_footer_stays_outside_scroll_area(qtbot, tmp_path: Path) -> None:
    settings_store = DummySettingsStore(tmp_path)
    settings = AppSettings(default_download_root=str(tmp_path))
    service = HuggingFaceService(auth_resolver=AuthResolver())
    queue_manager = QueueManager(lambda **kwargs: None, settings_store=settings_store, settings=settings)
    window = MainWindow(service=service, queue_manager=queue_manager, settings_store=settings_store, settings=settings, logger=None)
    qtbot.addWidget(window)
    window.resize(1200, 760)
    window.show()
    qtbot.wait(50)

    assert window.add_to_queue_button.parentWidget() is window.repo_footer_widget
    assert window.repo_footer_widget.isVisible()
    assert window.add_to_queue_button.isVisible()

    parent = window.add_to_queue_button.parentWidget()
    while parent is not None:
        assert not isinstance(parent, QtWidgets.QScrollArea)
        parent = parent.parentWidget()
