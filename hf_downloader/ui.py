from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from PyQt6 import QtCore, QtGui, QtWidgets

from .formatting import format_bytes, format_duration, format_speed
from .hf_service import HubServiceError, HuggingFaceService
from .models import DownloadJob, JobStatus, NodeCheckState, ProgressSnapshot, RepoDetails, RepoRef, RevisionOption, SearchResult
from .parsing import parse_repo_input
from .queue_manager import QueueManager
from .repo_analysis import mode_description, select_paths_for_mode
from .settings import SettingsStore
from .tree_ops import find_node, selected_allow_patterns, set_node_check_state, summarize_selection

logger = logging.getLogger("hf_downloader")

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
TOKENIZER_FILENAMES = {
    "added_tokens.json",
    "chat_template.jinja",
    "merges.txt",
    "special_tokens_map.json",
    "spiece.model",
    "sentencepiece.bpe.model",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "vocab.json",
}
STATUS_COLORS = {
    JobStatus.PENDING: "#5c6b73",
    JobStatus.RUNNING: "#1f7a8c",
    JobStatus.PAUSED: "#d17b0f",
    JobStatus.FAILED: "#b00020",
    JobStatus.COMPLETED: "#2d6a4f",
    JobStatus.CANCELLED: "#6c757d",
}
APP_STYLESHEET = """
QMainWindow, QWidget {
    background-color: #1f1b1a;
    color: #f2eee9;
    font-size: 10pt;
}
QGroupBox {
    border: 1px solid #5e5048;
    border-radius: 10px;
    margin-top: 10px;
    padding-top: 12px;
    font-weight: 600;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 6px;
    color: #f6d8be;
}
QLineEdit, QComboBox, QSpinBox, QListWidget, QTreeWidget, QTableWidget, QTabWidget::pane {
    background-color: #2a2625;
    border: 1px solid #5a4d46;
    border-radius: 7px;
    padding: 5px 7px;
}
QLineEdit:disabled, QComboBox:disabled, QListWidget:disabled, QTreeWidget:disabled, QTableWidget:disabled, QPushButton:disabled, QToolButton:disabled {
    color: #8d817a;
    background-color: #262221;
}
QPushButton, QToolButton {
    background-color: #3a3431;
    border: 1px solid #6a5950;
    border-radius: 8px;
    padding: 6px 12px;
    min-height: 16px;
}
QPushButton:hover, QToolButton:hover {
    background-color: #4a423e;
}
QPushButton[accent="true"] {
    background-color: #b85c38;
    border: 1px solid #d08055;
    color: white;
    font-weight: 700;
}
QPushButton[accent="true"]:hover {
    background-color: #c96c47;
}
QTabBar::tab {
    background: #2a2625;
    border: 1px solid #5a4d46;
    border-bottom: none;
    padding: 6px 12px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    margin-right: 4px;
}
QTabBar::tab:selected {
    background: #3a3431;
    color: #f6d8be;
}
QHeaderView::section {
    background-color: #3a3431;
    color: #f2eee9;
    border: none;
    border-right: 1px solid #5a4d46;
    padding: 6px;
    font-weight: 600;
}
QProgressBar {
    border: 1px solid #5a4d46;
    border-radius: 6px;
    background-color: #2a2625;
    text-align: center;
}
QProgressBar::chunk {
    background-color: #b85c38;
    border-radius: 5px;
}
QSplitter::handle {
    background-color: #312b29;
}
QWidget#repoFooter {
    background-color: #241f1e;
    border-top: 1px solid #5a4d46;
}
QLabel#searchHint, QLabel#emptyState, QLabel#mutedText {
    color: #c2b4aa;
}
QLabel#repoTitle {
    font-size: 16pt;
    font-weight: 700;
    color: #fff5ee;
}
QLabel#classificationPrimary {
    color: #ffd4b8;
    font-size: 11pt;
    font-weight: 700;
}
QLabel#chipLabel {
    background-color: #302925;
    border: 1px solid #6f5d50;
    border-radius: 999px;
    padding: 4px 10px;
    color: #f4e7da;
}
"""


class WorkerSignals(QtCore.QObject):
    finished = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)


class FunctionWorker(QtCore.QRunnable):
    def __init__(self, fn: Callable[[], object]) -> None:
        super().__init__()
        self.fn = fn
        self.signals = WorkerSignals()

    @QtCore.pyqtSlot()
    def run(self) -> None:
        try:
            result = self.fn()
        except Exception as exc:  # pragma: no cover - thread wrapper
            logger.exception("Background task failed")
            self.signals.failed.emit(str(exc))
            return
        self.signals.finished.emit(result)


class QueueTableWidget(QtWidgets.QTableWidget):
    rows_reordered = QtCore.pyqtSignal(int, int)

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.viewport().setAcceptDrops(True)
        self.setDropIndicatorShown(True)
        self.setDragDropOverwriteMode(False)
        self.setDefaultDropAction(QtCore.Qt.DropAction.MoveAction)
        self.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.InternalMove)

    def dropEvent(self, event: QtGui.QDropEvent) -> None:
        source_row = self.currentRow()
        if source_row < 0:
            event.ignore()
            return

        target_row = self._target_row(event)
        self.rows_reordered.emit(source_row, target_row)
        event.acceptProposedAction()

    def _target_row(self, event: QtGui.QDropEvent) -> int:
        pos = event.position().toPoint()
        row = self.rowAt(pos.y())
        if row < 0:
            return self.rowCount()

        rect = self.visualRect(self.model().index(row, 0))
        if pos.y() > rect.center().y():
            return row + 1
        return row


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, service: HuggingFaceService, queue_manager: QueueManager, settings_store: SettingsStore, settings, logger, parent=None) -> None:
        super().__init__(parent)
        self.service = service
        self.queue_manager = queue_manager
        self.settings_store = settings_store
        self.settings = settings
        self.logger = logger
        self.thread_pool = QtCore.QThreadPool.globalInstance()
        self.current_repo_details: RepoDetails | None = None
        self._syncing_tree = False
        self._job_snapshots: dict[str, ProgressSnapshot] = {}
        self._queue_row_by_job_id: dict[str, int] = {}
        self._tree_search_matches: list[QtWidgets.QTreeWidgetItem] = []
        self._tree_search_index = -1
        self._auto_open_timer = QtCore.QTimer(self)
        self._auto_open_timer.setSingleShot(True)
        self._auto_open_timer.setInterval(250)
        self._splitter_balanced = False

        self.setWindowTitle("HF Repo Downloader")
        self.resize(1440, 960)

        self._build_ui()
        self._apply_styles()
        self._load_settings_into_ui()
        self._connect_signals()
        self._refresh_queue_table()
        self._reset_repo_panel()
        self._update_results_panel_state()
        self._update_queue_controls_enabled()
        self._update_search_actions()

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        super().showEvent(event)
        if not self._splitter_balanced:
            self._rebalance_main_splitter()
            self._splitter_balanced = True

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        root_layout = QtWidgets.QVBoxLayout(central)
        root_layout.setContentsMargins(10, 10, 10, 10)
        root_layout.setSpacing(10)

        root_layout.addWidget(self._build_search_bar())

        top_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        top_splitter.addWidget(self._build_results_panel())
        top_splitter.addWidget(self._build_repo_panel())
        top_splitter.setStretchFactor(0, 1)
        top_splitter.setStretchFactor(1, 2)

        bottom_tabs = QtWidgets.QTabWidget()
        bottom_tabs.addTab(self._build_queue_panel(), "Queue")
        bottom_tabs.addTab(self._build_settings_panel(), "Settings")
        bottom_tabs.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )

        self.main_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        self.main_splitter.addWidget(top_splitter)
        self.main_splitter.addWidget(bottom_tabs)
        self.main_splitter.setStretchFactor(0, 1)
        self.main_splitter.setStretchFactor(1, 4)
        self.main_splitter.setChildrenCollapsible(False)
        self.main_splitter.setSizes([300, 700])
        root_layout.addWidget(self.main_splitter, stretch=1)

        self.setCentralWidget(central)
        self.statusBar().showMessage("Ready")

    def _rebalance_main_splitter(self) -> None:
        total_height = self.main_splitter.size().height()
        if total_height <= 0:
            sizes = self.main_splitter.sizes()
            total_height = sum(sizes)
        if total_height <= 0:
            return
        top_height = max(380, int(total_height * 0.42))
        max_top = total_height - 360
        if max_top > 0:
            top_height = min(top_height, max_top)
        bottom_height = max(360, total_height - top_height)
        self.main_splitter.setSizes([top_height, bottom_height])

    def _build_search_bar(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        search_row = QtWidgets.QHBoxLayout()
        search_row.setSpacing(8)

        self.input_edit = QtWidgets.QLineEdit()
        self.input_edit.setPlaceholderText("Search Hugging Face or paste a repo ID / URL")
        self.scope_combo = QtWidgets.QComboBox()
        self.scope_combo.addItem("All", "all")
        self.scope_combo.addItem("Models", "model")
        self.scope_combo.addItem("Datasets", "dataset")
        self.scope_combo.addItem("Spaces", "space")
        self.search_button = QtWidgets.QPushButton("Search / Open")
        self.search_button.setProperty("accent", True)
        self.search_button.style().unpolish(self.search_button)
        self.search_button.style().polish(self.search_button)
        self.search_hint_label = QtWidgets.QLabel("Paste a Hugging Face URL or search by repo name, author, or organization.")
        self.search_hint_label.setObjectName("searchHint")

        search_row.addWidget(self.input_edit, stretch=1)
        search_row.addWidget(self.scope_combo)
        search_row.addWidget(self.search_button)
        layout.addLayout(search_row)
        layout.addWidget(self.search_hint_label)
        return widget

    def _build_results_panel(self) -> QtWidgets.QWidget:
        group = QtWidgets.QGroupBox("Search Results")
        layout = QtWidgets.QVBoxLayout(group)
        layout.setSpacing(8)

        self.results_status_label = QtWidgets.QLabel("No search results yet")
        self.results_status_label.setObjectName("mutedText")
        layout.addWidget(self.results_status_label)

        self.results_stack = QtWidgets.QStackedWidget()
        self.results_empty_label = QtWidgets.QLabel("Search by repo name, author, or paste a Hugging Face link to open it directly.")
        self.results_empty_label.setObjectName("emptyState")
        self.results_empty_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.results_empty_label.setWordWrap(True)
        self.results_list = QtWidgets.QListWidget()
        self.results_list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.results_list.setSpacing(4)
        self.results_list.setAlternatingRowColors(True)
        self.open_result_button = QtWidgets.QPushButton("Open Repo")
        self.open_result_button.setProperty("accent", True)
        self.open_result_button.style().unpolish(self.open_result_button)
        self.open_result_button.style().polish(self.open_result_button)
        self.results_stack.addWidget(self.results_empty_label)
        self.results_stack.addWidget(self.results_list)
        layout.addWidget(self.results_stack, stretch=1)
        layout.addWidget(self.open_result_button)
        return group

    def _build_repo_panel(self) -> QtWidgets.QWidget:
        group = QtWidgets.QGroupBox("Repository Details")
        outer_layout = QtWidgets.QVBoxLayout(group)
        outer_layout.setContentsMargins(10, 18, 10, 10)
        outer_layout.setSpacing(8)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )

        body = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(body)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.setSizeConstraint(QtWidgets.QLayout.SizeConstraint.SetMinimumSize)

        self.repo_title_label = QtWidgets.QLabel("No repository loaded")
        self.repo_title_label.setObjectName("repoTitle")
        self.repo_title_label.setWordWrap(True)
        repo_title_font = self.repo_title_label.font()
        repo_title_font.setPointSize(repo_title_font.pointSize() + 2)
        repo_title_font.setBold(True)
        self.repo_title_label.setFont(repo_title_font)
        self.repo_revision_label = QtWidgets.QLabel("")
        self.repo_description_label = QtWidgets.QLabel("")
        self.repo_description_label.setWordWrap(True)
        self.classification_primary_label = QtWidgets.QLabel("")
        self.classification_primary_label.setObjectName("classificationPrimary")
        self.classification_badges_label = QtWidgets.QLabel("")
        self.classification_meta_label = QtWidgets.QLabel("")
        self.classification_badges_label.setWordWrap(True)
        self.classification_meta_label.setWordWrap(True)
        self.repo_revision_label.setObjectName("mutedText")
        self.repo_description_label.setObjectName("mutedText")
        self.classification_meta_label.setObjectName("mutedText")
        self.classification_badges_label.setObjectName("chipLabel")

        layout.addWidget(self.repo_title_label)
        layout.addWidget(self.repo_revision_label)
        layout.addWidget(self.repo_description_label)
        layout.addWidget(self.classification_primary_label)
        layout.addWidget(self.classification_badges_label)
        layout.addWidget(self.classification_meta_label)

        details_form = QtWidgets.QFormLayout()
        details_form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)
        details_form.setFormAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        details_form.setHorizontalSpacing(10)
        details_form.setVerticalSpacing(8)

        revision_row = QtWidgets.QHBoxLayout()
        revision_row.setContentsMargins(0, 0, 0, 0)
        self.revision_combo = QtWidgets.QComboBox()
        self.revision_combo.setEditable(True)
        self.revision_combo.setInsertPolicy(QtWidgets.QComboBox.InsertPolicy.NoInsert)
        self.load_revision_button = QtWidgets.QPushButton("Load Revision")
        revision_row.addWidget(self.revision_combo, stretch=1)
        revision_row.addWidget(self.load_revision_button)
        revision_widget = QtWidgets.QWidget()
        revision_widget.setLayout(revision_row)
        details_form.addRow("Revision", revision_widget)

        destination_row = QtWidgets.QHBoxLayout()
        destination_row.setContentsMargins(0, 0, 0, 0)
        self.destination_edit = QtWidgets.QLineEdit()
        self.destination_browse_button = QtWidgets.QPushButton("Browse...")
        destination_row.addWidget(self.destination_edit, stretch=1)
        destination_row.addWidget(self.destination_browse_button)
        destination_widget = QtWidgets.QWidget()
        destination_widget.setLayout(destination_row)
        details_form.addRow("Destination", destination_widget)
        layout.addLayout(details_form)

        self.selection_summary_label = QtWidgets.QLabel("Selected: 0 files, 0 B")
        self.selection_summary_label.setObjectName("mutedText")
        layout.addWidget(self.selection_summary_label)

        guided_mode_row = QtWidgets.QHBoxLayout()
        self.guided_mode_combo = QtWidgets.QComboBox()
        self.guided_mode_combo.addItem("Minimal", "minimal")
        self.guided_mode_combo.addItem("Recommended", "recommended")
        self.guided_mode_combo.addItem("Full", "full")
        self.apply_guided_mode_button = QtWidgets.QPushButton("Apply Mode")
        guided_mode_row.addWidget(QtWidgets.QLabel("Guided Mode"))
        guided_mode_row.addWidget(self.guided_mode_combo)
        guided_mode_row.addWidget(self.apply_guided_mode_button)
        layout.addLayout(guided_mode_row)

        self.guided_mode_description_label = QtWidgets.QLabel("")
        self.guided_mode_description_label.setWordWrap(True)
        layout.addWidget(self.guided_mode_description_label)

        preset_row = QtWidgets.QHBoxLayout()
        self.selection_preset_combo = QtWidgets.QComboBox()
        self.selection_preset_combo.addItem("Quick Select...", "none")
        self.selection_preset_combo.addItem("All Files", "all")
        self.selection_preset_combo.addItem("Safetensors Only", "safetensors")
        self.selection_preset_combo.addItem("Images Only", "images")
        self.selection_preset_combo.addItem("Config + Tokenizer", "configs")
        self.selection_preset_combo.addItem("Exclude Tokenizer", "exclude_tokenizer")
        self.apply_preset_button = QtWidgets.QPushButton("Apply Preset")
        self.extension_filter_edit = QtWidgets.QLineEdit()
        self.extension_filter_edit.setPlaceholderText(".safetensors,.gguf")
        self.select_extension_button = QtWidgets.QPushButton("Select Ext")
        self.exclude_extension_button = QtWidgets.QPushButton("Exclude Ext")
        preset_row.addWidget(self.selection_preset_combo)
        preset_row.addWidget(self.apply_preset_button)
        preset_row.addWidget(self.extension_filter_edit, stretch=1)
        preset_row.addWidget(self.select_extension_button)
        preset_row.addWidget(self.exclude_extension_button)
        layout.addLayout(preset_row)

        search_row = QtWidgets.QHBoxLayout()
        self.file_search_edit = QtWidgets.QLineEdit()
        self.file_search_edit.setPlaceholderText("Find file or folder in this repo")
        self.find_previous_button = QtWidgets.QPushButton("Previous")
        self.find_next_button = QtWidgets.QPushButton("Next")
        self.file_search_status_label = QtWidgets.QLabel("")
        search_row.addWidget(self.file_search_edit, stretch=1)
        search_row.addWidget(self.find_previous_button)
        search_row.addWidget(self.find_next_button)
        search_row.addWidget(self.file_search_status_label)
        layout.addLayout(search_row)

        self.repo_tree_stack = QtWidgets.QStackedWidget()
        self.repo_tree_empty_label = QtWidgets.QLabel("Open a repository to inspect files, detect the repo type, and prepare a clean download.")
        self.repo_tree_empty_label.setObjectName("emptyState")
        self.repo_tree_empty_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.repo_tree_empty_label.setWordWrap(True)
        self.tree_widget = QtWidgets.QTreeWidget()
        self.tree_widget.setHeaderLabels(["Name", "Size", "Files"])
        self.tree_widget.setUniformRowHeights(True)
        self.tree_widget.setAlternatingRowColors(True)
        self.tree_widget.setIndentation(18)
        self.tree_widget.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        tree_header = self.tree_widget.header()
        tree_header.setStretchLastSection(False)
        tree_header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        tree_header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        tree_header.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.repo_tree_stack.addWidget(self.repo_tree_empty_label)
        self.repo_tree_stack.addWidget(self.tree_widget)
        self.repo_tree_stack.setMinimumHeight(210)
        layout.addWidget(self.repo_tree_stack, stretch=1)

        scroll.setWidget(body)
        outer_layout.addWidget(scroll, stretch=1)

        self.repo_footer_widget = QtWidgets.QWidget()
        self.repo_footer_widget.setObjectName("repoFooter")
        button_row = QtWidgets.QHBoxLayout(self.repo_footer_widget)
        button_row.setContentsMargins(8, 8, 8, 8)
        button_row.setSpacing(8)
        self.expand_all_button = QtWidgets.QPushButton("Expand All")
        self.collapse_all_button = QtWidgets.QPushButton("Collapse All")
        self.select_all_button = QtWidgets.QPushButton("Select All")
        self.clear_selection_button = QtWidgets.QPushButton("Clear")
        self.add_to_queue_button = QtWidgets.QPushButton("Queue Download")
        self.add_to_queue_button.setProperty("accent", True)
        self.add_to_queue_button.style().unpolish(self.add_to_queue_button)
        self.add_to_queue_button.style().polish(self.add_to_queue_button)
        button_row.addWidget(self.expand_all_button)
        button_row.addWidget(self.collapse_all_button)
        button_row.addWidget(self.select_all_button)
        button_row.addWidget(self.clear_selection_button)
        button_row.addStretch(1)
        button_row.addWidget(self.add_to_queue_button)
        outer_layout.addWidget(self.repo_footer_widget)
        return group

    def _build_queue_panel(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setSpacing(8)

        self.queue_summary_label = QtWidgets.QLabel("Queue is empty")
        self.queue_summary_label.setObjectName("mutedText")
        layout.addWidget(self.queue_summary_label)

        self.queue_stack = QtWidgets.QStackedWidget()
        self.queue_empty_label = QtWidgets.QLabel("Add a repo selection to the queue to manage downloads here.")
        self.queue_empty_label.setObjectName("emptyState")
        self.queue_empty_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.queue_empty_label.setWordWrap(True)
        self.queue_table = QueueTableWidget(0, 9)
        self.queue_table.setHorizontalHeaderLabels(
            ["Repo", "Type", "Status", "Progress", "Speed", "ETA", "Files", "Destination", "Actions"]
        )
        self.queue_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.queue_table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.queue_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.queue_table.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.queue_table.setAlternatingRowColors(True)
        self.queue_table.setShowGrid(False)
        self.queue_table.verticalHeader().setVisible(False)
        self.queue_table.horizontalHeader().setStretchLastSection(True)
        self.queue_stack.addWidget(self.queue_empty_label)
        self.queue_stack.addWidget(self.queue_table)
        layout.addWidget(self.queue_stack, stretch=2)

        self.active_progress_bar = QtWidgets.QProgressBar()
        self.active_progress_bar.setRange(0, 1000)
        self.active_progress_bar.setValue(0)
        self.active_progress_label = QtWidgets.QLabel("No active download")
        self.active_progress_label.setObjectName("mutedText")
        layout.addWidget(self.active_progress_bar)
        layout.addWidget(self.active_progress_label)

        buttons = QtWidgets.QHBoxLayout()
        self.start_button = QtWidgets.QPushButton("Start")
        self.pause_button = QtWidgets.QPushButton("Pause")
        self.resume_button = QtWidgets.QPushButton("Resume")
        self.cancel_button = QtWidgets.QPushButton("Cancel")
        self.clear_queue_button = QtWidgets.QPushButton("Clear Queue")
        self.retry_button = QtWidgets.QPushButton("Retry Failed")
        self.open_folder_button = QtWidgets.QPushButton("Open Folder")
        for button in (
            self.start_button,
            self.pause_button,
            self.resume_button,
            self.cancel_button,
            self.clear_queue_button,
            self.retry_button,
            self.open_folder_button,
        ):
            buttons.addWidget(button)
        buttons.addStretch(1)
        layout.addLayout(buttons)

        self.task_summary_label = QtWidgets.QLabel("No job selected")
        self.task_summary_label.setObjectName("mutedText")
        layout.addWidget(self.task_summary_label)

        self.task_stack = QtWidgets.QStackedWidget()
        self.task_empty_label = QtWidgets.QLabel("Select a queued job to inspect per-file progress and errors.")
        self.task_empty_label.setObjectName("emptyState")
        self.task_empty_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.task_empty_label.setWordWrap(True)
        self.task_table = QtWidgets.QTableWidget(0, 5)
        self.task_table.setHorizontalHeaderLabels(["File", "Status", "Progress", "Size", "Error"])
        self.task_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.task_table.setAlternatingRowColors(True)
        self.task_table.setShowGrid(False)
        self.task_table.verticalHeader().setVisible(False)
        self.task_table.horizontalHeader().setStretchLastSection(True)
        self.task_stack.addWidget(self.task_empty_label)
        self.task_stack.addWidget(self.task_table)
        layout.addWidget(self.task_stack, stretch=1)

        self.error_label = QtWidgets.QLabel("")
        self.error_label.setWordWrap(True)
        self.error_label.setStyleSheet("color: #b00020;")
        layout.addWidget(self.error_label)
        return widget

    def _build_settings_panel(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)

        form = QtWidgets.QFormLayout()
        self.token_edit = QtWidgets.QLineEdit()
        self.token_edit.setEchoMode(QtWidgets.QLineEdit.EchoMode.Password)
        token_buttons = QtWidgets.QHBoxLayout()
        self.save_token_button = QtWidgets.QPushButton("Save Token")
        self.clear_token_button = QtWidgets.QPushButton("Clear Token")
        token_buttons.addWidget(self.save_token_button)
        token_buttons.addWidget(self.clear_token_button)
        token_widget = QtWidgets.QWidget()
        token_widget_layout = QtWidgets.QVBoxLayout(token_widget)
        token_widget_layout.setContentsMargins(0, 0, 0, 0)
        token_widget_layout.addWidget(self.token_edit)
        token_widget_layout.addLayout(token_buttons)
        form.addRow("Session Token", token_widget)

        self.default_root_edit = QtWidgets.QLineEdit()
        self.default_root_browse_button = QtWidgets.QPushButton("Browse...")
        default_root_row = QtWidgets.QHBoxLayout()
        default_root_row.addWidget(self.default_root_edit)
        default_root_row.addWidget(self.default_root_browse_button)
        default_root_widget = QtWidgets.QWidget()
        default_root_widget.setLayout(default_root_row)
        form.addRow("Default Root", default_root_widget)

        self.worker_spin = QtWidgets.QSpinBox()
        self.worker_spin.setRange(1, 16)
        self.retry_spin = QtWidgets.QSpinBox()
        self.retry_spin.setRange(0, 10)
        form.addRow("Worker Count", self.worker_spin)
        form.addRow("Retry Count", self.retry_spin)
        layout.addLayout(form)

        self.save_settings_button = QtWidgets.QPushButton("Save Settings")
        self.token_source_label = QtWidgets.QLabel("")
        self.log_path_label = QtWidgets.QLabel("")
        self.log_path_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.save_settings_button)
        layout.addWidget(self.token_source_label)
        layout.addWidget(self.log_path_label)
        layout.addStretch(1)
        return widget

    def _connect_signals(self) -> None:
        self.search_button.clicked.connect(self._on_search_clicked)
        self.input_edit.returnPressed.connect(self._on_search_clicked)
        self.input_edit.textChanged.connect(lambda _text: self._update_search_actions())
        self.open_result_button.clicked.connect(self._open_selected_search_result)
        self.results_list.itemDoubleClicked.connect(lambda _item: self._open_selected_search_result())
        self.results_list.itemSelectionChanged.connect(self._update_search_actions)
        self.scope_combo.currentIndexChanged.connect(lambda _index: self._update_search_actions())
        self.destination_browse_button.clicked.connect(self._browse_destination)
        self.load_revision_button.clicked.connect(self._load_selected_revision)
        self.expand_all_button.clicked.connect(self._expand_all)
        self.collapse_all_button.clicked.connect(self._collapse_all)
        self.select_all_button.clicked.connect(self._select_all)
        self.clear_selection_button.clicked.connect(self._clear_selection)
        self.add_to_queue_button.clicked.connect(self._add_current_selection_to_queue)
        self.guided_mode_combo.currentIndexChanged.connect(self._update_guided_mode_description)
        self.apply_guided_mode_button.clicked.connect(self._apply_guided_mode)
        self.apply_preset_button.clicked.connect(self._apply_selection_preset)
        self.select_extension_button.clicked.connect(lambda: self._apply_extension_filter(select_mode=True))
        self.exclude_extension_button.clicked.connect(lambda: self._apply_extension_filter(select_mode=False))
        self.file_search_edit.textChanged.connect(self._on_tree_search_changed)
        self.file_search_edit.returnPressed.connect(self._find_next_tree_match)
        self.find_previous_button.clicked.connect(self._find_previous_tree_match)
        self.find_next_button.clicked.connect(self._find_next_tree_match)
        self.tree_widget.itemChanged.connect(self._on_tree_item_changed)
        self.tree_widget.itemExpanded.connect(lambda _item: self._resize_tree_columns())
        self.tree_widget.itemCollapsed.connect(lambda _item: self._resize_tree_columns())
        self.queue_table.itemSelectionChanged.connect(self._on_queue_selection_changed)
        self.queue_table.customContextMenuRequested.connect(self._show_queue_context_menu)
        self.queue_table.rows_reordered.connect(self._on_queue_rows_reordered)

        self.start_button.clicked.connect(self._start_selected_job)
        self.pause_button.clicked.connect(self.queue_manager.pause_active)
        self.resume_button.clicked.connect(self._resume_selected_job)
        self.cancel_button.clicked.connect(self._cancel_selected_job)
        self.clear_queue_button.clicked.connect(self._clear_queue)
        self.retry_button.clicked.connect(self._retry_selected_job)
        self.open_folder_button.clicked.connect(self._open_selected_folder)

        self.save_settings_button.clicked.connect(self._save_settings)
        self.default_root_browse_button.clicked.connect(self._browse_default_root)
        self.save_token_button.clicked.connect(self._save_token)
        self.clear_token_button.clicked.connect(self._clear_token)

        self.queue_manager.job_updated.connect(self._on_job_updated)
        self.queue_manager.queue_changed.connect(self._on_queue_changed)
        self.queue_manager.message_emitted.connect(self.statusBar().showMessage)
        self.input_edit.textEdited.connect(self._on_input_edited)
        self._auto_open_timer.timeout.connect(self._auto_open_repo_input)

    @property
    def session_token(self) -> str | None:
        token = self.token_edit.text().strip()
        return token or None

    def _load_settings_into_ui(self) -> None:
        self.default_root_edit.setText(self.settings.default_download_root)
        self.worker_spin.setValue(self.settings.worker_count)
        self.retry_spin.setValue(self.settings.retry_count)
        self.log_path_label.setText(f"Log file: {self.settings_store.log_dir / 'app.log'}")
        resolved = self.service.resolve_token(self.session_token)
        self.token_source_label.setText(f"Current token source: {resolved.source}")

    def _apply_styles(self) -> None:
        self.setStyleSheet(APP_STYLESHEET)
        self.statusBar().setSizeGripEnabled(False)

    def _reset_repo_panel(self) -> None:
        self.current_repo_details = None
        self.repo_title_label.setText("No repository loaded")
        self.repo_revision_label.setText("Load a Hugging Face repo to inspect files before downloading.")
        self.repo_description_label.setText("")
        self.classification_primary_label.setText("")
        self.classification_badges_label.setText("")
        self.classification_meta_label.setText("")
        self.selection_summary_label.setText("Selected: 0 files of 0, 0 B of 0 B")
        self.guided_mode_combo.setCurrentIndex(0)
        self.selection_preset_combo.setCurrentIndex(0)
        self.guided_mode_description_label.setText("")
        self.revision_combo.clear()
        self.destination_edit.clear()
        self.extension_filter_edit.clear()
        self.file_search_edit.clear()
        self.file_search_status_label.setText("")
        self.tree_widget.clear()
        self._clear_tree_search_matches()
        self.repo_tree_stack.setCurrentWidget(self.repo_tree_empty_label)
        self._set_repo_controls_enabled(False)

    def _set_repo_controls_enabled(self, enabled: bool) -> None:
        controls = (
            self.revision_combo,
            self.load_revision_button,
            self.destination_edit,
            self.destination_browse_button,
            self.guided_mode_combo,
            self.apply_guided_mode_button,
            self.selection_preset_combo,
            self.apply_preset_button,
            self.extension_filter_edit,
            self.select_extension_button,
            self.exclude_extension_button,
            self.file_search_edit,
            self.find_previous_button,
            self.find_next_button,
            self.tree_widget,
            self.expand_all_button,
            self.collapse_all_button,
            self.select_all_button,
            self.clear_selection_button,
            self.add_to_queue_button,
        )
        for control in controls:
            control.setEnabled(enabled)

    def _update_results_panel_state(self) -> None:
        count = self.results_list.count()
        query = self.input_edit.text().strip()
        if count == 0:
            self.results_stack.setCurrentWidget(self.results_empty_label)
            if query:
                self.results_status_label.setText("No matching repos")
                self.results_empty_label.setText(
                    "No repos matched that search. Try a shorter name, switch scope, or paste a full Hugging Face URL."
                )
            else:
                self.results_status_label.setText("No search results yet")
                self.results_empty_label.setText(
                    "Search by repo name, author, or paste a Hugging Face link to open it directly."
                )
            return

        self.results_stack.setCurrentWidget(self.results_list)
        self.results_status_label.setText(f"{count} result{'s' if count != 1 else ''}")

    def _update_search_actions(self) -> None:
        text = self.input_edit.text().strip()
        parsed = parse_repo_input(text) if text else None
        has_selection = self.results_list.currentItem() is not None
        scope_label = self.scope_combo.currentText().rstrip("s")

        self.search_button.setEnabled(bool(text))
        self.open_result_button.setEnabled(has_selection)

        if parsed is not None:
            self.search_button.setText("Open Repo")
            self.search_hint_label.setText("Direct repo link detected. Press Enter to open it immediately.")
        elif text:
            button_label = "Search Hugging Face" if self.scope_combo.currentData() == "all" else f"Search {scope_label}s"
            self.search_button.setText(button_label)
            self.search_hint_label.setText("Search by repo name, author, or organization. Press Enter to run the search.")
        else:
            self.search_button.setText("Search / Open")
            self.search_hint_label.setText("Paste a Hugging Face URL or search by repo name, author, or organization.")

    def _update_queue_controls_enabled(self) -> None:
        jobs = self.queue_manager.jobs
        active_job = self.queue_manager.active_job
        selected_job_id = self._selected_job_id()
        selected_job = self.queue_manager.get_job(selected_job_id) if selected_job_id else None
        detail_job = selected_job or active_job

        self.queue_stack.setCurrentWidget(self.queue_table if jobs else self.queue_empty_label)
        self.task_stack.setCurrentWidget(self.task_table if detail_job is not None else self.task_empty_label)

        if not jobs:
            self.queue_summary_label.setText("Queue is empty")
        else:
            pending = sum(job.status == JobStatus.PENDING for job in jobs)
            running = sum(job.status == JobStatus.RUNNING for job in jobs)
            paused = sum(job.status == JobStatus.PAUSED for job in jobs)
            completed = sum(job.status == JobStatus.COMPLETED for job in jobs)
            failed = sum(job.status in {JobStatus.FAILED, JobStatus.CANCELLED} for job in jobs)
            self.queue_summary_label.setText(
                f"{len(jobs)} job{'s' if len(jobs) != 1 else ''} | "
                f"{pending} pending | {running + paused} active/paused | "
                f"{completed} completed | {failed} failed/cancelled"
            )

        if detail_job is None:
            self.task_summary_label.setText("Select a queued job to inspect file-level progress and errors.")
        else:
            snapshot = self._job_snapshots.get(detail_job.job_id) or self._snapshot_for_job(detail_job)
            self.task_summary_label.setText(
                f"{detail_job.title} | {snapshot.completed_files + snapshot.skipped_files}/"
                f"{len(detail_job.tasks)} files done | "
                f"{snapshot.failed_files} failed | {format_bytes(snapshot.downloaded_bytes)} / "
                f"{format_bytes(snapshot.total_bytes)}"
            )

        can_start = selected_job is not None and selected_job.status == JobStatus.PENDING and active_job is None
        can_pause = active_job is not None and active_job.status == JobStatus.RUNNING
        can_resume = (
            selected_job is not None
            and selected_job.status in {JobStatus.PAUSED, JobStatus.FAILED, JobStatus.CANCELLED}
            and (active_job is None or active_job.job_id == selected_job.job_id)
        )
        can_cancel = selected_job is not None and selected_job.status in {JobStatus.PENDING, JobStatus.RUNNING, JobStatus.PAUSED}
        can_retry = selected_job is not None and selected_job.status in {JobStatus.FAILED, JobStatus.CANCELLED}

        self.start_button.setEnabled(can_start)
        self.pause_button.setEnabled(can_pause)
        self.resume_button.setEnabled(can_resume)
        self.cancel_button.setEnabled(can_cancel)
        self.clear_queue_button.setEnabled(bool(jobs))
        self.retry_button.setEnabled(can_retry)
        self.open_folder_button.setEnabled(selected_job is not None)

    def _run_async(self, fn: Callable[[], object], on_success: Callable[[object], None], label: str) -> None:
        self.statusBar().showMessage(label)
        worker = FunctionWorker(fn)
        worker.signals.finished.connect(on_success)
        worker.signals.failed.connect(self._handle_background_error)
        self.thread_pool.start(worker)

    def _handle_background_error(self, message: str) -> None:
        self.statusBar().showMessage("Operation failed")
        self.error_label.setText(message)
        QtWidgets.QMessageBox.critical(self, "Operation Failed", message)

    def _on_search_clicked(self) -> None:
        self._auto_open_timer.stop()
        self._submit_current_input()

    def _on_input_edited(self, text: str) -> None:
        if self._is_auto_open_candidate(text):
            self._auto_open_timer.start()
        else:
            self._auto_open_timer.stop()

    def _auto_open_repo_input(self) -> None:
        if self._is_auto_open_candidate(self.input_edit.text()):
            self._submit_current_input()

    def _submit_current_input(self) -> None:
        text = self.input_edit.text().strip()
        if not text:
            return

        parsed = parse_repo_input(text)
        if parsed is not None:
            self._run_async(lambda: self.service.load_repo_details(parsed, self.session_token), self._on_repo_loaded, "Opening repository...")
            return

        scope = self.scope_combo.currentData()
        self._run_async(
            lambda: self.service.search_repos(text, scope, self.session_token),
            self._on_search_results,
            "Searching Hugging Face...",
        )

    def _is_auto_open_candidate(self, text: str) -> bool:
        value = text.strip()
        if not value:
            return False
        if value.startswith(("http://", "https://", "hf://")):
            return parse_repo_input(value) is not None
        return False

    def _on_search_results(self, results: object) -> None:
        search_results = list(results)
        self.results_list.clear()
        for result in search_results:
            item = QtWidgets.QListWidgetItem(self._format_result_text(result))
            item.setData(QtCore.Qt.ItemDataRole.UserRole, result)
            item.setToolTip(self._format_result_tooltip(result))
            item.setSizeHint(QtCore.QSize(0, 52))
            self.results_list.addItem(item)
        self._update_results_panel_state()
        self._update_search_actions()
        self.statusBar().showMessage(f"Found {len(search_results)} results")

    def _format_result_text(self, result: SearchResult) -> str:
        parts = [result.repo_type.title()]
        if result.downloads is not None:
            parts.append(f"{result.downloads} downloads")
        if result.likes is not None:
            parts.append(f"{result.likes} likes")
        if result.last_modified is not None:
            parts.append(result.last_modified.strftime("%Y-%m-%d"))
        return f"{result.repo_id}\n" + " | ".join(parts)

    def _format_result_tooltip(self, result: SearchResult) -> str:
        lines = [result.repo_id, f"Type: {result.repo_type.title()}"]
        if result.summary:
            lines.append(result.summary.strip())
        return "\n".join(lines)

    def _open_selected_search_result(self) -> None:
        current_item = self.results_list.currentItem()
        if current_item is None:
            return
        result = current_item.data(QtCore.Qt.ItemDataRole.UserRole)
        self.input_edit.setText(result.repo_id)
        repo = RepoRef(repo_type=result.repo_type, repo_id=result.repo_id)
        self._run_async(lambda: self.service.load_repo_details(repo, self.session_token), self._on_repo_loaded, "Opening repository...")

    def _on_repo_loaded(self, payload: object) -> None:
        details = payload
        if not isinstance(details, RepoDetails):
            return
        self.current_repo_details = details
        self.settings_store.record_recent_repo(self.settings, details.repo)
        self.repo_title_label.setText(details.title)
        self.repo_revision_label.setText(f"Revision: {details.revision or 'latest'}")
        self.repo_description_label.setText(details.description or "")
        self._populate_classification_panel(details)
        self._populate_revision_combo(details)

        default_destination = Path(self.default_root_edit.text().strip() or self.settings.default_download_root) / details.repo.repo_type / details.repo.repo_id
        self.destination_edit.setText(str(default_destination))
        self.file_search_edit.clear()
        self.guided_mode_combo.setCurrentIndex(1)
        self.selection_preset_combo.setCurrentIndex(0)
        self.extension_filter_edit.clear()
        self._clear_tree_search_matches()
        self._populate_tree(details)
        self.repo_tree_stack.setCurrentWidget(self.tree_widget)
        self._set_repo_controls_enabled(True)
        self._update_search_actions()
        self.statusBar().showMessage(f"Loaded {details.repo.repo_id}")
        self.error_label.setText("")

    def _populate_classification_panel(self, details: RepoDetails) -> None:
        classification = details.classification
        self.classification_primary_label.setText(f"Detected: {classification.primary_label}")
        badges = " | ".join(classification.detected_labels) if classification.detected_labels else classification.modality_label
        self.classification_badges_label.setText(f"Kinds: {badges}")
        meta_parts = [classification.summary]
        if classification.license_name:
            meta_parts.append(f"License: {classification.license_name}")
        if classification.has_readme:
            meta_parts.append("README: available")
        self.classification_meta_label.setText(" | ".join(meta_parts))
        self._update_guided_mode_description()

    def _populate_revision_combo(self, details: RepoDetails) -> None:
        current_value = details.repo.requested_revision
        self.revision_combo.blockSignals(True)
        self.revision_combo.clear()
        for option in details.available_revisions:
            self.revision_combo.addItem(option.label, option.value)

        target_text = "Latest / Default"
        if current_value:
            target_text = current_value
        for index in range(self.revision_combo.count()):
            item_value = self.revision_combo.itemData(index)
            item_label = self.revision_combo.itemText(index)
            if item_value == current_value or item_label == current_value:
                self.revision_combo.setCurrentIndex(index)
                target_text = item_label
                break
        self.revision_combo.setEditText(target_text)
        self.revision_combo.blockSignals(False)

    def _load_selected_revision(self) -> None:
        if self.current_repo_details is None:
            return
        selected_revision = self._selected_revision_value()
        repo = RepoRef(
            repo_type=self.current_repo_details.repo.repo_type,
            repo_id=self.current_repo_details.repo.repo_id,
            requested_revision=selected_revision,
            deep_path=self.current_repo_details.repo.deep_path,
        )
        self._run_async(lambda: self.service.load_repo_details(repo, self.session_token), self._on_repo_loaded, "Loading revision...")

    def _selected_revision_value(self) -> str | None:
        text = self.revision_combo.currentText().strip()
        if not text or text == "Latest / Default":
            return None
        for index in range(self.revision_combo.count()):
            if self.revision_combo.itemText(index) == text:
                value = self.revision_combo.itemData(index)
                return value or None
        return text

    def _populate_tree(self, details: RepoDetails) -> None:
        self._syncing_tree = True
        self.tree_widget.clear()
        for path in details.preselected_paths:
            node = find_node(details.root, path)
            if node is not None:
                set_node_check_state(node, NodeCheckState.CHECKED)
        for child in details.root.children:
            self.tree_widget.addTopLevelItem(self._build_tree_item(child))
        self.tree_widget.expandToDepth(1)
        self._syncing_tree = False
        self._resize_tree_columns()
        self._update_selection_summary()

    def _build_tree_item(self, node) -> QtWidgets.QTreeWidgetItem:
        item = QtWidgets.QTreeWidgetItem(
            [
                node.name,
                format_bytes(node.size),
                str(node.file_count if node.is_dir else 1),
            ]
        )
        item.setData(0, QtCore.Qt.ItemDataRole.UserRole, node)
        item.setToolTip(0, node.path or node.name)
        item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
        item.setCheckState(0, QtCore.Qt.CheckState(node.check_state))
        if node.is_dir:
            item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsAutoTristate)
        for child in node.children:
            item.addChild(self._build_tree_item(child))
        return item

    def _on_tree_item_changed(self, item: QtWidgets.QTreeWidgetItem, _column: int) -> None:
        if self._syncing_tree:
            return
        node = item.data(0, QtCore.Qt.ItemDataRole.UserRole)
        if node is None:
            return
        state = NodeCheckState(item.checkState(0).value)
        set_node_check_state(node, state)
        self._syncing_tree = True
        self._refresh_tree_states(self.tree_widget.invisibleRootItem())
        self._syncing_tree = False
        self._update_selection_summary()

    def _refresh_tree_states(self, parent_item: QtWidgets.QTreeWidgetItem) -> None:
        for index in range(parent_item.childCount()):
            child_item = parent_item.child(index)
            node = child_item.data(0, QtCore.Qt.ItemDataRole.UserRole)
            if node is not None:
                child_item.setCheckState(0, QtCore.Qt.CheckState(node.check_state))
            self._refresh_tree_states(child_item)

    def _update_selection_summary(self) -> None:
        if self.current_repo_details is None:
            self.selection_summary_label.setText("Selected: 0 files of 0, 0 B of 0 B")
            return
        count, size = summarize_selection(self.current_repo_details.root)
        self.selection_summary_label.setText(
            f"Selected: {count} files of {self.current_repo_details.total_files}, "
            f"{format_bytes(size)} of {format_bytes(self.current_repo_details.total_size)}"
        )

    def _select_all(self) -> None:
        if self.current_repo_details is None:
            return
        set_node_check_state(self.current_repo_details.root, NodeCheckState.CHECKED)
        self._sync_tree_from_model()

    def _expand_all(self) -> None:
        self.tree_widget.expandAll()
        self._resize_tree_columns()

    def _collapse_all(self) -> None:
        self.tree_widget.collapseAll()
        self._resize_tree_columns()

    def _clear_selection(self) -> None:
        if self.current_repo_details is None:
            return
        set_node_check_state(self.current_repo_details.root, NodeCheckState.UNCHECKED)
        self._sync_tree_from_model()

    def _update_guided_mode_description(self) -> None:
        if self.current_repo_details is None:
            self.guided_mode_description_label.setText("")
            return
        mode = self.guided_mode_combo.currentData()
        self.guided_mode_description_label.setText(mode_description(mode, self.current_repo_details.classification))

    def _apply_guided_mode(self) -> None:
        if self.current_repo_details is None:
            return
        file_paths = [node.path for node in self._iter_file_nodes(self.current_repo_details.root)]
        selected_paths = select_paths_for_mode(file_paths, self.current_repo_details.classification, self.guided_mode_combo.currentData())
        self._apply_explicit_paths(selected_paths)

    def _sync_tree_from_model(self) -> None:
        self._syncing_tree = True
        self._refresh_tree_states(self.tree_widget.invisibleRootItem())
        self._syncing_tree = False
        self._update_selection_summary()

    def _apply_selection_preset(self) -> None:
        if self.current_repo_details is None:
            return

        preset = self.selection_preset_combo.currentData()
        if preset == "none":
            return
        if preset == "all":
            self._select_all()
            return

        if preset == "exclude_tokenizer":
            set_node_check_state(self.current_repo_details.root, NodeCheckState.CHECKED)
            self._apply_match_to_files(self._is_tokenizer_related, select_mode=False, reset_root=False)
            return

        matchers = {
            "safetensors": lambda path: path.lower().endswith(".safetensors"),
            "images": lambda path: Path(path).suffix.lower() in IMAGE_EXTENSIONS,
            "configs": self._is_config_or_tokenizer_related,
        }
        matcher = matchers.get(preset)
        if matcher is None:
            return
        set_node_check_state(self.current_repo_details.root, NodeCheckState.UNCHECKED)
        self._apply_match_to_files(matcher, select_mode=True, reset_root=False)

    def _apply_extension_filter(self, select_mode: bool) -> None:
        if self.current_repo_details is None:
            return

        extensions = self._parse_extensions(self.extension_filter_edit.text())
        if not extensions:
            return

        if select_mode:
            set_node_check_state(self.current_repo_details.root, NodeCheckState.UNCHECKED)
        else:
            set_node_check_state(self.current_repo_details.root, NodeCheckState.CHECKED)
        self._apply_match_to_files(lambda path: Path(path).suffix.lower() in extensions, select_mode=select_mode, reset_root=False)

    def _apply_match_to_files(self, matcher: Callable[[str], bool], select_mode: bool, reset_root: bool) -> None:
        if self.current_repo_details is None:
            return
        if reset_root:
            target_state = NodeCheckState.UNCHECKED if select_mode else NodeCheckState.CHECKED
            set_node_check_state(self.current_repo_details.root, target_state)
        for node in self._iter_file_nodes(self.current_repo_details.root):
            if matcher(node.path):
                set_node_check_state(node, NodeCheckState.CHECKED if select_mode else NodeCheckState.UNCHECKED)
        self._sync_tree_from_model()

    def _apply_explicit_paths(self, paths: list[str]) -> None:
        if self.current_repo_details is None:
            return
        selected_set = set(paths)
        set_node_check_state(self.current_repo_details.root, NodeCheckState.UNCHECKED)
        for node in self._iter_file_nodes(self.current_repo_details.root):
            if node.path in selected_set:
                set_node_check_state(node, NodeCheckState.CHECKED)
        self._sync_tree_from_model()

    def _iter_file_nodes(self, node) -> list:
        files = []
        for child in node.children:
            if child.is_dir:
                files.extend(self._iter_file_nodes(child))
            else:
                files.append(child)
        return files

    def _parse_extensions(self, raw_text: str) -> set[str]:
        extensions: set[str] = set()
        for token in raw_text.replace(";", ",").split(","):
            cleaned = token.strip().lower()
            if not cleaned:
                continue
            if not cleaned.startswith("."):
                cleaned = f".{cleaned}"
            extensions.add(cleaned)
        return extensions

    def _is_tokenizer_related(self, path: str) -> bool:
        lower_path = path.lower()
        file_name = Path(path).name.lower()
        return "tokenizer" in lower_path or file_name in TOKENIZER_FILENAMES

    def _is_config_or_tokenizer_related(self, path: str) -> bool:
        lower_path = path.lower()
        file_name = Path(path).name.lower()
        return (
            self._is_tokenizer_related(path)
            or file_name.endswith(".json")
            or file_name.endswith(".yaml")
            or file_name.endswith(".yml")
            or "config" in lower_path
        )

    def _resize_tree_columns(self) -> None:
        for column in range(self.tree_widget.columnCount()):
            self.tree_widget.resizeColumnToContents(column)

    def _on_tree_search_changed(self, text: str) -> None:
        self._apply_tree_search(text)

    def _find_next_tree_match(self) -> None:
        if not self._tree_search_matches:
            self._apply_tree_search(self.file_search_edit.text())
            if not self._tree_search_matches:
                return
        self._tree_search_index = (self._tree_search_index + 1) % len(self._tree_search_matches)
        self._focus_tree_match(self._tree_search_index)

    def _find_previous_tree_match(self) -> None:
        if not self._tree_search_matches:
            self._apply_tree_search(self.file_search_edit.text())
            if not self._tree_search_matches:
                return
        self._tree_search_index = (self._tree_search_index - 1) % len(self._tree_search_matches)
        self._focus_tree_match(self._tree_search_index)

    def _apply_tree_search(self, text: str) -> None:
        self._clear_tree_search_matches()
        query = text.strip().lower()
        if not query:
            self.file_search_status_label.setText("")
            return

        root = self.tree_widget.invisibleRootItem()
        self._tree_search_matches = self._collect_tree_matches(root, query)
        for item in self._tree_search_matches:
            self._set_tree_item_match_style(item, matched=True, active=False)

        if not self._tree_search_matches:
            self._tree_search_index = -1
            self.file_search_status_label.setText("No matches")
            return

        self._tree_search_index = 0
        self.file_search_status_label.setText(f"1 / {len(self._tree_search_matches)}")
        self._focus_tree_match(self._tree_search_index)

    def _collect_tree_matches(self, parent_item: QtWidgets.QTreeWidgetItem, query: str) -> list[QtWidgets.QTreeWidgetItem]:
        matches: list[QtWidgets.QTreeWidgetItem] = []
        for index in range(parent_item.childCount()):
            item = parent_item.child(index)
            node = item.data(0, QtCore.Qt.ItemDataRole.UserRole)
            haystack = ""
            if node is not None:
                haystack = f"{node.name} {node.path}".lower()
            else:
                haystack = item.text(0).lower()
            if query in haystack:
                matches.append(item)
            matches.extend(self._collect_tree_matches(item, query))
        return matches

    def _focus_tree_match(self, index: int) -> None:
        if not self._tree_search_matches:
            return

        for match_index, item in enumerate(self._tree_search_matches):
            self._set_tree_item_match_style(item, matched=True, active=match_index == index)

        item = self._tree_search_matches[index]
        self._expand_tree_item_parents(item)
        self.tree_widget.setCurrentItem(item)
        self.tree_widget.scrollToItem(item, QtWidgets.QAbstractItemView.ScrollHint.PositionAtCenter)
        self.file_search_status_label.setText(f"{index + 1} / {len(self._tree_search_matches)}")
        self._resize_tree_columns()

    def _expand_tree_item_parents(self, item: QtWidgets.QTreeWidgetItem) -> None:
        parent = item.parent()
        while parent is not None:
            parent.setExpanded(True)
            parent = parent.parent()

    def _clear_tree_search_matches(self) -> None:
        for item in self._tree_search_matches:
            self._set_tree_item_match_style(item, matched=False, active=False)
        self._tree_search_matches = []
        self._tree_search_index = -1

    def _set_tree_item_match_style(self, item: QtWidgets.QTreeWidgetItem, matched: bool, active: bool) -> None:
        if not matched:
            for column in range(self.tree_widget.columnCount()):
                item.setBackground(column, QtGui.QBrush())
            return

        color = QtGui.QColor("#4a6fa5") if active else QtGui.QColor("#2f4f2f")
        brush = QtGui.QBrush(color)
        for column in range(self.tree_widget.columnCount()):
            item.setBackground(column, brush)

    def _browse_destination(self) -> None:
        selected = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose Destination", self.destination_edit.text())
        if selected:
            self.destination_edit.setText(selected)

    def _browse_default_root(self) -> None:
        selected = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose Default Download Root", self.default_root_edit.text())
        if selected:
            self.default_root_edit.setText(selected)

    def _add_current_selection_to_queue(self) -> None:
        if self.current_repo_details is None:
            QtWidgets.QMessageBox.warning(self, "No Repository", "Load a repository before adding a download job.")
            return

        selected_files = selected_allow_patterns(self.current_repo_details.root)
        if not selected_files:
            QtWidgets.QMessageBox.warning(self, "No Files Selected", "Select at least one file or folder to download.")
            return

        destination = Path(self.destination_edit.text().strip())
        if not destination:
            QtWidgets.QMessageBox.warning(self, "Missing Destination", "Choose a destination folder first.")
            return

        self._save_settings(show_message=False)
        self._run_async(
            lambda: self.service.plan_download(
                repo=self.current_repo_details.repo,
                title=self.current_repo_details.title,
                selected_files=selected_files,
                destination=destination,
                session_token=self.session_token,
                worker_count=self.worker_spin.value(),
                retry_count=self.retry_spin.value(),
            ),
            self._on_job_planned,
            "Planning download...",
        )

    def _on_job_planned(self, payload: object) -> None:
        job = payload
        if not isinstance(job, DownloadJob):
            return
        self.queue_manager.add_job(job)
        self._refresh_queue_table(select_job_id=job.job_id)
        self.statusBar().showMessage(
            f"Queued {job.title}: {len(job.tasks)} files, {format_bytes(job.total_selected_bytes)} selected, "
            f"{format_bytes(job.bytes_to_download)} to download"
        )

    def _refresh_queue_table(self, select_job_id: str | None = None) -> None:
        selected_job_id = select_job_id or self._selected_job_id()
        jobs = self.queue_manager.jobs
        self._queue_row_by_job_id = {job.job_id: row for row, job in enumerate(jobs)}
        self.queue_table.setRowCount(len(jobs))
        for row, job in enumerate(jobs):
            snapshot = self._job_snapshots.get(job.job_id) or self._snapshot_for_job(job)
            self._update_queue_row(row, job, snapshot)

        self.queue_table.resizeColumnsToContents()

        if selected_job_id:
            for row in range(self.queue_table.rowCount()):
                item = self.queue_table.item(row, 0)
                if item is not None and item.data(QtCore.Qt.ItemDataRole.UserRole) == selected_job_id:
                    self.queue_table.selectRow(row)
                    break

        self._refresh_progress_panel()
        self._refresh_task_table()
        self._update_queue_controls_enabled()

    def _update_queue_row(self, row: int, job: DownloadJob, snapshot: ProgressSnapshot) -> None:
        values = [
            job.title,
            job.repo.repo_type,
            job.status,
            "",
            format_speed(snapshot.speed_bps),
            format_duration(snapshot.eta_seconds),
            f"{snapshot.completed_files + snapshot.skipped_files}/{len(job.tasks)}",
            str(job.destination),
        ]
        for column, value in enumerate(values):
            if column == 3:
                self._update_queue_progress_widget(row, snapshot)
                continue
            item = self.queue_table.item(row, column)
            if item is None:
                item = QtWidgets.QTableWidgetItem()
                self.queue_table.setItem(row, column, item)
            item.setText(str(value))
            if column == 0:
                item.setData(QtCore.Qt.ItemDataRole.UserRole, job.job_id)
            if column == 2:
                item.setForeground(QtGui.QBrush(QtGui.QColor("white")))
                item.setBackground(QtGui.QBrush(QtGui.QColor(STATUS_COLORS.get(job.status, "#5c6b73"))))
                item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            elif column in {1, 4, 5, 6}:
                item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                item.setBackground(QtGui.QBrush())
                item.setForeground(QtGui.QBrush())
            else:
                item.setBackground(QtGui.QBrush())
                item.setForeground(QtGui.QBrush())
        self._update_queue_actions_widget(row, job)

    def _update_queue_progress_widget(self, row: int, snapshot: ProgressSnapshot) -> None:
        widget = self.queue_table.cellWidget(row, 3)
        if widget is None:
            progress_bar = QtWidgets.QProgressBar()
            progress_bar.setRange(0, 1000)
            progress_bar.setTextVisible(True)
            progress_bar.setFormat("%p%")
            self.queue_table.setCellWidget(row, 3, progress_bar)
            widget = progress_bar
        if isinstance(widget, QtWidgets.QProgressBar):
            widget.setValue(int(snapshot.percent * 10))
            widget.setToolTip(
                f"{format_bytes(snapshot.downloaded_bytes)} / {format_bytes(snapshot.total_bytes)} at {format_speed(snapshot.speed_bps)}"
            )

    def _update_queue_actions_widget(self, row: int, job: DownloadJob) -> None:
        widget = self.queue_table.cellWidget(row, 8)
        buttons: dict[str, QtWidgets.QToolButton]
        if widget is None:
            container = QtWidgets.QWidget()
            layout = QtWidgets.QHBoxLayout(container)
            layout.setContentsMargins(4, 2, 4, 2)
            layout.setSpacing(4)
            buttons = {}
            for name in ("primary", "secondary", "folder"):
                button = QtWidgets.QToolButton()
                button.setProperty("role_name", name)
                buttons[name] = button
                layout.addWidget(button)
            layout.addStretch(1)
            self.queue_table.setCellWidget(row, 8, container)
            widget = container

        buttons = {
            child.property("role_name"): child
            for child in widget.findChildren(QtWidgets.QToolButton)
        }
        primary = buttons["primary"]
        secondary = buttons["secondary"]
        folder = buttons["folder"]

        self._reset_tool_button(primary)
        self._reset_tool_button(secondary)
        self._reset_tool_button(folder)

        if job.status == JobStatus.PENDING:
            self._configure_tool_button(primary, "Start", lambda _checked=False, job_id=job.job_id: self.queue_manager.start_job(job_id))
            self._configure_tool_button(secondary, "Delete", lambda _checked=False, job_id=job.job_id: self.queue_manager.delete_job(job_id))
        elif job.status in {JobStatus.RUNNING, JobStatus.PAUSED}:
            if job.status == JobStatus.RUNNING:
                self._configure_tool_button(primary, "Pause", lambda _checked=False: self.queue_manager.pause_active())
            else:
                self._configure_tool_button(primary, "Resume", lambda _checked=False, job_id=job.job_id: self.queue_manager.resume_job(job_id))
            self._configure_tool_button(secondary, "Cancel", lambda _checked=False, job_id=job.job_id: self.queue_manager.cancel_job(job_id))
        elif job.status in {JobStatus.FAILED, JobStatus.CANCELLED}:
            self._configure_tool_button(primary, "Retry", lambda _checked=False, job_id=job.job_id: self._retry_job_from_row(job_id))
            self._configure_tool_button(secondary, "Delete", lambda _checked=False, job_id=job.job_id: self.queue_manager.delete_job(job_id))
        else:
            primary.hide()
            self._configure_tool_button(secondary, "Delete", lambda _checked=False, job_id=job.job_id: self.queue_manager.delete_job(job_id))

        self._configure_tool_button(folder, "Folder", lambda _checked=False, destination=job.destination: self._open_folder_path(destination))

    def _reset_tool_button(self, button: QtWidgets.QToolButton) -> None:
        try:
            button.clicked.disconnect()
        except TypeError:
            pass
        button.show()

    def _configure_tool_button(self, button: QtWidgets.QToolButton, text: str, callback: Callable[[], None]) -> None:
        button.setText(text)
        button.clicked.connect(callback)
        button.setAutoRaise(True)
        button.show()

    def _snapshot_for_job(self, job: DownloadJob) -> ProgressSnapshot:
        total = sum(task.size for task in job.tasks)
        downloaded = sum(task.downloaded_bytes for task in job.tasks)
        percent = 100.0 if total == 0 else (downloaded / total) * 100
        return ProgressSnapshot(
            job_id=job.job_id,
            status=job.status,
            total_bytes=total,
            downloaded_bytes=downloaded,
            percent=percent,
            speed_bps=0.0,
            eta_seconds=None,
            completed_files=sum(task.status.value == "Completed" for task in job.tasks),
            failed_files=sum(task.status.value == "Failed" for task in job.tasks),
            skipped_files=sum(task.status.value == "Skipped" for task in job.tasks),
            queued_files=sum(task.status.value == "Pending" for task in job.tasks),
            active_files=sum(task.status.value in {"Running", "Paused"} for task in job.tasks),
            cancelled_files=sum(task.status.value == "Cancelled" for task in job.tasks),
        )

    def _on_job_updated(self, job: DownloadJob, snapshot: ProgressSnapshot) -> None:
        self._job_snapshots[job.job_id] = snapshot
        if job.last_error:
            self.error_label.setText(job.last_error)
        row = self._queue_row_by_job_id.get(job.job_id)
        if row is None:
            self._refresh_queue_table(select_job_id=job.job_id)
        else:
            self._update_queue_row(row, job, snapshot)
            self._refresh_progress_panel()
            self._refresh_task_table()
            self._update_queue_controls_enabled()

    def _on_queue_changed(self, jobs: object) -> None:
        current_ids = {job.job_id for job in jobs}
        self._job_snapshots = {job_id: snapshot for job_id, snapshot in self._job_snapshots.items() if job_id in current_ids}
        self._refresh_queue_table()

    def _selected_job_id(self) -> str | None:
        selected_items = self.queue_table.selectedItems()
        if not selected_items:
            return None
        return selected_items[0].data(QtCore.Qt.ItemDataRole.UserRole)

    def _selected_job(self) -> DownloadJob | None:
        job_id = self._selected_job_id()
        if not job_id:
            return self.queue_manager.active_job
        return self.queue_manager.get_job(job_id)

    def _on_queue_selection_changed(self) -> None:
        self._refresh_progress_panel()
        self._refresh_task_table()
        self._update_queue_controls_enabled()

    def _show_queue_context_menu(self, position: QtCore.QPoint) -> None:
        item = self.queue_table.itemAt(position)
        if item is None:
            return

        self.queue_table.selectRow(item.row())
        job = self._selected_job()
        if job is None:
            return

        active_job = self.queue_manager.active_job
        can_start = job.status == JobStatus.PENDING and active_job is None
        can_cancel = job.status in {JobStatus.PENDING, JobStatus.RUNNING, JobStatus.PAUSED}

        menu = QtWidgets.QMenu(self)
        start_action = menu.addAction("Start")
        cancel_action = menu.addAction("Cancel")
        delete_action = menu.addAction("Delete")

        start_action.setEnabled(can_start)
        cancel_action.setEnabled(can_cancel)

        start_action.triggered.connect(self._start_selected_job)
        cancel_action.triggered.connect(self._cancel_selected_job)
        delete_action.triggered.connect(self._delete_selected_job)
        menu.exec(self.queue_table.viewport().mapToGlobal(position))

    def _on_queue_rows_reordered(self, source_row: int, target_row: int) -> None:
        moved_job_id = None
        if 0 <= source_row < len(self.queue_manager.jobs):
            moved_job_id = self.queue_manager.jobs[source_row].job_id
        selected_job_id = self._selected_job_id()
        moved = self.queue_manager.move_job(source_row, target_row)
        if moved:
            self._refresh_queue_table(select_job_id=moved_job_id or selected_job_id)
        else:
            self._refresh_queue_table(select_job_id=selected_job_id)

    def _refresh_progress_panel(self) -> None:
        job = self._selected_job() or self.queue_manager.active_job
        if job is None:
            self.active_progress_bar.setValue(0)
            self.active_progress_label.setText("No active download")
            return

        snapshot = self._job_snapshots.get(job.job_id) or self._snapshot_for_job(job)
        self.active_progress_bar.setValue(int(snapshot.percent * 10))
        self.active_progress_label.setText(
            f"{job.title}: {snapshot.percent:.1f}% | {format_bytes(snapshot.downloaded_bytes)} / "
            f"{format_bytes(snapshot.total_bytes)} | {format_speed(snapshot.speed_bps)} | ETA "
            f"{format_duration(snapshot.eta_seconds)} | "
            f"{snapshot.completed_files + snapshot.skipped_files}/{len(job.tasks)} files done"
        )

    def _refresh_task_table(self) -> None:
        job = self._selected_job()
        tasks = [] if job is None else job.tasks
        self.task_stack.setCurrentWidget(self.task_table if tasks else self.task_empty_label)
        self.task_table.setRowCount(len(tasks))
        for row, task in enumerate(tasks):
            progress = 100.0 if task.size == 0 else (task.downloaded_bytes / task.size) * 100
            values = [
                task.filename,
                task.status,
                f"{progress:.1f}%",
                format_bytes(task.size),
                task.error,
            ]
            for column, value in enumerate(values):
                self.task_table.setItem(row, column, QtWidgets.QTableWidgetItem(str(value)))
        self.task_table.resizeColumnsToContents()

    def _start_selected_job(self) -> None:
        selected_job = self._selected_job()
        job_id = selected_job.job_id if selected_job and selected_job.status == JobStatus.PENDING else None
        self.queue_manager.start_job(job_id)

    def _resume_selected_job(self) -> None:
        selected_job = self._selected_job()
        if selected_job is None:
            return
        self.queue_manager.resume_job(selected_job.job_id)

    def _cancel_selected_job(self) -> None:
        selected_job = self._selected_job()
        if selected_job is None:
            return
        self.queue_manager.cancel_job(selected_job.job_id)

    def _delete_selected_job(self) -> None:
        selected_job = self._selected_job()
        if selected_job is None:
            return
        self.queue_manager.delete_job(selected_job.job_id)

    def _clear_queue(self) -> None:
        self.queue_manager.clear_queue()

    def _retry_selected_job(self) -> None:
        selected_job = self._selected_job()
        if selected_job is None:
            return
        self.queue_manager.retry_failed(selected_job.job_id)
        self.queue_manager.start_job(selected_job.job_id)

    def _open_selected_folder(self) -> None:
        selected_job = self._selected_job()
        if selected_job is None:
            return
        self._open_folder_path(selected_job.destination)

    def _retry_job_from_row(self, job_id: str) -> None:
        self.queue_manager.retry_failed(job_id)
        self.queue_manager.start_job(job_id)

    def _open_folder_path(self, destination: Path) -> None:
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(destination)))

    def _save_settings(self, show_message: bool = True) -> None:
        self.settings.default_download_root = self.default_root_edit.text().strip() or self.settings.default_download_root
        self.settings.worker_count = self.worker_spin.value()
        self.settings.retry_count = self.retry_spin.value()
        self.settings_store.save(self.settings)
        self.token_source_label.setText(f"Current token source: {self.service.resolve_token(self.session_token).source}")
        if show_message:
            self.statusBar().showMessage("Settings saved")

    def _save_token(self) -> None:
        token = self.token_edit.text().strip()
        if not token:
            QtWidgets.QMessageBox.warning(self, "Missing Token", "Paste a token before saving it.")
            return
        try:
            self.service.auth_resolver.token_store.save_token(token)
        except Exception as exc:  # pragma: no cover - platform-specific keyring failure
            QtWidgets.QMessageBox.critical(self, "Token Save Failed", str(exc))
            return
        self.token_source_label.setText("Current token source: session/keyring")
        self.statusBar().showMessage("Token saved to keyring")

    def _clear_token(self) -> None:
        self.token_edit.clear()
        try:
            self.service.auth_resolver.token_store.clear_token()
        except Exception as exc:  # pragma: no cover - platform-specific keyring failure
            QtWidgets.QMessageBox.critical(self, "Token Clear Failed", str(exc))
            return
        self.token_source_label.setText(f"Current token source: {self.service.resolve_token(None).source}")
        self.statusBar().showMessage("Token cleared")
