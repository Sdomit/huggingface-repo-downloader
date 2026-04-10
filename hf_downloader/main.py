from __future__ import annotations

import sys

from PyQt6 import QtWidgets

from .auth import AuthResolver
from .hf_service import HuggingFaceService
from .logging_utils import configure_logging
from .queue_manager import QueueManager
from .settings import SettingsStore
from .ui import MainWindow


def main() -> int:
    app = QtWidgets.QApplication(sys.argv)
    settings_store = SettingsStore()
    settings = settings_store.load()
    logger = configure_logging(settings_store.log_dir)
    service = HuggingFaceService(auth_resolver=AuthResolver())
    queue_manager = QueueManager(service.download_func, settings_store=settings_store, settings=settings)
    window = MainWindow(service=service, queue_manager=queue_manager, settings_store=settings_store, settings=settings, logger=logger)
    window.show()
    return app.exec()
