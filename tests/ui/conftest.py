"""Conftest da suíte de UI: garante offscreen ANTES de qualquer import do Qt
e uma única `QApplication` para toda a sessão (Qt não aceita mais de uma)."""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication


@pytest.fixture(scope="session", autouse=True)
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app
