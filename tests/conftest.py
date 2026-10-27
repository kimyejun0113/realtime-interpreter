import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="session")
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


class FakeTranslator:
    """Stands in for any engine: records calls, returns canned text or raises."""

    def __init__(self, name="fake", reply="번역", error=None):
        self.model = self.name = name
        self.reply = reply
        self.error = error
        self.calls = []

    def stream(self, text, src, tgt, context, partial=False):
        self.calls.append((text, src, tgt, context))
        if self.error:
            raise self.error
        yield self.reply

    def complete(self, messages, max_tokens=250):
        self.calls.append(messages)
        if self.error:
            raise self.error
        return self.reply


@pytest.fixture
def fake_translator():
    return FakeTranslator
