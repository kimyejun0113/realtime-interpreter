import time

import pytest

from ui import overlay as overlay_mod
from ui.overlay import SubtitleOverlay


@pytest.fixture
def window(qapp):
    w = SubtitleOverlay()
    w.settings.remove("geometry")
    w.resize(700, 230)
    w.show()
    qapp.processEvents()
    yield w
    w.hide()
    w.deleteLater()


def shown(w):
    return w.body.text()


def finish(w, seq, src, tr):
    w.on_live(seq, src)
    w.on_final(seq, tr)


def test_newest_translation_always_visible_even_when_huge(window):
    finish(window, 1, "First sentence.", "첫 번째 문장입니다.")
    long_src = " ".join(f"word{i}" for i in range(120))
    window.on_live(2, long_src)
    window.on_partial(2, "아주 긴 임시 번역 " * 30 + "끝부분")
    html = shown(window)
    assert "끝부분" in html and "word119" in html  # the newest words survive trimming
    assert "…" in html


def test_unread_translation_kept_over_live_english(window):
    finish(window, 1, "Short.", "읽는 중인 번역입니다.")
    window.on_live(2, " ".join(f"live{i}" for i in range(80)))
    assert "읽는 중인 번역입니다." in shown(window)


def test_read_translations_make_room_first(window, monkeypatch):
    for i in range(1, 4):
        finish(window, i, f"Sentence {i}.", f"번역{i}")
    for e in list(window.entries.values())[:2]:
        e["final_at"] = time.monotonic() - 60  # long since read
    window.on_live(4, " ".join(f"live{i}" for i in range(60)))
    html = shown(window)
    assert "번역3" in html and "번역1" not in html


def test_partial_ignored_after_final(window):
    finish(window, 1, "Done.", "확정")
    window.on_partial(1, "늦게 온 임시 번역")
    assert "확정" in shown(window) and "늦게 온" not in shown(window)


def test_merged_entry_disappears(window):
    window.on_live(1, "which I")
    window.on_live(2, "we tried something new.")
    window.on_merged(1)
    assert 1 not in window.entries and "which I" not in shown(window)


def test_read_time_scales_with_length():
    short = {"done": True, "final_at": 0.0, "translation": "짧다"}
    long = {"done": True, "final_at": 0.0, "translation": "가" * 200}
    pending = {"done": False, "final_at": None, "translation": ""}
    assert SubtitleOverlay._read_until(short) == overlay_mod.READ_MIN_SEC
    assert SubtitleOverlay._read_until(long) == overlay_mod.READ_MAX_SEC
    assert SubtitleOverlay._read_until(pending) == float("inf")
