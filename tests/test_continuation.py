"""An unfinished sentence is re-translated together with the next piece, replacing its subtitle."""
import queue
import threading
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import pipeline
from pipeline import Events, Pipeline, Utterance
from quota import QuotaTracker
from translate.glossary import Glossary


class EchoTranslator:
    name = model = "echo"

    def __init__(self):
        self.sources = []

    def stream(self, text, src, tgt, context, partial=False):
        self.sources.append(text)
        yield f"<{text}>"


class FakeMemo:
    memo = ""

    def __init__(self):
        self.added = []

    def add(self, source, translation, src, tgt):
        self.added.append(source)


@pytest.fixture
def running_pipeline():
    p = Pipeline.__new__(Pipeline)  # no models/threads except the final loop under test
    p.cfg = SimpleNamespace(final_stt="local", conversation_memo=True, context_recent=2, context_older=6,
                            main_target_lang="ko", other_lang="en")
    p.events = Events()
    p.finals = queue.Queue()
    p.quota = QuotaTracker({})
    p.translator = EchoTranslator()
    p.final_translators = [p.translator]
    p.history = deque(maxlen=8)
    p.memo = FakeMemo()
    p.glossary = Glossary(Path("no-glossary.txt"))
    p._open = None
    p.logger = SimpleNamespace(write=lambda *a, **k: None)
    p.running = threading.Event()
    p.device, p.output_name = None, ""
    log = []
    p.events.on_final = lambda seq, text: log.append(("final", seq, text))
    p.events.on_merged = lambda seq: log.append(("merged", seq))
    threading.Thread(target=p._final_loop, daemon=True).start()
    return p, log


def say(p, seq, text, lang="en"):
    p.finals.put(Utterance(seq, text, lang, np.zeros(0, np.int16), time.perf_counter()))


def wait_for(cond, timeout=3.0):
    end = time.time() + timeout
    while time.time() < end and not cond():
        time.sleep(0.02)
    return cond()


def test_unfinished_sentence_is_retranslated_with_the_next_piece(running_pipeline):
    p, log = running_pipeline
    say(p, 1, "if everything goes according to plan and the legal team signs off")
    assert wait_for(lambda: ("final", 1, "<if everything goes according to plan and the legal team signs off>") in log)
    say(p, 2, "we will ship it to every customer next month.")
    combined = "if everything goes according to plan and the legal team signs off we will ship it to every customer next month."
    assert wait_for(lambda: ("final", 2, f"<{combined}>") in log)
    assert ("merged", 1) in log and log.index(("merged", 1)) < log.index(("final", 2, f"<{combined}>"))
    assert list(p.history) == [(combined, f"<{combined}>")]  # the half sentence was replaced, not kept
    assert p.memo.added == [combined]


def test_finished_sentence_is_not_merged(running_pipeline):
    p, log = running_pipeline
    say(p, 1, "We trained the model on more data.")
    say(p, 2, "Accuracy went up quite a bit.")
    assert wait_for(lambda: len([e for e in log if e[0] == "final"]) == 2)
    assert not any(e[0] == "merged" for e in log)
    assert p.translator.sources == ["We trained the model on more data.", "Accuracy went up quite a bit."]


def test_only_two_pieces_are_joined(running_pipeline):
    p, log = running_pipeline
    say(p, 1, "so the first thing we tried was a much bigger batch size")
    say(p, 2, "and then a smaller learning rate on top of that")
    say(p, 3, "which finally made the loss go down")
    assert wait_for(lambda: len([e for e in log if e[0] == "final"]) == 3)
    assert [e for e in log if e[0] == "merged"] == [("merged", 1)]
    assert p.translator.sources[-1] == "which finally made the loss go down"


@pytest.mark.parametrize("second,lang,delay_patch", [
    ("다음 주에 다시 이야기해요.", "ko", None),  # language changed
    ("we will ship it next month.", "en", 0.0),   # waited too long
    (" ".join(["word"] * 35) + ".", "en", None),   # too long together
])
def test_not_merged_when_it_would_not_help(running_pipeline, monkeypatch, second, lang, delay_patch):
    p, log = running_pipeline
    if delay_patch is not None:
        monkeypatch.setattr(pipeline, "OPEN_SENTENCE_SEC", delay_patch)
    say(p, 1, "if everything goes according to plan")
    assert wait_for(lambda: any(e[:2] == ("final", 1) for e in log))
    say(p, 2, second, lang)
    assert wait_for(lambda: any(e[:2] == ("final", 2) for e in log))
    assert not any(e[0] == "merged" for e in log)
    assert p.memo.added[0] == "if everything goes according to plan"  # reaches the notes as it was


def test_overlay_ignores_late_updates_for_a_merged_line(qapp):
    from ui.overlay import SubtitleOverlay

    w = SubtitleOverlay()
    w.on_live(1, "if everything goes")
    w.on_final(1, "모든 것이 순조롭다면…")
    w.on_live(2, "we ship it.")
    w.on_merged(1)
    w.on_source(1, "if everything goes (spacing fix)", "en")  # arrives after the merge
    w.on_live(1, "stale")
    assert 1 not in w.entries
    assert "모든 것이 순조롭다면" not in w.body.text()


@pytest.mark.parametrize("source,translation,unfinished", [
    ("If the legal team signs off on it.", "법무팀의 승인을 받는다면…", True),   # Whisper added a full stop
    ("So the main thing we still need...", "우리가 아직 필요한 것은", True),
    ("if everything goes according to plan", "계획대로 된다면", True),
    ("We will ship it next month.", "다음 달에 출시합니다.", False),
    ("Can you send it by Thursday?", "목요일까지 보내 주시겠습니까?", False),
])
def test_unfinished(source, translation, unfinished):
    assert pipeline._unfinished(source, translation) is unfinished


@pytest.mark.parametrize("source,filler", [
    ("and", True), ("So,", True), ("you know, so...", True), ("Um.", True), ("음", True), ("", True),
    ("I know.", False), ("Yeah.", False), ("so he's against", False), ("46", False), ("그래서 그건", False),
])
def test_filler_only(source, filler):
    assert pipeline._filler_only(source) is filler


def test_filler_fragment_is_dropped_without_a_request(running_pipeline, monkeypatch):
    p, log = running_pipeline
    monkeypatch.setattr(pipeline, "MERGE_WAIT_SEC", 0.05)  # nothing follows soon enough to absorb it
    say(p, 1, "you know, so")
    assert wait_for(lambda: ("merged", 1) in log)
    say(p, 2, "We shipped it.")
    assert wait_for(lambda: ("final", 2, "<We shipped it.>") in log)
    assert ("merged", 1) in log and p.translator.sources == ["We shipped it."]


def test_whisper_full_stop_does_not_hide_an_open_sentence(running_pipeline, monkeypatch):
    p, log = running_pipeline
    p.translator.stream = lambda text, *a, **k: iter([f"<{text}>" if text.endswith("month.") else "승인을 받는다면…"])
    say(p, 1, "If the legal team signs off on it.")
    say(p, 2, "we will ship it next month.")
    assert wait_for(lambda: ("merged", 1) in log)
