import time
from collections import deque
from types import SimpleNamespace

from pipeline import Pipeline
from quota import QuotaTracker
from translate import memo as memo_mod
from translate.memo import MemoKeeper


def pipeline_with_history(pairs, recent=2, older=6, memo=""):
    p = Pipeline.__new__(Pipeline)
    p.cfg = SimpleNamespace(context_recent=recent, context_older=older, conversation_memo=True)
    p.history = deque(pairs, maxlen=recent + older)
    p.memo = SimpleNamespace(memo=memo)
    return p


def test_recent_pairs_and_older_sources():
    pairs = [(f"src{i}", f"tr{i}") for i in range(8)]
    ctx = pipeline_with_history(pairs, memo="주제: 테스트")._context()
    assert ctx.recent == [("src6", "tr6"), ("src7", "tr7")]
    assert ctx.older == [f"src{i}" for i in range(6)]
    assert ctx.memo == "주제: 테스트"


def test_older_sources_capped_keeping_newest():
    pairs = [("x" * 250 + str(i), "t") for i in range(8)]
    ctx = pipeline_with_history(pairs)._context()
    assert sum(map(len, ctx.older)) <= Pipeline.OLDER_MAX_CHARS
    assert ctx.older[-1].endswith("5")  # the newest older sentence survives the cap


def test_empty_history():
    ctx = pipeline_with_history([])._context()
    assert ctx.recent == [] and ctx.older == []


def wait_for(cond, timeout=2.0):
    end = time.time() + timeout
    while time.time() < end and not cond():
        time.sleep(0.02)
    return cond()


def test_memo_updates_every_n_sentences(fake_translator):
    model = fake_translator(reply="주제: 로봇\n용어: agent = 에이전트")
    keeper = MemoKeeper([model], QuotaTracker({}))
    for i in range(memo_mod.MEMO_EVERY - 1):
        keeper.add(f"s{i}", f"t{i}", "en", "ko")
    time.sleep(0.1)
    assert model.calls == [] and keeper.memo == ""
    keeper.add("last", "마지막", "en", "ko")
    assert wait_for(lambda: keeper.memo.startswith("주제: 로봇"))
    assert "last => 마지막" in model.calls[0][1]["content"]


def test_memo_falls_back_and_keeps_pending_on_failure(fake_translator):
    broken = fake_translator(name="broken", error=RuntimeError("down"))
    keeper = MemoKeeper([broken], QuotaTracker({}))
    for i in range(memo_mod.MEMO_EVERY):
        keeper.add(f"s{i}", f"t{i}", "en", "ko")
    assert wait_for(lambda: len(broken.calls) == 1)
    time.sleep(0.05)
    assert keeper.memo == "" and len(keeper._pending) == memo_mod.MEMO_EVERY

    good = fake_translator(name="good", reply="주제: 복구")
    keeper.models = [broken, good]
    keeper.add("more", "더", "en", "ko")
    assert wait_for(lambda: keeper.memo == "주제: 복구")
    assert keeper._pending == []


def test_memo_reset():
    keeper = MemoKeeper([], QuotaTracker({}))
    keeper.memo = "x"
    keeper._pending = [("a", "b")]
    keeper.reset()
    assert keeper.memo == "" and keeper._pending == []
