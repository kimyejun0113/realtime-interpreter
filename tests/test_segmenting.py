import queue
import threading
import time

import numpy as np
import pytest

import pipeline
from pipeline import Events, Pipeline, Utterance, _commit_point, _SENTENCE_FINAL


def cut(text):
    i = _commit_point(text)
    return (text[:i].strip(), text[i:].strip()) if i else None


def test_sentence_end_followed_by_more_words_is_cut():
    assert cut("I trained the model. Then it") == ("I trained the model.", "Then it")


def test_short_sentence_heads_are_not_cut():
    assert cut("Hey there. How are you") is None  # "Hey there." is under 3 words


def test_last_sentence_end_wins():
    assert cut("One two three. Four five six. Seven") == ("One two three. Four five six.", "Seven")


def test_no_cut_without_punctuation_until_hard_limit():
    words = " ".join(f"w{i}" for i in range(pipeline.HARD_CUT_WORDS))
    assert cut(words) is None
    head, rest = cut(words + " extra")
    assert rest.split() == ["w37", "w38", "w39", "extra"]  # newest 4 words stay open


def test_comma_cut_only_after_soft_limit():
    short = "we went to the store, and then we bought apples"
    assert cut(short) is None
    long = ("so we went to the store and then we bought some apples and pears, "
            "and then we came back home and started cooking dinner for everyone")
    head, rest = cut(long)
    assert head.endswith("pears,") and rest.startswith("and then")


@pytest.mark.parametrize("text,final", [
    ("and I often do.", True), ('He said "go."', True), ("Why?", True),
    ("which I", False), ("for the reinforcement learning", False), ("e.g. this", False),
])
def test_sentence_final(text, final):
    assert bool(_SENTENCE_FINAL.search(text)) is final


def bare_pipeline():
    p = Pipeline.__new__(Pipeline)  # no threads/models: only what _next_final needs
    p.finals = queue.Queue()
    p.events = Events()
    return p


def utt(seq, text):
    return Utterance(seq, text, "en", np.zeros(0, np.int16), time.perf_counter())


def test_short_unfinished_piece_joins_the_next():
    p = bare_pipeline()
    merged = []
    p.events.on_merged = merged.append
    p.finals.put(utt(1, "which I"))
    p.finals.put(utt(2, "we're trying something new this year."))
    out = p._next_final()
    assert (out.seq, out.text) == (2, "which I we're trying something new this year.")
    assert merged == [1]


def test_finished_or_long_pieces_go_alone():
    p = bare_pipeline()
    p.finals.put(utt(1, "And I often do."))
    p.finals.put(utt(2, "Next one"))
    assert p._next_final().text == "And I often do."


def test_short_piece_alone_when_nothing_follows(monkeypatch):
    monkeypatch.setattr(pipeline, "MERGE_WAIT_SEC", 0.2)
    p = bare_pipeline()
    p.finals.put(utt(1, "which I"))
    t0 = time.perf_counter()
    assert p._next_final().text == "which I"
    assert time.perf_counter() - t0 < 1


@pytest.mark.parametrize("text,lang,ok", [
    ("오늘 회의는 여기까지 하겠습니다.", "ko", True), ("Today's meeting is here.", "ko", False),
    ("Hey, so I wanted to go.", "en", True), ("오늘 회의는 여기까지.", "en", False),
    ("Sarah가 팀에 합류했습니다.", "ko", True), ("...", "en", False),
])
def test_written_in(text, lang, ok):
    assert pipeline._written_in(text, lang) is ok


@pytest.mark.parametrize("a,b,ok", [
    ("Hey, so I wanted to quickly go over the numbers.", "Hey so I wanted to quickly go over the numbers", True),
    ("다음 주 목요일에 다시 이야기해요.", "다음주목요일에다시이야기해요.", True),
    ("Can you send me the updated deck by Thursday?", "Hey, so I wanted to quickly go over the numbers", False),
    ("Can you send me the updated deck by Thursday?", "Can you send me the update", True),
])
def test_similar(a, b, ok):
    assert pipeline._similar(a, b) is ok
