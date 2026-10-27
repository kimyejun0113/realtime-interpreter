import numpy as np

from quota import QuotaTracker
from stt.streaming import DualRecognizer
from translate import spacing

FRAME = np.zeros(1600, np.int16)


class FakeRecognizer:
    """Scripted recogniser: set .text/.ended/.conf/.n before each accept()."""

    def __init__(self, lang):
        self.lang = self.last_lang = lang
        self.text, self.ended, self.conf, self.n = "", False, float("-inf"), 0
        self.resets = 0

    def set(self, text="", ended=False, conf=float("-inf"), n=0):
        self.text, self.ended, self.conf, self.n = text, ended, conf, n

    def feed(self, pcm):
        return self.text, self.ended

    def reset(self):
        self.resets += 1
        self.set()

    def confidence(self, window=None):
        n = min(self.n, window) if window else self.n
        return (self.conf, n) if n else (float("-inf"), 0)

    def text_without_unsure_tail(self, threshold=-0.7):
        return self.text.rsplit(" ", 1)[0]  # pretend the last word was the other language


def dual(initial="en"):
    en, ko = FakeRecognizer("en"), FakeRecognizer("ko")
    return DualRecognizer({"en": en, "ko": ko}, initial=initial), en, ko


def test_follows_active_language_while_speaking():
    d, en, ko = dual()
    en.set("Hey so I wanted", conf=-0.3, n=8)
    ko.set("해쿠니까", conf=-1.8, n=5)
    assert d.accept(FRAME) == ("Hey so I wanted", False) and d.last_lang == "en"


def test_wrong_language_sentence_is_dropped():
    d, en, ko = dual()
    en.set("Hey so I wanted to", conf=-0.3, n=10)
    ko.set("쿠니까먹으면", ended=True, conf=-2.0, n=9)
    assert d.accept(FRAME) == ("Hey so I wanted to", False)
    assert ko.resets == 1 and en.resets == 0 and d.lang == "en"


def test_other_language_finishing_a_clear_sentence_takes_over():
    d, en, ko = dual()
    en.set("Humm to Moguy", conf=-1.3, n=8)
    ko.set("오늘 회의는 여기까지 하겠습니다.", ended=True, conf=-1.0, n=7)
    assert d.accept(FRAME) == ("오늘 회의는 여기까지 하겠습니다.", True)
    assert d.lang == d.last_lang == "ko" and en.resets >= 1 and ko.resets == 1


def test_noise_does_not_win_against_a_freshly_reset_side():
    d, en, ko = dual("ko")
    ko.set("", conf=float("-inf"), n=0)
    en.set("re Geo", ended=True, conf=-1.73, n=4)
    assert d.accept(FRAME) == ("", False) and d.lang == "ko"


def test_short_garbage_ending_restarts_only_that_side():
    d, en, ko = dual("ko")
    ko.set("해.", ended=True, conf=-1.7, n=2)
    en.set("Hey", conf=-0.5, n=2)
    assert d.accept(FRAME) == ("", False)
    assert ko.resets == 1 and en.resets == 0  # English keeps what it heard


def test_active_ending_early_while_other_leads_switches_without_finishing():
    d, en, ko = dual("ko")
    ko.set("예저코엑 스랑", ended=True, conf=-1.1, n=6)
    en.set("Hey, so I wanted to quickly", conf=-0.4, n=9)
    assert d.accept(FRAME) == ("Hey, so I wanted to quickly", False)
    assert d.lang == "en" and ko.resets >= 1 and en.resets == 0  # English keeps what it heard


def test_mid_utterance_switch_closes_the_old_sentence():
    d, en, ko = dual()
    en.set("We are fifteen percent behind Humm", conf=-0.6, n=20)
    ko.set("다음 주 목요일에", conf=-0.5, n=8)
    en.conf = -0.6

    # recent English tokens are poor (the speaker switched to Korean)
    def poor_recent(window=None):
        return (-1.3, min(20, window)) if window else (-0.6, 20)
    en.confidence = poor_recent
    text, ended = d.accept(FRAME)
    assert (text, ended) == ("We are fifteen percent behind", True)
    assert d.last_lang == "en" and d.lang == "ko"


def test_steady_english_never_switches_on_weak_korean():
    d, en, ko = dual()
    for _ in range(20):
        en.set("So what we basically did", conf=-0.3, n=12)
        ko.set("스랑고얼을", conf=-1.1, n=8)
        d.accept(FRAME)
    assert d.lang == "en"


class FakeModel:
    def __init__(self, reply):
        self.name = "fake"
        self.reply = reply

    def complete(self, messages, max_tokens=200):
        return self.reply


def test_needs_spacing():
    assert spacing.needs_spacing("다음주목요일에다시이야기해요.")
    assert not spacing.needs_spacing("다음 주 목요일에 다시 이야기해요.")
    assert not spacing.needs_spacing("Hello there, how are you")
    assert not spacing.needs_spacing("네.")


def test_space_korean_accepts_only_space_changes():
    q = QuotaTracker({})
    text = "다음주목요일에다시이야기해요."
    assert spacing.space_korean(text, [FakeModel("다음 주 목요일에  다시 이야기해요.")], q) == "다음 주 목요일에 다시 이야기해요."
    assert spacing.space_korean(text, [FakeModel("다음 주 수요일에 다시 이야기해요.")], q) is None  # changed a word
    assert spacing.space_korean(text, [FakeModel("Next Thursday.")], q) is None
