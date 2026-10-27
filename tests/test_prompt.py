import pytest

from translate import base
from translate.base import Context, build_messages


def system_of(messages):
    return messages[0]["content"]


@pytest.mark.parametrize("style,must,must_not", [
    ("합니다", "합니다체", "Always end sentences in 해요체"),
    ("해요", "Always end sentences in 해요체", "Always end sentences in 합니다체"),
    ("한다", "한다체", "Always end sentences in 해요체"),
])
def test_ko_style_rule_and_examples(monkeypatch, style, must, must_not):
    monkeypatch.setattr(base, "KO_STYLE", style)
    msgs = build_messages("Hello", "en", "ko", Context())
    assert must in system_of(msgs) and must_not not in system_of(msgs)
    examples = [m["content"] for m in msgs[1:-1] if m["role"] == "assistant"]
    assert examples == [out for _, out in base.KO_STYLES[style]["examples"]]
    assert msgs[-1] == {"role": "user", "content": "Hello"}


def test_unknown_style_falls_back_to_hamnida(monkeypatch):
    monkeypatch.setattr(base, "KO_STYLE", "엉뚱한값")
    assert "합니다체" in system_of(build_messages("Hi", "en", "ko", Context()))


def test_english_target_has_no_korean_rules_or_examples():
    msgs = build_messages("안녕하세요", "ko", "en", Context())
    assert "합니다체" not in system_of(msgs) and len(msgs) == 2


def test_context_sections_in_order():
    ctx = Context(recent=[("Sarah joined.", "Sarah가 합류했습니다.")], older=["We trained the agent."],
                  memo="용어: agent = 에이전트")
    system = system_of(build_messages("He said yes.", "en", "ko", ctx))
    memo_at, older_at, recent_at = (system.index("Conversation notes"), system.index("Earlier in the conversation"),
                                    system.index("Previous subtitles"))
    assert memo_at < older_at < recent_at
    assert "We trained the agent." in system and "Sarah joined. => Sarah가 합류했습니다." in system


def test_empty_context_adds_no_sections():
    system = system_of(build_messages("Hi", "en", "ko", Context()))
    assert "Conversation notes" not in system and "Previous subtitles" not in system


def test_partial_prompt_is_short_and_contextless():
    msgs = build_messages("so we went", "en", "ko", Context(memo="x"), partial=True)
    assert len(msgs) == 2 and len(system_of(msgs)) < 250 and "x" not in system_of(msgs)
