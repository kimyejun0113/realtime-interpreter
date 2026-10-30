import os

from translate import glossary as glossary_mod
from translate.base import Context, build_messages
from translate.glossary import Glossary, parse

TEXT = """\
# 용어집
Tilly = 틸리
Centillion = 센틸리언

George Orwell = 조지 오웰
Kubernetes
tilly = 틸리2
C# = 씨샵
센틸리언 교수
"""


def test_parse_skips_comments_blanks_and_duplicates():
    assert parse(TEXT) == [("Tilly", "틸리"), ("Centillion", "센틸리언"), ("George Orwell", "조지 오웰"),
                           ("Kubernetes", ""), ("C#", "씨샵"), ("센틸리언 교수", "")]


def test_whisper_hint_uses_the_side_written_in_the_speech_language(tmp_path):
    path = tmp_path / "glossary.txt"
    path.write_text(TEXT, encoding="utf-8")
    g = Glossary(path)
    assert g.whisper_hint("en") == "Tilly, Centillion, George Orwell, Kubernetes, C#."
    assert g.whisper_hint("ko") == "틸리, 센틸리언, 조지 오웰, 씨샵, 센틸리언 교수."


def test_whisper_hint_is_capped(tmp_path, monkeypatch):
    monkeypatch.setattr(glossary_mod, "WHISPER_MAX_CHARS", 20)
    path = tmp_path / "glossary.txt"
    path.write_text(TEXT, encoding="utf-8")
    assert Glossary(path).whisper_hint("en") == "Tilly, Centillion."


def test_missing_file_and_live_reload(tmp_path):
    path = tmp_path / "glossary.txt"
    g = Glossary(path)
    assert g.terms == [] and g.whisper_hint("en") == "" and g.prompt_lines() == []
    path.write_text("Tilly = 틸리\n", encoding="utf-8")
    assert g.prompt_lines() == ["Tilly = 틸리"]
    path.write_text("Tilly = 틸리\nSai = 사이\n", encoding="utf-8")
    os.utime(path, (1, 2))  # make sure the change is visible even within the same clock tick
    assert g.prompt_lines() == ["Tilly = 틸리", "Sai = 사이"]
    path.unlink()
    assert g.terms == []


def test_glossary_section_in_translation_prompt():
    system = build_messages("Killy does it for you.", "en", "ko",
                            Context(glossary=["Tilly = 틸리", "Kubernetes"]))[0]["content"]
    assert "- Tilly = 틸리\n- Kubernetes" in system and "sound-alike" in system and "Possibly misheard" not in system
    assert "Glossary" not in build_messages("Hi", "en", "ko", Context())[0]["content"]


def test_template_is_all_comments(tmp_path, monkeypatch):
    monkeypatch.setattr(glossary_mod.os, "startfile", lambda p: None, raising=False)
    path = tmp_path / "glossary.txt"
    glossary_mod.open_in_editor(path)
    assert path.exists() and parse(path.read_text(encoding="utf-8")) == []


LECTURE = "Tilly = 틸리\nCentillion = 센틸리언\nHuxley = 헉슬리\nJenny = 제니\n"


def lecture_glossary(tmp_path):
    path = tmp_path / "glossary.txt"
    path.write_text(LECTURE, encoding="utf-8")
    return Glossary(path)


def test_suspects_point_out_misheard_terms(tmp_path):
    g = lecture_glossary(tmp_path)
    assert g.suspects("Would you use Tilly or some versions of Killy already?") == [("Killy", "Tilly")]
    assert g.suspects("Stillion has put us in little bubbles.") == [("Stillion", "Centillion")]
    assert g.suspects("And Psy says you are accusing Centilian of censorship.") == [("Centilian", "Centillion")]
    assert g.suspects("we didn't read oxley, so") == [("oxley", "Huxley")]
    # written like a name mid-sentence, even an ordinary word is worth a look
    assert g.suspects("Jenny says Silly doesn't just tell you what you want.") == [("Silly", "Tilly")]


def test_suspects_leave_ordinary_words_alone(tmp_path):
    g = lecture_glossary(tmp_path)
    for text in ("The weather was chilly and silly things happened.", "It was truly a thrill.",
                 "Our journey continually goes on.", "Well, Jenny tells Tilly the truth.", ""):
        assert g.suspects(text) == [], text


def test_suspects_in_translation_prompt():
    ctx = Context(glossary=["Tilly = 틸리"], suspects=[("Killy", "Tilly")])
    system = build_messages("Killy does it.", "en", "ko", ctx)[0]["content"]
    assert '"Killy" → Tilly' in system and "makes no sense" in system
