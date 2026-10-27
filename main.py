import argparse
import sys
import threading
import time
from pathlib import Path

# Launched via pythonw (desktop shortcut) there is no console: keep output in logs/app.log
if sys.stdout is None or sys.stderr is None:
    log_dir = Path(__file__).with_name("logs")
    log_dir.mkdir(exist_ok=True)
    sys.stdout = sys.stderr = open(log_dir / "app.log", "a", encoding="utf-8", buffering=1)

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QMessageBox

from audio.capture import APP_PREFIX, FILE_PREFIX, SYSTEM_AUDIO
from config import Config
from pipeline import AUTO, Events, Pipeline
from ui.overlay import Bridge, SubtitleOverlay
from ui.settings import attach_menu


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="실시간 동시통역 자막")
    parser.add_argument("--smoke", metavar="WAV", help="smoke test: play a 16 kHz mono WAV instead of the mic, "
                                                       "exit 0 if translations appear without errors")
    parser.add_argument("--offline", action="store_true", help="with --smoke: no API calls (local translation only)")
    return parser.parse_args()


def install_smoke_check(app: QApplication, pipeline: Pipeline, events: Events) -> None:
    """Fail on any uncaught exception (incl. worker threads); pass once translations show up."""
    errors: list[str] = []
    finals: list[str] = []
    sys.excepthook = lambda *exc: errors.append(repr(exc[1]))
    threading.excepthook = lambda args: errors.append(repr(args.exc_value))
    on_final = events.on_final
    events.on_final = lambda seq, text: (finals.append(text), on_final(seq, text))
    state = {"deadline": None, "load_deadline": time.monotonic() + 60}

    def check() -> None:
        now = time.monotonic()
        if state["deadline"] is None and pipeline.running.is_set() and pipeline.mic is not None:
            state["deadline"] = now + pipeline.mic.duration + 8  # recording, then time for the last sentence
        done = state["deadline"] is not None and now >= state["deadline"]
        if errors or done or now >= state["load_deadline"] and state["deadline"] is None:
            ok = not errors and any(finals)
            print(f"SMOKE {'PASS' if ok else 'FAIL'}: {sum(map(bool, finals))} translations, errors={errors}", flush=True)
            app.exit(0 if ok else 2)

    timer = QTimer(app)
    timer.timeout.connect(check)
    timer.start(500)


def main() -> int:
    args = parse_args()
    app = QApplication(sys.argv[:1])
    cfg = Config()
    if args.smoke and args.offline:
        cfg.translate_engines, cfg.final_stt, cfg.conversation_memo = "", "local", False

    if not cfg.groq_api_key and not (args.smoke and args.offline):
        QMessageBox.warning(
            None, "API 키 필요",
            "GROQ_API_KEY 가 설정되지 않았습니다.\n.env.example 을 .env 로 복사한 뒤 키를 입력하세요.\n(README.md 참고)",
        )
        return 1

    overlay = SubtitleOverlay()
    bridge = Bridge()
    events = Events()
    for name in ("live", "partial", "source", "final", "merged", "level", "quota"):
        signal = getattr(bridge, name)
        setattr(events, f"on_{name}", signal.emit)
        if hasattr(overlay, f"on_{name}"):
            signal.connect(getattr(overlay, f"on_{name}"))

    pipeline = Pipeline(cfg, events)
    saved_lang = overlay.settings.value("lang")  # remember the language menu choice between runs
    if saved_lang in (AUTO, "en", "ko") and not args.smoke:
        pipeline.lang = saved_lang
    saved_input = overlay.settings.value("input")  # remember PC sound / app vs mic between runs
    if args.smoke:
        pipeline.device = FILE_PREFIX + str(Path(args.smoke).resolve())
        install_smoke_check(app, pipeline, events)
    elif saved_input == SYSTEM_AUDIO or (isinstance(saved_input, str) and saved_input.startswith(APP_PREFIX)):
        pipeline.device = saved_input

    def show_lang(*_) -> None:
        overlay.set_lang_label(pipeline.current_lang, pipeline.target_lang, auto=pipeline.lang == AUTO)

    def on_status(msg: str, kind: str) -> None:
        overlay.set_status(msg, kind)
        overlay.set_running(pipeline.running.is_set())
        show_lang()

    events.on_status = bridge.status.emit
    bridge.status.connect(on_status)
    events.on_lang = bridge.lang.emit
    bridge.lang.connect(show_lang)
    overlay.on_toggle = lambda: pipeline.stop() if pipeline.running.is_set() else pipeline.start()
    attach_menu(overlay, pipeline)

    show_lang()
    overlay.show()
    pipeline.start()

    code = app.exec()
    pipeline.stop()
    return code


if __name__ == "__main__":
    sys.exit(main())
