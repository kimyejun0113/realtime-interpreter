import sys
from pathlib import Path

# Launched via pythonw (desktop shortcut) there is no console: keep output in logs/app.log
if sys.stdout is None or sys.stderr is None:
    log_dir = Path(__file__).with_name("logs")
    log_dir.mkdir(exist_ok=True)
    sys.stdout = sys.stderr = open(log_dir / "app.log", "a", encoding="utf-8", buffering=1)

from PySide6.QtWidgets import QApplication, QMessageBox

from config import Config
from pipeline import Events, Pipeline
from ui.overlay import Bridge, SubtitleOverlay
from ui.settings import attach_menu


def main() -> int:
    app = QApplication(sys.argv)
    cfg = Config()

    if not cfg.groq_api_key:
        QMessageBox.warning(
            None, "API 키 필요",
            "GROQ_API_KEY 가 설정되지 않았습니다.\n.env.example 을 .env 로 복사한 뒤 키를 입력하세요.\n(README.md 참고)",
        )
        return 1

    overlay = SubtitleOverlay()
    bridge = Bridge()
    events = Events()
    for name in ("live", "partial", "source", "final", "level", "quota"):
        signal = getattr(bridge, name)
        setattr(events, f"on_{name}", signal.emit)
        if hasattr(overlay, f"on_{name}"):
            signal.connect(getattr(overlay, f"on_{name}"))

    pipeline = Pipeline(cfg, events)

    def on_status(msg: str, kind: str) -> None:
        overlay.set_status(msg, kind)
        overlay.set_running(pipeline.running.is_set())
        overlay.set_lang_label(pipeline.lang, pipeline.target_lang)

    events.on_status = bridge.status.emit
    bridge.status.connect(on_status)
    overlay.on_toggle = lambda: pipeline.stop() if pipeline.running.is_set() else pipeline.start()
    attach_menu(overlay, pipeline)

    overlay.set_lang_label(pipeline.lang, pipeline.target_lang)
    overlay.show()
    pipeline.start()

    code = app.exec()
    pipeline.stop()
    return code


if __name__ == "__main__":
    sys.exit(main())
