import sys

from PySide6.QtWidgets import QApplication, QMessageBox

from config import Config
from pipeline import Events, Pipeline
from ui.overlay import Bridge, SubtitleOverlay
from ui.settings import attach_context_menu


def main() -> int:
    app = QApplication(sys.argv)
    cfg = Config()

    missing = "GEMINI_API_KEY" if cfg.translate_provider == "gemini" and not cfg.gemini_api_key else None
    if not cfg.groq_api_key:
        missing = "GROQ_API_KEY"
    if missing:
        QMessageBox.warning(
            None, "API 키 필요",
            f"{missing} 가 설정되지 않았습니다.\n.env.example 을 .env 로 복사한 뒤 키를 입력하세요.\n"
            "(README.md 참고)",
        )
        return 1

    overlay = SubtitleOverlay()
    bridge = Bridge()
    bridge.source.connect(overlay.on_source)
    bridge.partial.connect(overlay.on_partial)
    bridge.final.connect(overlay.on_final)
    bridge.status.connect(overlay.on_status)
    bridge.level.connect(overlay.on_level)

    events = Events()
    events.on_source = bridge.source.emit
    events.on_partial = bridge.partial.emit
    events.on_final = bridge.final.emit
    events.on_status = bridge.status.emit
    events.on_level = bridge.level.emit

    pipeline = Pipeline(cfg, events)
    attach_context_menu(overlay, pipeline)
    overlay.show()
    pipeline.start()

    code = app.exec()
    pipeline.stop()
    return code


if __name__ == "__main__":
    sys.exit(main())
