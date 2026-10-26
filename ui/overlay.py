from collections import OrderedDict

from PySide6.QtCore import QObject, QPoint, Qt, Signal
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import QLabel, QProgressBar, QVBoxLayout, QWidget

MAX_LINES = 3


class Bridge(QObject):
    """Marshals pipeline callbacks (worker threads) onto the Qt main thread."""

    source = Signal(int, str, str)
    partial = Signal(int, str)
    final = Signal(int, str)
    status = Signal(str)
    level = Signal(float)


class SubtitleOverlay(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("동시통역")
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setStyleSheet("font-family: 'Malgun Gothic', 'Segoe UI', sans-serif;")
        self.font_size = 22
        self.bg_alpha = 170
        self.entries: OrderedDict[int, dict] = OrderedDict()
        self._drag_from: QPoint | None = None

        self.status_label = QLabel("오른쪽 클릭 → 시작")
        self.level_bar = QProgressBar()
        self.level_bar.setRange(0, 100)
        self.level_bar.setTextVisible(False)
        self.level_bar.setFixedHeight(4)
        self.body = QLabel()
        self.body.setWordWrap(True)
        self.body.setTextFormat(Qt.RichText)
        self.body.setAlignment(Qt.AlignLeft | Qt.AlignBottom)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 10, 18, 14)
        layout.addWidget(self.status_label)
        layout.addWidget(self.level_bar)
        layout.addWidget(self.body, 1)
        self._apply_style()
        self.resize(900, 260)
        screen = self.screen().availableGeometry()
        self.move(screen.center().x() - 450, screen.bottom() - 300)

    # ---- appearance ----------------------------------------------------
    def _apply_style(self) -> None:
        self.status_label.setStyleSheet("color:#9aa4b2; font-size:12px;")
        self.level_bar.setStyleSheet(
            "QProgressBar{background:#33ffffff;border:none;border-radius:2px}"
            "QProgressBar::chunk{background:#4ade80;border-radius:2px}"
        )
        self._render()

    def set_font_size(self, size: int) -> None:
        self.font_size = max(12, min(48, size))
        self._render()

    def set_opacity(self, alpha: int) -> None:
        self.bg_alpha = alpha
        self.update()

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setBrush(QColor(15, 18, 24, self.bg_alpha))
        p.setPen(Qt.NoPen)
        p.drawRoundedRect(self.rect(), 14, 14)

    # ---- content -------------------------------------------------------
    def _entry(self, seq: int) -> dict:
        if seq not in self.entries:
            self.entries[seq] = {"source": "", "lang": "", "translation": "", "done": False}
            while len(self.entries) > MAX_LINES:
                self.entries.popitem(last=False)
        return self.entries[seq]

    def _render(self) -> None:
        def esc(s: str) -> str:
            return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

        small = max(11, int(self.font_size * 0.55))
        parts = []
        for e in self.entries.values():
            color = "#ffffff" if e["done"] else "#e5e7eb"
            translation = esc(e["translation"]) or "…"
            parts.append(
                f'<div style="margin-bottom:8px">'
                f'<div style="color:#8b95a5;font-size:{small}px">[{e["lang"]}] {esc(e["source"])}</div>'
                f'<div style="color:{color};font-size:{self.font_size}px;font-weight:600">{translation}</div>'
                f"</div>"
            )
        self.body.setText("".join(parts))

    def on_source(self, seq: int, text: str, lang: str) -> None:
        e = self._entry(seq)
        e["source"], e["lang"] = text, lang
        self._render()

    def on_partial(self, seq: int, text: str) -> None:
        if seq in self.entries:
            self.entries[seq]["translation"] = text
            self._render()

    def on_final(self, seq: int, text: str) -> None:
        if seq in self.entries:
            self.entries[seq].update(translation=text, done=True)
            self._render()

    def on_status(self, msg: str) -> None:
        self.status_label.setText(msg)

    def on_level(self, level: float) -> None:
        self.level_bar.setValue(int(level * 100))

    # ---- dragging ------------------------------------------------------
    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self._drag_from = event.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, event) -> None:
        if self._drag_from is not None and event.buttons() & Qt.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_from)

    def mouseReleaseEvent(self, event) -> None:
        self._drag_from = None
