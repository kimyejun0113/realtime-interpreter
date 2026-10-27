import time
from collections import OrderedDict

from PySide6.QtCore import QEasingCurve, QEvent, QObject, QPoint, QPropertyAnimation, QSettings, Qt, Signal
from PySide6.QtGui import QColor, QFont, QFontDatabase, QTextDocument
from PySide6.QtWidgets import (
    QApplication, QFrame, QGraphicsDropShadowEffect, QGraphicsOpacityEffect, QHBoxLayout, QLabel,
    QSizeGrip, QSizePolicy, QToolButton, QVBoxLayout, QWidget,
)

MAX_ENTRIES = 6      # sentences kept in memory (older ones are forgotten)
HISTORY_SHOWN = 3    # past translations shown once everyone has had time to read them
# how long a finished translation stays on screen: base + per character, clamped
READ_BASE_SEC, READ_SEC_PER_CHAR, READ_MIN_SEC, READ_MAX_SEC = 1.5, 0.08, 3.0, 8.0

# Segoe Fluent Icons (Win11) / Segoe MDL2 Assets (Win10) glyphs
ICON_PLAY, ICON_PAUSE, ICON_MORE, ICON_CLOSE = "", "", "", ""

TEXT_FONTS = ["Segoe UI Variable Text", "Segoe UI", "Malgun Gothic"]
STATUS_COLORS = {"live": "#30D158", "paused": "#8E8E93", "warn": "#FF9F0A"}


def _icon_font() -> str:
    families = set(QFontDatabase.families())
    return "Segoe Fluent Icons" if "Segoe Fluent Icons" in families else "Segoe MDL2 Assets"


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class Bridge(QObject):
    """Marshals pipeline callbacks (worker threads) onto the Qt main thread."""

    live = Signal(int, str)
    partial = Signal(int, str)
    source = Signal(int, str, str)
    final = Signal(int, str)
    merged = Signal(int)
    status = Signal(str, str)
    level = Signal(float)
    quota = Signal(str)
    lang = Signal(str)


class IconButton(QToolButton):
    def __init__(self, glyph: str, tooltip: str, danger: bool = False):
        super().__init__()
        self.setText(glyph)
        self.setToolTip(tooltip)
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedSize(28, 28)
        font = QFont(_icon_font())
        font.setPixelSize(11)
        self.setFont(font)
        hover = "rgba(255,69,58,0.9)" if danger else "rgba(255,255,255,0.14)"
        self.setStyleSheet(
            "QToolButton{color:rgba(255,255,255,0.85);background:transparent;border:none;border-radius:14px}"
            f"QToolButton:hover{{background:{hover};color:white}}"
            "QToolButton:pressed{background:rgba(255,255,255,0.22)}"
        )


class SubtitleOverlay(QWidget):
    SHADOW_MARGIN = 16

    def __init__(self):
        super().__init__()
        self.setWindowTitle("동시통역")
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.settings = QSettings("realtime-interpreter", "overlay")
        self.font_size = int(self.settings.value("font_size", 24))
        self.bg_alpha = int(self.settings.value("bg_alpha", 215))
        self.entries: OrderedDict[int, dict] = OrderedDict()
        self._drag_from: QPoint | None = None
        self.on_toggle = lambda: None   # set by main
        self.on_menu = lambda pos: None

        font = QFont()
        font.setFamilies(TEXT_FONTS)
        self.setFont(font)

        # card with soft shadow inside a transparent margin
        self.card = QFrame(self)
        self.card.setObjectName("card")
        shadow = QGraphicsDropShadowEffect(self.card)
        shadow.setBlurRadius(36)
        shadow.setOffset(0, 8)
        shadow.setColor(QColor(0, 0, 0, 120))
        self.card.setGraphicsEffect(shadow)
        outer = QVBoxLayout(self)
        m = self.SHADOW_MARGIN
        outer.setContentsMargins(m, m, m, m)
        outer.addWidget(self.card)

        # header: status on the left, controls on the right
        self.dot = QLabel()
        self.dot.setFixedSize(8, 8)
        self.status_label = QLabel("준비 중")
        self.quota_label = QLabel("")
        self.lang_label = QLabel("")
        self.toggle_btn = IconButton(ICON_PAUSE, "일시정지")
        self.menu_btn = IconButton(ICON_MORE, "설정")
        self.close_btn = IconButton(ICON_CLOSE, "닫기", danger=True)
        self.toggle_btn.clicked.connect(lambda: self.on_toggle())
        self.menu_btn.clicked.connect(lambda: self.on_menu(self.menu_btn.mapToGlobal(QPoint(0, self.menu_btn.height() + 4))))
        self.close_btn.clicked.connect(self.close)

        controls = QWidget()
        cl = QHBoxLayout(controls)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(2)
        for w in (self.quota_label, self.toggle_btn, self.menu_btn, self.close_btn):
            cl.addWidget(w)
        cl.insertSpacing(1, 6)
        # controls stay faint until the pointer is over the window
        self.controls_fx = QGraphicsOpacityEffect(controls)
        self.controls_fx.setOpacity(0.45)
        controls.setGraphicsEffect(self.controls_fx)
        self.controls_anim = QPropertyAnimation(self.controls_fx, b"opacity", self)
        self.controls_anim.setDuration(180)
        self.controls_anim.setEasingCurve(QEasingCurve.OutCubic)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(8)
        header.addWidget(self.dot)
        header.addWidget(self.status_label)
        header.addWidget(self.lang_label)
        header.addStretch(1)
        header.addWidget(controls)

        self.body = QLabel()
        self.body.setWordWrap(True)
        self.body.setTextFormat(Qt.RichText)
        self.body.setAlignment(Qt.AlignHCenter | Qt.AlignBottom)
        # the window decides the size; long text is trimmed to fit instead of growing past the window
        self.body.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.body.setMinimumSize(1, 1)
        self.body.installEventFilter(self)

        grip_row = QHBoxLayout()
        grip_row.setContentsMargins(0, 0, 0, 0)
        grip_row.addStretch(1)
        grip = QSizeGrip(self)
        grip.setFixedSize(14, 14)
        grip.setStyleSheet("background:transparent")
        grip_row.addWidget(grip, 0, Qt.AlignRight | Qt.AlignBottom)

        inner = QVBoxLayout(self.card)
        inner.setContentsMargins(20, 12, 10, 4)
        inner.setSpacing(6)
        inner.addLayout(header)
        inner.addWidget(self.body, 1)
        inner.addLayout(grip_row)

        self.set_status("준비 중", "paused")
        self._apply_style()
        geometry = self.settings.value("geometry")
        if geometry is None or not self.restoreGeometry(geometry):
            self.resize(880, 250)
            screen = self.screen().availableGeometry()
            self.move(screen.center().x() - 440, screen.bottom() - 290)

    # ---- appearance ----------------------------------------------------
    def _apply_style(self) -> None:
        self.card.setStyleSheet(
            f"#card{{background:rgba(28,28,30,{self.bg_alpha});"
            "border:1px solid rgba(255,255,255,0.10);border-radius:20px}"
        )
        self.status_label.setStyleSheet("color:rgba(235,235,245,0.6);font-size:12px;font-weight:500")
        self.lang_label.setStyleSheet(
            "color:rgba(235,235,245,0.6);font-size:11px;font-weight:600;"
            "background:rgba(255,255,255,0.08);border-radius:6px;padding:2px 7px"
        )
        self.quota_label.setStyleSheet("color:rgba(235,235,245,0.45);font-size:11px")
        self._render()

    def set_font_size(self, size: int) -> None:
        self.font_size = max(14, min(48, size))
        self.settings.setValue("font_size", self.font_size)
        self._render()

    def set_opacity(self, alpha: int) -> None:
        self.bg_alpha = alpha
        self.settings.setValue("bg_alpha", alpha)
        self._apply_style()

    def set_lang_label(self, src: str, tgt: str, auto: bool = False) -> None:
        self.lang_label.setText(("자동 · " if auto else "") + f"{src.upper()} → {tgt.upper()}")

    def set_running(self, running: bool) -> None:
        self.toggle_btn.setText(ICON_PAUSE if running else ICON_PLAY)
        self.toggle_btn.setToolTip("일시정지" if running else "시작")

    def set_status(self, msg: str, kind: str) -> None:
        self.status_label.setText(msg)
        color = STATUS_COLORS.get(kind, STATUS_COLORS["paused"])
        self.dot.setStyleSheet(f"background:{color};border-radius:4px")

    # ---- content -------------------------------------------------------
    def _entry(self, seq: int) -> dict:
        if seq not in self.entries:
            self.entries[seq] = {"source": "", "translation": "", "done": False, "final_at": None}
            while len(self.entries) > MAX_ENTRIES:
                self.entries.popitem(last=False)
        return self.entries[seq]

    @staticmethod
    def _read_until(e: dict) -> float:
        """Until when a finished translation must stay on screen so it can be read."""
        if not e["done"] or e["final_at"] is None:
            return float("inf")  # still waiting for its final translation
        return e["final_at"] + min(READ_MAX_SEC, max(READ_MIN_SEC, READ_BASE_SEC + READ_SEC_PER_CHAR * len(e["translation"])))

    def _history_html(self, e: dict, newest: bool) -> str:
        # past sentences: translation only, so the space goes to what people actually read
        fs = self.font_size
        size, color, weight = (int(fs * 0.85), "rgba(255,255,255,0.9)", 600) if newest else (int(fs * 0.72), "rgba(255,255,255,0.55)", 400)
        text = _esc(e["translation"]) or '<span style="color:rgba(235,235,245,0.4)">…</span>'
        return f'<p align="center" style="margin:0 0 8px 0;color:{color};font-size:{size}px;font-weight:{weight}">{text}</p>'

    def _current_html(self, e: dict, source: str, translation: str) -> str:
        fs = self.font_size
        html = ""
        if source:
            html += (f'<p align="center" style="margin:0 0 3px 0;color:rgba(235,235,245,0.7);'
                     f'font-size:{int(fs * 0.72)}px">{_esc(source)}</p>')
        if translation:
            color = "#FFFFFF" if e["done"] else "rgba(255,255,255,0.72)"  # dim = tentative, still changing
            html += (f'<p align="center" style="margin:0 0 4px 0;color:{color};font-size:{fs}px;'
                     f'font-weight:700">{_esc(translation)}</p>')
        return html

    def _fits(self, html: str) -> bool:
        doc = QTextDocument()
        doc.setDocumentMargin(0)
        doc.setDefaultFont(self.body.font())
        doc.setHtml(html)
        doc.setTextWidth(max(100, self.body.width()))
        return doc.size().height() <= self.body.height() - 4

    def _render(self) -> None:
        if not self.entries:
            self.body.setText(
                f'<p align="center" style="color:rgba(235,235,245,0.35);font-size:{int(self.font_size * 0.7)}px">'
                "말을 시작하면 여기에 자막이 표시돼요</p>"
            )
            return
        now = time.monotonic()
        items = list(self.entries.values())
        cur, history = items[-1], items[:-1]
        src_words, tr_words = cur["source"].split(), cur["translation"].split()
        while len(history) > HISTORY_SHOWN and now >= self._read_until(history[0]):
            history.pop(0)

        def build() -> str:
            source = ("… " if len(src_words) < len(cur["source"].split()) else "") + " ".join(src_words)
            translation = ("… " if len(tr_words) < len(cur["translation"].split()) else "") + " ".join(tr_words)
            parts = [self._history_html(e, i == len(history) - 1) for i, e in enumerate(history)]
            return "".join(parts) + self._current_html(cur, source, translation)

        # When space runs out, give up the least useful text first:
        # 1. past translations that have already been on screen long enough to read
        # 2. the oldest words of the live English line
        # 3. the oldest words of the tentative translation
        # 4. only then past translations people may still be reading
        while not self._fits(html := build()):
            read = [e for e in history if now >= self._read_until(e)]
            if read:
                history.remove(read[0])
            elif len(src_words) > 6:
                src_words = src_words[2:]
            elif not cur["done"] and len(tr_words) > 4:
                tr_words = tr_words[1:]
            elif history:
                history.pop(0)
            elif len(src_words) > 2:
                src_words = src_words[1:]
            elif len(tr_words) > 2:
                tr_words = tr_words[1:]
            else:
                break
        self.body.setText(html)

    def eventFilter(self, obj, event) -> bool:
        # re-fit when the subtitle area itself changes size (it is laid out after the window resizes)
        if obj is self.body and event.type() == QEvent.Resize:
            self._render()
        return super().eventFilter(obj, event)

    def on_live(self, seq: int, text: str) -> None:
        e = self._entry(seq)
        if not e["done"]:
            e["source"] = text
            self._render()

    def on_partial(self, seq: int, text: str) -> None:
        e = self.entries.get(seq)
        if e and not e["done"]:
            e["translation"] = text
            self._render()

    def on_source(self, seq: int, text: str, lang: str) -> None:
        self._entry(seq)["source"] = text
        self._render()

    def on_final(self, seq: int, text: str) -> None:
        e = self.entries.get(seq)
        if e:
            e["done"] = True
            e["final_at"] = time.monotonic()
            if text:
                e["translation"] = text
            self._render()

    def on_merged(self, seq: int) -> None:
        # this piece is being translated together with the next one, which shows the combined text
        if self.entries.pop(seq, None) is not None:
            self._render()

    def on_quota(self, text: str) -> None:
        self.quota_label.setText(text)

    # ---- window behaviour ---------------------------------------------
    def _fade_controls(self, to: float) -> None:
        self.controls_anim.stop()
        self.controls_anim.setStartValue(self.controls_fx.opacity())
        self.controls_anim.setEndValue(to)
        self.controls_anim.start()

    def enterEvent(self, event) -> None:
        self._fade_controls(1.0)

    def leaveEvent(self, event) -> None:
        self._fade_controls(0.45)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self._drag_from = event.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, event) -> None:
        if self._drag_from is not None and event.buttons() & Qt.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_from)

    def mouseReleaseEvent(self, event) -> None:
        self._drag_from = None

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_Escape:
            self.close()

    def closeEvent(self, event) -> None:
        self.settings.setValue("geometry", self.saveGeometry())
        super().closeEvent(event)
        QApplication.instance().quit()  # Qt.Tool windows don't trigger quit-on-last-window
