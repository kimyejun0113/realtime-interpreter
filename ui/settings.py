from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QActionGroup
from PySide6.QtWidgets import QApplication, QMenu

from audio.capture import list_input_devices


def attach_context_menu(overlay, pipeline) -> None:
    """Right-click menu on the overlay: start/pause, microphone, font size, opacity, quit."""

    def show_menu(pos):
        menu = QMenu(overlay)

        toggle = menu.addAction("일시정지" if pipeline.running.is_set() else "시작")
        toggle.triggered.connect(lambda: pipeline.stop() if pipeline.running.is_set() else pipeline.start())

        mic_menu = menu.addMenu("마이크")
        group = QActionGroup(mic_menu)
        default = QAction("시스템 기본", mic_menu, checkable=True, checked=pipeline.device is None)
        default.triggered.connect(lambda: pipeline.set_device(None))
        group.addAction(default)
        mic_menu.addAction(default)
        for idx, name in list_input_devices():
            act = QAction(name, mic_menu, checkable=True, checked=pipeline.device == idx)
            act.triggered.connect(lambda _=False, i=idx: pipeline.set_device(i))
            group.addAction(act)
            mic_menu.addAction(act)

        font_menu = menu.addMenu("글자 크기")
        font_menu.addAction("크게").triggered.connect(lambda: overlay.set_font_size(overlay.font_size + 2))
        font_menu.addAction("작게").triggered.connect(lambda: overlay.set_font_size(overlay.font_size - 2))

        opacity_menu = menu.addMenu("배경 투명도")
        for label, alpha in (("진하게", 230), ("보통", 170), ("연하게", 90), ("투명", 0)):
            opacity_menu.addAction(label).triggered.connect(lambda _=False, a=alpha: overlay.set_opacity(a))

        menu.addSeparator()
        menu.addAction("종료").triggered.connect(QApplication.quit)
        menu.exec(overlay.mapToGlobal(pos))

    overlay.setContextMenuPolicy(Qt.CustomContextMenu)
    overlay.customContextMenuRequested.connect(show_menu)
