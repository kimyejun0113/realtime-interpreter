import os

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QActionGroup
from PySide6.QtWidgets import QMenu

from audio.capture import APP_PREFIX, SYSTEM_AUDIO, list_audio_apps, list_input_devices
from pipeline import LOG_DIR
from stt.streaming import model_available

MENU_STYLE = """
QMenu { background: rgba(44,44,46,245); border: 1px solid rgba(255,255,255,0.10);
        border-radius: 12px; padding: 6px; color: rgba(255,255,255,0.92); font-size: 13px; }
QMenu::item { padding: 6px 28px 6px 12px; border-radius: 6px; }
QMenu::item:selected { background: #0A84FF; color: white; }
QMenu::item:disabled { color: rgba(235,235,245,0.3); }
QMenu::separator { height: 1px; background: rgba(255,255,255,0.10); margin: 5px 8px; }
QMenu::indicator { width: 14px; height: 14px; left: 6px; }
QMenu::right-arrow { width: 8px; height: 8px; }
"""


def _menu(parent, title: str = "") -> QMenu:
    menu = QMenu(title, parent)
    menu.setWindowFlags(menu.windowFlags() | Qt.FramelessWindowHint | Qt.NoDropShadowWindowHint)
    menu.setAttribute(Qt.WA_TranslucentBackground)
    menu.setStyleSheet(MENU_STYLE)
    return menu


def build_menu(overlay, pipeline) -> QMenu:
    """Settings menu: language, microphone, text size, background, logs, quit."""
    menu = _menu(overlay)

    lang_menu = _menu(menu, "언어")
    lang_group = QActionGroup(lang_menu)
    for code, label in (("en", "영어 → 한국어"), ("ko", "한국어 → 영어")):
        act = QAction(label, lang_menu, checkable=True, checked=pipeline.lang == code)
        act.setEnabled(model_available(code))
        act.triggered.connect(lambda _=False, c=code: pipeline.set_lang(c))
        lang_group.addAction(act)
        lang_menu.addAction(act)
    menu.addMenu(lang_menu)

    def use_device(device) -> None:
        # mic indices change between runs, so only PC sound / app choices are remembered
        overlay.settings.setValue("input", device if isinstance(device, str) else "mic")
        pipeline.set_device(device)

    mic_menu = _menu(menu, "듣기")
    mic_group = QActionGroup(mic_menu)
    system = QAction("PC 소리 (영상·회의)", mic_menu, checkable=True, checked=pipeline.device == SYSTEM_AUDIO)
    system.triggered.connect(lambda: use_device(SYSTEM_AUDIO))
    mic_group.addAction(system)
    mic_menu.addAction(system)

    # per-app capture works even with the PC volume at 0 / speakers muted
    app_menu = _menu(mic_menu, "앱 소리만 (무음이어도 됨)")
    apps = list_audio_apps()
    current = pipeline.device if isinstance(pipeline.device, str) and pipeline.device.startswith(APP_PREFIX) else None
    if current and current[len(APP_PREFIX):] not in apps:
        apps.insert(0, current[len(APP_PREFIX):])
    for exe in apps:
        act = QAction(exe.removesuffix(".exe"), app_menu, checkable=True, checked=current == APP_PREFIX + exe)
        act.triggered.connect(lambda _=False, e=exe: use_device(APP_PREFIX + e))
        mic_group.addAction(act)
        app_menu.addAction(act)
    if not apps:
        app_menu.addAction("소리를 내는 앱이 없어요").setEnabled(False)
    mic_menu.addMenu(app_menu)
    if current:
        app_menu.menuAction().setCheckable(True)
        app_menu.menuAction().setChecked(True)
    mic_menu.addSeparator()
    default = QAction("마이크 (시스템 기본)", mic_menu, checkable=True, checked=pipeline.device is None)
    default.triggered.connect(lambda: use_device(None))
    mic_group.addAction(default)
    mic_menu.addAction(default)
    seen = set()
    for idx, name in list_input_devices():
        if name in seen:  # the same device is listed once per Windows audio API
            continue
        seen.add(name)
        act = QAction(name, mic_menu, checkable=True, checked=pipeline.device == idx)
        act.triggered.connect(lambda _=False, i=idx: use_device(i))
        mic_group.addAction(act)
        mic_menu.addAction(act)
    menu.addMenu(mic_menu)

    size_menu = _menu(menu, "글자 크기")
    for label, size in (("작게", 18), ("보통", 24), ("크게", 30), ("아주 크게", 38)):
        act = size_menu.addAction(label)
        act.setCheckable(True)
        act.setChecked(overlay.font_size == size)
        act.triggered.connect(lambda _=False, s=size: overlay.set_font_size(s))
    menu.addMenu(size_menu)

    bg_menu = _menu(menu, "배경")
    for label, alpha in (("진하게", 245), ("보통", 215), ("연하게", 150), ("아주 연하게", 90)):
        act = bg_menu.addAction(label)
        act.setCheckable(True)
        act.setChecked(overlay.bg_alpha == alpha)
        act.triggered.connect(lambda _=False, a=alpha: overlay.set_opacity(a))
    menu.addMenu(bg_menu)

    menu.addSeparator()
    menu.addAction("통역 기록 폴더 열기").triggered.connect(lambda: (LOG_DIR.mkdir(exist_ok=True), os.startfile(LOG_DIR)))
    menu.addSeparator()
    menu.addAction("종료").triggered.connect(overlay.close)
    return menu


def attach_menu(overlay, pipeline) -> None:
    overlay.on_menu = lambda global_pos: build_menu(overlay, pipeline).exec(global_pos)
    overlay.setContextMenuPolicy(Qt.CustomContextMenu)
    overlay.customContextMenuRequested.connect(lambda pos: overlay.on_menu(overlay.mapToGlobal(pos)))
