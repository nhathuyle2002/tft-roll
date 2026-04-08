#!/usr/bin/env python3
"""
TFT Roll Tool - Auto roll & buy helper for Teamfight Tactics
Set 17: Space Gods
Usage: python tft_roll_tool.py
"""

import os
import sys

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QScrollArea, QFrame, QGridLayout,
    QGroupBox, QSpinBox, QDoubleSpinBox, QCheckBox, QTabWidget,
    QTextEdit, QSizePolicy, QFileDialog, QButtonGroup, QRadioButton,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QPixmap, QPainter, QBrush, QColor

from tft_backend import (
    OCR_AVAILABLE,
    TFT_UNITS, COST_COLOR, COST_BG, COST_LABEL,
    DEFAULTS,
    ocr_all_slots, ocr_from_image_file,
)
from tft_v2_backend import (
    RollWorkerV2, AutoCaptureWorker,
    capture_once, run_train_on_image, save_strip_raw,
    get_hashmap, hashmap_path, set_active_resolution, TRAIN_DIR,
)


# ─────────────────────────────────────────────────────────────
#  Small UI widgets
# ─────────────────────────────────────────────────────────────

def make_dot(color: str, size: int = 10) -> QPixmap:
    px = QPixmap(size, size); px.fill(Qt.transparent)
    p = QPainter(px); p.setRenderHint(QPainter.Antialiasing)
    p.setBrush(QBrush(QColor(color))); p.setPen(Qt.NoPen)
    p.drawEllipse(0, 0, size, size); p.end()
    return px


class UnitChip(QFrame):
    remove_requested = pyqtSignal(str)

    def __init__(self, name: str, cost: int):
        super().__init__()
        self.name = name
        c, bg = COST_COLOR[cost], COST_BG[cost]
        self.setFixedHeight(32)
        self.setStyleSheet(
            f"UnitChip{{background:{bg};border:1px solid {c};border-radius:6px;}}"
        )
        lay = QHBoxLayout(self); lay.setContentsMargins(6, 2, 4, 2); lay.setSpacing(4)
        dot = QLabel(); dot.setPixmap(make_dot(c)); dot.setFixedSize(10, 10)
        lbl = QLabel(name)
        lbl.setStyleSheet(f"color:{c};font-size:11px;font-weight:bold;")
        btn = QPushButton("✕"); btn.setFixedSize(18, 18)
        btn.setStyleSheet(
            "QPushButton{background:transparent;color:#888;border:none;font-size:10px;}"
            "QPushButton:hover{color:#ff5555;}"
        )
        btn.clicked.connect(lambda: self.remove_requested.emit(self.name))
        lay.addWidget(dot); lay.addWidget(lbl); lay.addStretch(); lay.addWidget(btn)


class UnitButton(QPushButton):
    def __init__(self, name: str, cost: int):
        super().__init__(name)
        self.name = name; self._sel = False
        self._c = COST_COLOR[cost]; self._bg = COST_BG[cost]
        self.setFixedSize(120, 30); self._style()

    def _style(self):
        b = f"2px solid {self._c}" if self._sel else f"1px solid {self._c}55"
        self.setStyleSheet(
            f"QPushButton{{background:{self._bg};color:{self._c};border:{b};"
            f"border-radius:5px;font-size:11px;padding:2px 6px;}}"
            f"QPushButton:hover{{border:1px solid {self._c};color:white;}}"
        )

    def set_selected(self, v: bool):
        self._sel = v; self._style()


# ─────────────────────────────────────────────────────────────
#  Always-on-top log overlay
# ─────────────────────────────────────────────────────────────

class LogOverlay(QWidget):
    """
    Semi-transparent window that floats over TFT and shows
    live roll status + OCR results without alt-tabbing.
    Drag anywhere. Double-click title bar to collapse.
    Log accumulates indefinitely — never cleared on roll.
    """
    MAX_LINES = 500

    def __init__(self, parent=None):
        super().__init__(parent, Qt.Tool | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setWindowOpacity(0.92)
        self.setMinimumWidth(500)
        self.resize(560, 420)

        self._drag_pos = None
        self._collapsed = False
        self._line_count = 0
        self._build_ui()

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ── Title bar ──────────────────────────────────────────
        self._title_bar = QWidget()
        self._title_bar.setFixedHeight(28)
        self._title_bar.setStyleSheet(
            "background:#1a1a2e;border-radius:6px 6px 0 0;"
        )
        tb = QHBoxLayout(self._title_bar)
        tb.setContentsMargins(10, 0, 6, 0)

        icon = QLabel("⚡")
        icon.setStyleSheet("color:#ffd700;font-size:12px;")
        self._title_lbl = QLabel("TFT Roll Tool  ·  Ready")
        self._title_lbl.setStyleSheet(
            "color:#c9d1d9;font-size:11px;font-weight:bold;"
        )
        self._roll_badge = QLabel("Rolls: 0")
        self._roll_badge.setStyleSheet("color:#58a6ff;font-size:11px;")

        collapse_btn = QPushButton("—")
        collapse_btn.setFixedSize(20, 20)
        collapse_btn.setStyleSheet(
            "QPushButton{background:#2a2a4a;color:#aaa;border:none;"
            "border-radius:3px;font-size:11px;padding:0;}"
            "QPushButton:hover{background:#ffd700;color:#000;}"
        )
        collapse_btn.clicked.connect(self._toggle_collapse)

        close_btn = QPushButton("✕")
        close_btn.setFixedSize(20, 20)
        close_btn.setStyleSheet(
            "QPushButton{background:#2a2a4a;color:#aaa;border:none;"
            "border-radius:3px;font-size:11px;padding:0;}"
            "QPushButton:hover{background:#ff5555;color:#fff;}"
        )
        close_btn.clicked.connect(self.hide)

        tb.addWidget(icon)
        tb.addWidget(self._title_lbl)
        tb.addStretch()
        tb.addWidget(self._roll_badge)
        tb.addSpacing(8)
        tb.addWidget(collapse_btn)
        tb.addWidget(close_btn)
        outer.addWidget(self._title_bar)

        # ── Log body ───────────────────────────────────────────
        self._body = QWidget()
        self._body.setStyleSheet(
            "background:rgba(13,17,23,220);border-radius:0 0 6px 6px;"
        )
        body_lay = QVBoxLayout(self._body)
        body_lay.setContentsMargins(6, 6, 6, 6)

        self._log_edit = QTextEdit()
        self._log_edit.setReadOnly(True)
        self._log_edit.setFrameShape(QFrame.NoFrame)
        self._log_edit.setStyleSheet(
            "QTextEdit{background:transparent;color:#c9d1d9;"
            "font-size:11px;font-family:Consolas,monospace;border:none;}"
            "QScrollBar:vertical{width:6px;background:#111;}"
            "QScrollBar::handle:vertical{background:#333;border-radius:3px;}"
        )
        body_lay.addWidget(self._log_edit)

        outer.addWidget(self._body)

    # ── Public update methods ──────────────────────────────────

    def set_status(self, text: str):
        self._title_lbl.setText(f"TFT Roll Tool  ·  {text}")

    def set_roll_count(self, n: int):
        self._roll_badge.setText(f"Rolls: {n}")

    def append_log(self, html: str):
        self._line_count += 1
        if self._line_count > self.MAX_LINES:
            # Trim oldest ~50 lines to stay within limit
            cur = self._log_edit.toHtml()
            # split on <p> tags as line boundaries and drop first 50
            self._log_edit.clear()
            self._log_edit.append(html)
            self._line_count = 1
        else:
            self._log_edit.append(html)
        self._log_edit.verticalScrollBar().setValue(
            self._log_edit.verticalScrollBar().maximum()
        )

    def append_shop_row(self, results: list):
        parts = []
        for s in results:
            raw   = s.get("raw") or ""
            match = s["match"]
            cand  = s.get("best_candidate") or ""
            t     = f"<span style='color:#444;font-size:9px;'>[{s['crop_ms']}+{s['ocr_ms']}ms]</span>"
            if match:
                parts.append(
                    f"S{s['slot']} → <span style='color:#56d364;font-weight:bold;'>{match} ✓</span>{t}"
                )
            elif raw:
                label = cand if cand else raw
                parts.append(
                    f"S{s['slot']} → <span style='color:#666;'>{label} ✗</span>{t}"
                )
            else:
                parts.append(
                    f"S{s['slot']} <span style='color:#c9650a;'>⚠ failed</span>{t}"
                )
        self.append_log("  ".join(parts))

    # ── Drag to move ───────────────────────────────────────────

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_pos = event.globalPos() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, event):
        if self._drag_pos and event.buttons() == Qt.LeftButton:
            self.move(event.globalPos() - self._drag_pos)

    def mouseReleaseEvent(self, _event):
        self._drag_pos = None

    def _toggle_collapse(self):
        self._collapsed = not self._collapsed
        self._body.setVisible(not self._collapsed)
        self.adjustSize()


# ── Auto-scaling image preview label ─────────────────────────
class _ScaledImageLabel(QLabel):
    """QLabel that always fits its pixmap inside its bounds, preserving aspect ratio."""

    def __init__(self, placeholder: str = ""):
        super().__init__(placeholder)
        self._src_pix: QPixmap | None = None

    def setSourcePixmap(self, pix: QPixmap) -> None:
        self._src_pix = pix
        self._refresh()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._src_pix:
            self._refresh()

    def _refresh(self) -> None:
        scaled = self._src_pix.scaled(
            self.width(), self.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation,
        )
        super().setPixmap(scaled)


# ── One-shot manual capture thread ───────────────────────────
class _CaptureThread(QThread):
    done  = pyqtSignal(int, list)
    error = pyqtSignal(str)

    def __init__(self, regions: list, threshold: float):
        super().__init__()
        self._regions   = regions
        self._threshold = threshold

    def run(self):
        try:
            idx, results = capture_once(self._regions, self._threshold)
            self.done.emit(idx, results)
        except Exception as e:
            self.error.emit(str(e))


# ─────────────────────────────────────────────────────────────
class TFTRollTool(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TFT Roll Tool — Set 17: Space Gods")
        self.setMinimumSize(920, 680)
        self.resize(1000, 740)

        self._selected:       dict[str, int] = {}
        self._buttons:        dict[str, UnitButton] = {}
        self._worker:         RollWorkerV2 | None = None
        self._esc_watcher:    None = None
        self._auto_capture:   AutoCaptureWorker | None = None
        self._train_img_path: str | None = None
        self._cap_thread:     _CaptureThread | None = None

        # Resolution fixed to 1920×1080
        self._active_res: tuple[int, int] = (1920, 1080)
        set_active_resolution(1920, 1080)

        # Floating log overlay (always-on-top)
        self._overlay = LogOverlay()
        self._overlay.hide()

        self._build_ui()
        self._apply_theme()

    # ─────────────────────────────────────────────────────────
    #  Top-level UI
    # ─────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QWidget(); self.setCentralWidget(root)
        rl = QVBoxLayout(root); rl.setContentsMargins(10, 10, 10, 10); rl.setSpacing(8)

        hdr = QLabel("✦  TFT Roll Tool  ·  Set 17: Space Gods")
        hdr.setFont(QFont("Segoe UI", 15, QFont.Bold))
        hdr.setStyleSheet("color:#ffd700;padding:2px 0;")
        rl.addWidget(hdr)

        self._tabs = QTabWidget()
        self._tabs.setStyleSheet(
            "QTabWidget::pane{border:1px solid #333;}"
            "QTabBar::tab{background:#1e1e1e;color:#aaa;padding:6px 18px;}"
            "QTabBar::tab:selected{background:#2a2a2a;color:white;"
            "border-bottom:2px solid #ffd700;}"
        )
        rl.addWidget(self._tabs)
        self._tabs.addTab(self._tab_main(),     "🎮  Build & Roll")
        self._tabs.addTab(self._tab_settings(), "⚙  Settings")
        self._tabs.addTab(self._tab_ocr_test(),  "🔬  OCR Test")
        self._tabs.addTab(self._tab_train(),     "🧠  Train Mode")

    # ─────────────────────────────────────────────────────────
    #  Build & Roll tab  (no timing knobs here)
    # ─────────────────────────────────────────────────────────

    def _tab_main(self) -> QWidget:
        w = QWidget(); lay = QVBoxLayout(w); lay.setSpacing(8)
        row = QHBoxLayout(); row.setSpacing(10)

        # ── Left: unit picker ─────────────────────────────────
        pg = QGroupBox("Unit Picker  (click to add to build)")
        pg.setMinimumWidth(480)
        pl = QVBoxLayout(pg)
        sc = QScrollArea(); sc.setWidgetResizable(True)
        sc.setFrameShape(QFrame.NoFrame)
        sc.setStyleSheet("QScrollArea{background:transparent;}")
        inner = QWidget(); il = QVBoxLayout(inner); il.setSpacing(6)
        for cost in sorted(TFT_UNITS):
            lbl = QLabel(COST_LABEL[cost])
            lbl.setStyleSheet(
                f"color:{COST_COLOR[cost]};font-size:11px;"
                f"font-weight:bold;margin-top:4px;"
            )
            il.addWidget(lbl)
            rw = QWidget()
            gl = QGridLayout(rw); gl.setSpacing(4); gl.setContentsMargins(0, 0, 0, 0)
            for idx, name in enumerate(TFT_UNITS[cost]):
                btn = UnitButton(name, cost)
                btn.clicked.connect(lambda _, n=name, c=cost: self._toggle(n, c))
                gl.addWidget(btn, idx // 4, idx % 4)
                self._buttons[name] = btn
            il.addWidget(rw)
        il.addStretch()
        sc.setWidget(inner); pl.addWidget(sc)
        row.addWidget(pg, 3)

        # ── Right: build board + roll controls ────────────────
        rc = QVBoxLayout(); rc.setSpacing(8)

        # Build board
        bg = QGroupBox("Build Board"); bg.setMinimumWidth(260)
        bl = QVBoxLayout(bg)
        self._chip_w = QWidget()
        self._chip_l = QVBoxLayout(self._chip_w)
        self._chip_l.setSpacing(4); self._chip_l.setContentsMargins(0, 0, 0, 0)
        self._chip_l.addStretch()
        bsc = QScrollArea(); bsc.setWidgetResizable(True)
        bsc.setFrameShape(QFrame.NoFrame)
        bsc.setStyleSheet("QScrollArea{background:transparent;}")
        bsc.setWidget(self._chip_w); bl.addWidget(bsc)
        clr = QPushButton("Clear All")
        clr.setStyleSheet(
            "QPushButton{color:#ff5555;background:#2a2a2a;border:1px solid #555;"
            "border-radius:4px;padding:4px;}"
            "QPushButton:hover{background:#3a2020;}"
        )
        clr.clicked.connect(self._clear_all); bl.addWidget(clr)
        self._cnt_lbl = QLabel("0 units selected")
        self._cnt_lbl.setStyleSheet("color:#888;font-size:11px;")
        bl.addWidget(self._cnt_lbl)
        rc.addWidget(bg, 2)

        # Roll controls  (status + OCR toggle + START/STOP only)
        rg = QGroupBox("Roll Controls"); rl3 = QVBoxLayout(rg); rl3.setSpacing(6)

        self._status_lbl = QLabel("Ready")
        self._status_lbl.setStyleSheet(
            "color:#58a6ff;font-size:12px;font-weight:bold;"
        )
        rl3.addWidget(self._status_lbl)

        self._roll_lbl = QLabel("Rolls: 0")
        self._roll_lbl.setStyleSheet("color:#aaa;font-size:11px;")
        rl3.addWidget(self._roll_lbl)

        # DirectInput status badge
        di_badge = QLabel("⚡ DirectInput active — game inputs enabled")
        di_badge.setStyleSheet(
            "color:#56d364;font-size:10px;background:#0d2a14;"
            "border:1px solid #56d36455;border-radius:4px;padding:2px 6px;"
        )
        rl3.addWidget(di_badge)

        self._autoroll_cb = QCheckBox("Auto Roll – script presses D automatically")
        self._autoroll_cb.setChecked(False)
        self._autoroll_cb.setStyleSheet("color:#c9d1d9;font-size:11px;")
        self._autoroll_cb.setToolTip(
            "ON:  script presses D after every buy cycle (bot rolls for you).\n"
            "OFF: you press D manually; script detects it and buys."
        )
        rl3.addWidget(self._autoroll_cb)

        # ── Buy speed mode ────────────────────────────────────
        self._mode_group = QButtonGroup(self)
        mode_row = QHBoxLayout(); mode_row.setSpacing(12)

        self._mode_human = QRadioButton("👤  Human  (natural timing)")
        self._mode_human.setChecked(True)
        self._mode_human.setStyleSheet("color:#c9d1d9;font-size:11px;")
        self._mode_human.setToolTip(
            "Natural buy speed — shop_wait 0.15 s, buy_delay 0.05 s.\n"
            "Works with both manual and auto roll."
        )

        self._mode_bot = QRadioButton("🤖  BOT  (fastest buy)")
        self._mode_bot.setChecked(False)
        self._mode_bot.setStyleSheet("color:#c9d1d9;font-size:11px;")
        self._mode_bot.setToolTip(
            "Fastest buy speed — OCR runs parallel with shop_wait 0.1 s, buy_delay 0.005 s.\n"
            "Works with both manual and auto roll."
        )

        self._mode_group.addButton(self._mode_human, 0)
        self._mode_group.addButton(self._mode_bot,   1)
        mode_row.addWidget(self._mode_human)
        mode_row.addWidget(self._mode_bot)
        mode_row.addStretch()
        rl3.addLayout(mode_row)
        self._mode_human.toggled.connect(self._apply_mode_preset)

        if not OCR_AVAILABLE:
            warn = QLabel("⚠  OCR libs not installed — buying is disabled.\n"
                          "   Run:  pip install pytesseract opencv-python Pillow\n"
                          "   And install Tesseract from: github.com/UB-Mannheim/tesseract/wiki")
            warn.setStyleSheet(
                "color:#ffa500;font-size:10px;margin-left:4px;"
                "background:#2a1a00;border:1px solid #ffa50055;"
                "border-radius:4px;padding:4px 6px;"
            )
            warn.setWordWrap(True)
            rl3.addWidget(warn)

        br = QHBoxLayout()
        self._start_btn = QPushButton("▶  START"); self._start_btn.setFixedHeight(38)
        self._start_btn.setStyleSheet(
            "QPushButton{background:#1a4d1a;color:#56d364;border:1px solid #56d364;"
            "border-radius:6px;font-size:13px;font-weight:bold;}"
            "QPushButton:hover{background:#235c23;}"
            "QPushButton:disabled{background:#1e1e1e;color:#555;border-color:#333;}"
        )
        self._start_btn.clicked.connect(self._start)
        self._stop_btn = QPushButton("■  STOP")
        self._stop_btn.setFixedHeight(38); self._stop_btn.setEnabled(False)
        self._stop_btn.setStyleSheet(
            "QPushButton{background:#4d1a1a;color:#ff5555;border:1px solid #ff5555;"
            "border-radius:6px;font-size:13px;font-weight:bold;}"
            "QPushButton:hover{background:#5c2323;}"
            "QPushButton:disabled{background:#1e1e1e;color:#555;border-color:#333;}"
        )
        self._stop_btn.clicked.connect(self._stop)
        br.addWidget(self._start_btn); br.addWidget(self._stop_btn)
        rl3.addLayout(br)

        esc_hint = QLabel("Press  Esc  to stop instantly while in-game.")
        esc_hint.setStyleSheet("color:#555;font-size:10px;margin-left:2px;")
        rl3.addWidget(esc_hint)

        # Overlay toggle + auto-show option
        self._auto_overlay_cb = QCheckBox("Auto-show overlay on start")
        self._auto_overlay_cb.setChecked(True)
        self._auto_overlay_cb.setStyleSheet("color:#8b949e;font-size:10px;")
        self._auto_overlay_cb.setToolTip(
            "When checked, the log overlay is shown automatically\n"
            "whenever rolling or auto-capture starts."
        )
        rl3.addWidget(self._auto_overlay_cb)

        self._overlay_btn = QPushButton("📌  Show Log Overlay")
        self._overlay_btn.setFixedHeight(38)
        self._overlay_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._overlay_btn.setCheckable(True)
        self._overlay_btn.setStyleSheet(
            "QPushButton{background:#1a1a2e;color:#bc8cff;border:1px solid #bc8cff;"
            "border-radius:5px;font-size:11px;}"
            "QPushButton:checked{background:#2a2a4e;color:#ffd700;border-color:#ffd700;}"
            "QPushButton:hover{background:#2a2a4e;}"
        )
        self._overlay_btn.clicked.connect(self._toggle_overlay)
        rl3.addWidget(self._overlay_btn)

        tip = QLabel("Move mouse to top-left corner to emergency stop.")
        tip.setStyleSheet("color:#555;font-size:10px;")
        rl3.addWidget(tip)
        rc.addWidget(rg, 1)

        # _log kept as a hidden sink so existing append calls remain valid
        self._log = QTextEdit(); self._log.setReadOnly(True)
        self._log.hide()

        rw2 = QWidget(); rw2.setLayout(rc)
        row.addWidget(rw2, 2)
        lay.addLayout(row)
        return w

    # ─────────────────────────────────────────────────────────
    #  Settings tab
    # ─────────────────────────────────────────────────────────

    def _tab_settings(self) -> QWidget:
        w = QWidget()
        outer = QVBoxLayout(w); outer.setContentsMargins(10, 10, 10, 10)

        sc = QScrollArea(); sc.setWidgetResizable(True); sc.setFrameShape(QFrame.NoFrame)
        sc.setStyleSheet("QScrollArea{background:transparent;}")
        inner = QWidget(); lay = QVBoxLayout(inner)
        lay.setAlignment(Qt.AlignTop); lay.setSpacing(6)

        # ── Resolution (fixed) ────────────────────────────────
        lay.addWidget(self._sh("🖥  Display Resolution"))
        res_row = QHBoxLayout(); res_row.setSpacing(8)
        res_lbl = QLabel("Resolution:"); res_lbl.setFixedWidth(160)
        res_fixed = QLabel("1920×1080  (fixed)")
        res_fixed.setStyleSheet("color:#58a6ff;font-size:12px;font-weight:bold;")
        res_row.addWidget(res_lbl)
        res_row.addWidget(res_fixed)
        res_row.addStretch()
        lay.addLayout(res_row)

        # ── Timing ────────────────────────────────────────────
        lay.addWidget(self._sh("⏱  Timing"))

        self._pre_sp = self._ispin(
            1, 15, DEFAULTS["pre_delay"], " sec",
            "Start delay  (countdown before automation begins)"
        )
        self._shop_wait_sp = self._dspin(
            0.05, 3.0, DEFAULTS["shop_wait"], 0.05, " sec",
            "Shop load wait  (pause after rolling for shop to populate)"
        )
        self._buy_delay_sp = self._dspin(
            0.005, 1.0, DEFAULTS["buy_delay"], 0.005, " sec",
            "Buy speed  (delay between clicking each shop slot)"
        )
        self._ocr_thr_sp = self._dspin(
            0.3, 1.0, DEFAULTS["ocr_threshold"], 0.05, "",
            "OCR match threshold  (higher = stricter matching, 0.62 recommended)"
        )
        for label, spin in [
            ("Start delay:",    self._pre_sp),
            ("Shop load wait:", self._shop_wait_sp),
            ("Buy speed:",      self._buy_delay_sp),
            ("OCR threshold:",  self._ocr_thr_sp),
        ]:
            r = QHBoxLayout()
            lbl = QLabel(label); lbl.setFixedWidth(160)
            r.addWidget(lbl); r.addWidget(spin); r.addStretch()
            lay.addLayout(r)

        # ── Buttons ───────────────────────────────────────────
        lay.addWidget(QLabel(""))  # spacer
        btn_row = QHBoxLayout()

        save_btn = QPushButton("💾  Save Settings")
        save_btn.setFixedHeight(34)
        save_btn.setStyleSheet(
            "QPushButton{background:#1e3a1e;color:#56d364;border:1px solid #56d364;"
            "border-radius:5px;padding:4px 14px;font-size:12px;}"
            "QPushButton:hover{background:#235c23;}"
        )
        save_btn.clicked.connect(self._save_settings)

        reset_btn = QPushButton("↺  Reset to Default")
        reset_btn.setFixedHeight(34)
        reset_btn.setStyleSheet(
            "QPushButton{background:#2a2010;color:#ffd700;border:1px solid #ffd700;"
            "border-radius:5px;padding:4px 14px;font-size:12px;}"
            "QPushButton:hover{background:#3a3010;}"
        )
        reset_btn.clicked.connect(self._reset_settings)

        btn_row.addWidget(save_btn)
        btn_row.addWidget(reset_btn)
        btn_row.addStretch()
        lay.addLayout(btn_row)

        self._settings_status = QLabel("")
        self._settings_status.setStyleSheet("color:#58a6ff;font-size:11px;")
        lay.addWidget(self._settings_status)

        sc.setWidget(inner)
        outer.addWidget(sc)
        return w

    # ─────────────────────────────────────────────────────────
    #  OCR Test tab  (upload image → OCR → inspect results)
    # ─────────────────────────────────────────────────────────

    def _tab_ocr_test(self) -> QWidget:
        w = QWidget(); lay = QVBoxLayout(w)
        lay.setContentsMargins(12, 12, 12, 12); lay.setSpacing(8)

        # ── Header ────────────────────────────────────────────
        hdr = QLabel("Upload a TFT screenshot to see what OCR reads from each shop slot.")
        hdr.setStyleSheet("color:#aaa;font-size:11px;")
        hdr.setWordWrap(True)
        lay.addWidget(hdr)

        # ── File picker row ───────────────────────────────────
        picker_row = QHBoxLayout(); picker_row.setSpacing(8)

        self._ocr_test_path_lbl = QLabel("No image selected")
        self._ocr_test_path_lbl.setStyleSheet(
            "color:#666;font-size:11px;padding:4px 8px;"
            "background:#1a1a1a;border:1px solid #333;border-radius:4px;"
        )
        self._ocr_test_path_lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        browse_btn = QPushButton("📂  Browse…")
        browse_btn.setFixedHeight(30)
        browse_btn.setStyleSheet(
            "QPushButton{background:#1e2a3a;color:#58a6ff;border:1px solid #58a6ff;"
            "border-radius:4px;padding:2px 12px;font-size:11px;}"
            "QPushButton:hover{background:#253850;}"
        )
        browse_btn.clicked.connect(self._ocr_test_browse)

        picker_row.addWidget(self._ocr_test_path_lbl)
        picker_row.addWidget(browse_btn)
        lay.addLayout(picker_row)

        # ── Image preview (auto-scales with the window) ───────
        self._ocr_test_preview = _ScaledImageLabel("Image preview will appear here")
        self._ocr_test_preview.setAlignment(Qt.AlignCenter)
        self._ocr_test_preview.setMinimumHeight(140)
        self._ocr_test_preview.setMaximumHeight(320)
        self._ocr_test_preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self._ocr_test_preview.setStyleSheet(
            "background:#111;border:1px solid #2a2a2a;border-radius:4px;color:#444;"
        )
        lay.addWidget(self._ocr_test_preview)

        # ── Options row ───────────────────────────────────────
        opt_row = QHBoxLayout()
        self._ocr_test_use_hash_cb = QCheckBox("⚡ Use hash lookup first")
        self._ocr_test_use_hash_cb.setChecked(True)
        self._ocr_test_use_hash_cb.setToolTip(
            "When checked, each slot is looked up in the hashmap before running Tesseract.\n"
            "Hash hits are instant (⚡); only misses fall back to OCR."
        )
        opt_row.addWidget(self._ocr_test_use_hash_cb)
        opt_row.addStretch()
        lay.addLayout(opt_row)

        # ── Run button ────────────────────────────────────────
        run_btn = QPushButton("🔍  Run OCR")
        run_btn.setFixedHeight(34)
        run_btn.setStyleSheet(
            "QPushButton{background:#1a1a3a;color:#bc8cff;border:1px solid #bc8cff;"
            "border-radius:5px;padding:4px 18px;font-size:12px;font-weight:bold;}"
            "QPushButton:hover{background:#2a2a4a;}"
            "QPushButton:disabled{color:#444;border-color:#333;}"
        )
        run_btn.setEnabled(False)
        run_btn.clicked.connect(self._ocr_test_run)
        self._ocr_test_run_btn = run_btn
        lay.addWidget(run_btn)

        # ── Results log ───────────────────────────────────────
        res_lbl = QLabel("Results:")
        res_lbl.setStyleSheet("color:#ffd700;font-size:11px;font-weight:bold;")
        lay.addWidget(res_lbl)

        self._ocr_test_log = QTextEdit()
        self._ocr_test_log.setReadOnly(True)
        self._ocr_test_log.setMinimumHeight(180)
        self._ocr_test_log.setStyleSheet(
            "QTextEdit{background:#0d0d0d;color:#ccc;font-family:monospace;"
            "font-size:11px;border:1px solid #2a2a2a;border-radius:4px;padding:4px;}"
        )
        lay.addWidget(self._ocr_test_log, 1)

        # Status line
        self._ocr_test_status = QLabel("")
        self._ocr_test_status.setStyleSheet("color:#58a6ff;font-size:10px;")
        lay.addWidget(self._ocr_test_status)

        # Store selected path
        self._ocr_test_img_path = None
        return w

    def _ocr_test_browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open TFT Screenshot", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.webp)"
        )
        if not path:
            return
        self._ocr_test_img_path = path
        self._ocr_test_path_lbl.setText(os.path.basename(path))
        self._ocr_test_path_lbl.setStyleSheet(
            "color:#ccc;font-size:11px;padding:4px 8px;"
            "background:#1a1a1a;border:1px solid #444;border-radius:4px;"
        )
        self._ocr_test_run_btn.setEnabled(True)

        # Show scaled preview
        pix = QPixmap(path)
        if not pix.isNull():
            self._ocr_test_preview.setSourcePixmap(pix)

        self._ocr_test_log.clear()
        self._ocr_test_status.setText("")

    def _ocr_test_run(self):
        if not OCR_AVAILABLE:
            self._ocr_test_status.setText("OCR libs not installed.")
            return
        if not self._ocr_test_img_path:
            return

        self._ocr_test_status.setText("Running OCR…")
        QApplication.processEvents()

        regions   = [list(r) for r in DEFAULTS["name_regions"]]
        thr       = DEFAULTS["ocr_threshold"]
        use_hash  = self._ocr_test_use_hash_cb.isChecked()

        try:
            if use_hash:
                results = run_train_on_image(self._ocr_test_img_path, regions, thr)
            else:
                results = ocr_from_image_file(self._ocr_test_img_path, regions, thr)
        except Exception as e:
            self._ocr_test_status.setText(f"Error: {e}")
            return

        img_size = results[0]["img_size"] if results else ("?", "?")
        mode_tag = "hash+OCR" if use_hash else "OCR only"
        self._ocr_test_log.clear()
        _hdr = (
            f"<span style='color:#ffd700;font-weight:bold;'>"
            f"[OCR Test] Image: {os.path.basename(self._ocr_test_img_path)}"
            f"  ({img_size[0]}×{img_size[1]})  [{mode_tag}]"
            f"</span>"
        )
        self._ocr_test_log.append(_hdr)
        self._olog(_hdr)
        matched = 0
        total_ms = 0.0
        for r in results:
            raw     = r.get("raw") or ""
            cand    = r.get("best_candidate") or ""
            reg     = r["scaled_region"]
            hit     = r["match"] is not None
            src     = r.get("source", "ocr")
            crop_ms = r.get("crop_ms", 0)
            ocr_ms  = r.get("ocr_ms", 0)
            total_ms += crop_ms + ocr_ms
            if hit:
                matched += 1
                color   = "#56d364"
                src_tag = "⚡" if src == "hash" else "🔬"
                tag     = f"→ <b>{r['match']}</b> {src_tag} ✓"
            elif raw:
                color = "#666"
                label = cand if cand else raw
                tag   = f"→ {label} ✗"
            else:
                color = "#c9650a"
                tag   = "⚠ failed"
            coord  = f"[{reg[0]},{reg[1]} {reg[2]}×{reg[3]}]"
            timing = f"crop {crop_ms}ms | ocr {ocr_ms}ms"
            _line = (
                f"<span style='color:{color};'>Slot {r['slot']}: {tag}</span>"
                f"<span style='color:#444;font-size:9px;'>  {coord}  {timing}</span>"
            )
            self._ocr_test_log.append(_line)
            self._olog(_line)

        if use_hash:
            hash_hits    = sum(1 for r in results if r.get("source") == "hash")
            hash_ms      = sum(r.get("crop_ms", 0) for r in results if r.get("source") == "hash")
            ocr_only_ms  = sum(
                r.get("crop_ms", 0) + r.get("ocr_ms", 0)
                for r in results if r.get("source") != "hash"
            )
            _summary = (
                f"<span style='color:#555;font-size:10px;'>─ "
                f"⚡{hash_hits}/5 {hash_ms:.0f}ms hash  |  "
                f"🔬{5 - hash_hits}/5 {ocr_only_ms:.0f}ms ocr  |  "
                f"{matched}/{len(results)} matched</span>"
            )
        else:
            _summary = (
                f"<span style='color:#555;font-size:10px;'>─ total {total_ms:.0f} ms — {matched}/{len(results)} matched</span>"
            )
        self._ocr_test_log.append(_summary)
        self._olog(_summary)
        self._ocr_test_status.setText(
            f"Done — {matched} / {len(results)} slots matched above threshold."
        )

    # ── Settings helpers ─────────────────────────────────────

    @staticmethod
    def _sh(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(
            "color:#ffd700;font-size:11px;font-weight:bold;margin-top:10px;"
        )
        return lbl

    @staticmethod
    def _slot_lbl(i: int) -> QLabel:
        lbl = QLabel(f"Slot {i + 1}:"); lbl.setFixedWidth(55)
        return lbl

    @staticmethod
    def _dspin(lo, hi, val, step, suffix, tooltip="") -> QDoubleSpinBox:
        sp = QDoubleSpinBox()
        sp.setRange(lo, hi); sp.setValue(val)
        sp.setSingleStep(step); sp.setSuffix(suffix)
        sp.setFixedWidth(100)
        if tooltip:
            sp.setToolTip(tooltip)
        return sp

    @staticmethod
    def _ispin(lo, hi, val, suffix, tooltip="") -> QSpinBox:
        sp = QSpinBox()
        sp.setRange(lo, hi); sp.setValue(val)
        sp.setSuffix(suffix); sp.setFixedWidth(100)
        if tooltip:
            sp.setToolTip(tooltip)
        return sp

    def _test_ocr(self):
        """Scan all 5 name regions and log raw OCR text + best match + timing."""
        if not OCR_AVAILABLE:
            self._settings_status.setText("OCR libs not installed — cannot test.")
            return

        self._settings_status.setText("Scanning…")
        QApplication.processEvents()

        regions  = [list(r) for r in DEFAULTS["name_regions"]]
        thr      = self._ocr_thr_sp.value()
        t0       = __import__('time').perf_counter()
        results  = ocr_all_slots(regions, thr)
        total_ms = (__import__('time').perf_counter() - t0) * 1000

        self._log.append("<span style='color:#bc8cff;font-weight:bold;'>── OCR Test ──</span>")
        for r in results:
            raw  = r.get("raw") or ""
            cand = r.get("best_candidate") or ""
            hit  = r["match"] is not None
            if hit:
                color = "#56d364"
                tag   = f"→ <b>{r['match']}</b> ✓"
            elif raw:
                color = "#666"
                label = cand if cand else raw
                tag   = f"→ {label} ✗"
            else:
                color = "#c9650a"
                tag   = "⚠ failed"
            timing = f"<span style='color:#444;font-size:9px;'>  crop {r['crop_ms']}ms | ocr {r['ocr_ms']}ms</span>"
            self._log.append(
                f"<span style='color:{color};'>"
                f"S{r['slot']}: {tag}"
                f"</span>{timing}"
            )
        self._log.append(
            f"<span style='color:#555;font-size:10px;'>─ total {total_ms:.0f} ms</span>"
        )
        self._settings_status.setText(f"OCR test done — {total_ms:.0f} ms total.")

    def _apply_mode_preset(self):
        """Apply Human or BOT timing presets to the spinboxes."""
        if self._mode_human.isChecked():
            self._shop_wait_sp.setValue(0.05)
            self._buy_delay_sp.setValue(0.02)
        else:
            self._shop_wait_sp.setValue(0.05)
            self._buy_delay_sp.setValue(0.005)

    def _save_settings(self):
        DEFAULTS["pre_delay"]     = self._pre_sp.value()
        DEFAULTS["shop_wait"]     = self._shop_wait_sp.value()
        DEFAULTS["buy_delay"]     = self._buy_delay_sp.value()
        DEFAULTS["ocr_threshold"] = self._ocr_thr_sp.value()
        self._train_refresh_hm()
        self._settings_status.setText(
            f"✓ Settings saved  (1920×1080, {get_hashmap().size} hash entries)."
        )

    def _reset_settings(self):
        self._mode_human.setChecked(True)
        self._pre_sp.setValue(1)
        self._ocr_thr_sp.setValue(0.70)
        self._settings_status.setText("↺ Reset to defaults. Click Save to apply.")

    # ─────────────────────────────────────────────────────────
    #  Unit selection logic
    # ─────────────────────────────────────────────────────────

    def _toggle(self, name: str, cost: int):
        if name in self._selected:
            self._remove(name)
        else:
            self._add(name, cost)

    def _add(self, name: str, cost: int):
        if name in self._selected:
            return
        self._selected[name] = cost
        chip = UnitChip(name, cost)
        chip.remove_requested.connect(self._remove)
        self._chip_l.insertWidget(self._chip_l.count() - 1, chip)
        self._buttons[name].set_selected(True)
        self._update_count()

    def _remove(self, name: str):
        if name not in self._selected:
            return
        del self._selected[name]
        for i in range(self._chip_l.count()):
            item = self._chip_l.itemAt(i)
            if item and isinstance(item.widget(), UnitChip) \
                    and item.widget().name == name:
                wgt = item.widget()
                self._chip_l.removeWidget(wgt)
                wgt.deleteLater()
                break
        if name in self._buttons:
            self._buttons[name].set_selected(False)
        self._update_count()

    def _clear_all(self):
        for n in list(self._selected.keys()):
            self._remove(n)

    def _update_count(self):
        n = len(self._selected)
        self._cnt_lbl.setText(f"{n} unit{'s' if n != 1 else ''} selected")

    # ─────────────────────────────────────────────────────────
    #  Automation
    # ─────────────────────────────────────────────────────────

    def _show_overlay_if_auto(self) -> None:
        """Show the log overlay automatically if the auto-show option is enabled."""
        if self._auto_overlay_cb.isChecked() and not self._overlay.isVisible():
            screen = QApplication.primaryScreen().availableGeometry()
            self._overlay.move(screen.right() - self._overlay.width() - 20,
                               screen.top() + 80)
            self._overlay.show()
            self._overlay_btn.setText("📌  Hide Log Overlay")
            self._overlay_btn.setChecked(True)

    def _start(self):
        self._show_overlay_if_auto()
        self._log.clear()
        cfg = {
            "pre_delay":     self._pre_sp.value(),
            "shop_wait":     self._shop_wait_sp.value(),
            "buy_delay":     self._buy_delay_sp.value(),
            "ocr_threshold": self._ocr_thr_sp.value(),
            "auto_roll":     self._autoroll_cb.isChecked(),
            "bot_mode":      self._mode_bot.isChecked(),
            "wanted":        list(self._selected.keys()),
            "click_pos":     [list(p) for p in DEFAULTS["click_pos"]],
            "name_regions":  [list(r) for r in DEFAULTS["name_regions"]],
        }
        self._worker = RollWorkerV2(cfg)
        self._worker.status_signal.connect(self._on_status)
        self._worker.roll_signal.connect(self._on_roll)
        self._worker.found_signal.connect(self._on_found)
        self._worker.shop_signal.connect(self._on_shop)
        self._worker.finished.connect(self._on_done)
        self._worker.start()

        self._start_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)

    def _on_esc(self):
        if self._worker:
            self._worker.stop("Stopped by ESC.")
        if self._esc_watcher:
            self._esc_watcher.stop()

    def _stop(self):
        if self._worker:
            self._worker.stop()
        if self._esc_watcher:
            self._esc_watcher.stop()

    def _toggle_overlay(self):
        if self._overlay.isVisible():
            self._overlay.hide()
            self._overlay_btn.setText("📌  Show Log Overlay")
            self._overlay_btn.setChecked(False)
        else:
            # Position near top-right of screen on first show
            screen = QApplication.primaryScreen().availableGeometry()
            self._overlay.move(screen.right() - self._overlay.width() - 20,
                               screen.top() + 80)
            self._overlay.show()
            self._overlay_btn.setText("📌  Hide Log Overlay")
            self._overlay_btn.setChecked(True)

    def _on_roll(self, n: int):
        self._roll_lbl.setText(f"Rolls: {n}")
        self._overlay.set_roll_count(n)

    def _on_status(self, msg: str):
        self._status_lbl.setText(msg)
        self._overlay.set_status(msg)

    def _olog(self, html: str) -> None:
        """Append html to the global log overlay (all tabs share it)."""
        self._overlay.append_log(html)

    def _on_found(self, msg: str):
        html = f"<span style='color:#56d364;'>✓ {msg}</span>"
        self._log.append(html)
        self._olog(html)

    def _on_shop(self, results: list):
        self._overlay.append_shop_row(results)
        parts = []
        for s in results:
            raw     = s.get("raw") or ""
            match   = s["match"]
            cand    = s.get("best_candidate") or ""
            src_tag = "⚡" if s.get("source") == "hash" else ""
            t       = f"<span style='color:#444;font-size:9px;'>[{s['crop_ms']}+{s['ocr_ms']}ms]</span>"
            if match:
                parts.append(f"<span style='color:#56d364;'>S{s['slot']} → <b>{match}</b> ✓{src_tag}</span>{t}")
            elif raw:
                label = cand if cand else raw
                parts.append(f"<span style='color:#555;'>S{s['slot']} → {label} ✗</span>{t}")
            else:
                parts.append(f"<span style='color:#c9650a;'>S{s['slot']} ⚠ failed</span>{t}")
        self._log.append("  ".join(parts))

    def _on_done(self):
        if self._esc_watcher:
            self._esc_watcher.stop()
            self._esc_watcher = None
        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._worker = None

    # ─────────────────────────────────────────────────────────
    #  Train Mode tab
    # ─────────────────────────────────────────────────────────

    def _tab_train(self) -> QWidget:
        w   = QWidget()
        lay = QVBoxLayout(w)
        lay.setSpacing(10)
        lay.setContentsMargins(12, 12, 12, 12)

        # Hashmap stats bar
        hm_row = QHBoxLayout()
        self._hm_lbl = QLabel("Hashmap: loading…")
        self._hm_lbl.setStyleSheet("color:#ffd700;font-weight:bold;")
        btn_reload = QPushButton("⟳ Refresh")
        btn_reload.setFixedWidth(90)
        btn_reload.clicked.connect(self._train_refresh_hm)
        # Train-mode OCR threshold (kept separate from roll threshold)
        hm_row.addWidget(self._hm_lbl)
        hm_row.addStretch()
        hm_row.addWidget(QLabel("Train threshold (score ≥0.90 accepted):"))
        self._train_thr_sp = QDoubleSpinBox()
        self._train_thr_sp.setRange(0.50, 1.00)
        self._train_thr_sp.setSingleStep(0.01)
        self._train_thr_sp.setValue(0.99)
        self._train_thr_sp.setFixedWidth(72)
        self._train_thr_sp.setToolTip(
            "Score range: 0.0 (no match) – 1.0 (identical).\n"
            "Recommended ≥0.99 for training so only high-confidence\n"
            "OCR results are saved to the hash map."
        )
        hm_row.addWidget(self._train_thr_sp)
        hm_row.addSpacing(8)
        hm_row.addWidget(btn_reload)
        lay.addLayout(hm_row)

        # ── Train from file ───────────────────────────────────
        img_grp = QGroupBox("Train from Screenshot File")
        iglay   = QVBoxLayout(img_grp)
        row1 = QHBoxLayout()
        btn_browse = QPushButton("📂  Browse…")
        btn_browse.setFixedWidth(100)
        btn_browse.clicked.connect(self._train_browse)
        self._train_file_lbl = QLabel("No file selected")
        self._train_file_lbl.setStyleSheet("color:#8b949e;")
        self._btn_run_img = QPushButton("▶  Run Hash+OCR")
        self._btn_run_img.setEnabled(False)
        self._btn_run_img.clicked.connect(self._train_run_file)
        row1.addWidget(btn_browse)
        row1.addWidget(self._train_file_lbl, 1)
        row1.addWidget(self._btn_run_img)
        iglay.addLayout(row1)
        self._train_thumb = QLabel()
        self._train_thumb.setFixedHeight(80)
        self._train_thumb.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self._train_thumb.hide()
        iglay.addWidget(self._train_thumb)
        lay.addWidget(img_grp)

        # ── In-game capture ───────────────────────────────────
        cap_grp = QGroupBox("In-Game Capture")
        cglay   = QVBoxLayout(cap_grp)
        row2 = QHBoxLayout()
        self._btn_manual_cap = QPushButton("📸  Manual Capture (now)")
        self._btn_manual_cap.setFixedSize(200, 26)
        self._btn_manual_cap.setStyleSheet(
            "QPushButton{background:#1a2a1a;color:#56d364;border:1px solid #56d364;"
            "border-radius:5px;font-size:11px;}"
            "QPushButton:hover{background:#235c23;}"
            "QPushButton:disabled{background:#1e1e1e;color:#555;border-color:#333;}"
        )
        self._btn_manual_cap.clicked.connect(self._train_manual_capture)
        self._btn_snap_strip = QPushButton("📷  Name Strip Preview")
        self._btn_snap_strip.setFixedSize(180, 26)
        self._btn_snap_strip.setStyleSheet(
            "QPushButton{background:#1a1a2e;color:#bc8cff;border:1px solid #bc8cff;"
            "border-radius:5px;font-size:11px;}"
            "QPushButton:hover{background:#2a2a4e;}"
        )
        self._btn_snap_strip.clicked.connect(self._train_snap_strip)
        self._btn_save_strip = QPushButton("💾  Save Strip")
        self._btn_save_strip.setFixedSize(130, 26)
        self._btn_save_strip.setStyleSheet(
            "QPushButton{background:#1a2a1a;color:#ffd700;border:1px solid #ffd700;"
            "border-radius:5px;font-size:11px;}"
            "QPushButton:hover{background:#2a3a0a;}"
        )
        self._btn_save_strip.clicked.connect(self._train_save_strip_only)
        row2.addWidget(self._btn_manual_cap)
        row2.addWidget(self._btn_snap_strip)
        row2.addWidget(self._btn_save_strip)
        row2.addStretch()
        cglay.addLayout(row2)
        self._strip_preview_lbl = _ScaledImageLabel("Name strip preview will appear here")
        self._strip_preview_lbl.setFixedHeight(36)
        self._strip_preview_lbl.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self._strip_preview_lbl.setStyleSheet(
            "background:#0d0d0d;border:1px solid #2a2a2a;border-radius:4px;color:#444;"
        )
        self._strip_preview_lbl.hide()
        cglay.addWidget(self._strip_preview_lbl)
        row3 = QHBoxLayout()
        self._btn_auto_cap = QPushButton("▶  Start Auto Capture")
        self._btn_auto_cap.setFixedSize(200, 26)
        self._btn_auto_cap.setStyleSheet(
            "QPushButton{background:#1a1a3a;color:#58a6ff;border:1px solid #58a6ff;"
            "border-radius:5px;font-size:11px;}"
            "QPushButton:checked{background:#0d3a5c;color:#ffd700;border-color:#ffd700;}"
            "QPushButton:hover{background:#253850;}"
        )
        self._btn_auto_cap.setCheckable(True)
        self._btn_auto_cap.clicked.connect(self._train_toggle_auto)
        self._cap_interval_sp = QSpinBox()
        self._cap_interval_sp.setRange(5, 300)
        self._cap_interval_sp.setValue(15)
        self._cap_interval_sp.setFixedWidth(70)
        row3.addWidget(self._btn_auto_cap)
        row3.addWidget(QLabel("  Interval (s):"))
        row3.addWidget(self._cap_interval_sp)
        row3.addStretch()
        cglay.addLayout(row3)
        self._cap_status = QLabel("Idle")
        self._cap_status.setStyleSheet("color:#8b949e;font-size:11px;")
        cglay.addWidget(self._cap_status)
        lay.addWidget(cap_grp)

        # ── Results log ───────────────────────────────────────
        res_grp = QGroupBox("Results  (⚡ = hash hit, 🔬 = OCR fallback)")
        rglay   = QVBoxLayout(res_grp)
        self._train_log = QTextEdit()
        self._train_log.setReadOnly(True)
        self._train_log.setFont(QFont("Consolas,monospace", 10))
        self._train_log.setStyleSheet(
            "background:#0d1117;color:#c9d1d9;border:none;"
        )
        rglay.addWidget(self._train_log)
        lay.addWidget(res_grp, 1)

        self._train_refresh_hm()
        return w

    def _train_refresh_hm(self):
        hm = get_hashmap()
        hm.load()
        n = hm.size
        self._hm_lbl.setText(
            f"Hashmap: {n} entr{'y' if n == 1 else 'ies'}  ·  {hm._path.name}"
        )

    def _train_snap_strip(self):
        """Grab a live screenshot of the name-region strip and show it as a preview."""
        try:
            from PIL import ImageGrab
            import numpy as np
            name_regions = DEFAULTS["name_regions"]
            bx0 = min(r[0] for r in name_regions)
            by0 = min(r[1] for r in name_regions)
            bx1 = max(r[0] + r[2] for r in name_regions)
            by1 = max(r[1] + r[3] for r in name_regions)
            pil_img = ImageGrab.grab(bbox=(bx0, by0, bx1, by1))
            img_np  = np.array(pil_img)
            sh, sw, sch = img_np.shape
            from PyQt5.QtGui import QImage
            qimg = QImage(img_np.data, sw, sh, sw * sch, QImage.Format_RGB888)
            pix  = QPixmap.fromImage(qimg)
            self._strip_preview_lbl.setSourcePixmap(pix)
            self._strip_preview_lbl.show()
            self._cap_status.setText(
                f"Name strip  {sw}×{sh} px  "
                f"(x {bx0}–{bx1}, y {by0}–{by1})"
            )
        except Exception as e:
            self._cap_status.setText(f"Snap error: {e}")

    def _train_save_strip_only(self):
        """Capture the name-region strip and save it to train/ — no OCR or hashing."""
        try:
            from PIL import ImageGrab
            import cv2 as _cv2
            import numpy as np
            name_regions = DEFAULTS["name_regions"]
            bx0 = min(r[0] for r in name_regions)
            by0 = min(r[1] for r in name_regions)
            bx1 = max(r[0] + r[2] for r in name_regions)
            by1 = max(r[1] + r[3] for r in name_regions)
            pil_img = ImageGrab.grab(bbox=(bx0, by0, bx1, by1))
            img_rgb = np.array(pil_img)
            img_bgr = _cv2.cvtColor(img_rgb, _cv2.COLOR_RGB2BGR)
            idx = save_strip_raw(img_bgr)
            sh, sw = img_bgr.shape[:2]
            # Update preview
            from PyQt5.QtGui import QImage
            qimg = QImage(img_rgb.data, sw, sh, sw * img_rgb.shape[2], QImage.Format_RGB888)
            pix  = QPixmap.fromImage(qimg)
            self._strip_preview_lbl.setSourcePixmap(pix)
            self._strip_preview_lbl.show()
            path = TRAIN_DIR / f"{idx}_image.png"
            self._cap_status.setText(
                f"Saved strip → {path.name}  ({sw}×{sh} px)"
            )
            _line = f"<span style='color:#ffd700;'>[Save Strip] #{idx} saved → {path.name}  ({sw}×{sh})</span>"
            self._train_log.append(_line)
        except Exception as e:
            self._cap_status.setText(f"Save error: {e}")

    def _train_browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Screenshot", "",
            "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if not path:
            return
        self._train_img_path = path
        self._train_file_lbl.setText(path.split("/")[-1])
        self._btn_run_img.setEnabled(True)
        pix = QPixmap(path)
        if not pix.isNull():
            self._train_thumb.setPixmap(
                pix.scaledToHeight(76, Qt.SmoothTransformation)
            )
            self._train_thumb.show()

    def _train_run_file(self):
        if not self._train_img_path:
            return
        if not OCR_AVAILABLE:
            self._train_log.append(
                "<span style='color:#f85149;'>OCR not available.</span>"
            )
            return
        self._train_log.clear()
        results = run_train_on_image(
            self._train_img_path,
            [list(r) for r in DEFAULTS["name_regions"]],
            self._train_thr_sp.value(),
        )
        self._train_append(results, f"File: {self._train_img_path.split('/')[-1]}")
        self._train_refresh_hm()

    def _train_manual_capture(self):
        if not OCR_AVAILABLE:
            self._cap_status.setText("OCR not available.")
            return
        self._btn_manual_cap.setEnabled(False)
        self._cap_status.setText("Capturing…")
        self._cap_thread = _CaptureThread(
            [list(r) for r in DEFAULTS["name_regions"]],
            self._train_thr_sp.value(),
        )
        self._cap_thread.done.connect(self._on_manual_done)
        self._cap_thread.error.connect(
            lambda e: self._cap_status.setText(f"Error: {e}")
        )
        self._cap_thread.finished.connect(
            lambda: self._btn_manual_cap.setEnabled(True)
        )
        self._cap_thread.start()

    def _on_manual_done(self, idx: int, results: list):
        self._cap_status.setText(
            f"#{idx} saved  |  hashmap: {get_hashmap().size} entries"
        )
        self._train_append(results, f"Manual capture #{idx}")
        self._train_refresh_hm()

    def _train_toggle_auto(self, checked: bool):
        if checked:
            self._show_overlay_if_auto()
            self._btn_auto_cap.setText("■  Stop Auto Capture")
            cfg = {
                "capture_interval": self._cap_interval_sp.value(),
                "name_regions":     [list(r) for r in DEFAULTS["name_regions"]],
                "ocr_threshold":    self._train_thr_sp.value(),
            }
            self._auto_capture = AutoCaptureWorker(cfg)
            self._auto_capture.status_signal.connect(self._cap_status.setText)
            self._auto_capture.capture_done.connect(self._on_auto_done)
            self._auto_capture.start()
        else:
            self._btn_auto_cap.setText("▶  Start Auto Capture")
            if self._auto_capture:
                self._auto_capture.stop()
                self._auto_capture = None
            self._cap_status.setText("Idle")

    def _on_auto_done(self, idx: int, results: list):
        self._train_append(results, f"Auto capture #{idx}")
        self._train_refresh_hm()

    def _train_append(self, results: list, header: str = ""):
        if header:
            _hdr = f"<span style='color:#ffd700;font-weight:bold;'>[Train] {header}</span>"
            self._train_log.append(_hdr)
            self._olog(_hdr)
        for r in results:
            src      = r.get("source", "?")
            match    = r["match"] or "—"
            score    = r.get("score", 0.0)
            icon     = "⚡" if src == "hash" else "🔬"
            accepted = r["match"] is not None
            clr      = "#56d364" if accepted else "#f85149"
            t        = f"crop&nbsp;{r['crop_ms']:.0f}+ocr&nbsp;{r['ocr_ms']:.0f}ms"
            h8       = r.get("hash", "")[:8]
            conflict = r.get("hash_conflict")
            # Score colour: green ≥0.90, orange ≥0.70, red <0.70
            if score >= 0.90:
                score_clr = "#56d364"
            elif score >= 0.70:
                score_clr = "#ffa500"
            else:
                score_clr = "#f85149"
            score_tag = (
                f"<span style='color:{score_clr};font-weight:bold;'>{score:.2f}</span>"
            )
            _line = (
                f"<span style='color:{clr};'>{icon} S{r['slot']} → {match}</span>"
                f" score={score_tag}"
                f" <span style='color:#8b949e;font-size:10px;'>"
                f"[{t}] raw={r['raw']!r} h={h8}</span>"
            )
            self._train_log.append(_line)
            self._olog(_line)
            if conflict:
                _conflict_line = f"<span style='color:#ffa500;font-size:10px;'>⚠ {conflict}</span>"
                self._train_log.append(_conflict_line)
                self._olog(_conflict_line)

        # Summary line
        hash_hits   = sum(1 for r in results if r.get("source") == "hash")
        hash_ms     = sum(r.get("crop_ms", 0) for r in results if r.get("source") == "hash")
        ocr_only_ms = sum(
            r.get("crop_ms", 0) + r.get("ocr_ms", 0)
            for r in results if r.get("source") != "hash"
        )
        _summary = (
            f"<span style='color:#444;font-size:10px;'>"
            f"⚡ {hash_hits}/5 {hash_ms:.0f}ms hash  |  "
            f"🔬 {5 - hash_hits}/5 {ocr_only_ms:.0f}ms ocr"
            f"</span>"
        )
        self._train_log.append(_summary)
        self._olog(_summary)
        self._train_log.append("")

    # ─────────────────────────────────────────────────────────
    #  Theme
    # ─────────────────────────────────────────────────────────

    def _apply_theme(self):
        self.setStyleSheet("""
            QMainWindow,QWidget{background:#161b22;color:#c9d1d9;
                font-family:"Segoe UI",Arial,sans-serif;font-size:12px;}
            QGroupBox{border:1px solid #30363d;border-radius:6px;
                margin-top:10px;padding-top:12px;color:#8b949e;font-size:11px;}
            QGroupBox::title{subcontrol-origin:margin;left:10px;color:#c9d1d9;}
            QScrollBar:vertical{background:#1e2530;width:8px;border-radius:4px;}
            QScrollBar::handle:vertical{background:#444c56;border-radius:4px;}
            QScrollBar::add-line:vertical,QScrollBar::sub-line:vertical{height:0;}
            QSpinBox,QDoubleSpinBox{background:#21262d;border:1px solid #30363d;
                border-radius:4px;color:#c9d1d9;padding:2px 4px;}
            QCheckBox{color:#c9d1d9;}
            QCheckBox::indicator{width:14px;height:14px;}
            QLabel{color:#c9d1d9;}
        """)

    def closeEvent(self, event):
        if self._auto_capture:
            self._auto_capture.stop(); self._auto_capture.wait(1000)
        if self._cap_thread and self._cap_thread.isRunning():
            self._cap_thread.wait(2000)
        if self._worker:
            self._worker.stop(); self._worker.wait(2000)
        self._overlay.close()
        event.accept()


# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setApplicationName("TFT Roll Tool")
    win = TFTRollTool()
    win.show()
    sys.exit(app.exec_())
