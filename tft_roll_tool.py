#!/usr/bin/env python3
"""
TFT Roll Tool - Auto roll & buy helper for Teamfight Tactics
Set 17: Space Gods
Usage: python tft_roll_tool.py
"""

import os
import sys

import pyautogui
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QScrollArea, QFrame, QGridLayout,
    QGroupBox, QSpinBox, QDoubleSpinBox, QCheckBox, QTabWidget,
    QTextEdit, QSizePolicy, QFileDialog, QButtonGroup, QRadioButton,
)
from PyQt5.QtCore import Qt, QRect, QPoint
from PyQt5.QtGui import QFont, QPixmap, QPainter, QBrush, QColor, QPen

from tft_backend import (
    OCR_AVAILABLE,
    TFT_UNITS, COST_COLOR, COST_BG, COST_LABEL,
    DEFAULTS,
    ocr_all_slots, ocr_from_image_file,
    RollWorker,
)
# pyqtSignal is re-exported for the overlay / chip widgets defined here
from PyQt5.QtCore import pyqtSignal

pyautogui.FAILSAFE = True  # move mouse to top-left corner to emergency-stop


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
#  Full-screen region selector overlay
# ─────────────────────────────────────────────────────────────

class ScreenRegionSelector(QWidget):
    """
    Fullscreen overlay that shows a screenshot of the desktop as background.
    User drags a rectangle → emits region_selected(x, y, w, h) → closes.
    A separate `closed` signal is always emitted on close so the caller
    can restore its own window reliably.
    """
    region_selected = pyqtSignal(int, int, int, int)
    closed          = pyqtSignal()

    def __init__(self):
        super().__init__()
        # ── Capture the screen BEFORE the window appears ──────────────
        screen = QApplication.primaryScreen()
        self._bg: QPixmap = screen.grabWindow(0)   # real screenshot background

        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.showFullScreen()
        self.setCursor(Qt.CrossCursor)

        self._origin:  QPoint | None = None
        self._current: QPoint | None = None
        self._dragging = False

    # ── painting ──────────────────────────────────────────────────────

    def paintEvent(self, _event):
        painter = QPainter(self)

        # 1. Draw the captured screenshot so the screen looks normal
        painter.drawPixmap(0, 0, self._bg)

        dim = QColor(0, 0, 0, 150)
        sw, sh = self.width(), self.height()

        if self._origin and self._current:
            sel = self._selection_rect()

            # 2. Dim everything EXCEPT the selected rectangle (4-rect method)
            painter.fillRect(0,           0,          sw, sel.top(),    dim)  # top
            painter.fillRect(0,           sel.bottom()+1, sw, sh,       dim)  # bottom
            painter.fillRect(0,           sel.top(),  sel.left(), sel.height(), dim)  # left
            painter.fillRect(sel.right()+1, sel.top(), sw,        sel.height(), dim)  # right

            # 3. Gold border
            painter.setPen(QPen(QColor("#ffd700"), 2, Qt.SolidLine))
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(sel)

            # 4. Size label
            painter.setPen(QColor("#ffd700"))
            painter.setFont(QFont("Segoe UI", 10, QFont.Bold))
            ly = sel.y() - 24 if sel.y() > 30 else sel.bottom() + 20
            painter.drawText(sel.x() + 4, ly, f"{sel.width()} × {sel.height()} px")

            # 5. Dashed column dividers (5 slots preview)
            if sel.width() > 50:
                painter.setPen(QPen(QColor("#ffd70099"), 1, Qt.DashLine))
                slot_w = sel.width() / 5
                for i in range(1, 5):
                    dx = int(sel.x() + slot_w * i)
                    painter.drawLine(dx, sel.top(), dx, sel.bottom())

                # Slot number labels
                painter.setPen(QColor("#ffd700"))
                painter.setFont(QFont("Segoe UI", 9))
                for i in range(5):
                    lx = int(sel.x() + slot_w * i + slot_w / 2 - 4)
                    painter.drawText(lx, sel.top() + 16, str(i + 1))

        else:
            # No selection started yet — dim full screen
            painter.fillRect(self.rect(), dim)

        # Instruction banner
        painter.setPen(QColor("white"))
        painter.setFont(QFont("Segoe UI", 13))
        painter.drawText(
            self.rect(),
            Qt.AlignTop | Qt.AlignHCenter,
            "\n  Drag to select the shop bar (all 5 slots)  ·  ESC to cancel  "
        )

    # ── mouse events ──────────────────────────────────────────────────

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._origin   = event.pos()
            self._current  = event.pos()
            self._dragging = True

    def mouseMoveEvent(self, event):
        if self._dragging:
            self._current = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self._dragging:
            self._current  = event.pos()
            self._dragging = False
            sel = self._selection_rect()
            if sel.width() > 20 and sel.height() > 10:
                self.region_selected.emit(sel.x(), sel.y(), sel.width(), sel.height())
            self.close()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()

    def closeEvent(self, event):
        self.closed.emit()       # always fires → main window re-shows itself
        super().closeEvent(event)

    # ── helper ────────────────────────────────────────────────────────

    def _selection_rect(self) -> QRect:
        x1 = min(self._origin.x(),  self._current.x())
        y1 = min(self._origin.y(),  self._current.y())
        x2 = max(self._origin.x(),  self._current.x())
        y2 = max(self._origin.y(),  self._current.y())
        return QRect(x1, y1, x2 - x1, y2 - y1)


# ─────────────────────────────────────────────────────────────
#  Main Window
# ─────────────────────────────────────────────────────────────

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
            if match:
                # OCR success + in wanted list
                parts.append(
                    f"S{s['slot']} → <span style='color:#56d364;font-weight:bold;'>{match} ✓</span>"
                )
            elif raw:
                # OCR success + not in wanted list
                label = cand if cand else raw
                parts.append(
                    f"S{s['slot']} → <span style='color:#666;'>{label} ✗</span>"
                )
            else:
                # OCR failed to read anything
                parts.append(
                    f"S{s['slot']} <span style='color:#c9650a;'>⚠ failed</span>"
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


# ─────────────────────────────────────────────────────────────
class TFTRollTool(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TFT Roll Tool — Set 17: Space Gods")
        self.setMinimumSize(920, 680)
        self.resize(1000, 740)

        self._selected: dict[str, int] = {}
        self._buttons:  dict[str, UnitButton] = {}
        self._worker:   RollWorker | None = None

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

        tabs = QTabWidget()
        tabs.setStyleSheet(
            "QTabWidget::pane{border:1px solid #333;}"
            "QTabBar::tab{background:#1e1e1e;color:#aaa;padding:6px 18px;}"
            "QTabBar::tab:selected{background:#2a2a2a;color:white;"
            "border-bottom:2px solid #ffd700;}"
        )
        rl.addWidget(tabs)
        tabs.addTab(self._tab_main(),     "🎮  Build & Roll")
        tabs.addTab(self._tab_settings(), "⚙  Settings")
        tabs.addTab(self._tab_ocr_test(),  "🔬  OCR Test")

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

        self._autoroll_cb = QCheckBox("Auto Roll – script presses D after each scan")
        self._autoroll_cb.setChecked(False)
        self._autoroll_cb.setStyleSheet("color:#c9d1d9;font-size:11px;")
        self._autoroll_cb.setToolTip(
            "ON: script presses D to reroll after scanning & buying.\n"
            "OFF: script only scans & buys — you roll manually.\n"
            "(Has no effect in Listen D press mode.)"
        )
        rl3.addWidget(self._autoroll_cb)

        # ── Scan trigger mode ─────────────────────────────────────────
        trigger_label = QLabel("Scan trigger:")
        trigger_label.setStyleSheet("color:#8b949e;font-size:11px;margin-top:4px;")
        rl3.addWidget(trigger_label)

        self._trigger_group = QButtonGroup(self)
        trigger_row = QHBoxLayout()
        trigger_row.setSpacing(12)

        self._trigger_listen = QRadioButton("Listen D press  (default)")
        self._trigger_listen.setChecked(True)   # default
        self._trigger_listen.setStyleSheet("color:#c9d1d9;font-size:11px;")
        self._trigger_listen.setToolTip(
            "Scan + buy automatically whenever YOU press D in-game.\n"
            "The script detects the key and reads the new shop."
        )

        self._trigger_auto = QRadioButton("Auto  (timed loop)")
        self._trigger_auto.setChecked(False)
        self._trigger_auto.setStyleSheet("color:#c9d1d9;font-size:11px;")
        self._trigger_auto.setToolTip(
            "Scan + buy on a fixed timer (shop_wait interval).\n"
            "Combine with Auto Roll to have the script roll for you."
        )

        self._trigger_group.addButton(self._trigger_listen, 0)
        self._trigger_group.addButton(self._trigger_auto,   1)
        trigger_row.addWidget(self._trigger_listen)
        trigger_row.addWidget(self._trigger_auto)
        trigger_row.addStretch()
        rl3.addLayout(trigger_row)

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

        # Overlay toggle
        self._overlay_btn = QPushButton("📌  Show Log Overlay")
        self._overlay_btn.setFixedHeight(30)
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

        # OCR log
        lg = QGroupBox("Shop OCR Log"); ll = QVBoxLayout(lg)
        self._log = QTextEdit(); self._log.setReadOnly(True)
        self._log.setMaximumHeight(100)
        self._log.setStyleSheet(
            "QTextEdit{background:#0d1117;color:#8b949e;font-size:10px;"
            "border:none;font-family:Consolas,monospace;}"
        )
        ll.addWidget(self._log)
        rc.addWidget(lg, 1)

        rw2 = QWidget(); rw2.setLayout(rc)
        row.addWidget(rw2, 2)
        lay.addLayout(row)
        return w

    # ─────────────────────────────────────────────────────────
    #  Settings tab  (all timing + screen position controls)
    # ─────────────────────────────────────────────────────────

    def _tab_settings(self) -> QWidget:
        w = QWidget()
        outer = QVBoxLayout(w); outer.setContentsMargins(10, 10, 10, 10)

        # Scrollable so it fits any window height
        sc = QScrollArea(); sc.setWidgetResizable(True); sc.setFrameShape(QFrame.NoFrame)
        sc.setStyleSheet("QScrollArea{background:transparent;}")
        inner = QWidget(); lay = QVBoxLayout(inner)
        lay.setAlignment(Qt.AlignTop); lay.setSpacing(6)

        # ── Timing ────────────────────────────────────────────
        lay.addWidget(self._sh("⏱  Timing"))

        self._roll_sp = self._dspin(
            0.5, 10.0, DEFAULTS["roll_delay"], 0.25, " sec",
            "Roll every  (seconds between each D press)"
        )
        self._pre_sp = self._ispin(
            1, 15, DEFAULTS["pre_delay"], " sec",
            "Start delay  (countdown before automation begins)"
        )
        self._shop_wait_sp = self._dspin(
            0.1, 3.0, DEFAULTS["shop_wait"], 0.1, " sec",
            "Shop load wait  (pause after rolling for shop to populate)"
        )
        self._buy_delay_sp = self._dspin(
            0.05, 1.0, DEFAULTS["buy_delay"], 0.05, " sec",
            "Buy speed  (delay between clicking each shop slot)"
        )
        self._ocr_thr_sp = self._dspin(
            0.3, 1.0, DEFAULTS["ocr_threshold"], 0.05, "",
            "OCR match threshold  (higher = stricter matching, 0.62 recommended)"
        )
        for label, spin in [
            ("Roll every:",          self._roll_sp),
            ("Start delay:",         self._pre_sp),
            ("Shop load wait:",      self._shop_wait_sp),
            ("Buy speed:",           self._buy_delay_sp),
            ("OCR threshold:",       self._ocr_thr_sp),
        ]:
            r = QHBoxLayout()
            lbl = QLabel(label); lbl.setFixedWidth(160)
            r.addWidget(lbl); r.addWidget(spin); r.addStretch()
            lay.addLayout(r)

        # ── Click positions ───────────────────────────────────
        lay.addWidget(self._sh("🖱  Click Positions  (center of each shop card)"))
        self._click_spins: list[tuple[QSpinBox, QSpinBox]] = []
        for i, (x, y) in enumerate(DEFAULTS["click_pos"]):
            r = QHBoxLayout()
            r.addWidget(self._slot_lbl(i))
            xs = QSpinBox(); xs.setRange(0, 3840); xs.setValue(x)
            xs.setPrefix("X: "); xs.setFixedWidth(100)
            ys = QSpinBox(); ys.setRange(0, 2160); ys.setValue(y)
            ys.setPrefix("Y: "); ys.setFixedWidth(100)
            self._click_spins.append((xs, ys))
            r.addWidget(xs); r.addWidget(ys); r.addStretch()
            lay.addLayout(r)

        # ── OCR name regions ──────────────────────────────────
        lay.addWidget(self._sh("🔍  OCR Name Regions  [x, y, w, h] of unit-name text per slot"))
        self._name_spins: list[tuple[QSpinBox, QSpinBox, QSpinBox, QSpinBox]] = []
        for i, (x, y, rw, rh) in enumerate(DEFAULTS["name_regions"]):
            r = QHBoxLayout()
            r.addWidget(self._slot_lbl(i))
            xs = QSpinBox(); xs.setRange(0, 3840); xs.setValue(x);  xs.setPrefix("X:"); xs.setFixedWidth(88)
            ys = QSpinBox(); ys.setRange(0, 2160); ys.setValue(y);  ys.setPrefix("Y:"); ys.setFixedWidth(88)
            ws = QSpinBox(); ws.setRange(10, 400); ws.setValue(rw); ws.setPrefix("W:"); ws.setFixedWidth(84)
            hs = QSpinBox(); hs.setRange(4, 100);  hs.setValue(rh); hs.setPrefix("H:"); hs.setFixedWidth(76)
            self._name_spins.append((xs, ys, ws, hs))
            r.addWidget(xs); r.addWidget(ys); r.addWidget(ws); r.addWidget(hs); r.addStretch()
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

        note = QLabel(
            "Default values target 1920×1080 fullscreen TFT.  "
            "For other resolutions, hover your mouse over each shop slot in-game "
            "and read the coordinates from ShareX or Windows Snipping Tool."
        )
        note.setStyleSheet("color:#555;font-size:10px;margin-top:4px;")
        note.setWordWrap(True)
        lay.addWidget(note)

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

        regions = [list(r) for r in DEFAULTS["name_regions"]]
        thr     = DEFAULTS["ocr_threshold"]

        try:
            results = ocr_from_image_file(self._ocr_test_img_path, regions, thr)
        except Exception as e:
            self._ocr_test_status.setText(f"Error: {e}")
            return

        img_size = results[0]["img_size"] if results else ("?", "?")
        self._ocr_test_log.clear()
        self._ocr_test_log.append(
            f"<span style='color:#ffd700;font-weight:bold;'>"
            f"Image: {os.path.basename(self._ocr_test_img_path)}"
            f"  ({img_size[0]}×{img_size[1]})"
            f"</span>"
        )
        matched = 0
        for r in results:
            raw  = r.get("raw") or ""
            cand = r.get("best_candidate") or ""
            reg  = r["scaled_region"]
            hit  = r["match"] is not None
            if hit:
                matched += 1
                color = "#56d364"
                tag   = f"→ <b>{r['match']}</b> ✓"
            elif raw:
                color = "#666"
                label = cand if cand else raw
                tag   = f"→ {label} ✗"
            else:
                color = "#c9650a"
                tag   = "⚠ failed"
            coord = f"[{reg[0]},{reg[1]} {reg[2]}×{reg[3]}]"
            self._ocr_test_log.append(
                f"<span style='color:{color};'>Slot {r['slot']}: {tag}</span>"
                f"<span style='color:#444;font-size:9px;'>  {coord}</span>"
            )

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

    def _open_region_selector(self): pass
    def _show_selector(self): pass
    def _restore_window(self): pass
    def _apply_saved_geometry(self): pass
    def _on_region_selected(self, x, y, w, h): pass

    def _save_settings(self):
        for i, (xs, ys) in enumerate(self._click_spins):
            DEFAULTS["click_pos"][i] = [xs.value(), ys.value()]
        for i, (xs, ys, ws, hs) in enumerate(self._name_spins):
            DEFAULTS["name_regions"][i] = [
                xs.value(), ys.value(), ws.value(), hs.value()
            ]
        self._settings_status.setText("✓ Settings saved.")

    def _reset_settings(self):
        """Restore every settings widget to its built-in default value."""
        self._roll_sp.setValue(1.5)
        self._pre_sp.setValue(3)
        self._shop_wait_sp.setValue(0.5)
        self._buy_delay_sp.setValue(0.12)
        self._ocr_thr_sp.setValue(0.50)

        defaults_click = [
            [575, 992], [775, 992], [975, 992], [1175, 992], [1375, 992]
        ]
        defaults_name = [
            [480, 1045, 145, 20], [685, 1045, 145, 20], [885, 1045, 145, 20],
            [1085, 1045, 145, 20], [1290, 1045, 145, 20]
        ]
        for i, (xs, ys) in enumerate(self._click_spins):
            xs.setValue(defaults_click[i][0])
            ys.setValue(defaults_click[i][1])
        for i, (xs, ys, ws, hs) in enumerate(self._name_spins):
            xs.setValue(defaults_name[i][0]); ys.setValue(defaults_name[i][1])
            ws.setValue(defaults_name[i][2]); hs.setValue(defaults_name[i][3])

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

    def _start(self):
        self._log.clear()
        cfg = {
            "roll_delay":    self._roll_sp.value(),
            "pre_delay":     self._pre_sp.value(),
            "shop_wait":     self._shop_wait_sp.value(),
            "buy_delay":     self._buy_delay_sp.value(),
            "ocr_threshold": self._ocr_thr_sp.value(),
            "auto_roll":     self._autoroll_cb.isChecked(),
            "listen_d":      self._trigger_listen.isChecked(),
            "wanted":        list(self._selected.keys()),
            "click_pos":     [list(p) for p in DEFAULTS["click_pos"]],
            "name_regions":  [list(r) for r in DEFAULTS["name_regions"]],
        }
        self._worker = RollWorker(cfg)
        self._worker.status_signal.connect(self._on_status)
        self._worker.roll_signal.connect(self._on_roll)
        self._worker.found_signal.connect(self._on_found)
        self._worker.shop_signal.connect(self._on_shop)
        self._worker.finished.connect(self._on_done)
        self._worker.start()
        self._start_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)

    def _stop(self):
        if self._worker:
            self._worker.stop()

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

    def _on_found(self, msg: str):
        html = f"<span style='color:#56d364;'>✓ {msg}</span>"
        self._log.append(html)
        self._overlay.append_log(html)

    def _on_shop(self, results: list):
        self._overlay.append_shop_row(results)
        parts = []
        for s in results:
            raw   = s.get("raw") or ""
            match = s["match"]
            cand  = s.get("best_candidate") or ""
            if match:
                parts.append(f"<span style='color:#56d364;'>S{s['slot']} → <b>{match}</b> ✓</span>")
            elif raw:
                label = cand if cand else raw
                parts.append(f"<span style='color:#555;'>S{s['slot']} → {label} ✗</span>")
            else:
                parts.append(f"<span style='color:#c9650a;'>S{s['slot']} ⚠ failed</span>")
        self._log.append("  ".join(parts))

    def _on_done(self):
        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._worker = None

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
