import sys
import time
from collections import deque

import cv2
import mss
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QMutex, QMutexLocker
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QLabel,
                             QComboBox, QSlider, QPushButton, QFormLayout,
                             QLineEdit, QMessageBox, QHBoxLayout)

# This script now requires pygetwindow
# Install it using: pip install pygetwindow
try:
    import pygetwindow as gw
except ImportError:
    print("Error: pygetwindow library not found.")
    print("Please install it using: pip install pygetwindow")
    sys.exit(1)

# This script requires the pynput library for global hotkey support.
# Install it using: pip install pynput
try:
    from pynput import keyboard
except ImportError:
    print("Error: pynput library not found.")
    print("Please install it using: pip install pynput")
    sys.exit(1)


class HotkeyListenerThread(QThread):
    hotkey_pressed = pyqtSignal()
    def __init__(self, hotkey_str, parent=None):
        super().__init__(parent)
        self.hotkey_str, self.listener = hotkey_str, None
    def run(self):
        try:
            hotkey = keyboard.GlobalHotKeys({self.hotkey_str: self.hotkey_pressed.emit})
            self.listener = hotkey
            self.listener.start()
            self.listener.join()
        except Exception as e:
            print(f"Failed to set hotkey '{self.hotkey_str}': {e}")
    def stop(self):
        if self.listener: self.listener.stop()

class ScreenCaptureThread(QThread):
    new_frame = pyqtSignal(QImage)
    request_hide = pyqtSignal()
    request_show = pyqtSignal()
    window_geometry_changed = pyqtSignal(int, int, int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.running, self.paused = False, True
        self._window_title = ""
        self._fps, self._history_length, self._sensitivity = 30, 3, 20
        self.mutex = QMutex()
        self.last_geom = (0,0,0,0)

    def run(self):
        self.running = True
        frame_history = deque(maxlen=self._history_length)
        with mss.mss() as sct:
            while self.running:
                with QMutexLocker(self.mutex):
                    is_paused, window_title = self.paused, self._window_title
                    history_length, target_fps, sensitivity = self._history_length, self._fps, self._sensitivity
                if is_paused or not window_title:
                    time.sleep(0.1); continue

                frame_time_start = time.time()
                try:
                    target_windows = gw.getWindowsWithTitle(window_title)
                    if not target_windows:
                        print(f"Window '{window_title}' not found. Pausing."); self.pause(); continue
                    win = target_windows[0]
                    
                    current_geom = (win.left, win.top, win.width, win.height)
                    if current_geom != self.last_geom:
                        self.window_geometry_changed.emit(*current_geom)
                        self.last_geom = current_geom

                    capture_area = {'top': win.top, 'left': win.left, 'width': win.width, 'height': win.height}
                    self.request_hide.emit(); time.sleep(0.001)
                    sct_img = sct.grab(capture_area)
                    self.request_show.emit()
                    frame = np.array(sct_img)
                except Exception as e:
                    print(f"Capture error: {e}"); self.request_show.emit(); continue

                if frame_history.maxlen != history_length:
                    frame_history = deque(list(frame_history), maxlen=history_length)
                
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
                motion_output = np.zeros_like(gray_frame)
                if len(frame_history) > 0:
                    for old_frame in frame_history:
                        diff = cv2.absdiff(gray_frame, old_frame)
                        _, mask = cv2.threshold(diff, sensitivity, 255, cv2.THRESH_BINARY)
                        motion_output = cv2.bitwise_or(motion_output, mask)
                frame_history.append(gray_frame)
                
                h, w = motion_output.shape
                qt_image = QImage(motion_output.data, w, h, w, QImage.Format.Format_Grayscale8)
                self.new_frame.emit(qt_image)
                
                elapsed_time = time.time() - frame_time_start
                sleep_time = (1.0 / target_fps) - elapsed_time
                if sleep_time > 0: time.sleep(sleep_time)

    def stop(self):
        self.resume(); self.running = False; self.wait()
    def pause(self):
        with QMutexLocker(self.mutex): self.paused = True
    def resume(self):
        with QMutexLocker(self.mutex): self.paused = False
    def update_window(self, title):
        with QMutexLocker(self.mutex): self._window_title = title
    def update_fps(self, fps):
        with QMutexLocker(self.mutex): self._fps = fps
    def update_history_length(self, length):
        with QMutexLocker(self.mutex): self._history_length = length
    def update_sensitivity(self, value):
        with QMutexLocker(self.mutex): self._sensitivity = value

class OverlayWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool | Qt.WindowType.WindowTransparentForInput)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.layout = QVBoxLayout(self); self.layout.setContentsMargins(0,0,0,0)
        self.label = QLabel(); self.layout.addWidget(self.label); self.setLayout(self.layout)
    def set_image(self, q_image):
        pixmap = QPixmap.fromImage(q_image)
        self.label.setPixmap(pixmap)

class ControlPanel(QWidget):
    def __init__(self):
        super().__init__(); self.setWindowTitle("Motion Overlay Controls")
        self.overlay_window = OverlayWindow(); self.capture_thread = ScreenCaptureThread()
        self.hotkey_thread = None
        
        # --- GUI Widgets ---
        self.window_combo = QComboBox()
        self.refresh_button = QPushButton("Refresh List")
        self.history_slider = QSlider(Qt.Orientation.Horizontal); self.history_slider.setRange(1, 15); self.history_slider.setValue(3)
        self.history_label = QLabel()
        self.fps_slider = QSlider(Qt.Orientation.Horizontal); self.fps_slider.setRange(5, 60); self.fps_slider.setValue(30)
        self.fps_label = QLabel()
        self.sensitivity_slider = QSlider(Qt.Orientation.Horizontal); self.sensitivity_slider.setRange(5, 100); self.sensitivity_slider.setValue(20)
        self.sensitivity_label = QLabel()
        self.hotkey_input = QLineEdit("<ctrl>+<alt>+h")
        self.apply_hotkey_button = QPushButton("Apply Hotkey")
        self.start_button, self.show_button, self.hide_button = QPushButton("Start"), QPushButton("Show"), QPushButton("Hide")
        self.show_button.setEnabled(False); self.hide_button.setEnabled(False)

        # --- Layout ---
        layout = QFormLayout(self); window_layout = QHBoxLayout()
        window_layout.addWidget(self.window_combo); window_layout.addWidget(self.refresh_button)
        layout.addRow(QLabel("Select Window:"), window_layout)
        layout.addRow(self.history_label, self.history_slider); layout.addRow(self.fps_label, self.fps_slider)
        layout.addRow(self.sensitivity_label, self.sensitivity_slider); hotkey_layout = QVBoxLayout()
        hotkey_layout.addWidget(self.hotkey_input); hotkey_layout.addWidget(self.apply_hotkey_button)
        layout.addRow(QLabel("Hotkey:"), hotkey_layout); button_layout = QHBoxLayout()
        button_layout.addWidget(self.start_button); button_layout.addWidget(self.show_button); button_layout.addWidget(self.hide_button)
        layout.addRow(button_layout); self.setLayout(layout)

        # --- Connections ---
        self.refresh_button.clicked.connect(self.populate_windows)
        self.start_button.clicked.connect(self.start_capture)
        self.show_button.clicked.connect(self.show_overlay)
        self.hide_button.clicked.connect(self.hide_overlay)
        self.apply_hotkey_button.clicked.connect(self.update_hotkey)
        self.fps_slider.valueChanged.connect(self.on_fps_change)
        self.history_slider.valueChanged.connect(self.on_history_change)
        self.sensitivity_slider.valueChanged.connect(self.on_sensitivity_change)
        self.capture_thread.new_frame.connect(self.overlay_window.set_image)
        self.capture_thread.request_hide.connect(self.overlay_window.hide)
        self.capture_thread.request_show.connect(self.overlay_window.show)
        self.capture_thread.window_geometry_changed.connect(self.overlay_window.setGeometry)
        
        self.populate_windows(); self.on_fps_change(); self.on_history_change(); self.on_sensitivity_change()

    def populate_windows(self):
        self.window_combo.clear()
        for win in gw.getAllWindows():
            if win.title: self.window_combo.addItem(win.title)

    def start_capture(self):
        win_title = self.window_combo.currentText()
        if not win_title: QMessageBox.critical(self, "Error", "No window selected."); return
        self.capture_thread.update_window(win_title)
        self.capture_thread.start(); self.update_hotkey()
        self.start_button.setEnabled(False); self.show_button.setEnabled(True); self.hide_button.setEnabled(True)
        self.show_overlay()

    def stop_capture(self):
        self.capture_thread.stop()
        if self.hotkey_thread: self.hotkey_thread.stop()
        self.overlay_window.hide()

    def show_overlay(self): self.capture_thread.resume()
    def hide_overlay(self): self.capture_thread.pause()
    def toggle_overlay_visibility(self):
        if not self.capture_thread.paused: self.hide_overlay()
        else: self.show_overlay()

    def update_hotkey(self):
        if not self.capture_thread.isRunning(): return
        if self.hotkey_thread and self.hotkey_thread.isRunning(): self.hotkey_thread.stop()
        hotkey_str = self.hotkey_input.text()
        self.hotkey_thread = HotkeyListenerThread(hotkey_str, self)
        self.hotkey_thread.hotkey_pressed.connect(self.toggle_overlay_visibility)
        self.hotkey_thread.start()

    def on_fps_change(self):
        val = self.fps_slider.value(); self.fps_label.setText(f"Target FPS: {val}"); self.capture_thread.update_fps(val)
    def on_history_change(self):
        val = self.history_slider.value(); self.history_label.setText(f"Frames to Diff: {val}"); self.capture_thread.update_history_length(val)
    def on_sensitivity_change(self):
        val = self.sensitivity_slider.value(); self.sensitivity_label.setText(f"Sensitivity: {val}"); self.capture_thread.update_sensitivity(val)
    def closeEvent(self, event): self.stop_capture(); super().closeEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv); panel = ControlPanel(); panel.show(); sys.exit(app.exec())
