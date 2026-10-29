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
                             QLineEdit, QMessageBox, QHBoxLayout, QCheckBox)

# This script requires the pynput library for global hotkey support.
# Install it using: pip install pynput
try:
    from pynput import keyboard
except ImportError:
    print("Error: pynput library not found.")
    print("Please install it using: pip install pynput")
    sys.exit(1)


class HotkeyListenerThread(QThread):
    """A thread to listen for global hotkeys without blocking the GUI."""
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
    """This thread handles the continuous screen capturing and processing."""
    new_frame = pyqtSignal(QImage)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.running, self.paused = False, True
        self._monitor_number = 1
        self._fps, self._history_length, self._sensitivity = 30, 3, 20
        self._debug_mode = False 
        self.mutex = QMutex()
        self.last_size = (0,0)

    def run(self):
        self.running = True
        frame_history = deque(maxlen=self._history_length)

        with mss.mss() as sct:
            while self.running:
                with QMutexLocker(self.mutex):
                    is_paused, monitor_number = self.paused, self._monitor_number
                    history_length, target_fps, sensitivity = self._history_length, self._fps, self._sensitivity
                    debug_mode = self._debug_mode
                
                if is_paused:
                    time.sleep(0.1); continue

                frame_time_start = time.time()
                try:
                    monitor = sct.monitors[monitor_number]
                    
                    if (monitor['width'], monitor['height']) != self.last_size:
                        self.last_size = (monitor['width'], monitor['height'])
                        frame_history.clear()

                    sct_img = sct.grab(monitor)
                    frame = np.array(sct_img)
                except IndexError:
                    print(f"Error: Monitor {monitor_number} not found. Pausing."); self.pause(); continue
                except Exception as e:
                    print(f"Capture error: {e}"); continue

                if frame_history.maxlen != history_length:
                    frame_history = deque(list(frame_history), maxlen=history_length)
                
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)

                if debug_mode:
                    cv2.imshow("Debug - Raw Frame", gray_frame)
                
                cv2.waitKey(1)

                motion_output_float = np.zeros_like(gray_frame, dtype=np.float32)
                if len(frame_history) > 0:
                    history_len = len(frame_history)
                    for i, old_frame in enumerate(frame_history):
                        diff = cv2.absdiff(gray_frame, old_frame)
                        weight = (i + 1.0) / history_len
                        motion_output_float += diff.astype(np.float32) * weight
                
                motion_output = np.clip(motion_output_float, 0, 255).astype(np.uint8)
                
                # --- FIX: Use THRESH_TOZERO to preserve grayscale values ---
                # This sets pixels below the sensitivity to 0 (black), but leaves others untouched.
                _, motion_output = cv2.threshold(motion_output, sensitivity, 255, cv2.THRESH_TOZERO)
                
                frame_history.append(gray_frame)
                
                h, w = motion_output.shape
                qt_image = QImage(motion_output.data, w, h, w, QImage.Format.Format_Grayscale8)
                self.new_frame.emit(qt_image)
                
                elapsed_time = time.time() - frame_time_start
                sleep_time = (1.0 / target_fps) - elapsed_time
                if sleep_time > 0: time.sleep(sleep_time)
        
        cv2.destroyAllWindows()

    def stop(self):
        self.resume(); self.running = False; self.wait()
    def pause(self):
        with QMutexLocker(self.mutex): self.paused = True
    def resume(self):
        with QMutexLocker(self.mutex): self.paused = False
    def update_monitor(self, monitor_num):
        with QMutexLocker(self.mutex): self._monitor_number = monitor_num
    def update_fps(self, fps):
        with QMutexLocker(self.mutex): self._fps = fps
    def update_history_length(self, length):
        with QMutexLocker(self.mutex): self._history_length = length
    def update_sensitivity(self, value):
        with QMutexLocker(self.mutex): self._sensitivity = value
    def update_debug_mode(self, enabled):
        with QMutexLocker(self.mutex):
            self._debug_mode = enabled
            if not enabled:
                cv2.destroyWindow("Debug - Raw Frame"); cv2.waitKey(1)

class DisplayWindow(QWidget):
    """A standard window to display the video feed."""
    window_closed_by_user = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Motion Effect Output")
        self.layout = QVBoxLayout(self)
        self.label = QLabel("Waiting for feed...")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setScaledContents(True)
        self.layout.addWidget(self.label)
        self.setLayout(self.layout)
        self.resize(800, 600)

    def set_image(self, q_image):
        self.label.setPixmap(QPixmap.fromImage(q_image))

    def closeEvent(self, event):
        self.hide()
        self.window_closed_by_user.emit()
        event.ignore()

class ControlPanel(QWidget):
    def __init__(self):
        super().__init__(); self.setWindowTitle("Motion Effect Controls")
        self.display_window = DisplayWindow()
        self.capture_thread = ScreenCaptureThread()
        self.hotkey_thread = None
        
        self.monitor_combo = QComboBox()
        self.history_slider = QSlider(Qt.Orientation.Horizontal); self.history_slider.setRange(1, 15); self.history_slider.setValue(3)
        self.history_label = QLabel()
        self.fps_slider = QSlider(Qt.Orientation.Horizontal); self.fps_slider.setRange(5, 60); self.fps_slider.setValue(30)
        self.fps_label = QLabel()
        self.sensitivity_slider = QSlider(Qt.Orientation.Horizontal); self.sensitivity_slider.setRange(5, 100); self.sensitivity_slider.setValue(20)
        self.sensitivity_label = QLabel()
        self.hotkey_input = QLineEdit("<ctrl>+<alt>+h")
        self.apply_hotkey_button = QPushButton("Apply Hotkey")
        self.show_button, self.hide_button = QPushButton("Show Output"), QPushButton("Hide Output")
        self.hide_button.setEnabled(False) 
        self.debug_checkbox = QCheckBox("Show Raw Frame for Debugging")

        layout = QFormLayout(self)
        layout.addRow(QLabel("Select Monitor:"), self.monitor_combo)
        layout.addRow(self.history_label, self.history_slider); layout.addRow(self.fps_label, self.fps_slider)
        layout.addRow(self.sensitivity_label, self.sensitivity_slider)
        layout.addRow(self.debug_checkbox)
        hotkey_layout = QVBoxLayout(); hotkey_layout.addWidget(self.hotkey_input); hotkey_layout.addWidget(self.apply_hotkey_button)
        layout.addRow(QLabel("Hotkey:"), hotkey_layout); button_layout = QHBoxLayout()
        button_layout.addWidget(self.show_button); button_layout.addWidget(self.hide_button)
        layout.addRow(button_layout); self.setLayout(layout)

        self.show_button.clicked.connect(self.show_output_window)
        self.hide_button.clicked.connect(self.hide_output_window)
        self.apply_hotkey_button.clicked.connect(self.update_hotkey)
        self.monitor_combo.currentIndexChanged.connect(self.on_monitor_change)
        self.debug_checkbox.stateChanged.connect(self.on_debug_toggle)
        self.fps_slider.valueChanged.connect(self.on_fps_change)
        self.history_slider.valueChanged.connect(self.on_history_change)
        self.sensitivity_slider.valueChanged.connect(self.on_sensitivity_change)
        self.capture_thread.new_frame.connect(self.display_window.set_image)
        self.display_window.window_closed_by_user.connect(self.hide_output_window)
        
        self.populate_monitors(); self.on_fps_change(); self.on_history_change(); self.on_sensitivity_change()

    def populate_monitors(self):
        self.monitor_combo.clear()
        with mss.mss() as sct:
            for i, monitor in enumerate(sct.monitors):
                # Skip "All in one" monitor option
                if i == 0: continue
                self.monitor_combo.addItem(f"Monitor {i}: {monitor['width']}x{monitor['height']}", i)

    def stop_all_threads(self):
        self.capture_thread.stop()
        if self.hotkey_thread: self.hotkey_thread.stop()
        self.display_window.close()

    def show_output_window(self):
        monitor_number = self.monitor_combo.currentData()
        if monitor_number is None: 
            QMessageBox.critical(self, "Error", "No monitor selected."); return
        self.capture_thread.update_monitor(monitor_number)

        # Resize display window to match monitor resolution
        with mss.mss() as sct:
            try:
                monitor_geom = sct.monitors[monitor_number]
                self.display_window.resize(monitor_geom['width'], monitor_geom['height'])
            except IndexError:
                QMessageBox.critical(self, "Error", "Selected monitor not found.")
                return

        if not self.capture_thread.isRunning():
            self.capture_thread.start()
            self.update_hotkey()

        self.display_window.show()
        self.capture_thread.resume()
        self.show_button.setEnabled(False)
        self.hide_button.setEnabled(True)

    def hide_output_window(self):
        self.display_window.hide()
        self.capture_thread.pause()
        self.show_button.setEnabled(True)
        self.hide_button.setEnabled(False)
        
    def toggle_output_visibility(self):
        if self.display_window.isVisible():
            self.hide_output_window()
        else:
            self.show_output_window()
            
    def on_debug_toggle(self, state):
        is_checked = (state == Qt.CheckState.Checked.value)
        self.capture_thread.update_debug_mode(is_checked)

    def on_monitor_change(self):
        monitor_number = self.monitor_combo.currentData()
        if monitor_number is not None:
            self.capture_thread.update_monitor(monitor_number)

    def update_hotkey(self):
        if not self.capture_thread.isRunning(): return
        if self.hotkey_thread and self.hotkey_thread.isRunning(): self.hotkey_thread.stop()
        hotkey_str = self.hotkey_input.text()
        self.hotkey_thread = HotkeyListenerThread(hotkey_str, self)
        self.hotkey_thread.hotkey_pressed.connect(self.toggle_output_visibility)
        self.hotkey_thread.start()

    def on_fps_change(self):
        val = self.fps_slider.value(); self.fps_label.setText(f"Target FPS: {val}"); self.capture_thread.update_fps(val)
    def on_history_change(self):
        val = self.history_slider.value(); self.history_label.setText(f"Frames to Diff: {val}"); self.capture_thread.update_history_length(val)
    def on_sensitivity_change(self):
        val = self.sensitivity_slider.value(); self.sensitivity_label.setText(f"Sensitivity: {val}"); self.capture_thread.update_sensitivity(val)
    
    def closeEvent(self, event): 
        self.stop_all_threads()
        super().closeEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv); panel = ControlPanel(); panel.show(); sys.exit(app.exec())
