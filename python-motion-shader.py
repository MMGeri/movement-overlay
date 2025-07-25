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
        self.hotkey_str = hotkey_str
        self.listener = None

    def run(self):
        """Sets up and runs the hotkey listener."""
        try:
            def on_activate():
                self.hotkey_pressed.emit()

            hotkey = keyboard.GlobalHotKeys({
                self.hotkey_str: on_activate
            })
            
            self.listener = hotkey
            print(f"Hotkey '{self.hotkey_str}' is active.")
            self.listener.start()
            self.listener.join() 

        except Exception as e:
            print(f"Failed to set hotkey '{self.hotkey_str}': {e}")

    def stop(self):
        """Stops the listener."""
        if self.listener:
            self.listener.stop()
            print("Hotkey listener stopped.")


class ScreenCaptureThread(QThread):
    """This thread handles the continuous screen capturing and processing."""
    new_frame = pyqtSignal(QImage)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.running = False
        self.paused = True # Start in a paused state
        self._monitor_number = 1
        self._fps = 30
        self._history_length = 3
        self._sensitivity = 20
        self.mutex = QMutex()

    def run(self):
        self.running = True
        frame_history = deque(maxlen=self._history_length)

        with mss.mss() as sct:
            while self.running:
                # --- Check for paused state ---
                with QMutexLocker(self.mutex):
                    is_paused = self.paused
                if is_paused:
                    time.sleep(0.1) # Sleep to prevent busy-waiting
                    continue
                # --------------------------------

                with QMutexLocker(self.mutex):
                    monitor_number = self._monitor_number
                    history_length = self._history_length
                    target_fps = self._fps
                    sensitivity = self._sensitivity

                if frame_history.maxlen != history_length:
                    frame_history = deque(list(frame_history), maxlen=history_length)

                frame_time_start = time.time()
                
                try:
                    monitor = sct.monitors[monitor_number]
                    sct_img = sct.grab(monitor)
                    frame = np.array(sct_img)

                except IndexError:
                    print(f"Error: Monitor {monitor_number} not found. Using primary.")
                    self._monitor_number = 1
                    continue
                
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
                motion_output = np.zeros_like(gray_frame)

                if len(frame_history) > 0:
                    for old_frame in frame_history:
                        diff = cv2.absdiff(gray_frame, old_frame)
                        motion_output = cv2.add(motion_output, diff)
                
                _, motion_output = cv2.threshold(motion_output, sensitivity, 255, cv2.THRESH_TOZERO)

                frame_history.append(gray_frame)
                
                h, w = motion_output.shape
                qt_image = QImage(motion_output.data, w, h, w, QImage.Format.Format_Grayscale8)
                self.new_frame.emit(qt_image)

                elapsed_time = time.time() - frame_time_start
                sleep_time = (1.0 / target_fps) - elapsed_time
                if sleep_time > 0:
                    time.sleep(sleep_time)

    def stop(self):
        self.resume() # Ensure thread is not paused before stopping
        self.running = False
        self.wait()

    def pause(self):
        with QMutexLocker(self.mutex):
            self.paused = True
            
    def resume(self):
        with QMutexLocker(self.mutex):
            self.paused = False

    def update_monitor(self, monitor_num):
        with QMutexLocker(self.mutex):
            self._monitor_number = monitor_num

    def update_fps(self, fps):
        with QMutexLocker(self.mutex):
            self._fps = fps

    def update_history_length(self, length):
        with QMutexLocker(self.mutex):
            self._history_length = length
            
    def update_sensitivity(self, value):
        with QMutexLocker(self.mutex):
            self._sensitivity = value


class OverlayWindow(QWidget):
    """The window that displays the motion effect overlay."""
    def __init__(self):
        super().__init__()
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool |
            Qt.WindowType.WindowTransparentForInput 
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        
        self.layout = QVBoxLayout(self)
        self.label = QLabel()
        self.layout.addWidget(self.label)
        self.setLayout(self.layout)

    def set_image(self, q_image):
        pixmap = QPixmap.fromImage(q_image)
        self.label.setPixmap(pixmap)
        self.resize(pixmap.size())


class ControlPanel(QWidget):
    """The main GUI control panel."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Motion Overlay Controls")
        
        self.overlay_window = OverlayWindow()
        self.capture_thread = ScreenCaptureThread()
        self.hotkey_thread = None
        
        # --- Create GUI Widgets ---
        self.monitor_combo = QComboBox()
        self.populate_monitors()

        self.history_slider = QSlider(Qt.Orientation.Horizontal)
        self.history_slider.setRange(1, 15)
        self.history_slider.setValue(3)
        self.history_label = QLabel(f"Frames to Diff: {self.history_slider.value()}")

        self.fps_slider = QSlider(Qt.Orientation.Horizontal)
        self.fps_slider.setRange(5, 60)
        self.fps_slider.setValue(30)
        self.fps_label = QLabel(f"Target FPS: {self.fps_slider.value()}")
        
        self.sensitivity_slider = QSlider(Qt.Orientation.Horizontal)
        self.sensitivity_slider.setRange(5, 100)
        self.sensitivity_slider.setValue(20)
        self.sensitivity_label = QLabel(f"Sensitivity: {self.sensitivity_slider.value()}")

        self.hotkey_input = QLineEdit("<ctrl>+<alt>+h")
        self.apply_hotkey_button = QPushButton("Apply Hotkey")
        
        self.start_button = QPushButton("Start")
        self.show_button = QPushButton("Show")
        self.hide_button = QPushButton("Hide")
        
        self.show_button.setEnabled(False)
        self.hide_button.setEnabled(False)

        # --- Layout ---
        layout = QFormLayout(self)
        layout.addRow(QLabel("Select Screen:"), self.monitor_combo)
        layout.addRow(self.history_label, self.history_slider)
        layout.addRow(self.fps_label, self.fps_slider)
        layout.addRow(self.sensitivity_label, self.sensitivity_slider)
        hotkey_layout = QVBoxLayout()
        hotkey_layout.addWidget(self.hotkey_input)
        hotkey_layout.addWidget(self.apply_hotkey_button)
        layout.addRow(QLabel("Hotkey:"), hotkey_layout)
        
        # New side-by-side button layout
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.show_button)
        button_layout.addWidget(self.hide_button)
        layout.addRow(button_layout)
        
        self.setLayout(layout)

        # --- Connect Signals to Slots ---
        self.start_button.clicked.connect(self.start_capture)
        self.show_button.clicked.connect(self.show_overlay)
        self.hide_button.clicked.connect(self.hide_overlay)
        self.apply_hotkey_button.clicked.connect(self.update_hotkey)

        self.monitor_combo.currentIndexChanged.connect(self.on_monitor_change)
        self.fps_slider.valueChanged.connect(self.on_fps_change)
        self.history_slider.valueChanged.connect(self.on_history_change)
        self.sensitivity_slider.valueChanged.connect(self.on_sensitivity_change)
        
        self.capture_thread.new_frame.connect(self.overlay_window.set_image)
        
        # Initialize labels
        self.on_fps_change(self.fps_slider.value())
        self.on_history_change(self.history_slider.value())
        self.on_sensitivity_change(self.sensitivity_slider.value())

    def populate_monitors(self):
        with mss.mss() as sct:
            for i, monitor in enumerate(sct.monitors):
                if i == 0: continue
                self.monitor_combo.addItem(f"Monitor {i}: {monitor['width']}x{monitor['height']}", i)
    
    def start_capture(self):
        monitor_data = self.monitor_combo.currentData()
        if monitor_data is None:
            QMessageBox.critical(self, "Error", "Could not find a valid monitor to capture.")
            return

        self.capture_thread.update_monitor(monitor_data)
        
        with mss.mss() as sct:
            monitor_geom = sct.monitors[monitor_data]
            self.overlay_window.move(monitor_geom["left"], monitor_geom["top"])
            
        self.capture_thread.start()
        self.update_hotkey() 
        
        self.start_button.setEnabled(False)
        self.show_button.setEnabled(True)
        self.hide_button.setEnabled(True)

    def stop_capture(self):
        self.capture_thread.stop()
        if self.hotkey_thread:
            self.hotkey_thread.stop()
        self.overlay_window.hide()

    def show_overlay(self):
        self.capture_thread.resume()
        self.overlay_window.show()

    def hide_overlay(self):
        self.capture_thread.pause()
        self.overlay_window.hide()
        
    def toggle_overlay_visibility(self):
        """Toggles the overlay state for the hotkey."""
        if self.overlay_window.isVisible():
            self.hide_overlay()
        else:
            self.show_overlay()

    def update_hotkey(self):
        if not self.capture_thread.isRunning():
            return

        if self.hotkey_thread and self.hotkey_thread.isRunning():
            self.hotkey_thread.stop()
        
        hotkey_str = self.hotkey_input.text()
        self.hotkey_thread = HotkeyListenerThread(hotkey_str)
        self.hotkey_thread.hotkey_pressed.connect(self.toggle_overlay_visibility)
        self.hotkey_thread.start()
        
    def on_monitor_change(self, index):
        monitor_number = self.monitor_combo.itemData(index)
        if monitor_number is not None:
            self.capture_thread.update_monitor(monitor_number)

    def on_fps_change(self, value):
        self.fps_label.setText(f"Target FPS: {value}")
        self.capture_thread.update_fps(value)

    def on_history_change(self, value):
        self.history_label.setText(f"Frames to Diff: {value}")
        self.capture_thread.update_history_length(value)
        
    def on_sensitivity_change(self, value):
        self.sensitivity_label.setText(f"Sensitivity: {value}")
        self.capture_thread.update_sensitivity(value)

    def closeEvent(self, event):
        """Ensures all threads are stopped when the window is closed."""
        self.stop_capture()
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    panel = ControlPanel()
    panel.show()
    sys.exit(app.exec())
