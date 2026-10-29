# movement-overlay

A Windows screen overlay that highlights motion. It captures a monitor, compares recent frames and shows only the pixels that changed in a separate output window.

## Features

- Pick which monitor to capture
- Adjustable FPS, frame history length and motion sensitivity
- Global hotkey to show/hide the output window (default `Ctrl+Alt+H`, configurable in the UI)
- Debug mode that shows the raw captured frame

## Requirements

- Windows with Python 3
- Dependencies listed in [`requirements.txt`](requirements.txt): `opencv-python`, `mss`, `numpy`, `PyQt6`, `pynput`, `pygetwindow`

## Usage

```powershell
.\install.ps1   # creates venv/ and installs dependencies
.\run.ps1       # starts python-motion-shader.py
```

`backup-1-_drifting.py` is an earlier variant of the effect kept for reference.
