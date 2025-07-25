# run.ps1
# This script activates the virtual environment and runs the main Python application.

# IMPORTANT: Make sure you have saved the Python code from the chat
# into a file named 'python-motion-shader.py' in the same directory as this script.

$python_script_name = "python-motion-shader-window.py"
$venv_name = "venv"

# Check if the virtual environment exists
if (-not (Test-Path -Path ".\$($venv_name)\Scripts\Activate.ps1")) {
    Write-Host "Error: Virtual environment not found." -ForegroundColor Red
    Write-Host "Please run the 'install.ps1' script first." -ForegroundColor Yellow
    exit 1
}

# Check if the Python script exists
if (-not (Test-Path -Path $python_script_name)) {
    Write-Host "Error: The Python script '$python_script_name' was not found." -ForegroundColor Red
    Write-Host "Please save the Python code into a file with that name in this directory." -ForegroundColor Yellow
    exit 1
}

Write-Host "Activating virtual environment and starting the application..." -ForegroundColor Green

# Activate the environment and execute the script
& ".\$($venv_name)\Scripts\python.exe" $python_script_name

Write-Host "Application has been closed."
