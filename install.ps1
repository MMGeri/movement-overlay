# install.ps1
# This script creates a Python virtual environment and installs the required packages.

Write-Host "Setting up Python virtual environment..." -ForegroundColor Yellow

# Define the name for the virtual environment directory
$venv_name = "venv"

# Check if the virtual environment directory already exists
if (Test-Path -Path $venv_name) {
    Write-Host "Virtual environment '$venv_name' already exists." -ForegroundColor Green
} else {
    # Create the virtual environment
    Write-Host "Creating virtual environment: '$venv_name'..."
    python -m venv $venv_name
    if ($?) {
        Write-Host "Virtual environment created successfully." -ForegroundColor Green
    } else {
        Write-Host "Error: Failed to create virtual environment. Make sure Python 3 is installed and in your PATH." -ForegroundColor Red
        exit 1
    }
}

# --- Install packages ---
Write-Host "Activating virtual environment and installing packages from requirements.txt..."

# Activate the virtual environment and run pip install
# Note: The '&' operator executes the command.
& ".\$($venv_name)\Scripts\Activate.ps1"
pip install -r requirements.txt

if ($?) {
    Write-Host "Dependencies installed successfully!" -ForegroundColor Green
    Write-Host "You can now run the application using the 'run.ps1' script."
} else {
    Write-Host "Error: Failed to install dependencies." -ForegroundColor Red
}

Read-Host -Prompt "Setup complete. Press Enter to exit"
