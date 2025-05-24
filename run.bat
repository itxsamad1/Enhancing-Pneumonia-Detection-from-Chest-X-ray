@echo off
echo Pneumonia Detection App Launcher
echo ==============================
echo.

REM Check if Python is installed
where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Python is not installed or not in PATH. Please install Python 3.7 or newer.
    echo You can download Python from https://www.python.org/downloads/
    pause
    exit /b 1
)

REM Check Python version (must be 3.7+)
python -c "import sys; exit(0 if sys.version_info >= (3,7) else 1)" >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Your Python version is too old. Please install Python 3.7 or newer.
    echo You can download Python from https://www.python.org/downloads/
    pause
    exit /b 1
)

REM Create .streamlit config directory if it doesn't exist
if not exist "%USERPROFILE%\.streamlit" mkdir "%USERPROFILE%\.streamlit"

REM Create config.toml with error-only logging if it doesn't exist
if not exist "%USERPROFILE%\.streamlit\config.toml" (
    echo [logger] > "%USERPROFILE%\.streamlit\config.toml"
    echo level = "error" >> "%USERPROFILE%\.streamlit\config.toml"
    echo [browser] >> "%USERPROFILE%\.streamlit\config.toml"
    echo gatherUsageStats = false >> "%USERPROFILE%\.streamlit\config.toml"
)

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    if %ERRORLEVEL% NEQ 0 (
        echo Failed to create virtual environment. Please check your Python installation.
        pause
        exit /b 1
    )
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Check if pip is installed in the virtual environment
where pip >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Pip not found in virtual environment. Attempting to install...
    python -m ensurepip --upgrade
    if %ERRORLEVEL% NEQ 0 (
        echo Failed to install pip. Please check your Python installation.
        pause
        exit /b 1
    )
)

REM Install/update pip
echo Updating pip...
python -m pip install --upgrade pip

REM Install requirements
echo Installing dependencies...
pip install -r requirements.txt
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Warning: Some dependencies failed to install.
    echo The application might still work if core dependencies are installed.
    echo Press any key to continue anyway...
    pause >nul
)

REM Check for model file
if not exist "pneumonia_resnet18.pt" (
    echo.
    echo Warning: Model file "pneumonia_resnet18.pt" not found. 
    echo The application will run in demo mode.
    echo.
    timeout /t 5
)

REM Ensure assets/sample_images directory exists
if not exist "assets\sample_images" mkdir assets\sample_images

REM Run the Streamlit app
echo.
echo Starting Pneumonia Detection App...
echo.
set STREAMLIT_LOGGER_LEVEL=error
streamlit run app.py

REM Deactivate virtual environment when done
call venv\Scripts\deactivate.bat

pause 