@echo off
setlocal

echo ===========================================
echo     Financial RAG System Setup
echo ===========================================
echo.

REM -------------------------------------------------
REM Check Conda
REM -------------------------------------------------
where conda >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Conda is not installed or not in PATH.
    pause
    exit /b 1
)

REM -------------------------------------------------
REM Create Conda Environment
REM -------------------------------------------------
echo Creating Conda environment (Python 3.13.1)...

call conda create --prefix ./venv python=3.13.1 -y

if %errorlevel% neq 0 (
    echo Failed to create environment.
    pause
    exit /b 1
)

echo.
echo Activating environment...

call conda activate ./venv

if %errorlevel% neq 0 (
    echo Trying conda.bat...

    call "%USERPROFILE%\anaconda3\condabin\conda.bat" activate ./venv
)

echo.
echo Installing dependencies...

python -m pip install --upgrade pip
pip install -r requirements.txt

if %errorlevel% neq 0 (
    echo Failed installing requirements.
    pause
    exit /b 1
)

echo.
echo Creating .env file...

(
echo GROQ_API_KEY="Enter your api key"
)> .env

echo.
echo ===========================================
echo            SETUP COMPLETE
echo ===========================================
echo.

echo Open the .env file and replace:
echo.
echo     Enter your api key
echo.
echo with your actual Groq API key.
echo.

echo -------------------------------------------
echo Next Step 1:
echo -------------------------------------------
echo python -m ingestion.indexer
echo.

echo This will ingest your 10-K.pdf into ChromaDB.
echo.

pause

echo.
echo -------------------------------------------
echo Next Step 2:
echo -------------------------------------------
echo streamlit run frontend/app.py
echo.

pause