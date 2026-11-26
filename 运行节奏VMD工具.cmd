@echo off
rem Launch the lyrics -> VMD GUI using the local virtual environment.
cd /d "%~dp0"
if not exist ".venv-worker\Scripts\python.exe" (
  echo Missing .venv-worker. Run: py -3.10 -m venv .venv-worker
  echo Then: .venv-worker\Scripts\python.exe -m pip install -r requirements.txt
  pause
  exit /b 1
)
".venv-worker\Scripts\python.exe" "tools\rhythm_vmd_gui.py"
if errorlevel 1 pause
