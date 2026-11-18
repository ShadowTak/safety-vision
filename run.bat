@echo off
rem Start Safety Vision at http://127.0.0.1:8765 (set HOST=0.0.0.0 to open it to other devices)
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  uv venv --python 3.12 .venv || exit /b 1
  uv pip install --python .venv\Scripts\python.exe -r requirements.txt || exit /b 1
)
if not defined HOST set HOST=127.0.0.1
if not defined PORT set PORT=8765
rem secrets such as TELEGRAM_TOKEN live in .env (not in the code)
set ENVFILE=
if exist .env set ENVFILE=--env-file .env
.venv\Scripts\python.exe -m uvicorn server:app --host %HOST% --port %PORT% %ENVFILE%
