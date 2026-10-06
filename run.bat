@echo off
echo Starting ThirdEye on Python 3.11...
if defined PYTHON_EXE (
    "%PYTHON_EXE%" app.py
) else if exist "%USERPROFILE%\anaconda3\envs\ThirdEye311\python.exe" (
    "%USERPROFILE%\anaconda3\envs\ThirdEye311\python.exe" app.py
) else if exist "%CONDA_PREFIX%\python.exe" (
    "%CONDA_PREFIX%\python.exe" app.py
) else if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" app.py
) else (
    py -3.11 app.py 2>nul || python app.py
)
