Write-Host "Starting ThirdEye on Python 3.11..." -ForegroundColor Cyan

$condaPath = Join-Path $env:USERPROFILE "anaconda3\envs\ThirdEye311\python.exe"
if ($env:PYTHON_EXE) {
    & $env:PYTHON_EXE app.py
} elseif (Test-Path $condaPath) {
    & $condaPath app.py
} elseif ($env:CONDA_PREFIX -and (Test-Path "$env:CONDA_PREFIX\python.exe")) {
    & "$env:CONDA_PREFIX\python.exe" app.py
} elseif (Test-Path ".venv\Scripts\python.exe") {
    & ".venv\Scripts\python.exe" app.py
} elseif (Get-Command "py" -ErrorAction SilentlyContinue) {
    & py -3.11 app.py
} else {
    & python app.py
}
