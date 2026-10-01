$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$runtime = Join-Path $PSScriptRoot '.venv-runtime\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $runtime)) { $runtime = Join-Path $PSScriptRoot '.venv\Scripts\python.exe' }
if (-not (Test-Path -LiteralPath $runtime)) { throw 'Create a virtual environment and install the project as described in README.md.' }
& $runtime -c 'import ssl, streamlit'
if ($LASTEXITCODE -ne 0) { throw 'The Python environment is broken or missing dependencies. Follow the Python 3.12 setup in README.md.' }
& $runtime -m streamlit run app.py --server.address 127.0.0.1
