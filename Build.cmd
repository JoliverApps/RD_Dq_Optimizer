@echo off
setlocal
rem Use RD_PYTHON for a Python installation not registered with the launcher.
if defined RD_PYTHON goto custom_python
py -3.12 -c "import sys; assert sys.maxsize > 2**32" >nul 2>&1
if not errorlevel 1 goto python312
py -3.11 -c "import sys; assert sys.maxsize > 2**32" >nul 2>&1
if not errorlevel 1 goto python311
python -c "import sys; assert sys.version_info[:2] in [(3,11),(3,12)] and sys.maxsize > 2**32" >nul 2>&1
if not errorlevel 1 goto python_path
echo ERROR: Install 64-bit Python 3.12 with pip, venv, and Tcl/Tk.
echo Alternatively, set RD_PYTHON to the full path of python.exe.
exit /b 1

:custom_python
"%RD_PYTHON%" "%~dp0tools\build.py" %*
exit /b %errorlevel%
:python312
py -3.12 "%~dp0tools\build.py" %*
exit /b %errorlevel%
:python311
py -3.11 "%~dp0tools\build.py" %*
exit /b %errorlevel%
:python_path
python "%~dp0tools\build.py" %*
exit /b %errorlevel%
