# PyInstaller specification used by the Visual Studio build.
import os
from pathlib import Path

root = Path(SPECPATH)
configuration = os.environ.get('RD_BUILD_CONFIGURATION', 'Release')
debug_build = configuration == 'Debug'

a = Analysis(
    [str(root / 'src' / 'main.py')],
    pathex=[str(root), str(root / 'src')],
    binaries=[],
    datas=[],
    hiddenimports=[
        'PIL._tkinter_finder',
        'matplotlib.backends.backend_tkagg',
        'matplotlib.backends.backend_agg',
        'matplotlib.backends.backend_ps',
        'matplotlib.backends.backend_pdf',
        'matplotlib.backends.backend_svg',
    ],
    hookspath=[],
    hooksconfig={'matplotlib': {'backends': ['TkAgg', 'Agg', 'PS', 'PDF', 'SVG']}},
    runtime_hooks=[],
    excludes=['IPython', 'jupyter', 'notebook', 'pytest', 'PyQt5', 'PyQt6', 'PySide2', 'PySide6'],
    noarchive=False,
)
pyz = PYZ(a.pure)

common = dict(
    name='RD_EQD2_Optimizer', debug=False, bootloader_ignore_signals=False,
    strip=False, upx=False, console=debug_build,
    disable_windowed_traceback=False,
)
if debug_build:
    exe = EXE(pyz, a.scripts, [], exclude_binaries=True, **common)
    collect = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False,
                      name='RD_EQD2_Optimizer')
else:
    exe = EXE(pyz, a.scripts, a.binaries, a.datas, [], **common)
