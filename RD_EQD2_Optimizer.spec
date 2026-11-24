# Native desktop bundles for Windows, macOS, and Linux.
import os
import sys
import tomllib
from pathlib import Path

root = Path(SPECPATH)
configuration = os.environ.get('RD_BUILD_CONFIGURATION', 'Release')
debug_build = configuration == 'Debug'
macos_app = sys.platform == 'darwin' and not debug_build
app_name = 'RD_EQD2_Optimizer'
version = tomllib.loads((root / 'pyproject.toml').read_text(encoding='utf-8'))['project']['version']

a = Analysis(
    [str(root / 'src' / 'main.py')],
    pathex=[str(root), str(root / 'src')],
    binaries=[],
    datas=[(str(root / 'LICENSE'), '.')],
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
    name=app_name, debug=False, bootloader_ignore_signals=False,
    strip=False, upx=False, console=debug_build,
    disable_windowed_traceback=False,
)
if debug_build or macos_app:
    exe = EXE(pyz, a.scripts, [], exclude_binaries=True, **common)
    collect = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False,
                      name=app_name)
    if macos_app:
        app = BUNDLE(
            collect, name=f'{app_name}.app', icon=None,
            bundle_identifier='io.github.joliverapps.rd-eqd2-optimizer',
            version=version,
        )
else:
    exe = EXE(pyz, a.scripts, a.binaries, a.datas, [], **common)
