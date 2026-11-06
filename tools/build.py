"""Windows build entry point used by both Visual Studio and Build.cmd."""

import argparse
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV_DIR = ROOT / ".venv-build"
ENV_PYTHON = ENV_DIR / "Scripts" / "python.exe"
CHECK_CODE = (
    "import sys, struct; "
    'assert sys.version_info[:2] in [(3,11),(3,12)], "Use Python 3.11 or 3.12"; '
    'assert struct.calcsize("P")==8, "Use 64-bit Python"; '
    "import tkinter, _tkinter, sqlite3, numpy, matplotlib, sympy, PyInstaller"
)


def build_environment(configuration=None):
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    environment["PYTHONNOUSERSITE"] = "1"
    if configuration is not None:
        environment["RD_BUILD_CONFIGURATION"] = configuration
    return environment


def run(command, **kwargs):
    arguments = [str(value) for value in command]
    print("> " + subprocess.list2cmdline(arguments), flush=True)
    if "env" not in kwargs:
        kwargs["env"] = build_environment()
    return subprocess.run(arguments, cwd=ROOT, check=True, **kwargs)


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_environment():
    if not ENV_PYTHON.is_file():
        run([sys.executable, "-m", "venv", ENV_DIR])
    digest = hashlib.sha256()
    for name in ("requirements.txt", "requirements-build.txt"):
        digest.update((ROOT / name).read_bytes())
    digest.update(sys.version.encode())
    digest.update(sys.executable.encode())
    key = digest.hexdigest()
    marker = ENV_DIR / ".requirements-key"
    previous = marker.read_text(encoding="utf-8").strip() if marker.exists() else ""
    if previous != key:
        run(
            [
                ENV_PYTHON,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--only-binary=:all:",
                "-r",
                ROOT / "requirements-build.txt",
            ]
        )
        marker.write_text(key, encoding="utf-8")
    run([ENV_PYTHON, "-c", CHECK_CODE])
    return ENV_PYTHON


def clean(configuration):
    # These two paths are generated output only. Never touch application data.
    for directory in (ROOT / ".build" / configuration, ROOT / "dist" / configuration):
        if directory.is_dir():
            shutil.rmtree(directory)


def self_test(command, report):
    report.parent.mkdir(parents=True, exist_ok=True)
    # Remove a previous report so a failed process cannot leave a false pass.
    report.unlink(missing_ok=True)
    run([*command, "--self-test", report], timeout=240)
    payload = json.loads(report.read_text(encoding="utf-8"))
    if payload.get("status") != "passed" or payload.get("gui") != "passed":
        raise RuntimeError("Runtime self-test did not pass. See " + str(report))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", nargs="?", default="Build", choices=["Build", "Rebuild", "Clean", "Test", "Run"]
    )
    parser.add_argument("configuration", nargs="?", default="Release", choices=["Debug", "Release"])
    args = parser.parse_args()
    if os.name != "nt":
        parser.error("Build the Windows executable on Windows, using 64-bit Python 3.12 or 3.11.")
    if args.action == "Clean":
        clean(args.configuration)
        return 0
    if sys.version_info[:2] not in ((3, 11), (3, 12)) or struct.calcsize("P") != 8:
        parser.error("Use 64-bit CPython 3.12 or 3.11, including pip, venv, and Tcl/Tk.")
    if args.action == "Rebuild":
        clean(args.configuration)
    python = prepare_environment()
    if args.action == "Run":
        run([python, ROOT / "src" / "main.py"])
        return 0
    work = ROOT / ".build" / args.configuration
    self_test([python, ROOT / "src" / "main.py"], work / "source-test.json")
    if args.action == "Test":
        print("Source and GUI verification passed.")
        return 0
    environment = build_environment(args.configuration)
    dist = ROOT / "dist" / args.configuration
    run(
        [
            python,
            "-m",
            "PyInstaller",
            "--noconfirm",
            "--clean",
            "--distpath",
            dist,
            "--workpath",
            work / "pyinstaller",
            ROOT / "RD_EQD2_Optimizer.spec",
        ],
        env=environment,
    )
    executable = (
        dist / "RD_EQD2_Optimizer.exe"
        if args.configuration == "Release"
        else dist / "RD_EQD2_Optimizer" / "RD_EQD2_Optimizer.exe"
    )
    if not executable.is_file():
        raise RuntimeError("The build did not produce " + str(executable))
    self_test([executable], work / "executable-test.json")
    packages = subprocess.check_output(
        [str(python), "-m", "pip", "freeze"], text=True, env=build_environment()
    )
    manifest = {
        "configuration": args.configuration,
        "executable": str(executable.relative_to(ROOT)),
        "sha256": file_sha256(executable),
        "application_source_sha256": file_sha256(ROOT / "rd_eqd2_optimizer.py"),
        "dependencies": packages.splitlines(),
        "source_test": "passed",
        "executable_test": "passed",
    }
    (dist / "build-info.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("\nBuild and executable verification passed.\n" + str(executable))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
        print("BUILD FAILED: " + str(error), file=sys.stderr)
        raise SystemExit(1) from None
