"""Executable entry point. The scientific application remains in its source file."""

import sys
import traceback
from pathlib import Path


def main():
    try:
        # The standalone file is also the project's only application source.
        if not getattr(sys, "frozen", False):
            sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        # Windowed executables have no standard streams. Retain CLI output in a log.
        if sys.stdout is None or sys.stderr is None:
            log_dir = Path.home() / ".rd_dq_optimizer"
            log_dir.mkdir(parents=True, exist_ok=True)
            log = (log_dir / "application.log").open("a", encoding="utf-8", buffering=1)
            if sys.stdout is None:
                sys.stdout = log
            if sys.stderr is None:
                sys.stderr = log
        if len(sys.argv) > 1 and sys.argv[1] == "--self-test":
            import argparse

            parser = argparse.ArgumentParser(description="Verify the bundled application runtime")
            parser.add_argument("--self-test", metavar="REPORT_JSON", required=True)
            parser.add_argument("--no-gui", action="store_true")
            args = parser.parse_args()
            from runtime_check import run_checks

            run_checks(Path(args.self_test), include_gui=not args.no_gui)
            return 0
        import rd_eqd2_optimizer

        return rd_eqd2_optimizer.main() or 0
    except Exception:
        details = traceback.format_exc()
        if sys.stderr is not None:
            print(details, file=sys.stderr)
        if (
            getattr(sys, "frozen", False)
            and "--batch" not in sys.argv
            and "--self-test" not in sys.argv
        ):
            try:
                import ctypes

                ctypes.windll.user32.MessageBoxW(
                    None, details, "RD EQD2 Optimizer — startup error", 0x10
                )
            except Exception:
                pass
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
