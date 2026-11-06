"""Integration checks run against both source and the packaged executable."""

import csv
import json
import tempfile
from pathlib import Path


def check_gui(app, db_path, result):
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    original_tk = tk.Tk
    original_error = messagebox.showerror
    original_save = filedialog.asksaveasfilename
    failures = []
    completed = []
    # A user-created fixture tests switching sets without adding another default.
    fixtures = app.PrescriptionStore(db_path, app.DEFAULT_SETS)
    fixtures.save_set("Verification — saved set", 2.0, [("Reference", 20, 10), ("Schedule", 12, 3)])

    def fail_dialog(title, message, **_kwargs):
        raise RuntimeError(str(title) + ": " + str(message))

    def children(widget):
        for child in widget.winfo_children():
            yield child
            yield from children(child)

    def test_root(*args, **kwargs):
        root = original_tk(*args, **kwargs)

        def callback_error(_kind, value, _tb):
            failures.append(str(value))

        root.report_callback_exception = callback_error

        def exercise():
            try:
                buttons = [w for w in children(root) if isinstance(w, ttk.Button)]

                def button(text):
                    return next(w for w in buttons if str(w.cget("text")) == text)

                trees = [w for w in children(root) if isinstance(w, ttk.Treeview)]
                prescriptions = next(w for w in trees if "include" in w.cget("columns"))
                results = next(w for w in trees if "dq" in w.cget("columns"))
                ab_entry = next(
                    w
                    for w in children(root)
                    if isinstance(w, ttk.Entry) and w.master == button("Save α/β").master
                )
                optimize = next(w for w in buttons if str(w.cget("text")) == "Run optimization")
                root.update_idletasks()
                # Exercise the actual checkbox mouse binding.
                x, y, width, height = prescriptions.bbox("0", "#1")
                prescriptions.event_generate("<Button-1>", x=x + width // 2, y=y + height // 2)
                assert prescriptions.set("0", "include") == "☐"
                root.setvar(ab_entry.cget("textvariable"), "2.2")
                optimize.invoke()
                row = results.item(results.get_children()[0], "values")
                assert int(row[1]) == 7 and float(row[2]) == 2.2

                selector = next(w for w in children(root) if isinstance(w, ttk.Menubutton))
                menu = root.nametowidget(selector.cget("menu"))
                choices = {menu.entrycget(i, "label"): i for i in range(menu.index("end") + 1)}
                other = choices["Verification — saved set"]
                menu.invoke(other)
                assert float(ab_entry.get()) == 2.0
                menu.invoke(choices[app.PROSTATE_SET_NAME])
                assert float(ab_entry.get()) == 2.2 and prescriptions.set("0", "include") == "☐"

                button("Save α/β").invoke()
                saved = app.PrescriptionStore(db_path).find_set(app.PROSTATE_SET_NAME)
                assert saved["alpha_beta"] == 2.2 and len(saved["prescriptions"]) == 8
                export_path = Path(db_path).with_name("gui-export.csv")
                filedialog.asksaveasfilename = lambda **_kwargs: str(export_path)
                button("Export prescriptions set…").invoke()
                imported = app.read_prescriptions_set_csv(export_path)
                assert len(imported["prescriptions"]) == 7 and imported["alpha_beta"] == 2.2
                optimize.invoke()
                details = app.open_analysis_window(root, result)
                assert details is not None and details.winfo_exists(), (
                    "Analysis window did not open"
                )
                for text in ("Plot settings…", "Export image…"):
                    next(w for w in buttons if str(w.cget("text")) == text).invoke()
                root.update_idletasks()
                if failures:
                    raise RuntimeError("; ".join(failures))
                completed.append(True)
            except Exception as error:
                failures.append(str(error))
            finally:
                root.destroy()

        root.after(150, exercise)
        return root

    try:
        tk.Tk = test_root
        messagebox.showerror = fail_dialog
        app.launch_gui(db_path=db_path)
    finally:
        tk.Tk = original_tk
        messagebox.showerror = original_error
        filedialog.asksaveasfilename = original_save
    if not completed or failures:
        raise RuntimeError("GUI verification failed: " + "; ".join(failures))


def run_checks(report_path, include_gui=True):
    import numpy as np

    import rd_eqd2_optimizer as app

    report_path = Path(report_path).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {"status": "running", "gui": "not run", "checks": []}

    def passed(name):
        report["checks"].append(name)

    try:
        # Isolate verification from saved user prescriptions and preferences.
        with tempfile.TemporaryDirectory(prefix="rd-eqd2-check-") as temporary:
            folder = Path(temporary)
            result = app.analyze(1.5, 0.01, 50.0, 0.01, app.PROSTATE_PRESCRIPTIONS)
            i = result["idx"]
            # Baseline measured directly from the unchanged uploaded application.
            np.testing.assert_allclose(result["q"][i], 3.543859355746318, atol=2e-5, rtol=0)
            np.testing.assert_allclose(result["mean"][i], 75.81790297672867, atol=2e-5, rtol=0)
            np.testing.assert_allclose(result["rsd"][i], 2.8365977004593472, atol=2e-5, rtol=0)
            report.update(
                Dq_Gy=float(result["q"][i]),
                mean_EQD2_Gy=float(result["mean"][i]),
                RSD_percent=float(result["rsd"][i]),
            )
            passed("Original prostate calibration baseline")

            doses = np.array([dose / n for _, dose, n in app.PROSTATE_PRESCRIPTIONS])
            fractions = np.array([n for _, _, n in app.PROSTATE_PRESCRIPTIONS])
            symbolic = app.EQD2_FN(1.5, result["q"][i], doses, fractions)
            np.testing.assert_allclose(symbolic, result["eqd2"][i], rtol=1e-10)
            passed("SymPy lambdify and NumPy runtime")

            sd_fit = app.analyze(1.5, 0.01, 50, 0.01, app.PROSTATE_PRESCRIPTIONS, method="sd")
            np.testing.assert_allclose(
                sd_fit["q"][sd_fit["idx"]], result["q"][i], atol=2e-5, rtol=0
            )
            range_fit = app.analyze(1.5, 0.01, 50, 0.01, app.PROSTATE_PRESCRIPTIONS, method="range")
            assert np.isfinite(range_fit["spread"][range_fit["idx"]])
            passed("SSE, SD, and range optimization")

            store = app.PrescriptionStore(folder / "prescriptions.sqlite3", app.DEFAULT_SETS)
            assert len(store.list_sets()) == 1, "Only prostate may be seeded by default"
            prostate = store.find_set(app.PROSTATE_SET_NAME)
            assert len(prostate["prescriptions"]) == 8
            assert prostate["alpha_beta"] == 1.5
            reopened = app.PrescriptionStore(store.path, app.DEFAULT_SETS)
            assert reopened.list_sets() == store.list_sets(), "Defaults must not duplicate"

            existing = app.PrescriptionStore(folder / "existing.sqlite3")
            preserved_id = existing.save_set(
                "Saved user set", 3.0, [("Reference", 20, 10), ("Schedule", 12, 3)]
            )
            preserved = existing.get_set(preserved_id)
            seeded = app.PrescriptionStore(existing.path, app.DEFAULT_SETS)
            assert seeded.get_set(preserved_id) == preserved
            assert len(seeded.list_sets()) == 2
            default_id = seeded.find_set(app.PROSTATE_SET_NAME)["id"]
            seeded.update_alpha_beta(default_id, 1.8)
            reopened = app.PrescriptionStore(existing.path, app.DEFAULT_SETS)
            assert reopened.get_set(default_id)["alpha_beta"] == 1.8
            reopened.delete_set(default_id)
            reopened = app.PrescriptionStore(existing.path, app.DEFAULT_SETS)
            assert reopened.find_set(app.PROSTATE_SET_NAME) is None
            passed("Prostate-only defaults, idempotent initialization, and saved-set preservation")
            custom = store.save_set(
                "Verification — custom", 1.5, [("Reference", 78, 39), ("SBRT", 38, 4)]
            )
            assert len(store.get_set(custom)["prescriptions"]) == 2
            store.save_set(
                "Verification — edited",
                2.0,
                [("Reference", 78, 39), ("SBRT", 40, 5)],
                set_id=custom,
            )
            assert store.get_set(custom)["alpha_beta"] == 2.0
            assert store.delete_set(custom)
            passed("SQLite initial data, create, edit, and delete")

            state = app.PrescriptionsSelection()
            records = [store.get_set(row["id"]) for row in store.list_sets()]
            state.sync(records)
            state.set_included(prostate, [0, 2], False)
            state.alpha_beta_text[str(prostate["id"])] = "2.2"
            configured = state.configured_set(prostate)
            assert len(configured["prescriptions"]) == 6 and configured["alpha_beta"] == 2.2
            fitted_subset = app.analyze_selected([configured], 0.01, 50, 0.02)["sets"][0]["result"]
            assert (
                fitted_subset["schedules"] == configured["prescriptions"]
                and fitted_subset["ab"] == 2.2
            )
            state.sync(records)
            assert state.configured_set(prostate) == configured
            for bad in ("", "0", "-1", "nan", "inf", "abc"):
                state.alpha_beta_text[str(prostate["id"])] = bad
                try:
                    state.configured_set(prostate)
                except ValueError:
                    pass
                else:
                    raise AssertionError("Invalid alpha/beta was accepted")
            state.alpha_beta_text[str(prostate["id"])] = "2.2"
            with store._connection() as connection:
                ids_before = [
                    tuple(row)
                    for row in connection.execute(
                        "SELECT id,name,total_gy,fractions FROM prescriptions ORDER BY id"
                    )
                ]
            store.update_alpha_beta(prostate["id"], 2.2)
            with store._connection() as connection:
                ids_after = [
                    tuple(row)
                    for row in connection.execute(
                        "SELECT id,name,total_gy,fractions FROM prescriptions ORDER BY id"
                    )
                ]
            assert ids_before == ids_after
            assert store.get_set(prostate["id"])["alpha_beta"] == 2.2
            state.set_included(prostate, range(8), False)
            state.set_included(prostate, [0], True)
            try:
                state.configured_set(prostate)
            except ValueError:
                pass
            else:
                raise AssertionError("A one-prescription optimization was accepted")
            subset_path = folder / "selected-set.csv"
            app.export_prescriptions_set(configured, subset_path)
            subset = app.read_prescriptions_set_csv(subset_path)
            assert subset["prescriptions"] == configured["prescriptions"]
            assert subset["alpha_beta"] == configured["alpha_beta"]
            assert subset["name"] == configured["name"]
            passed(
                "Prescription subsets, independent alpha/beta drafts, validation, and CSV round trip"
            )

            csv_path = folder / "prescriptions.csv"
            with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
                writer = csv.writer(stream)
                writer.writerow(["name", "total_gy", "fractions", "tumor", "alpha_beta"])
                for name, dose, count in app.PROSTATE_PRESCRIPTIONS:
                    writer.writerow([name, dose, count, "Prostate", 1.5])
            imported = app.read_prescriptions_set_csv(csv_path)
            assert imported["prescriptions"] == app.PROSTATE_PRESCRIPTIONS
            passed("CSV import with Unicode and UTF-8 BOM")

            app.export(result, folder / "results")
            for extension in (".png", ".jpg", ".eps"):
                app.export_image(result, folder / ("profile" + extension), ppi=72)
            details = app.build_analysis({"name": "Prostate", "result": result})
            robustness = app.run_robustness(details)
            assert len(robustness["leave_one_out"]) == 8
            for extension in (".png", ".jpg", ".eps", ".pdf", ".svg"):
                app.export_dispersion_plot(details, folder / ("dispersion" + extension), ppi=72)
            for format_name in ("csv", "tsv", "json", "tex"):
                paths = app.export_analysis_tables(details, folder / "tables", format=format_name)
                assert paths and all(p.is_file() and p.stat().st_size for p in paths)
            for path in folder.glob("profile.*"):
                assert path.stat().st_size > 100
            passed("Image export: PNG, JPEG, EPS, PDF, SVG")
            passed("Analysis, robustness, and CSV/TSV/JSON/LaTeX export")

            store.save_set(
                "Verification — user set", 2.0, [("Reference", 20, 10), ("Schedule", 12, 3)]
            )
            selected = app.analyze_selected(
                [store.get_set(t["id"]) for t in store.list_sets()], 0.01, 50, 0.02
            )
            assert len(selected["sets"]) == 2
            app.export(selected, folder / "multiple")
            passed("Independent prescriptions sets with combined export")

            preferences = folder / "plot_settings.json"
            app.write_plot_settings(preferences, {"title": "Verification", "width_in": 9.0})
            assert app.read_plot_settings(preferences)["width_in"] == 9.0
            passed("Plot preference persistence")
            if include_gui:
                check_gui(app, folder / "gui.sqlite3", result)
                report["gui"] = "passed"
                passed("Main GUI, optimization, detailed analysis, settings, and export dialogs")
            else:
                report["gui"] = "skipped explicitly (--no-gui)"
        report["status"] = "passed"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = str(error)
        raise
    finally:
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Verification passed: " + str(report_path))
    return report
