"""Integration checks run against both source and the packaged executable."""

import csv
import json
import tempfile
from pathlib import Path


def check_math_and_plot_interval(app, folder):
    """Check RD values independently and verify the requested plotting interval."""
    from unittest.mock import patch

    import mpmath as mp
    import numpy as np
    from matplotlib.figure import Figure

    with mp.workdps(80):
        for sign in (1, -1):
            for magnitude in (1e-6, 1.5, 10, 1e6):
                ab = sign * magnitude
                for q_magnitude in (1e-14, 1e-6, 0.01, 1, 1e6, 1e14, 1e20):
                    q = sign * q_magnitude
                    a_mp, q_mp = mp.mpf(ab), mp.mpf(q)
                    root = mp.sqrt(1 + 2 * a_mp / q_mp)
                    rate = (root - 1) / a_mp if ab > 0 else (-1 - root) / a_mp
                    reserve = q_mp * rate
                    doses = (0.01, 1.8, 2.0, 9.5, 100)
                    expected_bed = [
                        float((mp.mpf(d) + q_mp * mp.expm1(-rate * d)) / (1 - reserve))
                        for d in doses
                    ]
                    bed = app.fraction_bed(ab, [q], doses)[0]
                    np.testing.assert_allclose(bed, expected_bed, rtol=1e-12)
                    expected_eqd2 = 40 * np.array(expected_bed) / expected_bed[2]
                    np.testing.assert_allclose(
                        app.EQD2_FN(ab, q, doses, 20), expected_eqd2, rtol=1e-12
                    )
        for ratio, candidate in ((1.5, 3.5), (-5, -3)):
            np.testing.assert_allclose(
                app.EQD2_FN(ratio, candidate, np.array([1.8, 2, 6]), 20),
                app._EQD2_SYMBOLIC_FN(ratio, candidate, np.array([1.8, 2, 6]), 20),
                rtol=1e-13,
            )

    schedules = [("Reference", 60, 30), ("Other", 48, 4)]
    grid = app.grid_values(-3.2, 4.9, 0.7)
    assert grid[0] == -3.2 and grid[-1] == 4.9 and 0 in grid
    cases = [(10, -20.3, 13.7), (-10, -14.3, 6.2)]
    for ab, start, stop in cases:
        fitted = app.analyze(ab, start, stop, 0.9, schedules)
        assert fitted["search_start"] == start and fitted["search_end"] == stop
        assert fitted["q"][0] == start and fitted["q"][-1] == stop
        assert np.count_nonzero(fitted["q"] == 0) == 1
        fitted["plot_settings"] = {"x_min": 0, "x_max": 1}
        fig = Figure(figsize=(8, 5))
        ax = fig.add_subplot(111)
        app.draw_range_plot(ax, fitted)
        np.testing.assert_array_equal(ax.get_xlim(), [start, stop])
        curve = ax.lines[0]
        assert curve.get_xdata().min() < 0 < curve.get_xdata().max()
        assert np.all(np.isfinite(curve.get_ydata()))
        app.export_image(fitted, folder / f"signed-profile-{ab}.png", ppi=100)

    # The profile interval belongs to the connected region around the selected minimum.
    points = np.linspace(-9, -1, 161)
    measures = 1 + np.sin(np.pi * (points + 5) / 4) ** 2

    def profile_values(_ab, q, _schedules):
        sd = 1 + np.sin(np.pi * (np.asarray(q) + 5) / 4) ** 2
        return np.column_stack((70 - sd / np.sqrt(2), 70 + sd / np.sqrt(2)))

    with patch.object(app, "evaluate", profile_values):
        bounds = app._profile_bounds(
            -10, -5, schedules, 1.0, lower=-9, upper=-1, profile=(points, measures)
        )
    radius = 4 * np.arcsin(np.sqrt(0.05)) / np.pi
    np.testing.assert_allclose(bounds, [-5 - radius, -5 + radius], atol=1e-11)
    flat = app.analyze(-10, -9, -1, 0.1, [("A", 60, 30), ("B", 60, 30)])
    assert flat["profile_bounds"] == (None, None)


def check_signed_model(app, folder):
    """Check the signed extension against independently integrated hazards."""
    import sqlite3

    import mpmath as mp
    import numpy as np
    from matplotlib.figure import Figure

    mp.mp.dps = 50
    schedules = [("Reference", 78, 39), ("Large fractions", 30, 5), ("Small fractions", 54, 30)]
    for reserve, rate in ((0.8, 0.3), (-0.8, 0.3), (-3.0, 0.5)):
        ab = 2 * (1 - reserve) / (reserve * rate)
        q = reserve / rate

        def effect(dose, reserve=reserve, rate=rate):
            return mp.quad(lambda x: 1 - mp.mpf(reserve) * mp.exp(-mp.mpf(rate) * x), [0, dose])

        expected = [float(2 * n * effect(total / n) / effect(2)) for _, total, n in schedules]
        np.testing.assert_allclose(app.evaluate(ab, [q], schedules)[0], expected, rtol=2e-13)
        np.testing.assert_allclose(
            app.EQD2_FN(
                ab, q, [total / n for _, total, n in schedules], [n for _, _, n in schedules]
            ),
            expected,
            rtol=2e-13,
        )
        actual_r, actual_s = app.rd_parameters(ab, [q])
        np.testing.assert_allclose([actual_r[0], actual_s[0]], [reserve, rate], rtol=2e-14)

    physical = np.array([total for _, total, _ in schedules])
    for ab in (1.5, -1.5, -2.0):
        np.testing.assert_array_equal(app.evaluate(ab, [0], schedules)[0], physical)
        assert app.EQD2_FN(ab, 0, 3, 10) == 30
        near = np.copysign(1e-12, ab)
        np.testing.assert_allclose(app.evaluate(ab, [near], schedules)[0], physical, atol=1e-9)
        boundary = app.analyze(ab, 0, 0, 0.1, schedules)
        detail = app.build_analysis({"name": "Linear boundary", "result": boundary})
        assert detail["r"] == 0 and detail["s"] is None
        assert all(p["RD_BED_Gy"] == p["D_Gy"] for p in detail["prescriptions"])
        app.export_analysis_tables(detail, folder / f"boundary-{ab}", "json")
        app.draw_dispersion_plot(Figure(), detail)
    for ab, q in ((1.5, -1), (-1.5, 1)):
        np.testing.assert_array_equal(
            app.evaluate(ab, [q], schedules), app.evaluate(-ab, [q], schedules)
        )
    for invalid in (0, float("nan"), float("inf"), -float("inf")):
        try:
            app.validated_alpha_beta(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid alpha/beta was accepted")

    # Construct isoeffective schedules using independent scalar inversion.
    ab, q = -5.0, -3.0
    rate = (1 + mp.sqrt(1 + 2 * ab / q)) / abs(ab)

    def effect(dose):
        return dose - q * (1 - mp.exp(-rate * dose))

    synthetic = []
    for n in (10, 20, 30, 39, 50):
        dose = float(mp.findroot(lambda d, n=n: n * effect(d) - 39 * effect(2), (0, 20)))
        synthetic.append((f"Schedule {n}", n * dose, n))
    for method in app.OPTIMIZATION_METHODS:
        fitted = app.analyze(ab, -20, 20, 0.02, synthetic, method=method)
        np.testing.assert_allclose(fitted["q"][fitted["idx"]], q, atol=2e-6, rtol=0)
        assert fitted["ab"] == abs(ab) and fitted["signed_ab"] == ab
        assert fitted["q"].min() < 0 < fitted["q"].max()
    fitted = app.analyze(ab, -20, 0, 0.02, schedules)
    detail = app.build_analysis({"name": "Inverse shoulder", "result": fitted})
    app.run_robustness(detail)
    app.export_analysis_tables(detail, folder / "negative-analysis", "json")
    app.draw_dispersion_plot(Figure(), detail)
    positive = app.analyze(1.5, -20, 20, 0.02, schedules)
    assert positive["q"][0] == -20 and positive["q"][-1] == 20
    assert app.evaluate(1.5, [1e12], schedules).min() > 0
    mixed = app.analyze_selected(
        [
            {"name": "Positive", "alpha_beta": 1.5, "prescriptions": schedules},
            {"name": "Negative", "alpha_beta": -5, "prescriptions": schedules},
        ],
        -20,
        20,
        0.05,
    )
    assert len(mixed["sets"]) == 2
    app.export(mixed, folder / "mixed-signs")
    crossing = app._profile_bounds(-5, -3, synthetic, 0.1, lower=-20, upper=0)
    assert crossing[0] < -3 < crossing[1]
    for q_crossing in crossing:
        np.testing.assert_allclose(
            app.evaluate(-5, [q_crossing], synthetic)[0].std(ddof=1),
            0.105,
            atol=1e-10,
        )

    # Existing positive-only databases must migrate without losing child rows.
    path = folder / "old-positive-only.sqlite3"
    with sqlite3.connect(path) as con:
        con.executescript("""
            CREATE TABLE tumors (id INTEGER PRIMARY KEY, name TEXT NOT NULL COLLATE NOCASE UNIQUE,
                alpha_beta REAL NOT NULL CHECK (alpha_beta > 0));
            CREATE TABLE prescriptions (id INTEGER PRIMARY KEY,
                tumor_id INTEGER NOT NULL REFERENCES tumors(id) ON DELETE CASCADE,
                name TEXT NOT NULL, total_gy REAL NOT NULL, fractions INTEGER NOT NULL,
                position INTEGER NOT NULL, UNIQUE(tumor_id,name));
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            INSERT INTO tumors VALUES (7,'Retained',1.5);
            INSERT INTO prescriptions VALUES (11,7,'Reference',78,39,0);
            INSERT INTO prescriptions VALUES (12,7,'Other',30,5,1);
            INSERT INTO metadata VALUES ('default_prescriptions_sets_seeded','1');
        """)
    store = app.PrescriptionStore(path, app.DEFAULT_SETS)
    assert len(store.list_sets()) == 1
    assert store.get_set(7)["prescriptions"] == [("Reference", 78, 39), ("Other", 30, 5)]
    store.update_alpha_beta(7, -5)
    reopened = app.PrescriptionStore(path)
    assert reopened.get_set(7)["alpha_beta"] == 5
    app.export_prescriptions_set(reopened.get_set(7), folder / "signed.csv")
    assert app.read_prescriptions_set_csv(folder / "signed.csv")["alpha_beta"] == 5
    with reopened._connection() as con:
        assert [row[0] for row in con.execute("SELECT id FROM prescriptions ORDER BY id")] == [
            11,
            12,
        ]
        assert not con.execute("PRAGMA foreign_key_check").fetchall()
    reopened.delete_set(7)
    with reopened._connection() as con:
        assert con.execute("SELECT count(*) FROM prescriptions").fetchone()[0] == 0


def check_full_signed_profile(app, folder):
    """Check both branches for one magnitude and the 15-prescription NSCLC set."""
    import numpy as np
    from matplotlib.figure import Figure

    doses = [
        (70, 35),
        (66, 33),
        (50, 20),
        (52.5, 15),
        (60, 15),
        (60, 8),
        (60, 5),
        (50, 4),
        (48, 4),
        (66, 3),
        (60, 3),
        (54, 3),
        (45, 3),
        (34, 1),
        (30, 1),
    ]
    nsclc = [
        (f"NSCLC {index + 1}", total, fractions) for index, (total, fractions) in enumerate(doses)
    ]
    result = app.analyze(10, -50, 50, 0.05, nsclc)
    assert result["ab"] == 10 and len(result["schedules"]) == 15
    assert result["q"][0] == -50 and result["q"][-1] == 50
    assert np.count_nonzero(result["q"] == 0) == 1
    assert np.all(np.isfinite(result["eqd2"])) and np.all(result["eqd2"] > 0)
    np.testing.assert_array_equal(
        result["eqd2"][result["q"] == 0][0], [total for _, total, _ in nsclc]
    )
    reserve, rate = app.rd_parameters(10, [-3, 3])
    np.testing.assert_allclose(2 * (1 - reserve) / (reserve * rate), [-10, 10], rtol=1e-13)
    for magnitude in (10, -10):
        zero = app.analyze(magnitude, 0, 0, 0.1, nsclc)
        assert zero["signed_ab"] == 10 and zero["q"].tolist() == [0]

    # Around zero EQD2 approaches the same smooth rational curve from both sides.
    q = np.array([-1e-4, -1e-6, 0, 1e-6, 1e-4])
    n = np.array([fractions for _, _, fractions in nsclc])
    d = np.array([total / fractions for _, total, fractions in nsclc])
    expected = 2 * n * (d - q[:, None]) / (2 - q[:, None])
    np.testing.assert_allclose(app.evaluate(10, q, nsclc), expected, rtol=1e-13)
    h = 1e-5
    around = app.evaluate(10, [-h, 0, h], nsclc)
    derivative = (around[2] - around[0]) / (2 * h)
    np.testing.assert_allclose(derivative, n * (d - 2) / 2, rtol=2e-8, atol=1e-8)

    fig = Figure()
    ax = fig.add_subplot(111)
    app.draw_range_plot(ax, result)
    assert ax.get_xlim() == (-50, 50)
    assert ax.lines[0].get_xdata().min() == -50
    assert ax.lines[0].get_xdata().max() == 50
    app.export(result, folder / "nsclc-full-profile")
    with (folder / "nsclc-full-profile" / "grid_results.csv").open(encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        q_value = float(row["Dq_Gy"])
        assert float(row["alpha_beta_Gy"]) == (10 if q_value >= 0 else -10)
    assert sum(float(row["Dq_Gy"]) == 0 for row in rows) == 1


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
                root.setvar(ab_entry.cget("textvariable"), "-2")
                optimize.invoke()
                row = results.item(results.get_children()[0], "values")
                assert float(row[2]) == 2
                button("Save α/β").invoke()
                assert (
                    app.PrescriptionStore(db_path).find_set(app.PROSTATE_SET_NAME)["alpha_beta"]
                    == 2
                )
                optimize.invoke()
                button("Analysis…").invoke()
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
    import sys

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
            if not getattr(sys, "frozen", False):
                root = Path(__file__).resolve().parents[1]
                assert Path(app.__file__).resolve() == root / "rd_eqd2_optimizer.py"
                assert not (root / "src" / "rd_eqd2_optimizer.py").exists()
                passed("Standalone and project use one application source")
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
            check_signed_model(app, folder)
            passed(
                "Signed fixed-ratio branches, zero limit, independent hazard integration, migration and exports"
            )
            check_math_and_plot_interval(app, folder)
            passed(
                "High-precision RD arithmetic, connected profile bounds, and requested graph intervals"
            )
            check_full_signed_profile(app, folder)
            passed("Full 15-prescription NSCLC profile, q-dependent signs, and smooth unique zero")

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
            for bad in ("", "0", "nan", "inf", "abc"):
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
