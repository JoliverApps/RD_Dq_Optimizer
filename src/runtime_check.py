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
            for ab in (1e-6, 1.5, 10, 1e6):
                for q_scale in (1e-14, 1e-6, 0.01, 1, 1e6, 1e14, 1e20):
                    q = sign * q_scale
                    a_mp, q_mp = sign * mp.mpf(ab), mp.mpf(q)
                    root = mp.sqrt(1 + 2 * a_mp / q_mp)
                    rate = (root - 1) / a_mp if sign > 0 else (-1 - root) / a_mp
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
        for ratio, candidate in ((1.5, 3.5), (5, -3)):
            np.testing.assert_allclose(
                app.EQD2_FN(ratio, candidate, np.array([1.8, 2, 6]), 20),
                app._EQD2_SYMBOLIC_FN(ratio, candidate, np.array([1.8, 2, 6]), 20),
                rtol=1e-13,
            )

    schedules = [("Reference", 60, 30), ("Other", 48, 4)]
    grid = app.grid_values(-3.2, 4.9, 0.7)
    assert grid[0] == -3.2 and grid[-1] == 4.9 and 0 in grid
    cases = [(10, -20.3, 13.7), (10, -14.3, 6.2)]
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
            10, -5, schedules, 1.0, lower=-9, upper=-1, profile=(points, measures)
        )
    radius = 4 * np.arcsin(np.sqrt(0.05)) / np.pi
    np.testing.assert_allclose(bounds, [-5 - radius, -5 + radius], atol=1e-11)
    flat = app.analyze(10, -9, -1, 0.1, [("A", 60, 30), ("B", 60, 30)])
    assert flat["profile_bounds"] == (None, None)


def check_signed_model(app, folder):
    """Check the signed extension against independently integrated hazards."""
    import sqlite3

    import mpmath as mp
    import numpy as np
    from matplotlib.figure import Figure

    mp.mp.dps = 50
    schedules = [("Reference", 78, 39), ("Large fractions", 30, 5), ("Small fractions", 54, 30)]
    for reserve, rate, ab in ((0.8, 0.3, 5 / 3), (-0.8, 0.3, 15), (-3.0, 0.5, 16 / 3)):
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
    for ab in (1.5, 2.0, 10):
        np.testing.assert_array_equal(app.evaluate(ab, [0], schedules)[0], physical)
        assert app.EQD2_FN(ab, 0, 3, 10) == 30
        near = 1e-12
        np.testing.assert_allclose(app.evaluate(ab, [near], schedules)[0], physical, atol=1e-9)
        boundary = app.analyze(ab, 0, 0, 0.1, schedules)
        detail = app.build_analysis({"name": "Linear boundary", "result": boundary})
        assert detail["r"] == 0 and detail["s"] is None
        assert all(p["RD_BED_Gy"] == p["D_Gy"] for p in detail["prescriptions"])
        app.export_analysis_tables(detail, folder / f"boundary-{ab}", "json")
        app.draw_dispersion_plot(Figure(), detail)
    values = app.evaluate(1.5, [-1, 0, 1], schedules)
    assert np.isfinite(values).all()
    for invalid in (0, -1.5, float("nan"), float("inf"), -float("inf")):
        try:
            app.validated_alpha_beta(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid alpha/beta was accepted")

    # Construct isoeffective schedules using independent scalar inversion.
    ab, q = 5.0, -3.0
    selected_ab = -ab
    rate = (-1 - mp.sqrt(1 + 2 * selected_ab / q)) / selected_ab

    def effect(dose):
        return dose - q * (1 - mp.exp(-rate * dose))

    synthetic = []
    for n in (10, 20, 30, 39, 50):
        dose = float(mp.findroot(lambda d, n=n: n * effect(d) - 39 * effect(2), (0, 20)))
        synthetic.append((f"Schedule {n}", n * dose, n))
    for method in app.OPTIMIZATION_METHODS:
        fitted = app.analyze(ab, -20, 20, 0.02, synthetic, method=method)
        np.testing.assert_allclose(fitted["q"][fitted["idx"]], q, atol=2e-6, rtol=0)
        assert fitted["ab"] == ab and fitted["signed_ab"] == -ab
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
            {"name": "Second", "alpha_beta": 5, "prescriptions": schedules},
        ],
        -20,
        20,
        0.05,
    )
    assert len(mixed["sets"]) == 2
    app.export(mixed, folder / "mixed-signs")
    crossing = app._profile_bounds(5, -3, synthetic, 0.1, lower=-20, upper=0)
    assert crossing[0] < -3 < crossing[1]
    for q_crossing in crossing:
        np.testing.assert_allclose(
            app.evaluate(5, [q_crossing], synthetic)[0].std(ddof=1),
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
    assert store.get_set(7)["prescriptions"] == app.validate_schedules(
        [("Reference", 78, 39), ("Other", 30, 5)]
    )
    store.update_alpha_beta(7, 5)
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
    """Check direct q-selected signs and the 15-prescription NSCLC set."""
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
    for ab in (10, 20):
        zero = app.analyze(ab, 0, 0, 0.1, nsclc)
        assert zero["signed_ab"] == ab and zero["q"].tolist() == [0]

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


def check_signed_reconstruction(app):
    """Recover classical descriptors from primitive r,s,k, retaining their signs."""
    import numpy as np

    for r in (-3.0, -0.8, -0.01, 0.2, 0.8):
        for s in (0.05, 0.3, 2.0):
            for k in (0.01, 1.0, 4.0):
                d0 = 1 / k
                alpha = k * (1 - r)
                beta = k * r * s / 2
                q = r / s
                actual_ab = alpha / beta
                input_ab = actual_ab if q > 0 else -actual_ab
                branch_ab, recovered_r, recovered_s = app._signed_rd_parameters(input_ab, [q])
                np.testing.assert_allclose(
                    [branch_ab[0], recovered_r[0], recovered_s[0]], [actual_ab, r, s], rtol=1e-12
                )
                direct_r = (np.sqrt(q * q + 2 * q * actual_ab) - q) / actual_ab
                np.testing.assert_allclose(recovered_r[0], direct_r, rtol=1e-12)
                np.testing.assert_allclose(recovered_s[0], direct_r / q, rtol=1e-12)
                np.testing.assert_allclose((1 - alpha * d0) ** 2 / (2 * q * d0), beta, rtol=1e-12)
                assert d0 > 0 and alpha > 0 and recovered_s[0] > 0
                if q < 0:
                    assert branch_ab[0] < 0 and recovered_r[0] < 0 and beta < 0
                else:
                    assert branch_ab[0] > 0 and 0 < recovered_r[0] < 1 and beta > 0
                doses = np.array([0.01, 2.0, 8.0])
                hazard = k * (doses + q * np.expm1(-s * doses))
                np.testing.assert_allclose(
                    app.fraction_bed(input_ab, [q], doses)[0], hazard / alpha, rtol=1e-12
                )
                expected_eqd2 = 10 * hazard / hazard[1]
                np.testing.assert_allclose(
                    app.EQD2_FN(input_ab, q, doses, 5), expected_eqd2, rtol=1e-12
                )
    _, rates = app.rd_parameters(10, [-2, 2])
    assert rates[0] > rates[1] > 0


def check_reference_prescriptions(app, folder):
    """Persist overrides and resolve automatic references outside the fitted subset."""
    import sqlite3

    import numpy as np

    rows = app.validate_schedules(
        [("Conventional", 78, 39), ("Other conventional", 60, 30), ("SBRT", 40, 5)]
    )
    store = app.PrescriptionStore(folder / "references.sqlite3")
    key = store.save_set("References", 1.5, rows)
    record = store.get_set(key)
    assert record["reference_prescription"] is None
    assert app.reference_schedule(record) == rows[0]
    state = app.PrescriptionsSelection()
    state.sync([record])
    state.set_included(record, [0], False)
    configured = state.configured_set(record)
    fitted = app.analyze_selected([configured], 0.01, 10, 0.1)["sets"][0]
    detail = app.build_analysis(fitted)
    assert detail["reference"] == 78 and detail["reference_prescription"] == rows[0]
    assert detail["models"]["RD"]["N"] == 2

    store.save_set("References", 1.5, rows, set_id=key, reference_prescription=rows[2])
    reopened = app.PrescriptionStore(store.path)
    record = reopened.get_set(key)
    assert record["reference_prescription"] == rows[2]
    state.sync([record])
    state.set_included(record, [0, 1], True)
    state.set_included(record, [2], False)
    for start, stop in ((0.01, 10), (-10, -0.01), (0, 0)):
        fitted = app.analyze_selected([state.configured_set(record)], start, stop, 0.1)["sets"][0]
        detail = app.build_analysis(fitted)
        expected = app.EQD2_FN(1.5, detail["D_q"], 8, 5)
        np.testing.assert_allclose(detail["reference"], expected, rtol=1e-13)
        assert detail["reference_prescription"] == rows[2]
        assert len(detail["prescriptions"]) == 2
    csv_path = folder / "reference-set.csv"
    app.export_prescriptions_set(record, csv_path)
    imported = app.read_prescriptions_set_csv(csv_path)
    assert imported["reference_prescription"] == rows[2]
    assert imported["prescriptions"] == record["prescriptions"]
    with csv_path.open(encoding="utf-8-sig") as stream:
        csv_rows = list(csv.DictReader(stream))
    csv_rows[0]["is_reference"] = "1"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=csv_rows[0])
        writer.writeheader()
        writer.writerows(csv_rows)
    try:
        app.read_prescriptions_set_csv(csv_path)
    except ValueError:
        pass
    else:
        raise AssertionError("CSV accepted two reference prescriptions")

    edited = [*rows[:2], app.Prescription("Edited SBRT", 38, 4)]
    store.save_set("References", 1.5, edited, set_id=key, reference_prescription=edited[2])
    assert store.get_set(key)["reference_prescription"] == edited[2]
    store.save_set("References", 1.5, edited, set_id=key)
    assert app.reference_schedule(store.get_set(key)) == rows[0]
    try:
        store.save_set(
            "References", 1.5, rows, set_id=key, reference_prescription=("Missing", 20, 10)
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Accepted a reference outside the saved prescriptions")
    assert store.get_set(key)["prescriptions"] == edited
    no_reference = {"prescriptions": [("A", 60, 20), ("B", 40, 5)]}
    assert app.reference_schedule(no_reference) is None
    fitted = app.analyze(1.5, 0.01, 10, 0.1, no_reference["prescriptions"])
    assert app.build_analysis({"result": fitted})["reference"] is None

    # Opening an older database must add the flag without changing saved data.
    old_path = folder / "legacy-references.sqlite3"
    with sqlite3.connect(old_path) as con:
        con.executescript("""
            CREATE TABLE tumors(id INTEGER PRIMARY KEY, name TEXT, alpha_beta REAL);
            CREATE TABLE prescriptions(id INTEGER PRIMARY KEY, tumor_id INTEGER,
                name TEXT, total_gy REAL, fractions INTEGER, position INTEGER);
            INSERT INTO tumors VALUES(1, 'Legacy', 1.5);
            INSERT INTO prescriptions VALUES(11, 1, 'Conventional', 78, 39, 0);
            INSERT INTO prescriptions VALUES(12, 1, 'SBRT', 40, 5, 1);
        """)
    migrated = app.PrescriptionStore(old_path)
    assert app.reference_schedule(migrated.get_set(1)) == rows[0]
    migrated = app.PrescriptionStore(old_path)
    with migrated._connection() as con:
        assert [r[0] for r in con.execute("SELECT id FROM prescriptions ORDER BY id")] == [11, 12]


def check_prostate_preset_upgrade(app, folder):
    previous_name = "Prostate — 8 prescriptions"
    previous_rows = [
        ("Shared conventional reference", 78.0, 39),
        ("RTOG 0126 IMRT", 79.2, 44),
        ("RTOG 0415 hypofractionated", 70.0, 28),
        ("PROFIT hypofractionated", 60.0, 20),
        ("PACE-B control", 62.0, 20),
        ("Mantz SBRT", 40.0, 5),
        ("HYPO-RT-PC ultrahypofractionated", 42.7, 7),
        ("Fuller SBRT", 38.0, 4),
    ]
    previous_defaults = {previous_name: {"prescriptions": previous_rows, "alpha_beta": 1.5}}
    path = folder / "upgrade-prostate.sqlite3"
    previous = app.PrescriptionStore(path, previous_defaults)
    set_id = previous.find_set(previous_name)["id"]
    previous.save_set(
        previous_name, 1.8, previous_rows, set_id=set_id, reference_prescription=previous_rows[-1]
    )
    current = app.PrescriptionStore(path, app.DEFAULT_SETS)
    assert current.get_set(set_id) == {
        "id": set_id,
        "name": app.PROSTATE_SET_NAME,
        "prescriptions": app.PROSTATE_SET["prescriptions"],
        "alpha_beta": 1.8,
        "reference_prescription": app.PROSTATE_SET["prescriptions"][-1],
    }
    assert len(app.PrescriptionStore(path, app.DEFAULT_SETS).list_sets()) == 1

    path = folder / "upgrade-customized-prostate.sqlite3"
    previous = app.PrescriptionStore(path, previous_defaults)
    set_id = previous.find_set(previous_name)["id"]
    previous.save_set(previous_name, 1.5, previous_rows[:-1], set_id=set_id)
    current = app.PrescriptionStore(path, app.DEFAULT_SETS)
    assert current.get_set(set_id) == previous.get_set(set_id)
    assert current.find_set(app.PROSTATE_SET_NAME)["prescriptions"] == app.PROSTATE_SET["prescriptions"]
    assert len(app.PrescriptionStore(path, app.DEFAULT_SETS).list_sets()) == 2

    path = folder / "upgrade-deleted-prostate.sqlite3"
    previous = app.PrescriptionStore(path, previous_defaults)
    previous.delete_set(previous.find_set(previous_name)["id"])
    assert app.PrescriptionStore(path, app.DEFAULT_SETS).list_sets() == []


def check_prescription_model(app, folder):
    """Outcomes belong to individual rows through edits, reordering, and analysis."""
    from dataclasses import replace

    rows = [app.Prescription("A", "78", "39", "0%"),
            app.Prescription("B", 38, 4, 100), app.Prescription("C", 60, 20)]
    assert rows[0].total_gy == 78.0 and rows[0].fractions == 39
    assert rows[0].dose_per_fraction_gy == 2 and rows[0].local_control_pct == 0
    assert rows[2].local_control_pct is None
    for values in (("", 78, 39), ("A", 0, 39), ("A", 78, 1.5),
                   ("A", 78, 39, -1), ("A", 78, 39, 101), ("A", 78, 39, "NaN")):
        try:
            app.Prescription(*values)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid prescription state was accepted")
    assert app.validate_schedules([("A", 78, 39, 0), ("B", 38, 4, 100)]) == rows[:2]
    assert app.validate_schedules([("A", 78, 39), ("B", 38, 4)])[0].local_control_pct is None
    store = app.PrescriptionStore(folder / "prescription-model.sqlite3")
    key = store.save_set("Model", 1.5, rows, reference_prescription=rows[0])
    record = store.get_set(key)
    assert "local_control_pct" not in record
    assert record["prescriptions"] == rows
    selection = app.PrescriptionsSelection()
    selection.sync([record])
    selection.set_included(record, [0], False)
    edited = [replace(rows[0], local_control_pct=12.5),
              replace(rows[2], name="Renamed C"), replace(rows[1], name="Renamed B")]
    store.save_set("Model", 1.5, edited, set_id=key, reference_prescription=rows[0])
    record = store.get_set(key)
    selection.sync([record])
    assert not selection.is_included(record, 0), "Outcome edits changed a row's selection"
    assert record["reference_prescription"] == edited[0]
    assert [p.local_control_pct for p in record["prescriptions"]] == [12.5, None, 100]
    subset = selection.configured_set(record)
    assert [p.local_control_pct for p in subset["prescriptions"]] == [None, 100]
    fit = app.analyze_selected([subset], 0.01, 10, 0.1)["sets"][0]
    assert "local_control_pct" not in fit["result"]
    observations = app.build_analysis(fit)["clinical"]["observations"]
    assert len(observations) == 1 and observations[0]["prescription"] == "Renamed B"
    path = app.export_prescriptions_set(record, folder / "model.csv")
    assert app.read_prescriptions_set_csv(path)["prescriptions"] == edited
    # Clinical data must not influence the physical-dose calibration.
    with_outcomes = app.analyze(1.5, 0.01, 10, 0.1, edited)
    without_outcomes = app.analyze(1.5, 0.01, 10, 0.1,
                                   [replace(p, local_control_pct=None) for p in edited])
    import numpy as np
    np.testing.assert_array_equal(with_outcomes["eqd2"], without_outcomes["eqd2"])
    assert with_outcomes["q"][with_outcomes["idx"]] == without_outcomes["q"][without_outcomes["idx"]]
    app.export(fit["result"], folder / "model-export")
    with (folder / "model-export" / "optimal_schedules.csv").open(encoding="utf-8-sig") as fp:
        exported = list(csv.DictReader(fp))
    assert exported[-1]["local_control_pct"] == "100.0"


def check_prescription_plot_labels(app, detail):
    """Labels remain inside the drawing area at screen and export sizes."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    endpoints = [app.Prescription("Zero", 52, 26, 0),
                 app.Prescription("Full", 60, 30, 100)]
    endpoint_detail = app.build_analysis({"name": "Endpoints", "result":
        app.analyze(1.5, .01, 10, .1, endpoints)})
    expected = {f"{p['D_Gy']:g}/{p['n']}" for p in detail["prescriptions"]}
    for size, dpi in (((8, 7.2), 100), ((6, 4.5), 144)):
        for kind in ("dispersion", "clinical", "endpoints"):
            fig = Figure(figsize=size, dpi=dpi)
            FigureCanvasAgg(fig)
            if kind == "dispersion":
                app.draw_dispersion_plot(fig, detail)
            else:
                clinical = (detail if kind == "clinical" else endpoint_detail)["clinical"]
                app.draw_clinical_plot(fig, clinical, show_summary=False)
            fig.canvas.draw()
            ax = fig.axes[0]
            bounds = ax.get_window_extent()
            labels = [text for text in ax.texts if text.get_gid() == "prescription-label"]
            assert labels
            if kind == "dispersion":
                assert {label.get_text() for label in labels} == expected
            else:
                assert len(labels) == len(clinical["observations"])
            for label in labels:
                box = label.get_bbox_patch().get_window_extent(fig.canvas.get_renderer())
                assert bounds.contains(box.x0, box.y0) and bounds.contains(box.x1, box.y1), (
                    kind, label.get_text(), bounds.bounds, box.bounds
                )
                assert label.get_rotation() == 0 and label.get_clip_on()
            fig.clear()
    for show_lq in (True, False):
        fig = Figure()
        app.draw_dispersion_plot(fig, detail, show_lq=show_lq, show_labels=False)
        assert not any(t.get_gid() == "prescription-label" for t in fig.axes[0].texts)
        app.draw_clinical_plot(fig, detail["clinical"], show_labels=False)
        assert not any(t.get_gid() == "prescription-label" for t in fig.axes[0].texts)
    fig = Figure()
    app.draw_dispersion_plot(fig, detail, show_lq=False)
    assert len([t for t in fig.axes[0].texts if t.get_gid() == "prescription-label"]) == len(expected)


def check_clinical_evaluation(app, folder):
    from dataclasses import replace
    import numpy as np
    from matplotlib.figure import Figure

    old_rows = [replace(p, local_control_pct=None) for p in app.PROSTATE_SET["prescriptions"]]
    old_defaults = {app.PROSTATE_SET_NAME: {"alpha_beta": 1.5, "prescriptions": old_rows}}
    path = folder / "clinical-default-upgrade.sqlite3"
    previous = app.PrescriptionStore(path, old_defaults)
    record = previous.find_set(app.PROSTATE_SET_NAME)
    changed = [*old_rows[:-1], replace(old_rows[-1], local_control_pct=98.5)]
    previous.save_set(record["name"], 1.8, changed, set_id=record["id"],
                      reference_prescription=changed[-1])
    current = app.PrescriptionStore(path, app.DEFAULT_SETS)
    record = current.find_set(app.PROSTATE_SET_NAME)
    assert record["prescriptions"] == [*app.PROSTATE_SET["prescriptions"][:-1],
                                      replace(app.PROSTATE_SET["prescriptions"][-1], local_control_pct=98.5)]
    assert record["alpha_beta"] == 1.8
    assert record["reference_prescription"] == record["prescriptions"][-1]
    current.save_set(record["name"], 1.8, old_rows, set_id=record["id"])
    assert all(p.local_control_pct is None for p in
               app.PrescriptionStore(path, app.DEFAULT_SETS).get_set(record["id"])["prescriptions"])
    path = folder / "clinical-customized-default.sqlite3"
    previous = app.PrescriptionStore(path, old_defaults)
    record = previous.find_set(app.PROSTATE_SET_NAME)
    previous.save_set(record["name"], 1.5, old_rows[:-1], set_id=record["id"])
    assert all(p.local_control_pct is None for p in
               app.PrescriptionStore(path, app.DEFAULT_SETS).get_set(record["id"])["prescriptions"])

    store = app.PrescriptionStore(folder / "clinical.sqlite3")
    grouped_rows = [app.Prescription(f"Example {i}", 2 * n, n, observed)
                    for i, (n, observed) in enumerate(((18, 38), (20, 79), (29, 74),
                                                       (30, 81), (31, 76), (32, 85)))]
    group_id = store.save_set("Grouped examples", 1.5, grouped_rows)
    record = store.get_set(group_id)
    fit = app.analyze_selected([record], 0.01, 50, 0.01)["sets"][0]
    detail = app.build_analysis(fit)
    clinical = detail["clinical"]
    np.testing.assert_allclose(clinical["mean"], [320 / 6, 433 / 6])
    assert 0 < clinical["linear"]["r_squared"] < 1
    assert len(clinical["observations"]) == 6
    cluster = clinical["cluster"]
    np.testing.assert_array_equal(np.sort(cluster["indices"]), [2, 3, 4, 5])
    np.testing.assert_allclose(cluster["maximum"] - cluster["minimum"], [6, 11])
    np.testing.assert_allclose(cluster["area_percent"], 100)
    mean, lower, upper = app._beta_control_prediction(clinical["beta"], clinical["mean"][0])
    assert 0 < lower < mean < upper < 100
    selection = app.PrescriptionsSelection()
    selection.sync([record])
    selection.set_included(record, [0], False)
    subset = selection.configured_set(record)
    result = app.analyze_selected([subset], 0.01, 50, 0.01)["sets"][0]
    assert len(app.build_analysis(result)["clinical"]["observations"]) == 5

    rows = [app.Prescription("A", 52, 26, 0), app.Prescription("B", 60, 30, 25),
            app.Prescription("C", 70, 35, 75), app.Prescription("D", 85, 42, 100),
            app.Prescription("E", 78, 39)]
    set_id = store.save_set("Observed", 1.5, rows)
    store.update_alpha_beta(set_id, 2)
    store.save_set("Observed", 2, rows, set_id=set_id)
    record = store.get_set(set_id)
    assert record["prescriptions"] == rows
    csv_path = app.export_prescriptions_set(record, folder / "clinical-observations.csv")
    assert app.read_prescriptions_set_csv(csv_path)["prescriptions"] == rows
    for invalid in (-0.01, 100.01, "NaN", "inf", "bad"):
        try:
            store.save_set("Observed", 2, [replace(rows[0], local_control_pct=invalid), *rows[1:]],
                           set_id=set_id)
        except ValueError:
            pass
        else:
            raise AssertionError("Accepted invalid local tumor control")
    assert store.get_set(set_id) == record
    result = app.analyze_selected([record], 0.01, 50, 0.01)["sets"][0]
    clinical = app.build_analysis(result)["clinical"]
    np.testing.assert_array_equal(clinical["points"][:, 1], [0, 25, 75, 100])
    assert clinical["cluster"] is None and clinical["beta"]["adjusted_endpoints"]
    mean, low, high = app._beta_control_prediction(clinical["beta"], clinical["points"][:, 0])
    assert np.all((0 <= low) & (low <= mean) & (mean <= high) & (high <= 100))
    fig = Figure()
    app.draw_clinical_plot(fig, clinical)
    assert len(fig.axes[0].patches) == 0
    png = folder / "clinical.png"
    app.export_clinical_plot(detail["clinical"], png, ppi=72, show_interval=True)
    assert png.read_bytes().startswith(b"\x89PNG")
    cleared = [replace(p, local_control_pct=None) for p in rows]
    store.save_set("Observed", 2, cleared, set_id=set_id)
    assert store.get_set(set_id)["prescriptions"] == cleared
    result = app.analyze_selected([store.get_set(set_id)], 0.01, 50, 0.01)["sets"][0]
    assert app.build_analysis(result)["clinical"] is None


def check_calculation_progress(app):
    """Progress preserves numerical results and cancellation leaves no partial result."""
    import numpy as np

    rows = app.PROSTATE_SET["prescriptions"]
    baseline = app.analyze(1.5, -.5, 10, .01, rows)
    updates = []

    def report(fraction, message):
        assert 0 <= fraction <= 1 and message
        if updates:
            assert fraction >= updates[-1][0], (fraction, updates[-1])
        updates.append((fraction, message))

    actual = app.analyze(1.5, -.5, 10, .01, rows, progress=report)
    np.testing.assert_array_equal(actual["q"], baseline["q"])
    np.testing.assert_allclose(actual["eqd2"], baseline["eqd2"], rtol=1e-13)
    assert actual["idx"] == baseline["idx"] and updates[-1][0] == 1
    updates.clear()
    detail = app.build_analysis({"name": "Progress", "result": actual}, progress=report)
    assert updates[-1][0] == 1 and len(detail["clinical"]["observations"]) == 6

    def cancel_when(message_part):
        def cancel(fraction, message):
            if message_part in message:
                raise app.CalculationCancelled("Requested by the user")
        return cancel

    for calculate in (
        lambda: app.analyze(1.5, -50, 50, .001, rows,
                            progress=cancel_when("Evaluating candidate doses")),
        lambda: app.build_analysis({"name": "Cancelled", "result": actual},
                                   progress=cancel_when("Fitting beta regression")),
    ):
        try:
            calculate()
        except app.CalculationCancelled:
            pass
        else:
            raise AssertionError("Calculation ignored cancellation")
    sentinel = {"previous": "completed robustness"}
    detail["robustness"] = sentinel
    try:
        app.run_robustness(detail, progress=cancel_when("Leave-one-out"))
    except app.CalculationCancelled:
        pass
    else:
        raise AssertionError("Robustness ignored cancellation")
    assert detail["robustness"] is sentinel
    updates.clear()
    actual_robustness = app.run_robustness(detail, [1.5], progress=report)
    assert updates[-1][0] == 1 and len(actual_robustness["leave_one_out"]) == 10


def check_progress_dialog(app, root):
    import threading
    import tkinter as tk
    from tkinter import ttk

    main_thread = threading.get_ident()
    previous_close = root.protocol("WM_DELETE_WINDOW")
    errors = []

    def children(widget):
        for child in widget.winfo_children():
            yield child
            yield from children(child)

    # A controlled worker lets the event loop demonstrate progress and modality.
    gate = threading.Event()
    worker_ids = []

    def work(report):
        worker_ids.append(threading.get_ident())
        report(.42, "Evaluating test prescriptions")
        if not gate.wait(5):
            raise AssertionError("The modal dialog blocked the Tk event loop")
        return "finished"

    def observe(cancel=False):
        try:
            dialog = root.grab_current()
            assert isinstance(dialog, tk.Toplevel) and root._calculation_active
            bar = next(w for w in children(dialog) if isinstance(w, ttk.Progressbar))
            assert float(bar.cget("value")) == 42
            labels = [str(root.getvar(w.cget("textvariable"))) for w in children(dialog)
                      if isinstance(w, ttk.Label) and w.cget("textvariable")]
            assert "42%" in labels
            root.tk.call(root.protocol("WM_DELETE_WINDOW"))
            assert root.winfo_exists()
            try:
                assert root.attributes("-disabled")
            except tk.TclError:
                pass
            if cancel:
                next(w for w in children(dialog) if isinstance(w, ttk.Button)
                     and w.cget("text") == "Cancel").invoke()
        except Exception as exc:
            errors.append(str(exc))
        finally:
            gate.set()

    root.after(100, observe)
    assert app.run_with_progress(root, "Test progress", work) == "finished"
    assert worker_ids[0] != main_thread and not errors, errors
    assert not root._calculation_active and root.grab_current() is None
    assert root.protocol("WM_DELETE_WINDOW") == previous_close

    gate.clear()
    root.after(100, lambda: observe(cancel=True))
    try:
        app.run_with_progress(root, "Test cancellation", work)
    except app.CalculationCancelled:
        pass
    else:
        raise AssertionError("Cancel applied a worker's partial result")
    assert not errors and not root._calculation_active, errors
    assert root.protocol("WM_DELETE_WINDOW") == previous_close

    owner = tk.Toplevel(root)
    owner.update()
    owner.grab_set()
    try:
        def fail(report):
            raise ValueError("Test failure")
        try:
            app.run_with_progress(owner, "Test failure", fail)
        except ValueError as exc:
            assert str(exc) == "Test failure"
        else:
            raise AssertionError("Worker error was swallowed")
        assert root.grab_current() is owner and not root._calculation_active
    finally:
        owner.destroy()


def check_gui(app, db_path, result):
    import tkinter as tk
    import traceback
    from tkinter import filedialog, messagebox, ttk

    original_tk = tk.Tk
    original_error = messagebox.showerror
    original_save = filedialog.asksaveasfilename
    failures = []
    completed = []
    # A user-created fixture tests switching sets without adding another default.
    fixtures = app.PrescriptionStore(db_path, app.DEFAULT_SETS)
    fixture_rows = [(f"Schedule {i}", 20 + i, 10 + i) for i in range(15)]
    fixtures.save_set("Verification — saved set", 2.0, fixture_rows)

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
                from unittest.mock import patch

                root.update()
                check_progress_dialog(app, root)

                def button(text, parent=None):
                    from types import SimpleNamespace

                    owner = parent or root
                    for widget in children(owner):
                        if isinstance(widget, ttk.Button) and str(widget.cget("text")) == text:
                            return widget
                    # Secondary actions are available through the desktop menus.
                    for menu in children(owner):
                        if not isinstance(menu, tk.Menu) or menu.index("end") is None:
                            continue
                        for index in range(menu.index("end") + 1):
                            if menu.type(index) == "command" and menu.entrycget(index, "label") == text:
                                return SimpleNamespace(invoke=lambda m=menu, i=index: m.invoke(i))
                    raise AssertionError("Action not found: " + text)

                notebook = next(
                    w
                    for w in children(root)
                    if isinstance(w, ttk.Notebook)
                    and isinstance(w.master, ttk.LabelFrame)
                    and w.master.cget("text") == "Prescription details"
                )

                workspace_tabs = notebook.master.master
                assert [workspace_tabs.tab(page, "text") for page in workspace_tabs.tabs()] == [
                    "Prescriptions", "Results"
                ]

                def menu_item(label, parent=root):
                    for menu in children(parent):
                        if not isinstance(menu, tk.Menu) or menu.index("end") is None:
                            continue
                        for index in range(menu.index("end") + 1):
                            if menu.type(index) == "command" and menu.entrycget(index, "label") == label:
                                return menu, index
                    raise AssertionError("Menu action missing: " + label)

                def menu_state(label, parent=root):
                    menu, index = menu_item(label, parent)
                    return menu.entrycget(index, "state")

                def view(name):
                    workspace_tabs.select(notebook.master)
                    page = next(tab for tab in notebook.tabs() if notebook.tab(tab, "text") == name)
                    notebook.select(page)
                    notebook.event_generate("<<NotebookTabChanged>>")
                    frame = root.nametowidget(page)
                    tree = next(w for w in children(frame) if isinstance(w, ttk.Treeview))
                    ab_entry = next(
                        w
                        for w in children(frame)
                        if isinstance(w, ttk.Entry) and w.master == button("Save α/β", frame).master
                    )
                    return frame, tree, ab_entry

                def set_checkbox(name):
                    return next(
                        w
                        for w in children(root)
                        if isinstance(w, ttk.Checkbutton)
                        and w.cget("textvariable")
                        and str(root.getvar(w.cget("textvariable"))).split("\n")[0] == name
                    )

                results = next(
                    w
                    for w in children(root)
                    if isinstance(w, ttk.Treeview) and "dq" in w.cget("columns")
                )
                step_entry = next(
                    w
                    for w in children(root)
                    if isinstance(w, ttk.Entry)
                    and isinstance(w.master, ttk.LabelFrame)
                    and w.master.cget("text") == "Optimization settings"
                    and w.grid_info().get("row") == 5
                )
                root.setvar(step_entry.cget("textvariable"), "0.05")
                optimize = button("Run optimization")
                root.update_idletasks()
                menubar = root.nametowidget(root.cget("menu"))
                assert [
                    menubar.entrycget(i, "label") for i in range(menubar.index("end") + 1)
                    if menubar.type(i) == "cascade"
                ] == ["File", "Prescriptions", "Analysis", "View", "Help"]
                assert button("Analysis…").instate(["disabled"])
                assert menu_state("Analysis…") == "disabled"
                assert optimize.instate(["!disabled"])
                assert menu_state("Run optimization") == "normal"
                file_menu = root.nametowidget(menubar.entrycget("File", "menu"))
                set_menu = root.nametowidget(menubar.entrycget("Prescriptions", "menu"))
                for label in ("New prescriptions set", "Import prescriptions set…", "Export prescriptions set…"):
                    menu, _index = menu_item(label)
                    assert menu is set_menu
                assert not any(
                    file_menu.type(i) == "command" and "prescriptions" in file_menu.entrycget(i, "label")
                    for i in range(file_menu.index("end") + 1)
                )
                left_panes = next(
                    w for w in children(root) if isinstance(w, ttk.Panedwindow)
                    and len(w.panes()) == 2
                    and isinstance(root.nametowidget(w.panes()[0]), ttk.LabelFrame)
                    and root.nametowidget(w.panes()[0]).cget("text") == "Prescription sets"
                )
                left_target = int(left_panes.winfo_height() * 0.45)
                left_panes.sashpos(0, left_target)
                root.update_idletasks()
                assert abs(left_panes.sashpos(0) - left_target) < 4, (
                    left_panes.sashpos(0), left_target, left_panes.winfo_height()
                )
                management = [button(text) for text in ("New set", "Edit set", "Import…", "Export…")]
                assert len({(w.winfo_width(), w.winfo_height()) for w in management}) == 1
                result_tabs = next(
                    w for w in children(root) if isinstance(w, ttk.Notebook)
                    and any(w.tab(page, "text") == "Optimization profile" for page in w.tabs())
                )
                assert all(
                    result_tabs.tab(page, "state") == "disabled"
                    for page in result_tabs.tabs()
                    if result_tabs.tab(page, "text") != "Optimization profile"
                )
                assert len(notebook.tabs()) == 1
                frame, prescriptions, ab_entry = view(app.PROSTATE_SET_NAME)
                original_count = len(app.PROSTATE_SET["prescriptions"])
                root.update_idletasks()
                x, y, width, height = prescriptions.bbox("0", "#1")
                prescriptions.event_generate("<Button-1>", x=x + width // 2, y=y + height // 2)
                assert prescriptions.set("0", "include") == "☐"
                root.setvar(ab_entry.cget("textvariable"), "2.2")
                optimize.invoke()
                row = results.item(results.get_children()[0], "values")
                assert int(row[1]) == original_count - 1 and float(row[2]) == 2.2
                assert button("Analysis…").instate(["!disabled"])
                assert menu_state("Analysis…") == "normal"
                assert workspace_tabs.tab(workspace_tabs.select(), "text") == "Results"
                # Cancel actual optimization/clinical commands before they can replace a valid run.
                import threading
                saved_results = [results.item(item, "values") for item in results.get_children()]
                for target in ("analyze", "clinical_evaluation"):
                    entered = threading.Event()

                    def slow_calculation(*args, **kwargs):
                        report = kwargs.get("progress") if target == "analyze" else args[1]
                        entered.set()
                        for _ in range(500):
                            report(.25, "Testing cancellation")
                            threading.Event().wait(.01)
                        raise AssertionError("Cancellation was not delivered")

                    def cancel_running():
                        if not entered.is_set():
                            root.after(10, cancel_running)
                            return
                        owner = root.grab_current()
                        next(w for w in children(owner) if isinstance(w, ttk.Button)
                             and w.cget("text") == "Cancel").invoke()

                    root.after(10, cancel_running)
                    with patch.object(app, target, side_effect=slow_calculation):
                        optimize.invoke()
                    assert saved_results == [results.item(item, "values") for item in results.get_children()]
                    assert button("Analysis…").instate(["!disabled"])
                    assert not root._calculation_active and root.grab_current() is None
                optimized_selector = next(
                    w for w in children(root) if isinstance(w, ttk.Combobox)
                    and app.PROSTATE_SET_NAME in w.cget("values")
                )
                assert optimized_selector.get() == app.PROSTATE_SET_NAME
                # A failed run must never leave the previous analysis enabled.
                with patch.object(messagebox, "showerror") as invalid_input:
                    root.setvar(step_entry.cget("textvariable"), "invalid")
                    optimize.invoke()
                    assert invalid_input.called
                assert button("Analysis…").instate(["disabled"])
                assert menu_state("Analysis…") == "disabled"
                root.setvar(step_entry.cget("textvariable"), "0.05")
                optimize.invoke()
                assert [result_tabs.tab(page, "text") for page in result_tabs.tabs()] == [
                    "Optimization profile", "EQD2 dispersion", "Clinical evaluation", "Robustness"
                ]
                for page in result_tabs.tabs():
                    result_tabs.select(page)
                    root.update_idletasks()
                embedded_clinical = next(
                    page for page in result_tabs.tabs()
                    if result_tabs.tab(page, "text") == "Clinical evaluation"
                )
                result_tabs.select(embedded_clinical)
                clinical_subtabs = next(
                    w for w in children(root.nametowidget(embedded_clinical))
                    if isinstance(w, ttk.Notebook)
                )
                assert [clinical_subtabs.tab(page, "text") for page in clinical_subtabs.tabs()] == [
                    "Analysis", "Plot"
                ]
                root.update()
                assert menu_state("Save plot…") == "disabled"
                assert menu_state("Print plot…") == "disabled"
                clinical_subtabs.select(clinical_subtabs.tabs()[1])
                root.update()
                assert menu_state("Save plot…") == "normal"
                assert menu_state("Print plot…") == "normal"
                clinical_subtabs.select(clinical_subtabs.tabs()[0])
                root.update()
                assert menu_state("Save plot…") == "disabled"
                result_tabs.select(result_tabs.tabs()[0])
                for geometry in ("1100x780", "1440x1040"):
                    root.geometry(geometry)
                    root.update_idletasks()
                    assert optimize.winfo_rootx() + optimize.winfo_width() <= root.winfo_rootx() + root.winfo_width()
                    assert prescriptions.winfo_width() > 500
                    assert result_tabs.winfo_height() > 80, (
                        geometry, result_tabs.winfo_height(),
                        result_tabs.master.winfo_height(),
                        result_tabs.master.master.winfo_height(),
                    )
                root.geometry("1440x1040")

                other_name = "Verification — saved set"
                set_checkbox(other_name).invoke()
                assert len(notebook.tabs()) == 2
                other_frame, other_tree, other_ab = view(other_name)
                assert len(other_tree.get_children()) == 15 and float(other_ab.get()) == 2.0
                root.setvar(other_ab.cget("textvariable"), "3.1")
                root.update_idletasks()
                x, y, width, height = other_tree.bbox("0", "#1")
                other_tree.event_generate("<Button-1>", x=x + width // 2, y=y + height // 2)
                assert other_tree.set("0", "include") == "☐"
                optimize.invoke()
                rows_before = [results.item(item, "values") for item in results.get_children()]
                assert len(rows_before) == 2
                # Switching tabs must reuse widgets and avoid database reads or result invalidation.
                with patch.object(
                    app.PrescriptionStore,
                    "get_set",
                    side_effect=AssertionError("A tab switch loaded the database"),
                ):
                    for _ in range(3):
                        shown, tree, entry = view(app.PROSTATE_SET_NAME)
                        assert shown is frame and tree is prescriptions
                        assert float(entry.get()) == 2.2 and tree.set("0", "include") == "☐"
                        shown, tree, entry = view(other_name)
                        assert shown is other_frame and tree is other_tree
                        assert float(entry.get()) == 3.1 and tree.set("0", "include") == "☐"
                assert rows_before == [
                    results.item(item, "values") for item in results.get_children()
                ]

                set_checkbox(other_name).invoke()
                assert len(notebook.tabs()) == 1
                set_checkbox(other_name).invoke()
                _, other_tree, other_ab = view(other_name)
                assert float(other_ab.get()) == 3.1 and other_tree.set("0", "include") == "☐"
                button("Clear all sets").invoke()
                assert not notebook.tabs()
                assert optimize.instate(["disabled"])
                assert menu_state("Run optimization") == "disabled"
                assert button("Analysis…").instate(["disabled"])
                assert menu_state("Analysis…") == "disabled"
                # Keyboard callbacks share the same selection guard as menus and buttons.
                root.event_generate("<Control-r>")
                root.update_idletasks()
                assert not results.get_children()
                button("Select all sets").invoke()
                assert optimize.instate(["!disabled"])
                assert menu_state("Run optimization") == "normal"
                assert len(notebook.tabs()) == 2
                _, other_tree, other_ab = view(other_name)
                assert float(other_ab.get()) == 3.1 and other_tree.set("0", "include") == "☐"
                set_checkbox(other_name).invoke()
                frame, prescriptions, ab_entry = view(app.PROSTATE_SET_NAME)
                assert float(ab_entry.get()) == 2.2 and prescriptions.set("0", "include") == "☐"

                button("Save α/β", frame).invoke()
                saved = app.PrescriptionStore(db_path).find_set(app.PROSTATE_SET_NAME)
                assert saved["alpha_beta"] == 2.2 and len(saved["prescriptions"]) == original_count
                export_path = Path(db_path).with_name("gui-export.csv")
                filedialog.asksaveasfilename = lambda **_kwargs: str(export_path)
                button("Export prescriptions set…").invoke()
                imported = app.read_prescriptions_set_csv(export_path)
                assert len(imported["prescriptions"]) == original_count - 1
                assert imported["alpha_beta"] == 2.2
                assert imported["prescriptions"] == app.PROSTATE_SET["prescriptions"][1:]
                optimize.invoke()
                details = app.open_analysis_window(root, result)
                assert details is not None and details.winfo_exists(), (
                    "Analysis window did not open"
                )
                for text in ("Plot settings…", "Export image…"):
                    button(text).invoke()
                root.setvar(ab_entry.cget("textvariable"), "2")
                assert button("Analysis…").instate(["disabled"])
                assert menu_state("Analysis…") == "disabled"
                optimize.invoke()
                row = results.item(results.get_children()[0], "values")
                assert float(row[2]) == 2
                button("Save α/β", frame).invoke()
                assert (
                    app.PrescriptionStore(db_path).find_set(app.PROSTATE_SET_NAME)["alpha_beta"]
                    == 2
                )
                optimize.invoke()
                button("Analysis…").invoke()
                root.update_idletasks()
                # Close earlier dialogs before exercising the reference editor.
                for child in root.winfo_children():
                    if isinstance(child, tk.Toplevel):
                        child.destroy()

                def dialog(title):
                    return next(
                        w
                        for w in children(root)
                        if isinstance(w, tk.Toplevel) and w.title() == title
                    )

                button("Edit prescriptions set").invoke()
                editor = dialog("Edit prescriptions set")
                radios = [w for w in children(editor) if isinstance(w, ttk.Radiobutton)]
                assert root.getvar(radios[0].cget("variable")) == "auto"
                control_entries = [
                    w for w in children(editor)
                    if isinstance(w, tk.Entry) and w.grid_info().get("column") == 4
                ]
                assert [entry.get() for entry in control_entries] == [
                    "" if p.local_control_pct is None else str(p.local_control_pct)
                    for p in app.PROSTATE_SET["prescriptions"]
                ]
                root.setvar(control_entries[-1].cget("textvariable"), "53.25")
                radios[-1].invoke()
                button("Save prescriptions set", editor).invoke()
                saved = app.PrescriptionStore(db_path).find_set(app.PROSTATE_SET_NAME)
                assert saved["reference_prescription"].schedule_key == app.PROSTATE_SET["prescriptions"][-1].schedule_key
                assert saved["prescriptions"][-1].local_control_pct == 53.25
                _, prescriptions, _ = view(app.PROSTATE_SET_NAME)
                assert prescriptions.set(str(original_count - 1), "reference") == "Yes"
                optimize.invoke()
                button("Analysis…").invoke()
                analysis = dialog("RD EQD2 — Detailed analysis")
                analysis_tabs = next(w for w in children(analysis) if isinstance(w, ttk.Notebook))
                clinical_tab = next(
                    page for page in analysis_tabs.tabs()
                    if analysis_tabs.tab(page, "text") == "Clinical evaluation"
                )
                analysis_tabs.select(clinical_tab)
                root.update()
                clinical_widgets = list(children(root.nametowidget(clinical_tab)))
                clinical_text = next(w for w in clinical_widgets if isinstance(w, tk.Text))
                assert "Mean local control" in clinical_text.get("1.0", "end")
                assert "Beta regression" in clinical_text.get("1.0", "end")
                clinical_subtabs = next(w for w in clinical_widgets if isinstance(w, ttk.Notebook))
                assert [clinical_subtabs.tab(page, "text") for page in clinical_subtabs.tabs()] == [
                    "Analysis", "Plot"
                ]
                assert clinical_text.winfo_ismapped()
                assert not any(isinstance(w, ttk.Panedwindow) for w in clinical_widgets)
                assert menu_state("Save plot…", analysis) == "disabled"
                assert menu_state("Print plot…", analysis) == "disabled"
                clinical_subtabs.select(clinical_subtabs.tabs()[1])
                root.update()
                assert not clinical_text.winfo_ismapped()
                assert button("Export clinical evaluation plot…", analysis).winfo_ismapped()
                assert menu_state("Save plot…", analysis) == "normal"
                assert menu_state("Print plot…", analysis) == "normal"
                save_path = Path(db_path).with_name("saved-clinical.pdf")
                filedialog.asksaveasfilename = lambda **_kwargs: str(save_path)
                button("Save plot…", analysis).invoke()
                assert save_path.read_bytes().startswith(b"%PDF")
                def choose_jpg(**kwargs):
                    kwargs["typevariable"].set("JPG")
                    return str(save_path)
                filedialog.asksaveasfilename = choose_jpg
                button("Save plot…", analysis).invoke()
                assert save_path.with_suffix(".jpg").read_bytes().startswith(b"\xff\xd8")
                with patch.object(app, "open_print_preview") as preview:
                    button("Print plot…", analysis).invoke()
                    assert preview.call_count == 1
                    printed = Path(preview.call_args.args[0])
                    assert printed.read_bytes().startswith(b"%PDF")
                    printed.unlink()
                    printed.parent.rmdir()
                clinical_subtabs.select(clinical_subtabs.tabs()[0])
                root.update()
                assert menu_state("Save plot…", analysis) == "disabled"
                assert menu_state("Print plot…", analysis) == "disabled"
                clinical_subtabs.select(clinical_subtabs.tabs()[1])
                root.update()
                table_tab = next(
                    page for page in analysis_tabs.tabs()
                    if analysis_tabs.tab(page, "text") == "Prescription values"
                )
                analysis_tabs.select(table_tab)
                analysis_tabs.event_generate("<<NotebookTabChanged>>")
                assert menu_state("Save plot…", analysis) == "disabled"
                assert menu_state("Print plot…", analysis) == "disabled"
                analysis_tabs.select(clinical_tab)
                analysis_tabs.event_generate("<<NotebookTabChanged>>")
                reference_entry = next(
                    w
                    for w in children(analysis)
                    if isinstance(w, ttk.Entry) and w.cget("width") == 9
                )
                expected = app.build_analysis(
                    app.analyze_selected(
                        [dict(saved, alpha_beta=2.0, prescriptions=saved["prescriptions"][1:])],
                        -50,
                        50,
                        0.05,
                    )["sets"][0]
                )["reference"]
                assert abs(float(reference_entry.get()) - expected) < 1e-8
                for owner, trigger, title in (
                    (analysis, "Export clinical evaluation plot…", "Export clinical evaluation plot"),
                    (analysis, "Export dispersion plot…", "Export dispersion plot"),
                    (root, "Export image…", "Export plot image"),
                ):
                    button(trigger, owner).invoke()
                    image_dialog = dialog(title)
                    fmt = next(w for w in children(image_dialog) if isinstance(w, ttk.Combobox))
                    assert tuple(fmt.cget("values")) == tuple(app.IMAGE_FORMATS)
                    assert fmt.get() == "PDF"
                    button("Cancel", image_dialog).invoke()
                analysis.destroy()

                # Reopening restores the explicit choice; returning to automatic uses 78 Gy.
                button("Edit prescriptions set").invoke()
                editor = dialog("Edit prescriptions set")
                radios = [w for w in children(editor) if isinstance(w, ttk.Radiobutton)]
                assert root.getvar(radios[0].cget("variable")) == str(radios[-1].cget("value"))
                radios[0].invoke()
                button("Save prescriptions set", editor).invoke()
                optimize.invoke()
                button("Analysis…").invoke()
                analysis = dialog("RD EQD2 — Detailed analysis")
                reference_entry = next(
                    w
                    for w in children(analysis)
                    if isinstance(w, ttk.Entry) and w.cget("width") == 9
                )
                assert float(reference_entry.get()) == 78
                analysis.destroy()
                button("New prescriptions set").invoke()
                editor = dialog("New prescriptions set")
                entries = [w for w in children(editor) if isinstance(w, tk.Entry)]
                values = ["New observed set", "1.5", "A", "50", "25", "0", "B", "27", "3", "100"]
                assert len(entries) == len(values)
                for entry, value in zip(entries, values):
                    root.setvar(entry.cget("textvariable"), value)
                button("Save prescriptions set", editor).invoke()
                assert [p.local_control_pct for p in app.PrescriptionStore(db_path).find_set(
                    "New observed set")["prescriptions"]] == [0, 100]
                if failures:
                    raise RuntimeError("; ".join(failures))
                completed.append(True)
            except Exception as error:
                failures.append(f"{type(error).__name__}: {error}\n" + traceback.format_exc())
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
                sample = app.read_prescriptions_set_csv(root / "samples" / "prostate.csv")
                assert sample["prescriptions"] == app.PROSTATE_SET["prescriptions"]
                assert sample["alpha_beta"] == app.PROSTATE_ALPHA_BETA
                assert app.reference_schedule(sample) == app.PROSTATE_SET["prescriptions"][0]
                passed("Sample CSV matches the built-in prostate set")
            result = app.analyze(1.5, 0.01, 50.0, 0.01, app.PROSTATE_SET["prescriptions"])
            i = result["idx"]
            # Expected calibration for the ten-prescription prostate set.
            np.testing.assert_allclose(result["q"][i], 3.4832728, atol=2e-5, rtol=0)
            np.testing.assert_allclose(result["mean"][i], 75.5906074, atol=2e-5, rtol=0)
            np.testing.assert_allclose(result["rsd"][i], 3.9191122, atol=2e-5, rtol=0)
            report.update(
                Dq_Gy=float(result["q"][i]),
                mean_EQD2_Gy=float(result["mean"][i]),
                RSD_percent=float(result["rsd"][i]),
            )
            passed("Prostate calibration")

            doses = np.array([p.dose_per_fraction_gy for p in app.PROSTATE_SET["prescriptions"]])
            fractions = np.array([p.fractions for p in app.PROSTATE_SET["prescriptions"]])
            symbolic = app.EQD2_FN(1.5, result["q"][i], doses, fractions)
            np.testing.assert_allclose(symbolic, result["eqd2"][i], rtol=1e-10)
            passed("SymPy lambdify and NumPy runtime")

            sd_fit = app.analyze(1.5, 0.01, 50, 0.01, app.PROSTATE_SET["prescriptions"], method="sd")
            np.testing.assert_allclose(
                sd_fit["q"][sd_fit["idx"]], result["q"][i], atol=2e-5, rtol=0
            )
            range_fit = app.analyze(1.5, 0.01, 50, 0.01, app.PROSTATE_SET["prescriptions"], method="range")
            assert np.isfinite(range_fit["spread"][range_fit["idx"]])
            passed("SSE, SD, and range optimization")
            check_signed_model(app, folder)
            passed(
                "Q-selected branches, zero limit, independent hazard integration, storage and exports"
            )
            check_math_and_plot_interval(app, folder)
            passed(
                "High-precision RD arithmetic, connected profile bounds, and requested graph intervals"
            )
            check_full_signed_profile(app, folder)
            passed("Full 15-prescription NSCLC profile, q-dependent signs, and smooth unique zero")
            check_signed_reconstruction(app)
            passed("Primitive RD parameters, signed classical descriptors, and direct H/BED/EQD2")
            check_reference_prescriptions(app, folder)
            passed(
                "Automatic and chosen reference prescriptions, subsets, CSV, and legacy migration"
            )

            store = app.PrescriptionStore(folder / "prescriptions.sqlite3", app.DEFAULT_SETS)
            assert len(store.list_sets()) == 1, "Only prostate may be seeded by default"
            prostate = store.find_set(app.PROSTATE_SET_NAME)
            assert len(prostate["prescriptions"]) == 10
            assert prostate["prescriptions"] == app.PROSTATE_SET["prescriptions"]
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
            check_prostate_preset_upgrade(app, folder)
            passed("Existing prostate preset upgrade and preservation of customized or deleted sets")
            check_prescription_model(app, folder)
            passed("Prescription objects retain outcomes through edits, selections, storage and exports")
            check_calculation_progress(app)
            passed("Calculation progress, cancellation, numerical equivalence and atomic results")
            check_clinical_evaluation(app, folder)
            passed("Observed local control storage, CSV round trip, clustering, beta fit and plot")
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
            assert len(configured["prescriptions"]) == len(prostate["prescriptions"]) - 2
            assert configured["alpha_beta"] == 2.2
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
            state.set_included(prostate, range(len(prostate["prescriptions"])), False)
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
                for p in app.PROSTATE_SET["prescriptions"]:
                    writer.writerow([p.name, p.total_gy, p.fractions, "Prostate", 1.5])
            imported = app.read_prescriptions_set_csv(csv_path)
            assert [p.schedule_key for p in imported["prescriptions"]] == [
                p.schedule_key for p in app.PROSTATE_SET["prescriptions"]
            ]
            assert all(p.local_control_pct is None for p in imported["prescriptions"])
            passed("CSV import with Unicode and UTF-8 BOM")

            app.export(result, folder / "results")
            assert list(app.IMAGE_FORMATS) == ["PDF", "JPG", "PNG", "EPS", "SVG"]
            for extension in app.IMAGE_FORMATS.values():
                app.export_image(result, folder / ("profile" + extension), ppi=72)
            details = app.build_analysis({"name": "Prostate", "result": result})
            check_prescription_plot_labels(app, details)
            passed("Prescription point labels stay inside dispersion and clinical plots")
            robustness = app.run_robustness(details)
            assert len(robustness["leave_one_out"]) == len(prostate["prescriptions"])
            for extension in app.IMAGE_FORMATS.values():
                app.export_dispersion_plot(details, folder / ("dispersion" + extension), ppi=72)
            for format_name in ("csv", "tsv", "json", "tex"):
                paths = app.export_analysis_tables(details, folder / "tables", format=format_name)
                assert paths and all(p.is_file() and p.stat().st_size for p in paths)
            signatures = {
                ".pdf": b"%PDF",
                ".jpg": b"\xff\xd8",
                ".png": b"\x89PNG",
                ".eps": b"%!PS",
                ".svg": b"<?xml",
            }
            for stem in ("profile", "dispersion"):
                for extension, signature in signatures.items():
                    data = (folder / (stem + extension)).read_bytes()
                    assert len(data) > 100 and data.startswith(signature)
            passed("Both plot exports: PDF, JPG, PNG, EPS, SVG with correct file signatures")
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
                passed("Modal progress, worker threads, Cancel, failure recovery and restored input")
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
