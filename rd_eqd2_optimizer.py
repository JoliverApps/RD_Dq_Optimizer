#!/usr/bin/env python3
"""RD EQD2 Optimizer: estimate Dq from radiotherapy prescriptions.

Run this file to open the GUI. Dependencies: numpy, matplotlib, scipy, sympy, and Tkinter.
The prostate set contains ten prescriptions at alpha/beta=1.5 Gy.
CSV import: name,total_gy,fractions; optional prescriptions_set,alpha_beta,is_reference,local_control_pct columns.
Select individual prescriptions and edit alpha/beta directly on the main page.
Batch example: python rd_eqd2_optimizer.py --batch --dataset prostate --out results
"""

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sqlite3
import unicodedata
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import sympy as sp

# Dose is expressed in Gy. The reference fraction size defines EQD2.
REFERENCE_DOSE_GY = 2.0
_SERIES_THRESHOLD = 1e-2
_DIRECT_Q_FRACTION = 0.25

# Symbolic signed-RD reconstruction and cumulative-hazard ratio.
AB = sp.symbols("alpha_over_beta", positive=True, real=True)
Q = sp.symbols("D_q", real=True)
D, N = sp.symbols("d n", positive=True, real=True)
SIGNED_AB = sp.Piecewise((AB, Q >= 0), (-AB, True))
R = (sp.sqrt(Q**2 + 2 * Q * SIGNED_AB) - Q) / SIGNED_AB
S = sp.Piecewise((R / Q, sp.Ne(Q, 0)), (sp.nan, True))


def _symbolic_fraction_effect(dose):
    return dose - Q * (1 - sp.exp(-S * dose))


EQD2_EXPR = sp.Piecewise(
    (N * D, sp.Eq(Q, 0) & sp.Ne(AB, 0)),
    (
        sp.Integer(REFERENCE_DOSE_GY)
        * N
        * _symbolic_fraction_effect(D)
        / _symbolic_fraction_effect(sp.Integer(REFERENCE_DOSE_GY)),
        True,
    ),
)
_EQD2_SYMBOLIC_FN = sp.lambdify((AB, Q, D, N), EQD2_EXPR, modules="numpy")


def EQD2_FN(ab, q, d, n):
    """Return EQD2 = d0*n*F(d)/F(d0), with F=H/k and d0=2 Gy.

    ``ab`` is the positive-q reference α/β; the signed value is selected from q.
    Inputs broadcast with NumPy. At q=0, the linear law gives EQD2=n*d.
    """
    ab, q, d, n = np.broadcast_arrays(*[np.asarray(x, dtype=float) for x in (ab, q, d, n)])
    if np.any(~np.isfinite(ab)) or np.any(ab <= 0):
        raise ValueError("Enter a positive finite α/β value for the q ≥ 0 branch.")
    if np.any(~np.isfinite(q)):
        raise ValueError("Candidate q values must be finite.")
    result = np.empty(ab.shape)
    for input_ab in np.unique(ab):
        selected = ab == input_ab
        branch_ab, reserve, rates = _signed_rd_parameters(float(input_ab), q[selected])
        effect = _fraction_effect_values(q[selected], branch_ab, reserve, rates, d[selected])
        reference = _fraction_effect_values(
            q[selected], branch_ab, reserve, rates, REFERENCE_DOSE_GY
        )
        result[selected] = (effect / reference) * (REFERENCE_DOSE_GY * n[selected])
    return result.item() if result.ndim == 0 else result


def validated_local_control(value):
    """Read an optional observed percentage, keeping both 0 and 100 valid."""
    if value is None or isinstance(value, str) and not value.strip():
        return None
    if isinstance(value, str):
        value = value.strip().removesuffix("%")
    message = "Local tumor control must be a percentage from 0 to 100, or left blank."
    try:
        percent = float(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(message) from exc
    if not 0 <= percent <= 100:
        raise ValueError(message)
    return percent


@dataclass(frozen=True, slots=True)
class Prescription:
    """One prescription and its optional observed local tumor control percentage."""

    name: str
    total_gy: float
    fractions: int
    local_control_pct: float | None = None

    def __post_init__(self):
        name = str(self.name).strip()
        try:
            total, fractions = float(self.total_gy), float(self.fractions)
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError("D must be positive and n must be a positive integer.") from exc
        if (not name or not math.isfinite(total) or total <= 0
                or not math.isfinite(fractions) or fractions <= 0 or not fractions.is_integer()):
            raise ValueError("A prescription needs a name, positive dose, and positive integer fractions.")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "total_gy", total)
        object.__setattr__(self, "fractions", int(fractions))
        object.__setattr__(self, "local_control_pct", validated_local_control(self.local_control_pct))

    @property
    def dose_per_fraction_gy(self):
        return self.total_gy / self.fractions

    @property
    def schedule_key(self):
        """Physical identity used by set selections and reference choices."""
        return self.name, self.total_gy, self.fractions

    @classmethod
    def from_value(cls, value):
        """Accept inline records and legacy three-value calculation inputs."""
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            return cls(value["name"], value["total_gy"], value["fractions"],
                       value.get("local_control_pct"))
        if isinstance(value, (tuple, list)) and len(value) in (3, 4):
            return cls(*value)
        raise ValueError("Use a Prescription, a field mapping, or a three/four-value row.")


PROSTATE_SET_NAME = "Prostate — 10 prescriptions"
PROSTATE_ALPHA_BETA = 1.5
# Each row carries its own outcome; None means no observed outcome is supplied.
PROSTATE_SET = {
    "name": PROSTATE_SET_NAME,
    "alpha_beta": PROSTATE_ALPHA_BETA,
    "prescriptions": [
        Prescription("Shared conventional reference", 78.00, 39, 91.6),
        Prescription("RTOG 0126 IMRT", 79.20, 44, 97.0),
        Prescription("GETUG 06", 80.00, 40),
        Prescription("RTOG 0415 conventional", 73.80, 41, 95.3),
        Prescription("RTOG 0415 hypofractionated", 70.00, 28, 99.4),
        Prescription("PROFIT hypofractionated", 60.00, 20, 86.92307692307692),
        Prescription("PACE-B control", 62.00, 20),
        Prescription("Mantz SBRT", 40.00, 5),
        Prescription("HYPO-RT-PC ultrahypofractionated", 42.70, 7),
        Prescription("Fuller SBRT", 38.00, 4, 99.0),
    ],
}
DEFAULT_Q_START, DEFAULT_Q_END, DEFAULT_Q_STEP = -50.0, 50.0, 0.001
DEFAULT_SETS = {PROSTATE_SET_NAME: PROSTATE_SET}
# Self-contained, SQLite-backed one-to-many prescription storage.
DEFAULT_DB_PATH = Path(
    os.environ.get("RD_DQ_DB_PATH", Path.home() / ".rd_dq_optimizer" / "prescriptions.sqlite3")
).expanduser()


class PrescriptionStore:
    def __init__(self, path=DEFAULT_DB_PATH, initial_sets=None):
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize(initial_sets or {}, seed_defaults=(initial_sets is not None))

    @contextmanager
    def _connection(self):
        con = sqlite3.connect(str(self.path))
        try:
            con.row_factory = sqlite3.Row
            con.execute("PRAGMA foreign_keys = ON")
            with con:
                yield con
        finally:
            con.close()

    def _initialize(self, datasets, seed_defaults=False):
        with self._connection() as con:
            con.executescript("""
                CREATE TABLE IF NOT EXISTS tumors (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL COLLATE NOCASE UNIQUE,
                    alpha_beta REAL NOT NULL CHECK (alpha_beta > 0)
                );
                CREATE TABLE IF NOT EXISTS prescriptions (
                    id INTEGER PRIMARY KEY,
                    tumor_id INTEGER NOT NULL REFERENCES tumors(id) ON DELETE CASCADE,
                    name TEXT NOT NULL,
                    total_gy REAL NOT NULL CHECK (total_gy > 0),
                    fractions INTEGER NOT NULL CHECK (fractions > 0),
                    position INTEGER NOT NULL,
                    UNIQUE (tumor_id, name)
                );
                CREATE INDEX IF NOT EXISTS idx_prescriptions_tumor ON prescriptions(tumor_id, position);
                CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            """)
            con.execute("BEGIN IMMEDIATE")
            columns = {row["name"] for row in con.execute("PRAGMA table_info(prescriptions)")}
            if "is_reference" not in columns:
                con.execute(
                    "ALTER TABLE prescriptions ADD COLUMN is_reference INTEGER NOT NULL "
                    "DEFAULT 0 CHECK (is_reference IN (0,1))"
                )
            if "local_control_pct" not in columns:
                con.execute(
                    "ALTER TABLE prescriptions ADD COLUMN local_control_pct REAL "
                    "CHECK (local_control_pct BETWEEN 0 AND 100)"
                )
            con.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_one_reference_per_set "
                "ON prescriptions(tumor_id) WHERE is_reference=1"
            )
            marker = "default_prescriptions_sets_seeded"
            if not seed_defaults or not datasets:
                return
            seeded = con.execute("SELECT value FROM metadata WHERE key=?", (marker,)).fetchone()
            if seeded is None:
                for name, dataset in datasets.items():
                    if con.execute("SELECT 1 FROM tumors WHERE name=?", (name,)).fetchone():
                        continue
                    self._seed_set(con, name, dataset)
                version = "2" if datasets is DEFAULT_SETS else "1"
                con.execute("INSERT INTO metadata(key,value) VALUES (?,?)", (marker, version))
            elif seeded["value"] == "1" and datasets is DEFAULT_SETS:
                self._upgrade_prostate_default(con)
                con.execute("UPDATE metadata SET value='2' WHERE key=?", (marker,))
            if datasets is DEFAULT_SETS:
                self._populate_prostate_local_control(con)

    def _populate_prostate_local_control(self, con):
        marker = "prostate_local_control_seeded"
        if con.execute("SELECT 1 FROM metadata WHERE key=?", (marker,)).fetchone():
            return
        prostate = con.execute("SELECT id FROM tumors WHERE name=?", (PROSTATE_SET_NAME,)).fetchone()
        if prostate is not None:
            rows = [
                tuple(row)
                for row in con.execute(
                    "SELECT name,total_gy,fractions FROM prescriptions "
                    "WHERE tumor_id=? ORDER BY position,id",
                    (prostate["id"],),
                )
            ]
            if rows == [p.schedule_key for p in PROSTATE_SET["prescriptions"]]:
                con.executemany(
                    "UPDATE prescriptions SET local_control_pct=? "
                    "WHERE tumor_id=? AND name=? AND total_gy=? AND fractions=? "
                    "AND local_control_pct IS NULL",
                    [
                        (p.local_control_pct, prostate["id"], *p.schedule_key)
                        for p in PROSTATE_SET["prescriptions"]
                        if p.local_control_pct is not None
                    ],
                )
        con.execute("INSERT INTO metadata(key,value) VALUES (?, '1')", (marker,))

    def _seed_set(self, con, name, dataset):
        cursor = con.execute(
            "INSERT INTO tumors (name,alpha_beta) VALUES (?,?)",
            (name, validated_alpha_beta(dataset["alpha_beta"])),
        )
        reference = dataset.get("reference_prescription")
        rows = validate_schedules(dataset["prescriptions"])
        reference_key = Prescription.from_value(reference).schedule_key if reference is not None else None
        con.executemany(
            "INSERT INTO prescriptions "
            "(tumor_id,name,total_gy,fractions,position,is_reference,local_control_pct) "
            "VALUES (?,?,?,?,?,?,?)",
            [
                (
                    cursor.lastrowid,
                    p.name,
                    p.total_gy,
                    p.fractions,
                    position,
                    int(p.schedule_key == reference_key),
                    p.local_control_pct,
                )
                for position, p in enumerate(rows)
            ],
        )

    def _upgrade_prostate_default(self, con):
        previous = con.execute(
            "SELECT id FROM tumors WHERE name=?", ("Prostate — 8 prescriptions",)
        ).fetchone()
        if previous is None or con.execute(
            "SELECT 1 FROM tumors WHERE name=?", (PROSTATE_SET_NAME,)
        ).fetchone():
            return

        previous_rows = [
            ("Shared conventional reference", 78.00, 39),
            ("RTOG 0126 IMRT", 79.20, 44),
            ("RTOG 0415 hypofractionated", 70.00, 28),
            ("PROFIT hypofractionated", 60.00, 20),
            ("PACE-B control", 62.00, 20),
            ("Mantz SBRT", 40.00, 5),
            ("HYPO-RT-PC ultrahypofractionated", 42.70, 7),
            ("Fuller SBRT", 38.00, 4),
        ]
        set_id = previous["id"]
        saved_rows = [
            tuple(row)
            for row in con.execute(
                "SELECT name,total_gy,fractions FROM prescriptions "
                "WHERE tumor_id=? ORDER BY position,id",
                (set_id,),
            )
        ]
        if saved_rows != previous_rows:
            self._seed_set(con, PROSTATE_SET_NAME, DEFAULT_SETS[PROSTATE_SET_NAME])
            return

        con.execute("UPDATE tumors SET name=? WHERE id=?", (PROSTATE_SET_NAME, set_id))
        con.executemany(
            "INSERT INTO prescriptions(tumor_id,name,total_gy,fractions,position) "
            "VALUES (?,?,?,?,?) "
            "ON CONFLICT(tumor_id,name) DO UPDATE SET position=excluded.position",
            [(set_id, *p.schedule_key, position) for position, p in enumerate(PROSTATE_SET["prescriptions"])],
        )

    def list_sets(self):
        with self._connection() as con:
            return [
                dict(row)
                for row in con.execute(
                    "SELECT id,name,alpha_beta FROM tumors ORDER BY name COLLATE NOCASE"
                )
            ]

    def get_set(self, set_id):
        with self._connection() as con:
            row = con.execute(
                "SELECT id,name,alpha_beta FROM tumors WHERE id=?", (set_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"Prescriptions set {set_id} does not exist.")
            data = dict(row)
            prescriptions = con.execute(
                "SELECT name,total_gy,fractions,is_reference,local_control_pct FROM prescriptions "
                "WHERE tumor_id=? ORDER BY position,id",
                (set_id,),
            ).fetchall()
            data["prescriptions"] = [
                Prescription(r["name"], r["total_gy"], r["fractions"], r["local_control_pct"])
                for r in prescriptions
            ]
            data["reference_prescription"] = next(
                (
                    row
                    for row, saved in zip(data["prescriptions"], prescriptions)
                    if saved["is_reference"]
                ),
                None,
            )
            return data

    def find_set(self, name):
        with self._connection() as con:
            row = con.execute("SELECT id FROM tumors WHERE name=?", (name,)).fetchone()
        return self.get_set(row["id"]) if row else None

    def save_set(
        self,
        name,
        alpha_beta,
        prescriptions,
        set_id=None,
        reference_prescription=None,
    ):
        name = str(name).strip()
        if not name:
            raise ValueError("Prescriptions set name is required.")
        if len(name) > 120:
            raise ValueError("Prescriptions set name must not exceed 120 characters.")
        ab = validated_alpha_beta(alpha_beta)
        rows = validate_schedules(prescriptions)
        if reference_prescription is not None:
            reference_key = Prescription.from_value(reference_prescription).schedule_key
            reference_prescription = next((p for p in rows if p.schedule_key == reference_key), None)
            if reference_prescription is None:
                raise ValueError("The reference prescription must belong to this prescriptions set.")
        labels = [p.name.casefold() for p in rows]
        if len(labels) != len(set(labels)):
            raise ValueError("Each prescription name must be unique within a prescriptions set.")
        with self._connection() as con:
            try:
                if set_id is None:
                    cur = con.execute(
                        "INSERT INTO tumors(name,alpha_beta) VALUES (?,?)", (name, ab)
                    )
                    set_id = cur.lastrowid
                else:
                    cur = con.execute(
                        "UPDATE tumors SET name=?,alpha_beta=? WHERE id=?", (name, ab, set_id)
                    )
                    if not cur.rowcount:
                        raise KeyError(f"Prescriptions set {set_id} does not exist.")
                    con.execute("DELETE FROM prescriptions WHERE tumor_id=?", (set_id,))
                con.executemany(
                    "INSERT INTO prescriptions "
                    "(tumor_id,name,total_gy,fractions,position,is_reference,local_control_pct) "
                    "VALUES (?,?,?,?,?,?,?)",
                    [
                        (set_id, *row.schedule_key, i, int(row == reference_prescription), row.local_control_pct)
                        for i, row in enumerate(rows)
                    ],
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError(
                    "A prescriptions set with this name, or a duplicate prescription, already exists."
                ) from exc
        return set_id

    def delete_set(self, set_id):
        with self._connection() as con:
            cur = con.execute("DELETE FROM tumors WHERE id=?", (set_id,))
        return bool(cur.rowcount)

    def update_alpha_beta(self, set_id, alpha_beta):
        """Update a saved set's alpha/beta without replacing its prescriptions."""
        ab = validated_alpha_beta(alpha_beta)
        with self._connection() as con:
            cur = con.execute("UPDATE tumors SET alpha_beta=? WHERE id=?", (ab, set_id))
            if not cur.rowcount:
                raise KeyError(f"Prescriptions set {set_id} does not exist.")
        return ab


def validated_alpha_beta(value):
    try:
        ab = float(value)
    except (ValueError, TypeError) as exc:
        raise ValueError("Enter a positive finite α/β value for the q ≥ 0 branch.") from exc
    if not math.isfinite(ab) or ab <= 0:
        raise ValueError("Enter a positive finite α/β value for the q ≥ 0 branch.")
    return ab


def reference_schedule(record):
    """Resolve the chosen reference by physical identity, including outside a fitted subset."""
    rows = [Prescription.from_value(row) for row in record["prescriptions"]]
    reference = record.get("reference_prescription")
    if reference is not None:
        reference = Prescription.from_value(reference)
        return next((p for p in rows if p.schedule_key == reference.schedule_key), reference)
    return next(
        (p for p in rows if math.isclose(p.dose_per_fraction_gy, REFERENCE_DOSE_GY,
                                       rel_tol=0, abs_tol=1e-10)),
        None,
    )


class PrescriptionsSelection:
    """Session selections and alpha/beta drafts, kept separately for each saved set."""

    def __init__(self):
        self.included = {}
        self.alpha_beta_text = {}
        self._saved_alpha_beta = {}

    def sync(self, records):
        keys = {str(record["id"]) for record in records}
        for mapping in (self.included, self.alpha_beta_text, self._saved_alpha_beta):
            for key in set(mapping) - keys:
                del mapping[key]
        for record in records:
            key = str(record["id"])
            previous = self.included.get(key, {})
            self.included[key] = {
                Prescription.from_value(row).schedule_key:
                    previous.get(Prescription.from_value(row).schedule_key, True)
                for row in record["prescriptions"]
            }
            ab = record["alpha_beta"]
            if key not in self.alpha_beta_text or self._saved_alpha_beta.get(key) != ab:
                self.alpha_beta_text[key] = str(ab)
            self._saved_alpha_beta[key] = ab

    def is_included(self, record, index):
        return self.included[str(record["id"])][Prescription.from_value(record["prescriptions"][index]).schedule_key]

    def set_included(self, record, indices, value):
        for index in indices:
            self.included[str(record["id"])][Prescription.from_value(record["prescriptions"][index]).schedule_key] = bool(value)

    def selected_rows(self, record):
        return [
            row
            for index, row in enumerate(record["prescriptions"])
            if self.is_included(record, index)
        ]

    def configured_set(self, record):
        rows = self.selected_rows(record)
        if len(rows) < 2:
            raise ValueError(f'Select at least two prescriptions in "{record["name"]}".')
        try:
            ab = validated_alpha_beta(self.alpha_beta_text[str(record["id"])])
        except ValueError as exc:
            raise ValueError(f"{record['name']}: {exc}") from exc
        return dict(
            record,
            prescriptions=rows,
            alpha_beta=ab,
            reference_prescription=reference_schedule(record),
        )


def validate_schedules(schedules):
    rows = [Prescription.from_value(row) for row in schedules]
    if len(rows) < 2:
        raise ValueError("Select or enter at least two prescriptions.")
    return rows


def _branch_alpha_beta(ab, q):
    """Select a=ab for q>=0 and a=-ab for q<0.

    The q=0 value records the selected reference; the linear law has β=0,
    so its physical α/β is undefined and is not used in reconstruction.
    """
    return np.where(np.asarray(q) < 0, -ab, ab)


def _signed_rd_parameters(ab, qgrid):
    """Reconstruct signed a, r=(sqrt(q²+2qa)-q)/a, and s=r/q.

    For q>0, 0<r<1; for q<0, r<0. Both nonzero branches have s>0.
    At q=0, r=0 and s is undetermined (NaN). Rates have units Gy⁻¹.
    """
    ab = validated_alpha_beta(ab)
    qgrid = np.asarray(qgrid, dtype=float)
    if qgrid.size == 0 or np.any(~np.isfinite(qgrid)):
        raise ValueError("All candidate q values must be finite.")
    branch_ab = _branch_alpha_beta(ab, qgrid)
    reserve = np.zeros(qgrid.shape)
    rates = np.full(qgrid.shape, np.nan)
    active = qgrid != 0
    q, a = qgrid[active], branch_ab[active]
    root = np.hypot(q, np.sqrt(2.0) * np.sqrt(q * a))
    r = np.empty_like(q)
    positive = q > 0
    # For q>0, r=2q/(sqrt(q²+2qa)+q) avoids subtraction of close values.
    r[positive] = 2 * q[positive] / (root[positive] + q[positive])
    r[~positive] = (root[~positive] - q[~positive]) / a[~positive]
    reserve[active] = r
    rates[active] = r / q
    return branch_ab, reserve, rates


def rd_parameters(ab, qgrid):
    """Return r and s for the signed α/β and q values."""
    _, reserve, rates = _signed_rd_parameters(ab, qgrid)
    return reserve, rates


def _initial_hazard_factor(q, branch_ab, reserve, rates):
    """Return the positive normalization α/k=1-r.

    The identity 1-r=a*r*s/2 avoids cancellation when r approaches one.
    Negative r uses 1-r directly; the linear case has normalization one.
    """
    q, branch_ab, reserve, rates = np.broadcast_arrays(q, branch_ab, reserve, rates)
    factor = np.ones(q.shape)
    positive, negative = q > 0, q < 0
    factor[positive] = branch_ab[positive] * reserve[positive] * rates[positive] / 2
    factor[negative] = 1 - reserve[negative]
    return factor


def _shoulder_weight(t):
    """Evaluate g(t)=1-(1-exp(-t))/t, including g(0)=0.

    The small-t expansion through t⁵ avoids cancellation in expm1(-t)/t.
    """
    t = np.asarray(t, dtype=float)
    weight = np.empty_like(t)
    small = t < _SERIES_THRESHOLD
    x = t[small]
    weight[small] = x * (1 / 2 + x * (-1 / 6 + x * (1 / 24 + x * (-1 / 120 + x / 720))))
    x = t[~small]
    weight[~small] = 1 + np.expm1(-x) / x
    return weight


def _fraction_effect_values(q, branch_ab, reserve, rates, doses):
    """Return F(d)=H(d)/k=d-q*(1-exp(-s*d)), in Gy.

    Direct evaluation is stable for q<0 and for 0<q<=d/4. Otherwise use
    F(d)=(1-r)*d+r*d*g(s*d), whose positive terms avoid cancellation.
    At q=0, F(d)=d; the undetermined rate is never evaluated.
    """
    q, branch_ab, reserve, rates, doses = np.broadcast_arrays(q, branch_ab, reserve, rates, doses)
    effect = np.array(doses, dtype=float, copy=True)
    direct = (q < 0) | ((q > 0) & (q <= _DIRECT_Q_FRACTION * doses))
    effect[direct] = doses[direct] + q[direct] * np.expm1(-rates[direct] * doses[direct])
    stable = (q > 0) & ~direct
    if np.any(stable):
        d, r, s = doses[stable], reserve[stable], rates[stable]
        t = s * d
        weight = _shoulder_weight(t)
        factor = _initial_hazard_factor(q[stable], branch_ab[stable], r, s)
        effect[stable] = factor * d + r * d * weight
    return effect


def fraction_bed(ab, qgrid, doses):
    """Return BED per fraction, F(d)/(1-r), for each q and dose.

    Rows correspond to candidate q values and columns to fraction doses.
    Multiply by the fraction count for a treatment course. At q=0, BED=d.
    """
    qgrid = np.atleast_1d(np.asarray(qgrid, dtype=float))
    branch_ab, reserve, rates = _signed_rd_parameters(ab, qgrid)
    doses = np.atleast_1d(np.asarray(doses, dtype=float))
    effect = _fraction_effect_values(
        qgrid[:, None], branch_ab[:, None], reserve[:, None], rates[:, None], doses[None, :]
    )
    factor = _initial_hazard_factor(qgrid, branch_ab, reserve, rates)
    return effect / factor[:, None]


def evaluate(ab, qgrid, schedules):
    schedules = validate_schedules(schedules)
    qgrid = np.atleast_1d(np.asarray(qgrid, dtype=float))
    d = np.array([p.dose_per_fraction_gy for p in schedules], dtype=float)
    n = np.array([p.fractions for p in schedules], dtype=float)
    return EQD2_FN(ab, qgrid[:, None], d[None, :], n[None, :])


def grid_values(start, stop, step):
    start = float(start)
    stop = float(stop)
    step = float(step)
    if not all(np.isfinite(x) for x in (start, stop, step)) or step <= 0 or stop < start:
        raise ValueError("Use finite q start <= q end and a positive step.")
    intervals = (stop - start) / step
    if not math.isfinite(intervals) or intervals > 200000:
        raise ValueError("Over 200,000 grid points; increase step.")
    count = math.floor(intervals) + 1
    a = start + step * np.arange(count)
    rounding = 8 * np.finfo(float).eps * max(abs(start), abs(stop), abs(step))
    a[np.abs(a - stop) <= rounding] = stop
    if start <= 0 <= stop:
        a[np.abs(a) <= rounding] = 0.0
    endpoints = [start, stop, 0.0] if start <= 0 <= stop else [start, stop]
    a = np.unique(np.concatenate((a[(a >= start) & (a <= stop)], endpoints)))
    a[a == 0] = 0.0
    if len(a) > 200001:
        raise ValueError("Over 200,000 grid points; increase step.")
    return a


OPTIMIZATION_METHODS = {
    "sse": "EQD2 sum of squared deviations (SSE)",
    "sd": "EQD2 standard deviation (SD)",
    "range": "EQD2 range (max − min)",
}

DEFAULT_PLOT_TITLES = {
    "sse": "RD EQD2 — SSE minimization profile",
    "sd": "RD EQD2 — SD minimization profile",
    "range": "RD EQD2 — range minimization profile",
}


# These are presentation settings only: no influence on RD or optimization.
# Blank titles and labels mean automatic RD-specific text for the chosen method.
DEFAULT_PLOT_SETTINGS = {
    "title": "",
    "x_label": "",
    "y_label": "",
    "show_title": True,
    "show_legend": True,
    "show_grid": True,
    "show_bounds": True,
    "show_minimum": True,
    "show_summary": True,
    "tight_crop": False,
    "width_in": 8.0,
    "height_in": 4.8,
    "font_family": "DejaVu Sans",
    "title_font": 12.0,
    "axis_font": 10.5,
    "tick_font": 9.0,
    "legend_font": 8.5,
    "summary_font": 8.2,
    "dq_font": 9.0,
    "line_width": 2.0,
    "minimum_size": 5.0,
    "bounds_alpha": 0.12,
    "bounds_width": 0.9,
    "grid_alpha": 0.25,
    "grid_style": ":",
    "line_style": "-",
    "legend_location": "upper left",
    "curve_color": "#1f77b4",
    "bounds_color": "",
    "y_min": "",
    "y_max": "",
}

PLOT_Y_LABELS = {
    "sse": r"EQD2 sum of squared deviations (Gy$^2$)",
    "sd": "EQD2 sample standard deviation (Gy)",
    "range": "EQD2 range (Gy)",
}


def normalized_plot_settings(values=None):
    """Return a validated plotting configuration, independent of the model."""
    from matplotlib.colors import to_rgba

    cfg = dict(DEFAULT_PLOT_SETTINGS)
    if values:
        cfg.update({k: v for k, v in values.items() if k in cfg})
    for key in ("title", "x_label", "y_label"):
        cfg[key] = str(cfg[key]).strip()
        if len(cfg[key]) > 160:
            raise ValueError(f"{key.replace('_', ' ').capitalize()} cannot exceed 160 characters.")
    for key in (
        "show_title",
        "show_legend",
        "show_grid",
        "show_bounds",
        "show_minimum",
        "show_summary",
        "tight_crop",
    ):
        if not isinstance(cfg[key], bool):
            raise ValueError(f"{key} must be enabled or disabled.")
    cfg["font_family"] = str(cfg["font_family"]).strip()
    if cfg["font_family"] not in (
        "DejaVu Sans",
        "DejaVu Serif",
        "Liberation Sans",
        "Liberation Serif",
        "monospace",
    ):
        raise ValueError("Select a valid plot font family.")
    for key, lo, hi in (
        ("width_in", 3, 24),
        ("height_in", 2.5, 18),
        ("title_font", 6, 36),
        ("axis_font", 6, 30),
        ("tick_font", 5, 25),
        ("legend_font", 5, 25),
        ("summary_font", 5, 25),
        ("dq_font", 5, 25),
        ("line_width", 0.2, 8),
        ("minimum_size", 1, 18),
        ("bounds_width", 0.2, 5),
        ("bounds_alpha", 0, 0.8),
        ("grid_alpha", 0, 1),
    ):
        try:
            num = float(cfg[key])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{key.replace('_', ' ').capitalize()} must be numeric.") from exc
        if not math.isfinite(num) or not lo <= num <= hi:
            raise ValueError(f"{key.replace('_', ' ').capitalize()} must be between {lo} and {hi}.")
        cfg[key] = num
    if cfg["grid_style"] not in ("-", "--", ":", "-."):
        raise ValueError("Invalid grid line style.")
    if cfg["line_style"] not in ("-", "--", ":", "-."):
        raise ValueError("Invalid optimization-curve line style.")
    if cfg["legend_location"] not in (
        "best",
        "upper right",
        "upper left",
        "lower right",
        "lower left",
        "upper center",
        "lower center",
    ):
        raise ValueError("Invalid legend location.")
    for key in ("curve_color", "bounds_color"):
        cfg[key] = str(cfg[key]).strip()
        if cfg[key]:
            try:
                to_rgba(cfg[key])
            except ValueError as exc:
                raise ValueError(
                    f"{key.replace('_', ' ').capitalize()} must be a valid Matplotlib color (e.g., #1f77b4)."
                ) from exc
    for key in ("y_min", "y_max"):
        if cfg[key] is None or str(cfg[key]).strip() == "":
            cfg[key] = ""
            continue
        try:
            value = float(cfg[key])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{key.replace('_', ' ').capitalize()} must be a number or blank (automatic)."
            ) from exc
        if not math.isfinite(value):
            raise ValueError(f"{key.replace('_', ' ').capitalize()} must be finite.")
        cfg[key] = value
    for axis in ("y",):
        if (
            cfg[f"{axis}_min"] != ""
            and cfg[f"{axis}_max"] != ""
            and cfg[f"{axis}_min"] >= cfg[f"{axis}_max"]
        ):
            raise ValueError(f"{axis.upper()}-axis minimum must be smaller than maximum.")
    return cfg


def plot_title_for(result):
    """Resolve a custom title or the selected method's RD EQD2 default."""
    settings = result.get("plot_settings") or {}
    return (
        str(settings.get("title") or result.get("plot_title") or "").strip()
        or DEFAULT_PLOT_TITLES[result["method"]]
    )


def write_plot_settings(path, settings):
    """Save only visualization preferences, separate from prescription storage."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(normalized_plot_settings(settings), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def read_plot_settings(path):
    try:
        return normalized_plot_settings(json.loads(Path(path).read_text(encoding="utf-8")))
    except (OSError, ValueError, TypeError, KeyError):
        return normalized_plot_settings()


def _bounded_minimum(objective, left, right, iterations=90):
    """Refine an interior grid minimum using golden-section search (no extra dependency)."""
    ratio = (math.sqrt(5) - 1) / 2
    c = right - ratio * (right - left)
    d = left + ratio * (right - left)
    fc, fd = objective(c), objective(d)
    for _ in range(iterations):
        if fc <= fd:
            right, d, fd = d, c, fc
            c = right - ratio * (right - left)
            fc = objective(c)
        else:
            left, c, fc = c, d, fd
            d = left + ratio * (right - left)
            fd = objective(d)
        if right - left < 1e-12 * max(1, abs(left), abs(right)):
            break
    return 0.5 * (left + right)


def _profile_bounds(
    ab,
    optimum,
    schedules,
    minimum_value,
    fraction=0.05,
    criterion="sd",
    lower=1e-4,
    upper=1e3,
    profile=None,
):
    """Find the profile's tolerance crossings on signed intervals, including zero.

    SSE and SD use sample-SD bounds; range uses range-profile bounds.
    None indicates no crossing before the search boundary.
    """
    if criterion not in ("sd", "range"):
        raise ValueError("Unknown profile criterion.")
    target = minimum_value * (1.0 + fraction)

    def residual(q):
        vals = evaluate(ab, np.array([q]), schedules)[0]
        val = float(np.ptp(vals)) if criterion == "range" else float(np.std(vals, ddof=1))
        return val - target

    def solve(lo, hi):
        flo, fhi = residual(lo), residual(hi)
        if flo == 0:
            return lo
        if fhi == 0:
            return hi
        if flo * fhi > 0:
            return None
        for _ in range(90):
            middle = lo + (hi - lo) / 2
            fm = residual(middle)
            if (flo <= 0) == (fm <= 0):
                lo, flo = middle, fm
            else:
                hi, fhi = middle, fm
            if hi - lo < 1e-12 * max(1, abs(lo), abs(hi)):
                break
        return lo + (hi - lo) / 2

    if profile is None:
        left = solve(lower, optimum) if optimum > lower else None
        right = solve(optimum, upper) if optimum < upper else None
        return left, right

    points, measures = profile

    def connected_crossing(indices):
        previous = optimum
        for index in indices:
            point = float(points[index])
            if measures[index] > target:
                return solve(min(point, previous), max(point, previous))
            previous = point
        return None

    left = connected_crossing(np.flatnonzero(points < optimum)[::-1])
    right = connected_crossing(np.flatnonzero(points > optimum))
    return left, right


def _profile_statistics(eqd2):
    """Summarize each candidate's equivalent-dose distribution."""
    mean = np.mean(eqd2, axis=1)
    return {
        "spread": np.ptp(eqd2, axis=1),
        "sd": np.std(eqd2, axis=1),
        "sample_sd": np.std(eqd2, axis=1, ddof=1),
        "mean": mean,
        "sse": np.sum((eqd2 - mean[:, None]) ** 2, axis=1),
    }


def _dispersion_value(eqd2, method):
    """Evaluate the selected minimization criterion for one candidate."""
    if method == "range":
        return float(np.ptp(eqd2))
    if method == "sd":
        return float(np.std(eqd2, ddof=1))
    return float(np.sum((eqd2 - np.mean(eqd2)) ** 2))


def analyze(ab, start, stop, step, schedules, method="sse", bound_pct=5.0):
    """Search q over the selected interval and refine the best interior grid cell.

    SSE and sample SD use the SD profile for tolerance bounds; range uses
    the range profile. The requested endpoints and zero are retained.
    """
    bound_pct = float(bound_pct)
    if not math.isfinite(bound_pct) or bound_pct <= 0:
        raise ValueError(
            "Near-optimal profile tolerance must be a positive, finite percentage (e.g., 5)."
        )
    if method not in OPTIMIZATION_METHODS:
        raise ValueError(f"Unknown optimization method: {method!r}")
    schedules = validate_schedules(schedules)
    ab = validated_alpha_beta(ab)
    q = grid_values(start, stop, step)
    z = evaluate(ab, q, schedules)
    statistics = _profile_statistics(z)
    objective_key = {"range": "spread", "sd": "sample_sd", "sse": "sse"}[method]
    idx = int(np.argmin(statistics[objective_key]))
    if 0 < idx < len(q) - 1:

        def objective(value):
            row = evaluate(ab, np.array([value]), schedules)[0]
            return _dispersion_value(row, method)

        refined = _bounded_minimum(objective, float(q[idx - 1]), float(q[idx + 1]))
        # Include the refined minimum in the plotted curve and the exported CSV.
        if float(q[0]) < refined < float(q[-1]) and abs(refined - q[idx]) > 1e-10:
            q = np.insert(q, int(np.searchsorted(q, refined)), refined)
            z = evaluate(ab, q, schedules)
            statistics = _profile_statistics(z)
            idx = int(np.argmin(statistics[objective_key]))
    dq = float(q[idx])
    profile_measure = "range" if method == "range" else "sd"
    mean = statistics["mean"]
    sample_sd = statistics["sample_sd"]
    minimum_measure = (
        float(statistics["spread"][idx]) if method == "range" else float(sample_sd[idx])
    )
    bounds = _profile_bounds(
        ab,
        dq,
        schedules,
        minimum_measure,
        fraction=bound_pct / 100.0,
        criterion=profile_measure,
        lower=float(q[0]),
        upper=float(q[-1]),
        profile=(q, statistics["spread"] if method == "range" else sample_sd),
    )
    fraction_doses = np.array([p.dose_per_fraction_gy for p in schedules])
    centered_doses = fraction_doses - fraction_doses.mean()
    dose_variation = float(np.sum(centered_doses**2))
    slope = (
        (z - mean[:, None]) @ centered_doses / dose_variation
        if dose_variation > 0
        else np.full(len(q), np.nan)
    )
    rsd = 100.0 * sample_sd / mean
    eta = 100.0 * np.abs(slope) * np.ptp(fraction_doses) / mean
    return dict(
        q=q,
        eqd2=z,
        profile_bounds=bounds,
        profile_measure=profile_measure,
        bound_pct=bound_pct,
        search_step=float(step),
        search_start=float(start),
        search_end=float(stop),
        rsd=rsd,
        eta=eta,
        slope=slope,
        idx=idx,
        ab=float(ab),
        signed_ab=float(_branch_alpha_beta(ab, dq)),
        method=method,
        schedules=schedules,
        **statistics,
    )


def analyze_selected(selections, start, stop, step, method="sse", bound_pct=5.0):
    """Optimize each configured prescriptions set independently at its own α/β.

    ``selections`` contain names, alpha_beta values, and selected prescriptions.
    Prescriptions from different sets are never pooled into one dispersion estimate.
    """
    if not selections:
        raise ValueError("Select at least one prescriptions set to optimize and plot.")
    result_sets = []
    for record in selections:
        res = analyze(
            record["alpha_beta"],
            start,
            stop,
            step,
            record["prescriptions"],
            method=method,
            bound_pct=bound_pct,
        )
        res["reference_prescription"] = reference_schedule(record)
        result_sets.append({"name": record["name"], "id": record.get("id"), "result": res})
    return {"method": method, "bound_pct": float(bound_pct), "sets": result_sets}


def export(result, folder):
    """Export optimization grids, calibrated prescriptions, and the profile plot."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    multi = "sets" in result
    datasets = result["sets"] if multi else [{"name": None, "result": result}]
    if not datasets:
        raise ValueError("No optimization results to export.")
    with (folder / "grid_results.csv").open("w", newline="", encoding="utf-8-sig") as fp:
        w = csv.writer(fp)
        if not multi or len(datasets) == 1:
            r = datasets[0]["result"]
            names = [p.name for p in r["schedules"]]
            w.writerow(
                ["Dq_Gy", "mean_EQD2_Gy", "population_SD_Gy", "range_Gy"]
                + [f"EQD2_{name}_Gy" for name in names]
                + ["sample_SD_Gy", "SSE_Gy2", "alpha_beta_Gy"]
            )
            for j, q in enumerate(r["q"]):
                w.writerow(
                    [q, r["mean"][j], r["sd"][j], r["spread"][j]]
                    + list(r["eqd2"][j])
                    + [r["sample_sd"][j], r["sse"][j], float(_branch_alpha_beta(r["ab"], q))]
                )
        else:
            header = ["Dq_Gy"]
            for dataset in datasets:
                name = dataset["name"]
                header += [
                    f"{name}: mean_EQD2_Gy",
                    f"{name}: population_SD_Gy",
                    f"{name}: sample_SD_Gy",
                    f"{name}: SSE_Gy2",
                    f"{name}: range_Gy",
                ]
                header += [f"{name}: EQD2_{row.name}_Gy" for row in dataset["result"]["schedules"]]
            w.writerow(header)
            # Each set has its own refined minimum. Evaluate all curves on the
            # union of candidate grids, rather than pairing unequal row indices.
            common_q = np.unique(np.concatenate([dataset["result"]["q"] for dataset in datasets]))
            aligned = []
            for dataset in datasets:
                r = dataset["result"]
                eqd2 = evaluate(r["ab"], common_q, r["schedules"])
                means = eqd2.mean(axis=1)
                aligned.append(
                    (
                        eqd2,
                        means,
                        eqd2.std(axis=1),
                        eqd2.std(axis=1, ddof=1),
                        np.sum((eqd2 - means[:, None]) ** 2, axis=1),
                        np.ptp(eqd2, axis=1),
                    )
                )
            for i, q in enumerate(common_q):
                values = [q]
                for eqd2, means, sd, sample_sd, sse, spread in aligned:
                    values += [means[i], sd[i], sample_sd[i], sse[i], spread[i]]
                    values += list(eqd2[i])
                w.writerow([value if np.isfinite(value) else None for value in values])
    with (folder / "optimal_schedules.csv").open("w", newline="", encoding="utf-8-sig") as fp:
        w = csv.writer(fp)
        if multi and len(datasets) > 1:
            w.writerow(
                [
                    "prescriptions_set",
                    "alpha_beta_Gy",
                    "optimization_method",
                    "optimal_Dq_Gy",
                    "schedule",
                    "total_Gy",
                    "fractions",
                    "dose_per_fraction_Gy",
                    "optimal_EQD2_Gy",
                    "local_control_pct",
                ]
            )
        else:
            w.writerow(
                ["schedule", "total_Gy", "fractions", "dose_per_fraction_Gy", "optimal_EQD2_Gy", "local_control_pct"]
            )
        for dataset in datasets:
            r = dataset["result"]
            i = r["idx"]
            for p, value in zip(r["schedules"], r["eqd2"][i]):
                values = [p.name, p.total_gy, p.fractions, p.dose_per_fraction_gy, value, p.local_control_pct]
                if multi and len(datasets) > 1:
                    values = [dataset["name"], r["signed_ab"], r["method"], r["q"][i]] + values
                w.writerow(values)
    # Export the optimization profile at 160 ppi.
    export_image(result, folder / "range.png", ppi=160)


IMAGE_FORMATS = {"PDF": ".pdf", "JPG": ".jpg", "PNG": ".png", "EPS": ".eps", "SVG": ".svg"}
IMAGE_EXTENSIONS = {*IMAGE_FORMATS.values(), ".jpeg"}


def _validated_ppi(value):
    try:
        ppi = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("PPI must be an integer between 1 and 1200.") from exc
    if not math.isfinite(ppi) or not ppi.is_integer() or not 1 <= ppi <= 1200:
        raise ValueError("PPI must be an integer between 1 and 1200.")
    return int(ppi)


def export_image(result, filename, ppi=300):
    """Save the current optimization plot in PDF, JPEG, PNG, EPS, or SVG format.

    PDF, EPS and SVG preserve vector linework and lettering. PPI controls
    raster output resolution and any rasterized artists in a vector file.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    if result is None or ("sets" in result and not result["sets"]):
        raise ValueError("Run optimization before exporting an image.")
    destination = Path(filename)
    ext = destination.suffix.casefold()
    if ext not in IMAGE_EXTENSIONS:
        raise ValueError("Image format must be PDF, JPG, PNG, EPS, or SVG.")
    ppi = _validated_ppi(ppi)
    if not destination.parent.is_dir():
        raise FileNotFoundError(f"Output folder does not exist: {destination.parent}")
    settings = normalized_plot_settings(result.get("plot_settings"))
    fig = Figure(figsize=(settings["width_in"], settings["height_in"]), dpi=100)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    draw_range_plot(ax, result)
    if ext == ".eps":
        # PostScript cannot represent transparency. Composite translucent
        # figure elements onto white before exporting the vector graphic.
        from matplotlib.colors import to_rgb

        def opaque_on_white(artist, color):
            alpha = artist.get_alpha()
            if alpha is not None and alpha < 1:
                rgb = to_rgb(color)
                artist.set_color(tuple(alpha * c + (1 - alpha) for c in rgb))
                artist.set_alpha(1)

        for patch in ax.patches:
            alpha = patch.get_alpha()
            if alpha is not None and alpha < 1:
                rgb = to_rgb(patch.get_facecolor())
                patch.set_facecolor(tuple(alpha * c + (1 - alpha) for c in rgb))
                patch.set_alpha(1)
        for line in ax.get_lines():
            opaque_on_white(line, line.get_color())
        for gridline in ax.get_xgridlines() + ax.get_ygridlines():
            opaque_on_white(gridline, gridline.get_color())
        for t in ax.texts:
            box = t.get_bbox_patch()
            if box is not None and box.get_alpha() is not None and box.get_alpha() < 1:
                a = box.get_alpha()
                rgb = to_rgb(box.get_facecolor())
                box.set_facecolor(tuple(a * c + (1 - a) for c in rgb))
                box.set_alpha(1)
        if ax.get_legend() is not None:
            ax.get_legend().get_frame().set_alpha(1.0)
    # The plotting routine already sets margins around the axis-bound Dq tag.
    # Avoid tight_layout here: it otherwise reintroduces a large empty bottom.
    ext_format = "jpeg" if ext in (".jpg", ".jpeg") else ext[1:]
    fig.savefig(
        destination,
        format=ext_format,
        dpi=ppi,
        facecolor="white",
        bbox_inches="tight" if settings["tight_crop"] else None,
        pad_inches=0.07,
    )
    return destination


def read_prescriptions_set_csv(path):
    """Read one prescriptions set without changing the database.

    Required columns: name,total_gy,fractions (also accepts prescription,D,n).
    Optional prescriptions_set and alpha_beta columns describe one consistent set.
    An is_reference column may mark one prescription as the reference.
    The optional local_control_pct column holds observed percentages.
    The tumor header is also accepted. Plain CSVs use the filename as set name
    and leave alpha/beta blank for the import editor. UTF-8 BOM and comma/semicolon/tab delimiters are accepted.
    """
    aliases = {
        "name": "name",
        "prescription": "name",
        "prescription_name": "name",
        "schedule": "name",
        "schedule_name": "name",
        "total_gy": "total_gy",
        "d": "total_gy",
        "d_gy": "total_gy",
        "total_dose": "total_gy",
        "total_dose_gy": "total_gy",
        "fractions": "fractions",
        "n": "fractions",
        "n_fractions": "fractions",
        "tumor": "prescriptions_set",
        "tumour": "prescriptions_set",
        "tumor_name": "prescriptions_set",
        "tumour_name": "prescriptions_set",
        "prescriptions_set": "prescriptions_set",
        "prescription_set": "prescriptions_set",
        "prescriptions_set_name": "prescriptions_set",
        "prescription_set_name": "prescriptions_set",
        "set": "prescriptions_set",
        "set_name": "prescriptions_set",
        "alpha_beta": "alpha_beta",
        "alpha_over_beta": "alpha_beta",
        "alpha_beta_gy": "alpha_beta",
        "ab": "alpha_beta",
        "is_reference": "is_reference",
        "reference": "is_reference",
        "local_control_pct": "local_control_pct",
        "local_tumor_control_pct": "local_control_pct",
        "local_tumor_control": "local_control_pct",
    }

    def column_name(value):
        value = str(value).strip().casefold().replace("α", "alpha").replace("β", "beta")
        value = value.replace("%", "pct")
        return aliases.get(re.sub(r"[\s/()\-]+", "_", value).strip("_"), value)

    schedules, set_names, alpha_betas, references = [], [], [], []
    try:
        with open(path, encoding="utf-8-sig", newline="") as fp:
            sample = fp.read(8192)
            fp.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
            except csv.Error:
                dialect = csv.excel
            reader = csv.DictReader(fp, dialect=dialect, strict=True)
            headers = [column_name(value) for value in (reader.fieldnames or [])]
            if len(headers) != len(set(headers)):
                raise ValueError("CSV has duplicate column headings.")
            if not {"name", "total_gy", "fractions"}.issubset(headers):
                raise ValueError(
                    "CSV requires name,total_gy,fractions columns (or prescription,D,n). "
                    "Optional columns: prescriptions_set,alpha_beta,is_reference,local_control_pct."
                )
            reader.fieldnames = headers
            for row in reader:
                line = reader.line_num
                if None in row:
                    raise ValueError(
                        f"CSV line {line}: too many fields; quote names containing the delimiter."
                    )
                if not any(str(value or "").strip() for value in row.values()):
                    continue
                if any(
                    row.get(key) is None or not row[key].strip()
                    for key in ("name", "total_gy", "fractions")
                ):
                    raise ValueError(
                        f"CSV line {line}: prescription name, total dose, and fractions are required."
                    )
                label = row["name"].strip()
                try:
                    total, fx = float(row["total_gy"]), float(row["fractions"])
                except ValueError as exc:
                    raise ValueError(
                        f"CSV line {line}: total dose and fractions must be numeric."
                    ) from exc
                if (
                    not math.isfinite(total)
                    or total <= 0
                    or not math.isfinite(fx)
                    or fx <= 0
                    or not fx.is_integer()
                ):
                    raise ValueError(
                        f"CSV line {line}: total dose must be positive and fractions a positive integer."
                    )
                try:
                    prescription = Prescription(label, total, int(fx), row.get("local_control_pct"))
                except ValueError as exc:
                    raise ValueError(f"CSV line {line}: {exc}") from exc
                schedules.append(prescription)
                flag = (row.get("is_reference") or "").strip().casefold()
                if flag not in ("", "0", "false", "no", "1", "true", "yes"):
                    raise ValueError(f"CSV line {line}: is_reference must be 0 or 1.")
                if flag in ("1", "true", "yes"):
                    references.append(schedules[-1])
                set_name = (row.get("prescriptions_set") or "").strip()
                if set_name:
                    set_names.append(set_name)
                ab = (row.get("alpha_beta") or "").strip()
                if ab:
                    try:
                        ab = float(ab)
                    except ValueError as exc:
                        raise ValueError(
                            f"CSV line {line}: enter a positive finite alpha_beta for q >= 0."
                        ) from exc
                    if not math.isfinite(ab) or ab <= 0:
                        raise ValueError(
                            f"CSV line {line}: enter a positive finite alpha_beta for q >= 0."
                        )
                    alpha_betas.append(validated_alpha_beta(ab))
    except (csv.Error, UnicodeError) as exc:
        raise ValueError(f"Cannot read CSV. Use UTF-8 text with a header row: {exc}") from exc
    schedules = validate_schedules(schedules)
    if len({p.name.casefold() for p in schedules}) != len(schedules):
        raise ValueError("CSV prescription names must be unique within the prescriptions set.")
    if len(references) > 1:
        raise ValueError("Choose only one reference prescription per prescriptions set.")
    if len({name.casefold() for name in set_names}) > 1:
        raise ValueError(
            "Import one prescriptions set per CSV: the set column contains different names."
        )
    if alpha_betas and any(
        not math.isclose(value, alpha_betas[0], rel_tol=1e-12) for value in alpha_betas
    ):
        raise ValueError(
            "Import one alpha/beta value per prescriptions set: values differ between rows."
        )
    return {
        "name": set_names[0] if set_names else Path(path).stem,
        "alpha_beta": alpha_betas[0] if alpha_betas else None,
        "prescriptions": schedules,
        "reference_prescription": references[0] if references else None,
    }


def export_prescriptions_set(record, filename):
    """Export a configured set as CSV, including its alpha/beta, for re-import."""
    name = str(record["name"]).strip()
    if not name:
        raise ValueError("Prescriptions set name is required.")
    ab = validated_alpha_beta(record["alpha_beta"])
    rows = validate_schedules(record["prescriptions"])
    path = Path(filename)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "name", "total_gy", "fractions", "prescriptions_set", "alpha_beta", "is_reference",
                "local_control_pct",
            ]
        )
        reference = record.get("reference_prescription")
        reference_key = Prescription.from_value(reference).schedule_key if reference is not None else None
        for p in rows:
            writer.writerow([p.name, p.total_gy, p.fractions, name, ab,
                             int(p.schedule_key == reference_key), p.local_control_pct])
    return path


def draw_range_plot(ax, result):
    """Draw the chosen RD EQD2 criterion using the shared GUI/export settings."""
    from textwrap import fill

    from matplotlib import rc_context, rcParams
    from matplotlib.colors import to_rgba
    from matplotlib.offsetbox import AnchoredOffsetbox, PaddedBox, TextArea, VPacker

    ax.clear()
    datasets = result["sets"] if "sets" in result else [{"name": None, "result": result}]
    if not datasets:
        return
    settings = normalized_plot_settings(result.get("plot_settings"))
    method = result["method"]
    metric = {"sse": "sse", "sd": "sample_sd", "range": "spread"}[method]
    cycle = rcParams["axes.prop_cycle"].by_key().get("color", ["#1f77b4", "#ff7f0e"])
    first_color = settings["curve_color"] or cycle[0]
    palette = [first_color] + [color for color in cycle if to_rgba(color) != to_rgba(first_color)]
    summary_boxes = []
    with rc_context({"font.family": settings["font_family"]}):
        for pos, dataset in enumerate(datasets):
            r = dataset["result"]
            values = r[metric]
            i = r["idx"]
            name = dataset["name"]
            label = (
                f"{name} (α/β={r['ab']:g} Gy at q ≥ 0)" if name else OPTIMIZATION_METHODS[method]
            )
            # Explicit colours prevent the second curve from reusing the first
            # cycle colour when a custom first-curve colour is supplied.
            color_arg = {"color": palette[pos % len(palette)]}
            (line,) = ax.plot(
                r["q"],
                values,
                label=label,
                linewidth=settings["line_width"],
                linestyle=settings["line_style"],
                **color_arg,
            )
            color = line.get_color()
            bounds_color = settings["bounds_color"] or color
            dq = float(r["q"][i])
            if settings["show_minimum"]:
                ax.plot(
                    [dq],
                    [values[i]],
                    marker="o",
                    linestyle="None",
                    zorder=5,
                    markersize=settings["minimum_size"],
                    color=color,
                )
            bound_label = (
                f"{r['bound_pct']:g}% {'range' if r['profile_measure'] == 'range' else 'SD'} bounds"
            )
            lower, upper = r["profile_bounds"]
            if lower is not None and upper is not None:
                if settings["show_bounds"]:
                    ax.axvspan(
                        lower, upper, alpha=settings["bounds_alpha"], color=bounds_color, zorder=0
                    )
                    for edge in (lower, upper):
                        ax.axvline(
                            edge,
                            ls="--",
                            lw=settings["bounds_width"],
                            alpha=0.7,
                            color=bounds_color,
                            zorder=1,
                        )
            objective_label = {"sse": "SSE", "sd": "Sample SD", "range": "Range"}[method]
            objective_unit = " Gy²" if method == "sse" else " Gy"
            if settings["show_summary"]:
                lines = (
                    [fill(name, width=29, break_long_words=False, break_on_hyphens=False)]
                    if name
                    else []
                )
                lines.append(f"{objective_label} min = {values[i]:.3f}{objective_unit}")
                lines.append(bound_label)
                lines.append(
                    f"{lower:.3f}–{upper:.3f} Gy"
                    if lower is not None and upper is not None
                    else "Outside search interval"
                )
                text = TextArea(
                    "\n".join(lines),
                    textprops={
                        "fontsize": settings["summary_font"],
                        "fontfamily": settings["font_family"],
                        "multialignment": "left",
                        "linespacing": 1.2,
                    },
                )
                summary_boxes.append(
                    PaddedBox(
                        text,
                        pad=5,
                        draw_frame=True,
                        patch_attrs={"facecolor": "white", "edgecolor": "#cccccc"},
                    )
                )
            if settings["show_minimum"]:
                ax.plot(
                    [dq],
                    [0],
                    marker="v",
                    markersize=5.5,
                    transform=ax.get_xaxis_transform(),
                    clip_on=False,
                    color=color,
                    zorder=5,
                )
                left_edge = float(r.get("search_start", r["q"][0]))
                right_edge = float(r.get("search_end", r["q"][-1]))
                span = right_edge - left_edge
                alignment = (
                    "left"
                    if dq <= left_edge + 0.15 * span
                    else "right"
                    if dq >= right_edge - 0.15 * span
                    else "center"
                )
                ax.annotate(
                    r"$D_q$" + f" = {dq:.4f} Gy",
                    xy=(dq, 0),
                    xycoords=ax.get_xaxis_transform(),
                    xytext=(0, -17 - 16 * pos),
                    textcoords="offset points",
                    fontsize=settings["dq_font"],
                    fontfamily=settings["font_family"],
                    ha=alignment,
                    va="top",
                    color=color,
                    clip_on=False,
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.5},
                )
        if summary_boxes:
            stack = VPacker(children=summary_boxes, align="right", pad=0, sep=8)
            summaries = AnchoredOffsetbox(
                loc="upper right", child=stack, frameon=False, pad=0.5, borderpad=0.6
            )
            summaries.set_zorder(7)
            ax.add_artist(summaries)
        ax.set_xlabel(
            settings["x_label"] or "Candidate parameter ($q$) [Gy]",
            fontsize=settings["axis_font"],
            fontfamily=settings["font_family"],
        )
        ax.set_ylabel(
            settings["y_label"] or PLOT_Y_LABELS[method],
            fontsize=settings["axis_font"],
            fontfamily=settings["font_family"],
        )
        if settings["show_title"]:
            ax.set_title(
                plot_title_for(result),
                fontsize=settings["title_font"],
                fontfamily=settings["font_family"],
            )
        else:
            ax.set_title("")
        ax.tick_params(axis="both", labelsize=settings["tick_font"])
        for label in (*ax.get_xticklabels(), *ax.get_yticklabels()):
            label.set_fontfamily(settings["font_family"])
        if settings["show_grid"]:
            ax.grid(True, alpha=settings["grid_alpha"], linestyle=settings["grid_style"])
        if settings["show_legend"]:
            ax.legend(
                loc="upper left",
                fontsize=settings["legend_font"],
                prop={"family": settings["font_family"], "size": settings["legend_font"]},
            )
        start = min(
            float(ds["result"].get("search_start", ds["result"]["q"][0])) for ds in datasets
        )
        stop = max(float(ds["result"].get("search_end", ds["result"]["q"][-1])) for ds in datasets)
        if start == stop:
            padding = max(1.0, abs(start)) * 0.05
            ax.set_xlim(start - padding, stop + padding)
        else:
            ax.set_xlim(start, stop)
        low, high = settings["y_min"], settings["y_max"]
        if low != "" or high != "":
            current = ax.get_ylim()
            ax.set_ylim(
                bottom=low if low != "" else current[0], top=high if high != "" else current[1]
            )
        ax.xaxis.labelpad = (
            (26 + 16 * max(0, len(datasets) - 1)) if settings["show_minimum"] else 10
        )
        # Responsive margins for the Dq axis tag and variable font settings.
        # These are proportional to the figure height and are used for both GUI and exports.
        bottom = (0.175 + 0.055 * max(0, len(datasets) - 1)) if settings["show_minimum"] else 0.15
        physical_bottom = (
            ax.xaxis.labelpad + settings["tick_font"] + settings["axis_font"] + 12
        ) / 72
        bottom = max(bottom, physical_bottom / max(1, ax.figure.get_figheight()))
        if not settings["show_title"]:
            top = 0.965
        else:
            top = (
                0.85
                if settings["title_font"] >= 22
                else (0.88 if settings["title_font"] >= 15 else 0.91)
            )
        ax.figure.subplots_adjust(left=0.14, right=0.975, top=top, bottom=min(0.48, bottom))


# Detailed post-optimization analysis, shared by the GUI and exports.
@lru_cache(maxsize=256)
def _t95(df):
    """Two-sided Student-t 95% critical value (mpmath ships with SymPy)."""
    import mpmath as mp

    lo, hi = 0.0, 128.0
    for _ in range(65):
        t = (lo + hi) / 2
        tail = float(mp.betainc(df / 2, 0.5, 0, df / (df + t * t), regularized=True)) / 2
        if tail > 0.025:
            lo = t
        else:
            hi = t
    return (lo + hi) / 2


def _distribution(values, doses):
    values, doses = np.asarray(values, float), np.asarray(doses, float)
    if not np.all(np.isfinite(values)):
        return dict(
            N=len(values),
            **dict.fromkeys(
                (
                    "mean",
                    "SD",
                    "RSD",
                    "minimum",
                    "maximum",
                    "full_range",
                    "SSE",
                    "intercept",
                    "slope",
                    "delta_trend",
                    "eta",
                    "CI95_lower",
                    "CI95_upper",
                )
            ),
        )
    mean, sd = float(values.mean()), float(values.std(ddof=1))
    x = doses - doses.mean()
    denominator = float(x @ x)
    slope = float(x @ (values - mean) / denominator) if denominator > 0 else None
    intercept = mean - slope * float(doses.mean()) if slope is not None else None
    delta = abs(slope) * float(np.ptp(doses)) if slope is not None else None
    half = _t95(len(values) - 1) * sd / math.sqrt(len(values))
    return {
        "N": len(values),
        "mean": mean,
        "SD": sd,
        "RSD": 100 * sd / abs(mean) if mean != 0 else None,
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "full_range": float(np.ptp(values)),
        "SSE": float(np.sum((values - mean) ** 2)),
        "intercept": intercept,
        "slope": slope,
        "delta_trend": delta,
        "eta": 100 * delta / abs(mean) if delta is not None and mean != 0 else None,
        "CI95_lower": mean - half,
        "CI95_upper": mean + half,
    }


def build_analysis(dataset):
    """Calculate per-prescription BED, EQD2, and descriptive statistics."""
    r = dataset["result"]
    i, ab = r["idx"], r["ab"]
    dq = float(r["q"][i])
    input_ab = ab
    ab = float(_branch_alpha_beta(input_ab, dq))
    reserves, rates = rd_parameters(input_ab, np.array([dq]))
    reserve = float(reserves[0])
    s = float(rates[0]) if dq != 0 else None
    beds = fraction_bed(input_ab, [dq], [p.dose_per_fraction_gy for p in r["schedules"]])[0]
    prescriptions = []
    for p, eqd2, per_fraction_bed in zip(r["schedules"], r["eqd2"][i], beds):
        name, total, n = p.schedule_key
        d = p.dose_per_fraction_gy
        bed = n * per_fraction_bed
        lq_bed = total * (1 + d / ab)
        lq_eqd2 = total * (ab + d) / (ab + REFERENCE_DOSE_GY) if ab != -REFERENCE_DOSE_GY else None
        prescriptions.append(
            {
                "prescription": name,
                "D_Gy": total,
                "n": n,
                "d_Gy": d,
                "RD_BED_Gy": bed,
                "RD_EQD2_Gy": float(eqd2),
                "LQ_BED_Gy": lq_bed,
                "LQ_EQD2_Gy": lq_eqd2,
                "local_control_pct": p.local_control_pct,
                "RD_deviation_Gy": float(eqd2 - r["mean"][i]),
                "RD_minus_LQ_Gy": float(eqd2 - lq_eqd2) if lq_eqd2 is not None else None,
            }
        )
    doses = [p["d_Gy"] for p in prescriptions]
    models = {
        model: _distribution([p[model + "_EQD2_Gy"] for p in prescriptions], doses)
        for model in ("RD", "LQ")
    }
    lo, hi = r["profile_bounds"]
    reference_row = reference_schedule(
        {
            "prescriptions": r["schedules"],
            "reference_prescription": dataset.get(
                "reference_prescription", r.get("reference_prescription")
            ),
        }
    )
    reference = None
    if reference_row is not None:
        reference = float(EQD2_FN(input_ab, dq, reference_row.dose_per_fraction_gy,
                                   reference_row.fractions))
    analysis = {
        "name": dataset.get("name") or "Prescriptions set",
        "alpha_beta": ab,
        "input_alpha_beta": input_ab,
        "D_q": dq,
        "s": s,
        "r": reserve,
        "method": r["method"],
        "prescriptions": prescriptions,
        "models": models,
        "bounds": (lo, hi),
        "bound_pct": r["bound_pct"],
        "profile_measure": r["profile_measure"],
        "reference": reference,
        "reference_prescription": reference_row,
        "boundary": i in (0, len(r["q"]) - 1),
        "source_result": r,
        "robustness": None,
    }
    analysis["clinical"] = clinical_evaluation(analysis)
    return analysis


def _clinical_cluster(points):
    """Find a compact strict majority separated by a gap on either axis."""
    count = len(points)
    minimum = max(3, count // 2 + 1)
    widths = np.ptp(points, axis=0)
    best = None
    best_score = None
    for axis in (0, 1):
        other = 1 - axis
        order = np.argsort(points[:, axis], kind="stable")
        ordered = points[order]
        for left in range(count - minimum + 1):
            for right in range(left + minimum, count + 1):
                gaps = []
                if left:
                    gaps.append(ordered[left, axis] - ordered[left - 1, axis])
                if right < count:
                    gaps.append(ordered[right, axis] - ordered[right - 1, axis])
                axis_span = ordered[right - 1, axis] - ordered[left, axis]
                if not gaps or min(gaps) <= axis_span:
                    continue
                span = np.ptp(ordered[left:right], axis=0)
                if span[other] > 0.75 * widths[other]:
                    continue
                relative_span = np.divide(span, widths, out=np.zeros(2), where=widths > 0)
                score = (-(right - left), np.prod(relative_span), np.sum(relative_span))
                if best_score is None or score < best_score:
                    best, best_score = order[left:right], score
    return best


def _fit_control_beta(eqd2, control, varying_dose):
    """Fit a logit mean and common precision by beta maximum likelihood."""
    n = len(control)
    if n < (4 if varying_dose else 3):
        return None, (
            "Beta regression requires at least four outcomes with varying EQD2, "
            "or three for a constant-EQD2 fit."
        )
    y = control / 100
    if np.ptp(y) <= 1e-12:
        return None, "A beta fit requires variation in observed local control."
    from scipy.optimize import minimize
    from scipy.special import betaln, digamma, expit

    endpoints = bool(np.any((y == 0) | (y == 1)))
    if endpoints:
        y = (y * (n - 1) + 0.5) / n

    center = float(eqd2.mean())
    scale = float(np.max(np.abs(eqd2 - center))) if varying_dose else 1.0
    design = np.ones((n, 2 if varying_dose else 1))
    if varying_dose:
        design[:, 1] = (eqd2 - center) / scale
    log_y, log_one_minus_y = np.log(y), np.log1p(-y)
    logit_y = log_y - log_one_minus_y

    def objective(params):
        mean = expit(design @ params[:-1])
        phi = np.exp(params[-1])
        a, b = mean * phi, (1 - mean) * phi
        likelihood = (a - 1) * log_y + (b - 1) * log_one_minus_y - betaln(a, b)
        a_score = log_y - digamma(a) + digamma(phi)
        b_score = log_one_minus_y - digamma(b) + digamma(phi)
        mean_score = phi * mean * (1 - mean) * (a_score - b_score)
        gradient = np.r_[design.T @ mean_score, a @ a_score + b @ b_score]
        return -likelihood.mean(), -gradient / n

    initial = np.clip(np.linalg.lstsq(design, logit_y, rcond=None)[0], -12, 12)
    mu = expit(design @ initial)
    residual_var = np.mean((y - mu) ** 2)
    start_phi = np.clip(np.mean(mu * (1 - mu)) / max(residual_var, 1e-12) - 1, 0.1, 1e4)
    bounds = [(-15, 15)] * len(initial) + [(math.log(1e-3), math.log(1e6))]
    fit = minimize(
        objective,
        np.r_[initial, math.log(start_phi)],
        jac=True,
        method="L-BFGS-B",
        bounds=bounds,
        options={"gtol": 1e-7, "ftol": 1e-12},
    )
    if not fit.success or not np.isfinite(fit.fun) or any(
        value - lo < 1e-4 or hi - value < 1e-4 for value, (lo, hi) in zip(fit.x, bounds)
    ):
        return None, "The outcomes do not support a stable beta fit."
    return {
        "center_Gy": center,
        "intercept": float(fit.x[0]),
        "slope_per_Gy": float(fit.x[1] / scale) if varying_dose else None,
        "precision": math.exp(fit.x[-1]),
        "adjusted_endpoints": endpoints,
    }, ""


def _beta_control_prediction(fit, eqd2):
    """Return the fitted mean and central 95% beta percentiles, in percent."""
    from scipy.special import betaincinv, expit

    eqd2 = np.asarray(eqd2)
    mean = expit(fit["intercept"] + (fit["slope_per_Gy"] or 0) * (eqd2 - fit["center_Gy"]))
    a, b = mean * fit["precision"], (1 - mean) * fit["precision"]
    return 100 * mean, 100 * betaincinv(a, b, 0.025), 100 * betaincinv(a, b, 0.975)


def _clinical_area_ratio(points, rows):
    """RD enclosing area as a percentage of the LQ area for the same observations."""
    lq = [row["LQ_EQD2_Gy"] for row in rows]
    if any(x is None or not math.isfinite(x) for x in lq):
        return None
    lq_width = max(lq) - min(lq)
    if lq_width <= 0 or np.ptp(points[:, 1]) <= 0:
        return None
    return float(100 * np.ptp(points[:, 0]) / lq_width)


def clinical_evaluation(analysis):
    """Describe observed local control against RD EQD2 at the calibrated parameters."""
    observations = [row for row in analysis["prescriptions"] if row["local_control_pct"] is not None]
    if not observations:
        return None
    points = np.array([(r["RD_EQD2_Gy"], r["local_control_pct"]) for r in observations])
    eqd2, control = points.T
    mean = points.mean(axis=0)
    trend = None
    varying_dose = not math.isclose(float(eqd2.min()), float(eqd2.max()), rel_tol=1e-12)
    if varying_dose:
        dx, dy = eqd2 - mean[0], control - mean[1]
        slope = float(dx @ dy / (dx @ dx))
        sst = float(dy @ dy)
        trend = {
            "slope": slope,
            "intercept": float(mean[1] - slope * mean[0]),
            "r_squared": (
                float(np.clip(1 - np.sum((dy - slope * dx) ** 2) / sst, 0, 1)) if sst > 0 else None
            ),
        }

    selected = _clinical_cluster(points)
    cluster = None
    if selected is not None:
        subgroup = points[selected]
        cluster = {
            "minimum": subgroup.min(axis=0),
            "maximum": subgroup.max(axis=0),
            "indices": selected,
            "area_percent": _clinical_area_ratio(subgroup, [observations[i] for i in selected]),
        }
    beta, beta_status = _fit_control_beta(eqd2, control, varying_dose)
    return {
        "name": analysis["name"],
        "observations": observations,
        "points": points,
        "mean": mean,
        "minimum": points.min(axis=0),
        "maximum": points.max(axis=0),
        "linear": trend,
        "cluster": cluster,
        "area_percent": _clinical_area_ratio(points, observations),
        "beta": beta,
        "beta_status": beta_status,
    }


def clinical_basic_summary(clinical):
    mean, low, high = (clinical[key] for key in ("mean", "minimum", "maximum"))
    trend = clinical["linear"]
    rows = [
        ("Observed prescriptions", str(len(clinical["observations"]))),
        ("Mean RD EQD2", f"{mean[0]:.2f} Gy"),
        ("Mean local control", f"{mean[1]:.2f}%"),
        ("RD EQD2 range", f"{low[0]:.2f}–{high[0]:.2f} Gy"),
        ("Local control range", f"{low[1]:.2f}–{high[1]:.2f}%"),
        (
            "Linear trend",
            f"{mean[1]:.2f} {trend['slope']:+.3g} × (EQD2 − {mean[0]:.2f})"
            if trend else "Unavailable at constant EQD2",
        ),
    ]
    if trend and trend["r_squared"] is not None:
        rows.append(("R²", f"{trend['r_squared']:.3f}"))
    if clinical["area_percent"] is not None:
        rows.append(("RD area / LQ area", f"{clinical['area_percent']:.1f}%"))
    if clinical["cluster"] is not None:
        cluster = clinical["cluster"]
        eqd2_width, control_width = cluster["maximum"] - cluster["minimum"]
        rows.append(("Cluster", f"{len(cluster['indices'])} prescriptions"))
        rows.append(("Cluster widths", f"{eqd2_width:.2f} Gy × {control_width:.2f} pp"))
        if cluster["area_percent"] is not None:
            rows.append(("Cluster RD area / LQ area", f"{cluster['area_percent']:.1f}%"))
    return rows


def clinical_beta_summary(clinical):
    fit = clinical["beta"]
    if fit is None:
        return clinical["beta_status"]
    mean, low, high = _beta_control_prediction(fit, fit["center_Gy"])
    slope = (
        f"Logit slope {fit['slope_per_Gy']:+.4g} per Gy"
        if fit["slope_per_Gy"] is not None else "Constant-EQD2 fit"
    )
    summary = (
        f"{slope} · Precision {fit['precision']:.4g}\n"
        f"At {fit['center_Gy']:.2f} Gy, fitted mean {mean:.2f}% and "
        f"central 95% beta interval {low:.2f}–{high:.2f}%."
    )
    if fit["adjusted_endpoints"]:
        summary += (
            "\nWith 0% or 100% present, the beta fit transforms all proportions y "
            "to (y × (N − 1) + 0.5) / N."
        )
    return summary


def analysis_summary(a):
    rd = a["models"]["RD"]
    rows = [
        ("Prescriptions set", a["name"]),
        ("Prescriptions (N)", rd["N"]),
        (
            "Reference prescription",
            a["reference_prescription"].name if a["reference_prescription"] else "N/A",
        ),
        ("Reference EQD2 (Gy)", a["reference"]),
        ("Selected α/β (Gy)", a["alpha_beta"]),
        ("Input α/β (Gy, q ≥ 0)", a["input_alpha_beta"]),
        ("Dq (Gy)", a["D_q"]),
        ("s (Gy⁻¹)", a["s"]),
        ("r", a["r"]),
        ("Optimization criterion", OPTIMIZATION_METHODS[a["method"]]),
        ("Minimum location", "Search boundary" if a["boundary"] else "Interior"),
        ("Minimum fraction dose (Gy)", min(p["d_Gy"] for p in a["prescriptions"])),
        ("Maximum fraction dose (Gy)", max(p["d_Gy"] for p in a["prescriptions"])),
    ]
    names = [
        ("Mean EQD2 (Gy)", "mean"),
        ("Sample SD (Gy)", "SD"),
        ("RSD (%)", "RSD"),
        ("Minimum EQD2 (Gy)", "minimum"),
        ("Maximum EQD2 (Gy)", "maximum"),
        ("Full EQD2 range (Gy)", "full_range"),
        ("SSE (Gy²)", "SSE"),
        ("Trend intercept (Gy)", "intercept"),
        ("Trend slope", "slope"),
        ("Absolute trend excursion (Gy)", "delta_trend"),
        ("η (%)", "eta"),
        ("Mean 95% CI lower (Gy)", "CI95_lower"),
        ("Mean 95% CI upper (Gy)", "CI95_upper"),
    ]
    for model in ("RD", "LQ"):
        rows.extend((model + " " + label, a["models"][model][key]) for label, key in names)
    rows.extend(
        [
            (
                f"{a['bound_pct']:g}% {a['profile_measure'].upper()} profile lower (Gy)",
                a["bounds"][0],
            ),
            (
                f"{a['bound_pct']:g}% {a['profile_measure'].upper()} profile upper (Gy)",
                a["bounds"][1],
            ),
        ]
    )
    return rows


def run_robustness(a, alpha_beta_values=None):
    r = a["source_result"]
    start = float(r.get("search_start", r["q"][0]))
    stop = float(r.get("search_end", r["q"][-1]))
    step = float(r["search_step"]) if "search_step" in r else float(np.median(np.diff(r["q"])))
    rows = r["schedules"]
    held_out = []
    if len(rows) >= 3:
        for omitted in range(len(rows)):
            subset = rows[:omitted] + rows[omitted + 1 :]
            fitted = analyze(
                a["input_alpha_beta"],
                start,
                stop,
                step,
                subset,
                method=a["method"],
                bound_pct=a["bound_pct"],
            )
            detail = build_analysis({"name": a["name"], "result": fitted})
            dq = detail["D_q"]
            name, total, n = rows[omitted].schedule_key
            e = float(evaluate(a["input_alpha_beta"], np.array([dq]), rows)[0, omitted])
            held_out.append(
                {
                    "omitted_prescription": name,
                    "D_q_Gy": dq,
                    "mean_EQD2_Gy": detail["models"]["RD"]["mean"],
                    "RSD_percent": detail["models"]["RD"]["RSD"],
                    "eta_percent": detail["models"]["RD"]["eta"],
                    "held_out_EQD2_Gy": e,
                    "held_out_deviation_Gy": e - detail["models"]["RD"]["mean"],
                    "boundary": detail["boundary"],
                }
            )
    sensitivity = []
    for ab in alpha_beta_values or [
        a["input_alpha_beta"] * 2 / 3,
        a["input_alpha_beta"],
        a["input_alpha_beta"] * 4 / 3,
    ]:
        ab = validated_alpha_beta(ab)
        fitted = analyze(ab, start, stop, step, rows, method=a["method"], bound_pct=a["bound_pct"])
        detail = build_analysis({"name": a["name"], "result": fitted})
        sensitivity.append(
            {
                "alpha_beta_Gy": ab,
                "fitted_alpha_beta_Gy": detail["alpha_beta"],
                "D_q_Gy": detail["D_q"],
                "mean_EQD2_Gy": detail["models"]["RD"]["mean"],
                "SD_Gy": detail["models"]["RD"]["SD"],
                "RSD_percent": detail["models"]["RD"]["RSD"],
                "eta_percent": detail["models"]["RD"]["eta"],
                "boundary": detail["boundary"],
            }
        )
    # Horizontal-trend calibration uses the same search interval as the optimization.
    grid = r["q"]
    doses = np.array([p.dose_per_fraction_gy for p in rows])
    centered = doses - doses.mean()
    flat = None
    if float(centered @ centered) > 0:

        def slope(q):
            vals = evaluate(a["input_alpha_beta"], np.array([q]), rows)[0]
            return abs(float(centered @ (vals - vals.mean()) / (centered @ centered)))

        j = min(range(len(grid)), key=lambda j: slope(float(grid[j])))
        qflat = (
            _bounded_minimum(slope, float(grid[j - 1]), float(grid[j + 1]))
            if 0 < j < len(grid) - 1
            else float(grid[j])
        )
        vals = evaluate(a["input_alpha_beta"], np.array([qflat]), rows)[0]
        flat = {
            "q_flat_Gy": qflat,
            "SD_Gy": float(vals.std(ddof=1)),
            "absolute_slope": slope(qflat),
        }
    a["robustness"] = {
        "leave_one_out": held_out,
        "alpha_beta_sensitivity": sensitivity,
        "horizontal_trend": flat,
        "held_out_RMSE_Gy": math.sqrt(np.mean([p["held_out_deviation_Gy"] ** 2 for p in held_out]))
        if held_out
        else None,
    }
    return a["robustness"]


def _analysis_stem(name):
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    clean = re.sub(r"[^A-Za-z0-9]+", "_", ascii_name).strip("_")[:65] or "set"
    return clean + "_" + hashlib.sha256(name.encode()).hexdigest()[:6]


def _latex_escape(value):
    mapping = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
        "α": r"$\alpha$",
        "β": r"$\beta$",
        "η": r"$\eta$",
        "²": r"$^2$",
        "⁻": r"$^-$",
        "¹": "1",
        "—": "---",
        "–": "--",
    }
    return "".join(mapping.get(c, c) for c in str(value)).replace(r"$^-$1", r"$^{-1}$")


def _analysis_value(value, digits=6):
    if value is None or isinstance(value, (float, np.floating)) and not math.isfinite(value):
        return "N/A"
    return f"{value:.{digits}g}" if isinstance(value, (float, np.floating)) else str(value)


def export_analysis_tables(a, folder, format="csv"):
    """Export one analyzed set; full-precision CSV/TSV/JSON or LaTeX tables."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    stem = _analysis_stem(a["name"])
    tables = {
        "prescriptions": a["prescriptions"],
        "summary": [{"metric": k, "value": v} for k, v in analysis_summary(a)],
        "trends": [dict(model=model, **stats) for model, stats in a["models"].items()],
    }
    clinical = a.get("clinical")
    if clinical:
        rows = clinical_basic_summary(clinical)
        rows.append(("Beta regression", clinical_beta_summary(clinical)))
        tables["clinical_evaluation"] = [{"metric": key, "value": value} for key, value in rows]
    if a["robustness"]:
        tables.update({k: v for k, v in a["robustness"].items() if isinstance(v, list)})
        robust_summary = {"held_out_RMSE_Gy": a["robustness"]["held_out_RMSE_Gy"]}
        robust_summary.update(a["robustness"]["horizontal_trend"] or {})
        tables["robustness_summary"] = [
            {"metric": k, "value": v} for k, v in robust_summary.items()
        ]
    written = []
    if format in ("csv", "tsv"):
        for label, rows in tables.items():
            if not rows:
                continue
            path = folder / f"{stem}_{label}.{format}"
            with path.open("w", encoding="utf-8-sig", newline="") as fp:
                writer = csv.DictWriter(
                    fp, fieldnames=list(rows[0]), delimiter="\t" if format == "tsv" else ","
                )
                writer.writeheader()
                writer.writerows(rows)
            written.append(path)
    elif format == "json":
        path = folder / f"{stem}_analysis.json"
        payload = {
            "name": a["name"],
            "alpha_beta": a["alpha_beta"],
            "D_q": a["D_q"],
            "tables": tables,
        }
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
        )
        written.append(path)
    elif format in ("tex", "latex"):
        path = folder / f"{stem}_tables.tex"
        lines = ["% Requires \\\\usepackage{graphicx} for wide tables."]
        headings = {
            "prescription": "Prescription",
            "D_Gy": r"$D$ (Gy)",
            "n": r"$n$",
            "d_Gy": r"$d$ (Gy)",
            "RD_BED_Gy": "RD BED (Gy)",
            "RD_EQD2_Gy": "RD EQD2 (Gy)",
            "LQ_BED_Gy": "LQ BED (Gy)",
            "LQ_EQD2_Gy": "LQ EQD2 (Gy)",
            "local_control_pct": r"Local tumor control (\%)",
            "RD_deviation_Gy": "RD deviation (Gy)",
            "RD_minus_LQ_Gy": "RD minus LQ (Gy)",
            "metric": "Quantity",
            "value": "Value",
            "model": "Model",
        }
        for label, rows in tables.items():
            if not rows:
                continue
            fields = list(rows[0])
            align = "l" + "r" * (len(fields) - 1)
            lines += [
                r"\begin{table}[htbp]",
                r"\centering",
                r"\small",
                r"\caption{" + _latex_escape(a["name"] + " — " + label.replace("_", " ")) + r"}",
                r"\resizebox{\linewidth}{!}{%",
                r"\begin{tabular}{" + align + "}",
                r"\hline",
                " & ".join(headings.get(k, _latex_escape(k.replace("_", " "))) for k in fields)
                + r" \\",
                r"\hline",
            ]
            lines += [
                " & ".join(_latex_escape(_analysis_value(row[k])) for k in fields) + r" \\"
                for row in rows
            ]
            lines += [r"\hline", r"\end{tabular}}", r"\end{table}", ""]
        path.write_text("\n".join(lines), encoding="utf-8")
        written.append(path)
    else:
        raise ValueError("Choose CSV, TSV, LaTeX, or JSON.")
    return written


def draw_dispersion_plot(fig, a, show_lq=True, show_labels=True, show_ci=True, reference=None):
    """Plot prescription EQD2 values and fitted trends against dose per fraction."""
    from matplotlib.ticker import MaxNLocator, MultipleLocator

    fig.clear()
    fig.set_facecolor("white")
    ax = fig.add_axes([0.115, 0.21, 0.86, 0.69])
    rows = sorted(a["prescriptions"], key=lambda p: p["d_Gy"])
    d = np.array([p["d_Gy"] for p in rows])
    rd = np.array([p["RD_EQD2_Gy"] for p in rows])
    lq = np.array([p["LQ_EQD2_Gy"] for p in rows])
    show_lq = show_lq and a["models"]["LQ"]["mean"] is not None
    stats = a["models"]["RD"]
    blue = "#1f77b4"
    orange = "#d95f02"
    # Use an opaque pastel SD band for consistent EPS and raster appearance.
    ax.axhspan(
        stats["mean"] - stats["SD"],
        stats["mean"] + stats["SD"],
        color="#e0ecf5",
        label=r"RD mean $\pm$ SD",
        zorder=0,
    )
    ax.axhline(stats["mean"], color=blue, lw=2, label=f"RD mean = {stats['mean']:.2f} Gy")
    if reference is not None:
        ax.axhline(
            reference, color="gold", ls="--", lw=1.8, label=f"Reference = {reference:.2f} Gy"
        )
    line = np.linspace(float(d.min()), float(d.max()), 200)
    if stats["slope"] is not None:
        ax.plot(
            line, stats["intercept"] + stats["slope"] * line, "k:", lw=1.8, label="RD linear trend"
        )
    ax.scatter(
        d, rd, s=60, c=blue, edgecolors="black", linewidths=0.7, label="RD schedule EQD2", zorder=4
    )
    if show_lq:
        ls = a["models"]["LQ"]
        ax.axhline(ls["mean"], color=orange, ls="-.", lw=2, label=f"LQ mean = {ls['mean']:.2f} Gy")
        if ls["slope"] is not None:
            ax.plot(
                line,
                ls["intercept"] + ls["slope"] * line,
                color=orange,
                ls=":",
                lw=1.8,
                label="LQ linear trend",
            )
        ax.scatter(
            d,
            lq,
            s=81,
            facecolors="none",
            edgecolors=orange,
            marker="s",
            lw=1.5,
            label="LQ schedule EQD2",
            zorder=5,
        )
    ax.set_xlim(float(d.min()) - 0.20, float(d.max()) + 0.25)
    visible = np.concatenate([rd, lq]) if show_lq else rd
    high = max(float(visible.max()), stats["mean"] + stats["SD"], reference or 0)
    # Use a minimum upper limit of 140 Gy and expand for larger values.
    upper = max(140.0, 20.0 * math.ceil(high * 1.08 / 20.0))
    lower = min(0.0, 20.0 * math.floor(float(visible.min()) * 1.08 / 20.0))
    ax.set_ylim(lower, upper)
    if float(np.ptp(d)) <= 10:
        ax.xaxis.set_major_locator(MultipleLocator(1.0))
    else:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=9, min_n_ticks=3))
    if upper <= 200:
        ax.yaxis.set_major_locator(MultipleLocator(20.0))
    else:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=7, min_n_ticks=3))
    ax.grid(True, color="#b7b7b7", alpha=0.25, ls=":", lw=0.8)
    ax.set_ylabel("EQD2 [Gy]", fontsize=13, fontweight="bold")
    ax.set_xlabel(r"Dose per fraction, $d$ [Gy]", fontsize=13, fontweight="bold")
    ax.xaxis.set_label_coords(0.5, -0.235 if show_labels else -0.10)
    ax.tick_params(axis="both", labelsize=10)
    if show_labels:
        levels = []
        for p, x in zip(rows, d):
            level = 0
            while any(abs(x - oldx) < 0.18 and level == oldlevel for oldx, oldlevel in levels):
                level += 1
            levels.append((x, level))
            ax.text(
                x,
                -0.085 - 0.05 * level,
                f"{p['D_Gy']:g}/{p['n']}",
                rotation=90,
                ha="center",
                va="top",
                transform=ax.get_xaxis_transform(),
                fontsize=8,
                clip_on=False,
            )
    fig.suptitle(
        "RD and LQ EQD2 vs dose per fraction" if show_lq else "RD EQD2 vs dose per fraction",
        fontsize=18,
        fontweight="bold",
        y=0.98,
    )
    panel = a["name"] + rf" ($\alpha/\beta={a['alpha_beta']:g}$ Gy, $D_q={a['D_q']:.4f}$ Gy)"
    title = ax.set_title(panel, fontsize=12, pad=12)
    # Keep long set names on one line without colliding with the main title.
    from matplotlib.backends.backend_agg import RendererAgg

    renderer = RendererAgg(max(1, int(fig.bbox.width)), max(1, int(fig.bbox.height)), fig.dpi)
    available = fig.bbox.width * 0.95
    while title.get_window_extent(renderer).width > available and title.get_fontsize() > 6:
        title.set_fontsize(title.get_fontsize() - 0.5)
    lines = [
        "RD statistics",
        f"N = {stats['N']}",
        f"Mean = {stats['mean']:.2f} Gy",
        f"SD = {stats['SD']:.2f} Gy",
    ]
    if show_ci:
        lines.append(f"95% CI:\n{stats['CI95_lower']:.2f}–{stats['CI95_upper']:.2f} Gy")
    # The reference locations are clear for the prostate preset. For lower
    # equivalent-dose sets, use the vacant upper area.
    low_distribution = float(rd.min()) < 0.5 * upper
    stats_y = 0.98 if low_distribution else 0.46
    equations_y = 0.98 if low_distribution else 0.42
    ax.text(
        0.025,
        stats_y,
        "\n".join(lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        linespacing=1.2,
        color="black",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.95, "pad": 2.0},
    )
    equations = []
    for model in ("RD", "LQ") if show_lq else ("RD",):
        s = a["models"][model]
        if s["slope"] is None:
            equations.append(f"{model}: trend undefined (identical fraction doses)")
        else:
            sign = "+" if s["slope"] >= 0 else "-"
            equations.append(
                rf"$\mathrm{{EQD2}}_{{\mathrm{{{model}}}}} = {s['intercept']:.4f} {sign} {abs(s['slope']):.4f}\,d$"
            )
    ax.text(
        0.65,
        equations_y,
        "\n".join(equations),
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10,
        linespacing=1.4,
        color="black",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.95, "pad": 4.0},
    )
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    order = [
        "RD schedule EQD2",
        "RD linear trend",
        f"RD mean = {stats['mean']:.2f} Gy",
        r"RD mean $\pm$ SD",
    ]
    if show_lq:
        order += [
            "LQ schedule EQD2",
            "LQ linear trend",
            f"LQ mean = {a['models']['LQ']['mean']:.2f} Gy",
        ]
    if reference is not None:
        order += [f"Reference = {reference:.2f} Gy"]
    order = [v for v in order if v in by_label]
    ax.legend(
        [by_label[v] for v in order],
        order,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=2,
        columnspacing=1.2,
        handlelength=2.6,
        fontsize=10,
        facecolor="white",
        framealpha=0.9,
        edgecolor="#cccccc",
    )
    return ax


def draw_clinical_plot(
    fig, clinical, show_beta=True, show_interval=False, show_labels=True, show_summary=True
):
    """Plot observations, descriptive statistics, and the bounded beta trend."""
    from matplotlib.patches import Rectangle
    from matplotlib.ticker import MaxNLocator, PercentFormatter

    fig.clear()
    fig.set_facecolor("white")
    if clinical is None:
        fig.text(0.5, 0.5, "Enter local tumor control for a selected prescription to see its EQD2 here.",
                 ha="center", va="center", wrap=True)
        return

    height_px = fig.get_figheight() * fig.dpi
    bottom = 0.20 if show_summary else min(0.32, 100 / max(1, height_px))
    top = 0.88 if show_summary else 1 - min(0.28, 72 / max(1, height_px))
    ax = fig.add_axes([0.10, bottom, 0.57 if show_summary else 0.85, top - bottom])
    eqd2, control = clinical["points"].T
    low, high = clinical["minimum"], clinical["maximum"]
    dose_grid = np.linspace(low[0], high[0], 160)
    y_ranges = [control]
    cluster = clinical["cluster"]
    if cluster is not None:
        span = cluster["maximum"] - cluster["minimum"]
        ax.add_patch(
            Rectangle(cluster["minimum"], *span, facecolor="#cfe8e1",
                      edgecolor="#397c70", linewidth=1.2, label="Prescription cluster", zorder=1)
        )
    ax.scatter(eqd2, control, s=44, color="#16384b", edgecolor="white", linewidth=0.7,
               label="Observed local control", zorder=5, clip_on=False)
    ax.axhline(clinical["mean"][1], color="#6b7785", linestyle=":", label="Observed mean")
    trend = clinical["linear"]
    if trend is not None:
        values = clinical["mean"][1] + trend["slope"] * (dose_grid - clinical["mean"][0])
        ax.plot(dose_grid, values, color="#44566c", linestyle="--", label="Linear trend")
        y_ranges.append(values)
    if (show_beta or show_interval) and clinical["beta"]:
        beta_grid = dose_grid if low[0] < high[0] else low[0] + np.array([-0.8, 0.8])
        mean, lower, upper = _beta_control_prediction(clinical["beta"], beta_grid)
        if show_beta:
            ax.plot(beta_grid, mean, color="#a34819", linewidth=1.8, label="Beta mean")
            y_ranges.append(mean)
        if show_interval:
            ax.fill_between(beta_grid, lower, upper, color="#f8e6d8",
                            label="Central 95% beta interval", zorder=0)
            y_ranges.extend([lower, upper])
    if show_labels:
        for index, (row, x, y) in enumerate(zip(clinical["observations"], eqd2, control)):
            above = index % 2 == 0 and y < 99
            ax.annotate(f"{row['D_Gy']:g}/{row['n']}", (x, y), xytext=(5, 6 if above else -6),
                        textcoords="offset points", va="bottom" if above else "top", fontsize=8)

    x_pad = max(1.0, (high[0] - low[0]) * 0.12)
    low_y, high_y = np.clip([min(values.min() for values in y_ranges),
                             max(values.max() for values in y_ranges)], 0, 100)
    y_pad = max(2.5, (high_y - low_y) * 0.12)
    ax.set(xlim=(low[0] - x_pad, high[0] + x_pad),
           ylim=(max(0, low_y - y_pad), min(100, high_y + y_pad)),
           xlabel="Prescription RD EQD2 (Gy)", ylabel="Observed local tumor control")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=7))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    ax.set_axisbelow(True)
    ax.grid(color="#dce4e8", linestyle=":")
    legend_y = -0.16 if show_summary else -52 / max(1, (top - bottom) * height_px)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, legend_y), ncol=2,
              fontsize=8.5, frameon=False)
    fig.suptitle("Local tumor control versus RD EQD2", fontsize=16, fontweight="bold", y=0.975)
    ax.set_title(clinical["name"], fontsize=10, pad=9, wrap=True)
    if show_summary:
        stats = clinical_basic_summary(clinical)
        fig.text(0.70, 0.87, "Basic analysis", fontsize=11, fontweight="bold", va="top")
        fig.text(0.70, 0.82, "\n\n".join(f"{key}\n{value}" for key, value in stats),
                 fontsize=8.5, va="top", linespacing=1.05)


def _export_analysis_plot(data, filename, ppi, draw, size, **options):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    ppi = _validated_ppi(ppi)
    path = Path(filename)
    if path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise ValueError("Image format must be PDF, JPG, PNG, EPS, or SVG.")
    fig = Figure(figsize=size, dpi=100)
    FigureCanvasAgg(fig)
    draw(fig, data, **options)
    if path.suffix.lower() == ".eps":
        for axis in fig.axes:
            for line in axis.get_xgridlines() + axis.get_ygridlines():
                alpha = line.get_alpha()
                if alpha is not None and alpha < 1:
                    line.set_color(tuple(alpha * x + 1 - alpha for x in (0.718, 0.718, 0.718)))
                    line.set_alpha(1)
            for text in axis.texts:
                box = text.get_bbox_patch()
                if box is not None:
                    box.set_alpha(1)
            if axis.get_legend() is not None:
                axis.get_legend().get_frame().set_alpha(1)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=int(ppi), facecolor="white", bbox_inches="tight", pad_inches=0.08)
    fig.clear()


def export_dispersion_plot(a, filename, ppi=300, **options):
    options.setdefault("reference", a["reference"])
    _export_analysis_plot(a, filename, ppi, draw_dispersion_plot, (8, 7.2), **options)


def export_clinical_plot(clinical, filename, ppi=300, **options):
    _export_analysis_plot(clinical, filename, ppi, draw_clinical_plot, (10, 6.4), **options)


def open_print_preview(filename):
    """Open a PDF in the system viewer, where the user chooses the printer."""
    import subprocess
    import sys

    path = str(Path(filename).resolve())
    if sys.platform == "win32":
        os.startfile(path)
    else:
        subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", path])


def configure_desktop_style(root):
    """Apply the desktop palette without changing exported plot styles."""
    from tkinter import font, ttk

    family = "Segoe UI" if os.name == "nt" else "DejaVu Sans"
    for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkHeadingFont"):
        font.nametofont(name, root=root).configure(family=family, size=10)
    style = ttk.Style(root)
    style.theme_use("clam")
    root.configure(background="#eef2f6")
    style.configure(".", background="#eef2f6", foreground="#172b4d", font=(family, 10))
    style.configure("TFrame", background="#eef2f6")
    style.configure("TLabel", background="#eef2f6")
    style.configure("Title.TLabel", font=(family, 21, "bold"))
    style.configure("Subtitle.TLabel", foreground="#627088")
    style.configure("TLabelframe", background="#ffffff", bordercolor="#cbd5e1", relief="solid")
    style.configure("TLabelframe.Label", font=(family, 11, "bold"))
    style.configure("TButton", width=20, padding=(10, 7), background="#f6f8fb",
                    bordercolor="#cbd5e1", focuscolor="#0d8b84")
    style.map("TButton", background=[("active", "#e4edf2"), ("pressed", "#d5e5eb")])
    style.configure("Accent.TButton", background="#00877f", foreground="white",
                    font=(family, 10, "bold"), bordercolor="#00877f")
    style.map("Accent.TButton", background=[("active", "#006f69"), ("pressed", "#005d58")],
              foreground=[("disabled", "#a5c9c6"), ("!disabled", "white")])
    style.configure("TMenubutton", width=20, padding=(10, 7), background="#f6f8fb",
                    bordercolor="#cbd5e1")
    style.configure("TEntry", padding=6, fieldbackground="white", bordercolor="#cbd5e1")
    style.configure("TCombobox", padding=6, fieldbackground="white", bordercolor="#cbd5e1")
    style.map("TCombobox", fieldbackground=[("readonly", "white")],
              foreground=[("readonly", "#172b4d")])
    style.configure("TCheckbutton", padding=(4, 5))
    style.configure("Treeview", rowheight=28, background="white", fieldbackground="white",
                    bordercolor="#cbd5e1")
    style.map("Treeview", background=[("selected", "#d4eeeb")],
              foreground=[("selected", "#123c42")])
    style.configure("Treeview.Heading", font=(family, 10, "bold"), padding=(6, 8),
                    background="#e8edf3", relief="flat")
    style.configure("TNotebook", background="#eef2f6", borderwidth=0)
    style.configure("TNotebook.Tab", padding=(12, 8), background="#e8edf3")
    style.map("TNotebook.Tab", background=[("selected", "white")],
              foreground=[("selected", "#007b74"), ("disabled", "#8994a4")])


def open_analysis_window(parent, result, *, container=None, tabs=None):
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure

    if result is None:
        return None
    datasets = (
        result["sets"] if "sets" in result else [{"name": "Prescriptions set", "result": result}]
    )
    embedded = container is not None
    window = parent.winfo_toplevel() if embedded else tk.Toplevel(parent)
    if not embedded:
        window.title("RD EQD2 — Detailed analysis")
        window.geometry("1320x1000")
        window.minsize(980, 740)
    outer = ttk.Frame(container if embedded else window, padding=8)
    outer.pack(fill="both", expand=True)
    header = ttk.Frame(outer)
    header.pack(fill="x")
    ttk.Label(header, text="Optimized set").pack(side="left", padx=(0, 8))
    selector = ttk.Combobox(
        header,
        state="readonly",
        width=68,
        values=[ds.get("name") or "Prescriptions set" for ds in datasets],
    )
    selector.pack(side="left", fill="x", expand=True)
    selector.current(0)
    meta = tk.StringVar()
    meta_label = ttk.Label(outer, textvariable=meta, wraplength=1150)
    meta_label.pack(anchor="w", pady=(6, 4))
    outer.bind("<Configure>", lambda event: meta_label.configure(
        wraplength=max(300, event.width - 20)
    ))
    toolbar = ttk.Frame(outer)
    toolbar.pack(fill="x", pady=(0, 6))
    if tabs is None:
        tabs = ttk.Notebook(outer)
        tabs.pack(fill="both", expand=True)
    plot_page = ttk.Frame(tabs)
    clinical_page = ttk.Frame(tabs, padding=6)
    prescription_page = ttk.Frame(tabs, padding=6)
    summary_page = ttk.Frame(tabs, padding=6)
    robust_page = ttk.Frame(tabs, padding=6)
    for page, label in [
        (plot_page, "EQD2 dispersion"),
        (clinical_page, "Clinical evaluation"),
        (prescription_page, "Prescription values"),
        (summary_page, "Summary & trends"),
        (robust_page, "Robustness"),
    ]:
        if not embedded or page not in (prescription_page, summary_page):
            tabs.add(page, text=label)
    options_bar = ttk.Frame(plot_page, padding=5)
    options_bar.pack(fill="x")
    show_lq = tk.BooleanVar(value=True)
    show_labels = tk.BooleanVar(value=True)
    show_ci = tk.BooleanVar(value=True)
    reference_var = tk.StringVar()
    cache = {}

    def current():
        index = selector.current()
        if index not in cache:
            cache[index] = build_analysis(datasets[index])
        return cache[index]

    def options():
        value = reference_var.get().strip()
        reference = float(value) if value else None
        if reference is not None and (not math.isfinite(reference) or reference <= 0):
            raise ValueError("Reference EQD2 must be positive, or left blank.")
        return {
            "show_lq": show_lq.get(),
            "show_labels": show_labels.get(),
            "show_ci": show_ci.get(),
            "reference": reference,
        }

    fig = Figure(figsize=(8, 7.2), dpi=100)
    canvas = FigureCanvasTkAgg(fig, master=plot_page)
    canvas.get_tk_widget().pack(fill="both", expand=True)

    def redraw():
        try:
            draw_dispersion_plot(fig, current(), **options())
            canvas.draw_idle()
        except Exception as exc:
            messagebox.showerror("Plot settings", str(exc), parent=window)

    for label, var in [
        ("Show LQ", show_lq),
        ("Prescription labels", show_labels),
        ("95% CI of mean", show_ci),
    ]:
        ttk.Checkbutton(options_bar, text=label, variable=var, command=redraw).pack(
            side="left", padx=5
        )
    ttk.Label(options_bar, text="Reference EQD2 (Gy)").pack(side="left", padx=(15, 3))
    ttk.Entry(options_bar, textvariable=reference_var, width=9).pack(side="left")
    ttk.Button(options_bar, text="Apply", command=redraw).pack(side="left", padx=5)

    clinical_tabs = ttk.Notebook(clinical_page)
    clinical_tabs.pack(fill="both", expand=True)
    clinical_information = ttk.Frame(clinical_tabs, padding=6)
    clinical_plot = ttk.Frame(clinical_tabs, padding=6)
    clinical_tabs.add(clinical_information, text="Analysis")
    clinical_tabs.add(clinical_plot, text="Plot")
    clinical_controls = ttk.Frame(clinical_plot)
    clinical_controls.pack(fill="x", pady=(0, 6))
    clinical_text = tk.Text(clinical_information, height=9, wrap="word", borderwidth=0,
                            background="#eef2f6", foreground="#172b4d", state="disabled")
    clinical_scroll = ttk.Scrollbar(clinical_information, command=clinical_text.yview)
    clinical_text.configure(yscrollcommand=clinical_scroll.set)
    clinical_scroll.pack(side="right", fill="y")
    clinical_text.pack(fill="both", expand=True)
    clinical_figure = Figure(figsize=(10, 6.4), dpi=100)
    clinical_canvas = FigureCanvasTkAgg(clinical_figure, master=clinical_plot)
    clinical_canvas.get_tk_widget().pack(fill="both", expand=True)
    show_beta = tk.BooleanVar(value=True)
    show_interval = tk.BooleanVar(value=False)
    clinical_labels = tk.BooleanVar(value=True)

    def clinical_options():
        return {
            "show_beta": show_beta.get(),
            "show_interval": show_interval.get(),
            "show_labels": clinical_labels.get(),
            "show_summary": False,
        }

    def redraw_clinical():
        draw_clinical_plot(clinical_figure, current()["clinical"], **clinical_options())
        clinical_canvas.draw_idle()

    clinical_canvas.get_tk_widget().bind("<Configure>", lambda event: redraw_clinical(), add="+")

    beta_controls = []
    for label, var in [
        ("Beta mean", show_beta),
        ("95% beta interval", show_interval),
        ("Prescription labels", clinical_labels),
    ]:
        control = ttk.Checkbutton(clinical_controls, text=label, variable=var, command=redraw_clinical)
        control.pack(side="left", padx=(0, 8))
        if var is not clinical_labels:
            beta_controls.append(control)

    def table(parent, fields, headings=None, name_width=260):
        frame = ttk.Frame(parent)
        frame.pack(fill="both", expand=True)
        tree = ttk.Treeview(frame, columns=fields, show="headings")
        for index, field in enumerate(fields):
            tree.heading(field, text=(headings or {}).get(field, field.replace("_", " ")))
            tree.column(
                field,
                width=name_width if index == 0 else 130,
                minwidth=70,
                stretch=index == 0,
                anchor="w" if index == 0 else "center",
            )
        vertical = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        horizontal = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
        tree.grid(row=0, column=0, sticky="nsew")
        vertical.grid(row=0, column=1, sticky="ns")
        horizontal.grid(row=1, column=0, sticky="ew")
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)
        return tree

    def populate(tree, rows):
        tree.delete(*tree.get_children())
        fields = tree["columns"]
        for row in rows:
            tree.insert("", "end", values=[_analysis_value(row.get(field), 7) for field in fields])

    pfields = (
        "prescription",
        "D_Gy",
        "n",
        "d_Gy",
        "RD_BED_Gy",
        "RD_EQD2_Gy",
        "LQ_BED_Gy",
        "LQ_EQD2_Gy",
        "local_control_pct",
        "RD_deviation_Gy",
        "RD_minus_LQ_Gy",
    )
    ptable = table(
        prescription_page,
        pfields,
        {
            "prescription": "Prescription",
            "D_Gy": "D (Gy)",
            "d_Gy": "d (Gy)",
            "RD_BED_Gy": "RD BED (Gy)",
            "RD_EQD2_Gy": "RD EQD2 (Gy)",
            "LQ_BED_Gy": "LQ BED (Gy)",
            "LQ_EQD2_Gy": "LQ EQD2 (Gy)",
            "local_control_pct": "Local control (%)",
            "RD_deviation_Gy": "RD minus mean (Gy)",
            "RD_minus_LQ_Gy": "RD minus LQ (Gy)",
        },
    )
    stable = table(
        summary_page,
        ("metric", "value"),
        {"metric": "Statistic / descriptor", "value": "Value"},
        430,
    )
    robust_controls = ttk.Frame(robust_page)
    robust_controls.pack(fill="x", pady=(0, 8))
    ttk.Label(robust_controls, text="Sensitivity α/β values (Gy, comma separated)").pack(
        side="left"
    )
    ab_values = tk.StringVar()
    ttk.Entry(robust_controls, textvariable=ab_values, width=25).pack(side="left", padx=7)
    robust_note = tk.StringVar(
        value="Run robustness to calculate leave-one-out and α/β sensitivity."
    )
    ttk.Label(robust_page, textvariable=robust_note, wraplength=1120).pack(anchor="w", pady=(0, 7))
    robust_tabs = ttk.Notebook(robust_page)
    robust_tabs.pack(fill="both", expand=True)
    loo_page = ttk.Frame(robust_tabs)
    ab_page = ttk.Frame(robust_tabs)
    robust_tabs.add(loo_page, text="Leave-one-out")
    robust_tabs.add(ab_page, text="α/β sensitivity")
    loo_table = table(
        loo_page,
        (
            "omitted_prescription",
            "D_q_Gy",
            "mean_EQD2_Gy",
            "RSD_percent",
            "eta_percent",
            "held_out_EQD2_Gy",
            "held_out_deviation_Gy",
            "boundary",
        ),
    )
    ab_table = table(
        ab_page,
        (
            "alpha_beta_Gy",
            "D_q_Gy",
            "mean_EQD2_Gy",
            "SD_Gy",
            "RSD_percent",
            "eta_percent",
            "boundary",
        ),
        name_width=140,
    )

    def refresh_robustness():
        robust = current()["robustness"]
        populate(loo_table, robust["leave_one_out"] if robust else [])
        populate(ab_table, robust["alpha_beta_sensitivity"] if robust else [])
        if robust:
            flat = robust["horizontal_trend"]
            robust_note.set(
                (
                    "Horizontal-trend Dq = " + _analysis_value(flat["q_flat_Gy"], 7) + " Gy; "
                    if flat
                    else ""
                )
                + "Held-out RMSE = "
                + _analysis_value(robust["held_out_RMSE_Gy"], 6)
                + " Gy. "
                + (
                    "Leave-one-out requires at least three prescriptions. "
                    if not robust["leave_one_out"]
                    else ""
                )
                + "Boundary = True means that the optimum reached the selected search limit."
            )
        else:
            robust_note.set("Run robustness to calculate leave-one-out and α/β sensitivity.")

    def calculate_robustness():
        try:
            values = [float(v.strip()) for v in ab_values.get().split(",") if v.strip()]
            if not values:
                raise ValueError("Enter at least one positive finite α/β value for q ≥ 0.")
            window.configure(cursor="watch")
            window.update_idletasks()
            run_robustness(current(), values)
            refresh_robustness()
        except Exception as exc:
            messagebox.showerror("Robustness analysis", str(exc), parent=window)
        finally:
            window.configure(cursor="")

    ttk.Button(robust_controls, text="Run robustness", command=calculate_robustness).pack(
        side="left", padx=5
    )

    def export_tables_dialog():
        dialog = tk.Toplevel(window)
        dialog.title("Export analysis tables")
        dialog.transient(window)
        dialog.grab_set()
        body = ttk.Frame(dialog, padding=16)
        body.pack(fill="both", expand=True)
        fmt = tk.StringVar(value="CSV")
        scope = tk.StringVar(value="Current set")
        with_plots = tk.BooleanVar(value=False)
        with_robust = tk.BooleanVar(value=False)
        ttk.Label(body, text="Table format").grid(row=0, column=0, sticky="w", pady=5)
        ttk.Combobox(
            body,
            textvariable=fmt,
            state="readonly",
            values=("CSV", "TSV", "LaTeX", "JSON"),
            width=25,
        ).grid(row=0, column=1, padx=10)
        ttk.Label(body, text="Export scope").grid(row=1, column=0, sticky="w", pady=5)
        ttk.Combobox(
            body,
            textvariable=scope,
            state="readonly",
            values=("Current set", "All optimized sets"),
            width=25,
        ).grid(row=1, column=1, padx=10)
        ttk.Checkbutton(
            body, text="Include dispersion plots (PNG, 300 PPI)", variable=with_plots
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=5)
        ttk.Checkbutton(body, text="Calculate and include robustness", variable=with_robust).grid(
            row=3, column=0, columnspan=2, sticky="w", pady=5
        )

        def save_tables():
            folder = filedialog.askdirectory(parent=dialog, title="Choose analysis export folder")
            if not folder:
                return
            try:
                indices = (
                    range(len(datasets))
                    if scope.get() == "All optimized sets"
                    else [selector.current()]
                )
                paths = []
                window.configure(cursor="watch")
                window.update_idletasks()
                for index in indices:
                    if index not in cache:
                        cache[index] = build_analysis(datasets[index])
                    detail = cache[index]
                    if with_robust.get() and detail["robustness"] is None:
                        run_robustness(detail)
                    paths += export_analysis_tables(
                        detail, folder, {"LaTeX": "tex"}.get(fmt.get(), fmt.get().lower())
                    )
                    if with_plots.get():
                        path = Path(folder) / (_analysis_stem(detail["name"]) + "_dispersion.png")
                        display_options = options()
                        if index != selector.current():
                            display_options["reference"] = detail["reference"]
                        export_dispersion_plot(detail, path, **display_options)
                        paths.append(path)
                refresh_robustness()
                dialog.destroy()
                messagebox.showinfo(
                    "Analysis exported", f"Saved {len(paths)} file(s) to {folder}", parent=window
                )
            except Exception as exc:
                messagebox.showerror("Export error", str(exc), parent=dialog)
            finally:
                window.configure(cursor="")

        ttk.Button(body, text="Cancel", command=dialog.destroy).grid(row=4, column=0, pady=(12, 0))
        ttk.Button(body, text="Choose folder and export…", command=save_tables).grid(
            row=4, column=1, pady=(12, 0)
        )

    def export_plot_dialog(clinical=False):
        plot_name = "clinical evaluation plot" if clinical else "dispersion plot"
        dialog = tk.Toplevel(window)
        dialog.title("Export " + plot_name)
        dialog.transient(window)
        dialog.grab_set()
        body = ttk.Frame(dialog, padding=16)
        body.pack()
        fmt = tk.StringVar(value=next(iter(IMAGE_FORMATS)))
        ppi = tk.StringVar(value="300")
        ttk.Label(body, text="Image format").grid(row=0, column=0, pady=5, sticky="w")
        ttk.Combobox(
            body,
            textvariable=fmt,
            state="readonly",
            values=list(IMAGE_FORMATS),
            width=18,
        ).grid(row=0, column=1, padx=8)
        ttk.Label(body, text="Resolution (PPI)").grid(row=1, column=0, pady=5, sticky="w")
        ttk.Entry(body, textvariable=ppi, width=20).grid(row=1, column=1, padx=8)

        def save_plot():
            try:
                value = _validated_ppi(ppi.get())
                suffix = "_clinical_evaluation" if clinical else "_dispersion"
                ext = IMAGE_FORMATS[fmt.get()]
                path = filedialog.asksaveasfilename(
                    parent=dialog,
                    title="Save " + plot_name,
                    initialfile=_analysis_stem(current()["name"]) + suffix + ext,
                    defaultextension=ext,
                    filetypes=[(fmt.get(), "*" + ext)],
                )
                if not path:
                    return
                path = Path(path).with_suffix(ext)
                if clinical:
                    export_clinical_plot(current()["clinical"], path, ppi=value, **clinical_options())
                else:
                    export_dispersion_plot(current(), path, ppi=value, **options())
                dialog.destroy()
                messagebox.showinfo("Plot exported", f"Saved {path}", parent=window)
            except Exception as exc:
                messagebox.showerror("Image export error", str(exc), parent=dialog)

        ttk.Button(body, text="Cancel", command=dialog.destroy).grid(row=2, column=0, pady=(12, 0))
        ttk.Button(body, text="Save image…", command=save_plot).grid(row=2, column=1, pady=(12, 0))

    ttk.Button(toolbar, text="Export tables…", command=export_tables_dialog).pack(
        side="left", padx=(0, 7)
    )
    ttk.Button(toolbar, text="Export dispersion plot…", command=export_plot_dialog).pack(
        side="left"
    )
    clinical_export = ttk.Button(
        clinical_controls, text="Export clinical evaluation plot…",
        command=lambda: export_plot_dialog(clinical=True),
    )
    clinical_export.pack(side="right")

    def select_set(_event=None):
        a = current()
        rd = a["models"]["RD"]
        meta.set(
            f"α/β = {a['alpha_beta']:g} Gy · Dq = {a['D_q']:.6f} Gy · N = {rd['N']} · "
            + f"Mean EQD2 = {rd['mean']:.3f} Gy · RSD = {rd['RSD']:.3f}% · η = {_analysis_value(rd['eta'], 5)}%"
            + (" · Optimum at search boundary" if a["boundary"] else "")
        )
        reference_var.set("" if a["reference"] is None else f"{a['reference']:.12g}")
        ab_values.set(
            ", ".join(
                f"{v:g}"
                for v in [
                    a["input_alpha_beta"] * 2 / 3,
                    a["input_alpha_beta"],
                    a["input_alpha_beta"] * 4 / 3,
                ]
            )
        )
        populate(ptable, a["prescriptions"])
        populate(stable, [{"metric": k, "value": v} for k, v in analysis_summary(a)])
        clinical = a["clinical"]
        if clinical:
            summary = clinical_basic_summary(clinical)
            information = (
                "Basic analysis\n" + "\n".join(f"{key}: {value}" for key, value in summary)
                + "\n\nBeta regression\n" + clinical_beta_summary(clinical)
            )
        else:
            information = "Enter observed local control in the prescription set editor."
        clinical_text.configure(state="normal")
        clinical_text.delete("1.0", "end")
        clinical_text.insert("1.0", information)
        clinical_text.configure(state="disabled")
        clinical_export.configure(state="normal" if clinical else "disabled")
        for control in beta_controls:
            control.configure(state="normal" if clinical and clinical["beta"] else "disabled")
        refresh_robustness()
        redraw()
        redraw_clinical()
        update_plot_actions()

    def selected_plot():
        page = tabs.select()
        if page == str(plot_page):
            return "dispersion"
        if (page == str(clinical_page) and clinical_tabs.select() == str(clinical_plot)
                and current()["clinical"] is not None):
            return "clinical"
        return None

    def write_selected_plot(path):
        if selected_plot() == "clinical":
            export_clinical_plot(current()["clinical"], path, **clinical_options())
        elif selected_plot() == "dispersion":
            export_dispersion_plot(current(), path, **options())
        else:
            return False
        return True

    def save_selected_plot():
        if selected_plot() is None:
            return
        selected_type = tk.StringVar(value="PDF")
        path = filedialog.asksaveasfilename(
            parent=window, title="Save plot", defaultextension=".pdf",
            initialfile=_analysis_stem(current()["name"]) + "_" + selected_plot() + ".pdf",
            filetypes=[(name, "*" + extension) for name, extension in IMAGE_FORMATS.items()],
            typevariable=selected_type,
        )
        if path:
            try:
                extension = IMAGE_FORMATS[selected_type.get()]
                write_selected_plot(Path(path).with_suffix(extension))
            except Exception as exc:
                messagebox.showerror("Save plot", str(exc), parent=window)

    def print_selected_plot():
        if selected_plot() is None:
            return
        import tempfile

        try:
            path = Path(tempfile.mkdtemp(prefix="rd-dq-print-")) / "analysis.pdf"
            write_selected_plot(path)
            open_print_preview(path)
        except Exception as exc:
            messagebox.showerror("Print plot", str(exc), parent=window)

    def update_plot_actions(_event=None):
        if embedded:
            tabs.event_generate("<<NotebookTabChanged>>", when="tail")
        else:
            state = "normal" if selected_plot() is not None else "disabled"
            analysis_file.entryconfigure("Save plot…", state=state)
            analysis_file.entryconfigure("Print plot…", state=state)

    clinical_tabs.bind("<<NotebookTabChanged>>", update_plot_actions, add="+")
    selector.bind("<<ComboboxSelected>>", select_set)
    if embedded:
        toolbar.pack_forget()
        outer.export_tables = export_tables_dialog
        outer.export_plot = export_plot_dialog
        outer.has_clinical = lambda: current()["clinical"] is not None
        outer.save_plot = save_selected_plot
        outer.print_plot = print_selected_plot
        outer.can_save_plot = lambda: selected_plot() is not None
    else:
        analysis_menubar = tk.Menu(window)
        window.configure(menu=analysis_menubar)
        analysis_file = tk.Menu(analysis_menubar, tearoff=False)
        analysis_menubar.add_cascade(label="File", menu=analysis_file)
        analysis_file.add_command(label="Save plot…", accelerator="Ctrl+S", command=save_selected_plot)
        analysis_file.add_command(label="Save tables…", command=export_tables_dialog)
        analysis_file.add_command(label="Print plot…", accelerator="Ctrl+P", command=print_selected_plot)
        analysis_file.add_separator()
        analysis_file.add_command(label="Close", command=window.destroy)
        analysis_exports = tk.Menu(analysis_menubar, tearoff=False)
        analysis_menubar.add_cascade(label="Export", menu=analysis_exports)
        analysis_exports.add_command(label="Export tables…", command=export_tables_dialog)
        analysis_exports.add_command(label="Export dispersion plot…", command=export_plot_dialog)
        analysis_exports.add_command(label="Export clinical evaluation plot…",
                                     command=lambda: export_plot_dialog(clinical=True))
        tabs.bind("<<NotebookTabChanged>>", update_plot_actions, add="+")
        window.bind("<Control-s>", lambda event: save_selected_plot())
        window.bind("<Control-p>", lambda event: print_selected_plot())
        window.bind("<Escape>", lambda event: window.destroy())
    select_set()
    update_plot_actions()
    return outer if embedded else window


def launch_gui(db_path=None):
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure

    store = (
        PrescriptionStore(path=db_path, initial_sets=DEFAULT_SETS)
        if db_path is not None
        else PrescriptionStore(initial_sets=DEFAULT_SETS)
    )
    root = tk.Tk()
    root.title("RD Dq Optimizer")
    configure_desktop_style(root)
    root.geometry("1440x1040")
    root.minsize(1100, 780)
    panel = ttk.Frame(root, padding=14)
    panel.pack(fill="both", expand=True)
    heading = ttk.Frame(panel)
    heading.pack(fill="x", pady=(0, 14))
    title = ttk.Frame(heading)
    title.pack(side="left")
    ttk.Label(title, text="Dq calibration", style="Title.TLabel").pack(anchor="w")
    ttk.Label(title, text="Resilience–Depletion", style="Subtitle.TLabel").pack(anchor="w")
    workspace = ttk.Panedwindow(panel, orient="horizontal")
    workspace.pack(fill="both", expand=True)
    sidebar = ttk.Panedwindow(workspace, orient="vertical", width=280)
    content = ttk.LabelFrame(workspace, text="Workspace", padding=8)
    workspace_tabs = ttk.Notebook(content)
    workspace_tabs.pack(fill="both", expand=True)
    workspace.add(sidebar, weight=0)
    workspace.add(content, weight=1)
    controls = ttk.LabelFrame(sidebar, text="Optimization settings", padding=12)

    var_start = tk.StringVar(value=f"{DEFAULT_Q_START:g}")
    var_end = tk.StringVar(value=f"{DEFAULT_Q_END:g}")
    var_step = tk.StringVar(value=f"{DEFAULT_Q_STEP:g}")
    var_bound_pct = tk.StringVar(value="5")
    method_labels = {
        "sse": "Sum of squared deviations",
        "sd": "Standard deviation",
        "range": "Range (max − min)",
    }
    var_method = tk.StringVar(value=method_labels["sse"])
    plot_settings_path = store.path.with_name("plot_settings.json")
    plot_settings = read_plot_settings(plot_settings_path)
    for index, (label, var) in enumerate(
        [("q start (Gy)", var_start), ("q end (Gy)", var_end), ("q step (Gy)", var_step)]
    ):
        ttk.Label(controls, text=label).grid(row=index * 2, column=0, sticky="w", pady=(0, 3))
        ttk.Entry(controls, textvariable=var, width=24).grid(
            row=index * 2 + 1, column=0, sticky="ew", pady=(0, 8)
        )
    ttk.Label(controls, text="Minimization criterion").grid(row=6, column=0, sticky="w", pady=(0, 3))
    methods = ttk.Combobox(
        controls, textvariable=var_method, state="readonly", width=26,
        values=list(method_labels.values()),
    )
    methods.grid(row=7, column=0, sticky="ew", pady=(0, 8))
    ttk.Label(controls, text="Near-optimal bounds (%)").grid(row=8, column=0, sticky="w", pady=(0, 3))
    ttk.Entry(controls, textvariable=var_bound_pct).grid(row=9, column=0, sticky="ew")
    controls.columnconfigure(0, weight=1)

    setbox = ttk.LabelFrame(sidebar, text="Prescription sets", padding=10)
    sidebar.add(setbox, weight=1)
    sidebar.add(controls, weight=1)
    focused = tk.StringVar()
    management_actions = ttk.Frame(setbox)
    management_actions.pack(fill="x", pady=(0, 8))
    for index, (label, command) in enumerate(
        [
            ("New set", lambda: edit_dialog()),
            ("Edit set", lambda: edit_selected()),
            ("Import…", lambda: import_set_csv()),
            ("Export…", lambda: export_set_csv()),
        ]
    ):
        ttk.Button(management_actions, text=label, width=11, command=command).grid(
            row=index // 2, column=index % 2, sticky="ew", padx=3, pady=3
        )
    for column in (0, 1):
        management_actions.columnconfigure(column, weight=1, uniform="set_actions")
    set_actions = ttk.Frame(setbox)
    set_actions.pack(fill="x", pady=(0, 8))
    for column, (label, selected) in enumerate((("Select all", True), ("Clear all", False))):
        ttk.Button(set_actions, text=label, width=11,
                   command=lambda value=selected: set_all_checks(value)).grid(
                       row=0, column=column, sticky="ew", padx=3
                   )
        set_actions.columnconfigure(column, weight=1, uniform="set_selection")
    check_frame = ttk.Frame(setbox)
    check_frame.pack(fill="both", expand=True)
    selection_canvas = tk.Canvas(check_frame, width=250, height=180, highlightthickness=0,
                                 background="#ffffff")
    check_scroll = ttk.Scrollbar(check_frame, orient="vertical", command=selection_canvas.yview)
    check_hscroll = ttk.Scrollbar(setbox, orient="horizontal", command=selection_canvas.xview)
    selection_canvas.configure(yscrollcommand=check_scroll.set, xscrollcommand=check_hscroll.set)
    check_hscroll.pack(side="bottom", fill="x")
    selection_canvas.pack(side="left", fill="both", expand=True)
    check_scroll.pack(side="right", fill="y")
    check_items = ttk.Frame(selection_canvas)
    selection_id = selection_canvas.create_window((0, 0), window=check_items, anchor="nw")
    check_items.bind(
        "<Configure>",
        lambda event: selection_canvas.configure(scrollregion=selection_canvas.bbox("all")),
    )
    selection_canvas.bind(
        "<Configure>",
        lambda event: selection_canvas.itemconfigure(
            selection_id, width=max(event.width, check_items.winfo_reqwidth())
        ),
    )
    prescription_list = ttk.LabelFrame(workspace_tabs, text="Prescription details", padding=8)
    workspace_tabs.add(prescription_list, text="Prescriptions")
    prescription_tabs = ttk.Notebook(prescription_list)
    prescription_tabs.enable_traversal()
    prescription_tabs.pack(fill="both", expand=True)
    empty_prescriptions = ttk.Label(
        prescription_list, text="Select a prescriptions set to view its prescriptions."
    )
    tab_views = {}
    dataset_cache = {}
    enabled_vars = {}
    set_labels = {}
    selections = PrescriptionsSelection()
    syncing_tabs = False
    result_holder = {"result": None, "analysis_window": None}

    bar = ttk.Frame(heading)
    bar.pack(side="right", anchor="center")
    status = tk.StringVar(value="Ready · Select prescriptions and run optimization.")
    status_bar = ttk.Label(root, textvariable=status, style="Subtitle.TLabel", padding=(14, 8),
                           wraplength=1380)
    status_bar.pack(side="bottom", fill="x", before=panel)
    root.bind("<Configure>", lambda event: status_bar.configure(
        wraplength=max(500, event.width - 30)
    ) if event.widget is root else None)
    results_area = ttk.Frame(workspace_tabs, padding=8)
    workspace_tabs.add(results_area, text="Results")
    result_toolbar = ttk.Frame(results_area)
    result_toolbar.pack(fill="x", pady=(0, 6))
    ttk.Label(result_toolbar, text="Results", font=("TkDefaultFont", 11, "bold")).pack(side="left")
    results_box = ttk.LabelFrame(results_area, text="Calibration summary", padding=(6, 3))
    results_box.pack(fill="x", pady=(0, 6))
    # Retain access to all result columns on narrower displays.
    results_grid = ttk.Frame(results_box)
    results_grid.pack(fill="x", expand=True)
    columns = (
        "prescriptions_set",
        "N",
        "ab",
        "dq",
        "criterion",
        "mean",
        "sample_sd",
        "rsd",
        "eta",
        "range",
        "bounds",
        "location",
    )
    results_table = ttk.Treeview(results_grid, columns=columns, show="headings", height=2)
    for key, label, width in [
        ("prescriptions_set", "Prescriptions set", 250),
        ("N", "Prescriptions", 90),
        ("ab", "Selected α/β (Gy)", 125),
        ("dq", "Dq (Gy)", 100),
        ("criterion", "Minimized objective", 155),
        ("mean", "Mean EQD2 (Gy)", 125),
        ("sample_sd", "EQD2 SD (Gy)", 120),
        ("rsd", "RSD (%)", 85),
        ("eta", "η (%)", 85),
        ("range", "EQD2 range (Gy)", 120),
        ("bounds", "Dq profile bounds (Gy)", 175),
        ("location", "Minimum", 95),
    ]:
        results_table.heading(key, text=label)
        results_table.column(
            key, width=width, anchor="w" if key == "prescriptions_set" else "center"
        )
    results_scroll = ttk.Scrollbar(results_grid, orient="vertical", command=results_table.yview)
    results_hscroll = ttk.Scrollbar(results_grid, orient="horizontal", command=results_table.xview)
    results_table.configure(yscrollcommand=results_scroll.set, xscrollcommand=results_hscroll.set)
    results_table.grid(row=0, column=0, sticky="nsew")
    results_scroll.grid(row=0, column=1, sticky="ns")
    results_hscroll.grid(row=1, column=0, sticky="ew")
    results_grid.columnconfigure(0, weight=1)
    analysis_controls = ttk.Frame(results_area)
    analysis_controls.pack(fill="x")
    result_tabs = ttk.Notebook(results_area)
    result_tabs.pack(fill="both", expand=True)
    plotbox = ttk.Frame(result_tabs)
    result_tabs.add(plotbox, text="Optimization profile")
    fig = Figure(figsize=(plot_settings["width_in"], plot_settings["height_in"]), dpi=100)
    ax = fig.add_subplot(111)
    canvas = FigureCanvasTkAgg(fig, master=plotbox)
    canvas.get_tk_widget().pack(fill="both", expand=True)

    def clear_analysis_tabs():
        for widget in result_tabs.winfo_children():
            if widget is not plotbox:
                if str(widget) in result_tabs.tabs():
                    result_tabs.forget(widget)
                widget.destroy()
        for child in analysis_controls.winfo_children():
            child.destroy()
        result_holder.pop("embedded_analysis", None)

    def draw_blank_plot():
        ax.clear()
        ax.set_axis_off()
        ax.text(0.5, 0.58, "Run optimization to view results", transform=ax.transAxes,
                ha="center", va="center", fontsize=16, fontweight="bold", color="#172b4d")
        ax.text(0.5, 0.43, "Select prescriptions and review the settings to begin.",
                transform=ax.transAxes, ha="center", va="center", fontsize=11, color="#627088")
        fig.subplots_adjust(left=0.03, right=0.97, top=0.95, bottom=0.05)

    def can_run():
        checked = [dataset_cache[key] for key, var in enabled_vars.items() if var.get()]
        return bool(checked) and all(len(selections.selected_rows(record)) >= 2 for record in checked)

    def sync_action_states(_event=None):
        run_state = "normal" if can_run() else "disabled"
        analysis_state = "normal" if result_holder["result"] is not None else "disabled"
        run_button.configure(state=run_state)
        analysis_button.configure(state=analysis_state)
        analysis_menu.entryconfigure("Run optimization", state=run_state)
        analysis_menu.entryconfigure("Analysis…", state=analysis_state)
        view = result_holder.get("embedded_analysis")
        plot_ready = analysis_state == "normal" and workspace_tabs.select() == str(results_area) and (
            result_tabs.select() == str(plotbox) or (view is not None and view.can_save_plot())
        )
        for label in ("Save plot…", "Print plot…"):
            file_menu.entryconfigure(label, state="normal" if plot_ready else "disabled")

    def invalidate_result(*_args):
        clear_analysis_tabs()
        for label in ("EQD2 dispersion", "Clinical evaluation", "Robustness"):
            page = ttk.Frame(result_tabs)
            result_tabs.add(page, text=label, state="disabled")
        result_tabs.select(plotbox)
        existing = result_holder.get("analysis_window")
        if existing is not None and existing.winfo_exists():
            existing.destroy()
        result_holder["analysis_window"] = None
        result_holder["result"] = None
        for item in results_table.get_children():
            results_table.delete(item)
        status.set("Inputs changed. Run optimization to update the plot and results.")
        draw_blank_plot()
        canvas.draw_idle()
        sync_action_states()

    def plot_settings_dialog():
        """Edit visualization parameters without changing any optimization data."""
        dialog = tk.Toplevel(root)
        dialog.title("Plot settings — RD EQD2 optimization")
        dialog.geometry("700x635")
        dialog.minsize(630, 520)
        dialog.transient(root)
        dialog.grab_set()
        wrapper = ttk.Frame(dialog, padding=12)
        wrapper.pack(fill="both", expand=True)
        notebook = ttk.Notebook(wrapper)
        notebook.pack(fill="both", expand=True)
        tab_labels = [
            ("Titles & axes", "titles"),
            ("Fonts & image", "type"),
            ("Curves & layout", "styles"),
        ]
        tab_frames = {}
        for text_label, code in tab_labels:
            tab = ttk.Frame(notebook, padding=12)
            notebook.add(tab, text=text_label)
            tab_frames[code] = tab
        vars_ = {}

        def reset_fields():
            for k, v in DEFAULT_PLOT_SETTINGS.items():
                vars_[k].set(v)

        def numeric_field(tab, key, label, row, col=0, width=15):
            ttk.Label(tab, text=label).grid(
                row=row, column=col * 2, sticky="w", padx=(0, 8), pady=6
            )
            ttk.Entry(tab, textvariable=vars_[key], width=width).grid(
                row=row, column=col * 2 + 1, sticky="ew", pady=6, padx=(0, 16)
            )

        def string_field(tab, key, label, row, width=51):
            ttk.Label(tab, text=label).grid(row=row, column=0, sticky="w", pady=7)
            ttk.Entry(tab, textvariable=vars_[key], width=width).grid(
                row=row, column=1, columnspan=3, sticky="ew", pady=7
            )

        def checkbox_field(tab, key, label, row, col=0):
            ttk.Checkbutton(tab, text=label, variable=vars_[key]).grid(
                row=row, column=col * 2, columnspan=2, sticky="w", padx=(0, 16), pady=6
            )

        for key, value in plot_settings.items():
            if isinstance(value, bool):
                vars_[key] = tk.BooleanVar(value=value)
            else:
                vars_[key] = tk.StringVar(value=str(value))
        titles = tab_frames["titles"]
        ttk.Label(
            titles,
            text="Leave a text field blank to use the automatic RD EQD2 label.",
            foreground="#555555",
        ).grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 10))
        string_field(titles, "title", "Graph title", 1)
        string_field(titles, "x_label", "Horizontal axis", 2)
        string_field(titles, "y_label", "Vertical axis", 3)
        ttk.Separator(titles, orient="horizontal").grid(
            row=4, column=0, columnspan=4, sticky="ew", pady=12
        )
        ttk.Label(titles, text="Axis limits (leave blank for automatic scaling)").grid(
            row=5, column=0, columnspan=4, sticky="w", pady=(0, 6)
        )
        numeric_field(titles, "y_min", "Y minimum", 7)
        numeric_field(titles, "y_max", "Y maximum", 7, 1)
        titles.columnconfigure(1, weight=1)
        titles.columnconfigure(3, weight=1)
        typography = tab_frames["type"]
        numeric_field(typography, "width_in", "Export width (inches)", 0)
        numeric_field(typography, "height_in", "Export height (inches)", 0, 1)
        ttk.Label(
            typography,
            text="Image dimensions affect exported PDF, JPG, PNG, EPS and SVG files.\n"
            "Image resolution (PPI) remains selectable in the Export image window.",
            foreground="#555555",
            justify="left",
        ).grid(row=1, column=0, columnspan=4, sticky="w", pady=(0, 16))
        ttk.Label(typography, text="Font family").grid(row=2, column=0, sticky="w", pady=6)
        ttk.Combobox(
            typography,
            textvariable=vars_["font_family"],
            state="readonly",
            values=(
                "DejaVu Sans",
                "DejaVu Serif",
                "Liberation Sans",
                "Liberation Serif",
                "monospace",
            ),
            width=22,
        ).grid(row=2, column=1, columnspan=3, sticky="w", pady=6)
        numeric_field(typography, "title_font", "Graph title font (pt)", 3)
        numeric_field(typography, "axis_font", "Axis label font (pt)", 3, 1)
        numeric_field(typography, "tick_font", "Tick label font (pt)", 4)
        numeric_field(typography, "legend_font", "Legend font (pt)", 4, 1)
        numeric_field(typography, "summary_font", "Statistics font (pt)", 5)
        numeric_field(typography, "dq_font", r"Dq label font (pt)", 5, 1)
        ttk.Checkbutton(
            typography,
            text="Tight-crop exported image (may change exact dimensions)",
            variable=vars_["tight_crop"],
        ).grid(row=6, column=0, columnspan=4, sticky="w", pady=(14, 2))
        typography.columnconfigure(1, weight=1)
        typography.columnconfigure(3, weight=1)
        styles = tab_frames["styles"]
        numeric_field(styles, "line_width", "Curve width (pt)", 0)
        numeric_field(styles, "minimum_size", "Minimum marker size", 0, 1)
        numeric_field(styles, "bounds_width", "Bounds line width", 1)
        numeric_field(styles, "bounds_alpha", "Bounds shading opacity", 1, 1)
        numeric_field(styles, "grid_alpha", "Grid opacity", 2)
        for key, label, row, col, choices in [
            ("line_style", "Curve line style", 3, 0, ("-", "--", ":", "-.")),
            ("grid_style", "Grid line style", 3, 1, ("-", "--", ":", "-.")),
        ]:
            ttk.Label(styles, text=label).grid(row=row, column=col * 2, sticky="w", pady=6)
            ttk.Combobox(
                styles, textvariable=vars_[key], values=choices, state="readonly", width=16
            ).grid(row=row, column=col * 2 + 1, sticky="ew", padx=(0, 16), pady=6)
        numeric_field(styles, "curve_color", "First curve color", 5, width=16)
        numeric_field(styles, "bounds_color", "Bounds color (optional)", 5, 1, width=16)
        ttk.Label(
            styles,
            text="Use named Matplotlib colors or hex codes (e.g., #1f77b4). "
            "Blank bounds color follows its curve.",
            foreground="#555555",
        ).grid(row=6, column=0, columnspan=4, sticky="w", pady=(4, 12))
        checkbox_field(styles, "show_title", "Show title", 7)
        checkbox_field(styles, "show_legend", "Show legend", 7, 1)
        checkbox_field(styles, "show_grid", "Show grid", 8)
        checkbox_field(styles, "show_bounds", "Show profile bounds", 8, 1)
        checkbox_field(styles, "show_minimum", "Show Dq minimum markers", 9)
        checkbox_field(styles, "show_summary", "Show optimization statistics", 9, 1)
        styles.columnconfigure(1, weight=1)
        styles.columnconfigure(3, weight=1)
        foot = ttk.Frame(wrapper)
        foot.pack(fill="x", pady=(12, 0))
        ttk.Button(foot, text="Restore defaults", command=reset_fields).pack(side="left")

        def apply_settings():
            try:
                validated = normalized_plot_settings({k: v.get() for k, v in vars_.items()})
            except (ValueError, TypeError) as exc:
                messagebox.showerror("Invalid plot setting", str(exc), parent=dialog)
                return
            plot_settings.clear()
            plot_settings.update(validated)
            if result_holder["result"] is not None:
                result_holder["result"]["plot_settings"] = dict(plot_settings)
                draw_range_plot(ax, result_holder["result"])
            else:
                draw_blank_plot()
            canvas.draw_idle()
            try:
                write_plot_settings(plot_settings_path, plot_settings)
            except OSError as exc:
                messagebox.showwarning(
                    "Plot preferences not saved",
                    f"The settings were applied but could not be saved for the next session: {exc}",
                    parent=dialog,
                )
            dialog.destroy()

        ttk.Button(foot, text="Cancel", command=dialog.destroy).pack(side="right", padx=(8, 0))
        ttk.Button(foot, text="Apply", command=apply_settings).pack(side="right")
        dialog.bind("<Escape>", lambda event: dialog.destroy())
        ttk.Label(
            wrapper,
            text="Settings affect the graph and exported images, not RD EQD2 calculations.",
            foreground="#555555",
        ).pack(anchor="w", pady=(8, 0))

    def method_key():
        return next(
            (key for key, label in method_labels.items() if label == var_method.get()), "sse"
        )

    def method_changed(_event=None):
        invalidate_result()

    def bind_scrolling(widget, target):
        def wheel(event):
            units = (
                (-1 if event.num == 4 else 1)
                if event.num in (4, 5)
                else (
                    -max(1, abs(event.delta) // 120)
                    if event.delta > 0
                    else max(1, abs(event.delta) // 120)
                )
            )
            target.yview_scroll(units, "units")
            return "break"

        for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            widget.bind(sequence, wheel)

    def refresh_set_label(key):
        full = dataset_cache[key]
        count = len(selections.selected_rows(full))
        ab = selections.alpha_beta_text[key].strip() or "—"
        set_labels[key].set(
            f"{full['name']}\nα/β = {ab} Gy · {count}/{len(full['prescriptions'])} selected"
        )

    def refresh_prescriptions(key):
        if key not in dataset_cache or key not in tab_views:
            return
        full, view = dataset_cache[key], tab_views[key]
        tree = view["tree"]
        selected = tree.selection()
        visible = tree.yview()[0]
        tree.delete(*tree.get_children())
        for index, prescription in enumerate(full["prescriptions"]):
            name, total, fx = prescription.schedule_key
            checked = selections.is_included(full, index)
            control = prescription.local_control_pct
            tree.insert(
                "",
                "end",
                iid=str(index),
                tags=("even" if index % 2 == 0 else "odd",),
                values=(
                    "☑" if checked else "☐",
                    name,
                    f"{total:g}",
                    fx,
                    f"{total / fx:.4g}",
                    "Yes" if prescription == reference_schedule(full) else "",
                    f"{control:.4g}%" if control is not None else "",
                ),
            )
        tree.selection_set([row for row in selected if tree.exists(row)])
        tree.yview_moveto(visible)
        view["summary"].set(
            f"{len(selections.selected_rows(full))} of {len(full['prescriptions'])} prescriptions selected · "
            f"{sum(p.local_control_pct is not None for p in full['prescriptions'])} observed outcomes"
        )
        refresh_set_label(key)

    def view_set(set_id):
        key = str(set_id)
        if key in tab_views:
            focused.set(key)
            if prescription_tabs.select() != str(tab_views[key]["frame"]):
                prescription_tabs.select(tab_views[key]["frame"])

    def prescription_tab_changed(_event=None):
        selected = prescription_tabs.select()
        key = next((key for key, view in tab_views.items() if str(view["frame"]) == selected), "")
        focused.set(key)

    def alpha_beta_changed(key):
        if syncing_tabs or key not in dataset_cache or key not in tab_views:
            return
        selections.alpha_beta_text[key] = tab_views[key]["ab"].get()
        refresh_set_label(key)
        invalidate_result()

    def save_main_alpha_beta(key):
        if key not in dataset_cache or key not in tab_views:
            return
        try:
            ab = store.update_alpha_beta(int(key), tab_views[key]["ab"].get())
        except (ValueError, KeyError, sqlite3.Error) as exc:
            messagebox.showerror("Cannot save α/β", str(exc), parent=root)
            return
        refresh_sets(prefer_id=int(key))
        status.set(
            f"Saved α/β = {ab:g} Gy for {dataset_cache[key]['name']}. Run optimization to update results."
        )

    def select_prescriptions(value, key):
        if key not in dataset_cache:
            return
        full = dataset_cache[key]
        selections.set_included(full, range(len(full["prescriptions"])), value)
        refresh_prescriptions(key)
        invalidate_result()

    def toggle_prescriptions(indices, key):
        if key not in dataset_cache or not indices:
            return
        full = dataset_cache[key]
        include = not all(selections.is_included(full, index) for index in indices)
        selections.set_included(full, indices, include)
        refresh_prescriptions(key)
        invalidate_result()

    def prescription_click(event, key):
        tree = event.widget
        row = tree.identify_row(event.y)
        if row and tree.identify_column(event.x) == "#1":
            tree.focus(row)
            tree.selection_set(row)
            tree.focus_set()
            toggle_prescriptions([int(row)], key)
            return "break"

    def prescription_space(event, key):
        toggle_prescriptions([int(row) for row in event.widget.selection()], key)
        return "break"

    def create_set_tab(key):
        frame = ttk.Frame(prescription_tabs, padding=(4, 6))
        ab_bar = ttk.Frame(frame)
        ab_bar.pack(fill="x", pady=(0, 5))
        ttk.Label(ab_bar, text="α/β (Gy)").pack(side="left", padx=(0, 6))
        ab = tk.StringVar(value=selections.alpha_beta_text[key])
        ttk.Entry(ab_bar, textvariable=ab, width=12).pack(side="left")
        ttk.Button(ab_bar, text="Save α/β", width=14, command=lambda: save_main_alpha_beta(key)).pack(
            side="left", padx=(6, 14)
        )
        ttk.Button(
            ab_bar, text="Select all", width=14, command=lambda: select_prescriptions(True, key)
        ).pack(side="left", padx=3)
        ttk.Button(
            ab_bar, text="Clear all", width=14, command=lambda: select_prescriptions(False, key)
        ).pack(side="left", padx=3)
        summary = tk.StringVar()
        ttk.Label(frame, textvariable=summary).pack(anchor="w", pady=(0, 5))
        details_area = ttk.Frame(frame)
        details_area.pack(fill="both", expand=True)
        tree = ttk.Treeview(
            details_area,
            columns=("include", "name", "D", "n", "d", "reference", "local_control"),
            show="headings",
            height=10,
        )
        tree.tag_configure("even", background="#ffffff")
        tree.tag_configure("odd", background="#f3f6fa")
        for column, heading, width in [
            ("include", "Optimize", 70),
            ("name", "Prescription", 300),
            ("D", "D (Gy)", 85),
            ("n", "Fractions", 80),
            ("d", "Gy/fx", 75),
            ("reference", "Reference", 80),
            ("local_control", "Local control (%)", 120),
        ]:
            tree.heading(column, text=heading)
            tree.column(
                column,
                width=width,
                minwidth=60,
                stretch=(column == "name"),
                anchor="w" if column == "name" else "center",
            )
        vertical = ttk.Scrollbar(details_area, orient="vertical", command=tree.yview)
        horizontal = ttk.Scrollbar(details_area, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
        tree.grid(row=0, column=0, sticky="nsew")
        vertical.grid(row=0, column=1, sticky="ns")
        horizontal.grid(row=1, column=0, sticky="ew")
        details_area.columnconfigure(0, weight=1)
        details_area.rowconfigure(0, weight=1)
        tree.bind("<Button-1>", lambda event: prescription_click(event, key))
        tree.bind("<space>", lambda event: prescription_space(event, key))
        tab_views[key] = {"frame": frame, "ab": ab, "summary": summary, "tree": tree}
        ab.trace_add("write", lambda *_args: alpha_beta_changed(key))
        prescription_tabs.add(frame, text=dataset_cache[key]["name"])
        refresh_prescriptions(key)

    def sync_prescription_tabs(prefer_id=None, refresh=False):
        nonlocal syncing_tabs
        selected_keys = [key for key, var in enabled_vars.items() if var.get()]
        old_focus = focused.get()
        syncing_tabs = True
        try:
            for key in list(tab_views):
                if key not in selected_keys:
                    frame = tab_views.pop(key)["frame"]
                    prescription_tabs.forget(frame)
                    frame.destroy()
            for index, key in enumerate(selected_keys):
                if key not in tab_views:
                    create_set_tab(key)
                elif refresh:
                    tab_views[key]["ab"].set(selections.alpha_beta_text[key])
                    refresh_prescriptions(key)
                prescription_tabs.tab(tab_views[key]["frame"], text=dataset_cache[key]["name"])
                prescription_tabs.insert(index, tab_views[key]["frame"])
            if selected_keys:
                empty_prescriptions.pack_forget()
                prescription_tabs.pack(fill="both", expand=True)
                preferred = str(prefer_id) if prefer_id is not None else old_focus
                view_set(preferred if preferred in selected_keys else selected_keys[0])
            else:
                focused.set("")
                prescription_tabs.pack_forget()
                empty_prescriptions.pack(anchor="w", padx=6, pady=12)
        finally:
            syncing_tabs = False

    prescription_tabs.bind("<<NotebookTabChanged>>", prescription_tab_changed)

    def export_set_csv():
        key = focused.get()
        if key not in dataset_cache:
            messagebox.showinfo(
                "Export prescriptions set", "Select a prescriptions set first.", parent=root
            )
            return
        try:
            record = selections.configured_set(dataset_cache[key])
        except ValueError as exc:
            messagebox.showerror("Cannot export prescriptions set", str(exc), parent=root)
            return
        path = filedialog.asksaveasfilename(
            parent=root,
            title="Export prescriptions set",
            defaultextension=".csv",
            initialfile=_analysis_stem(record["name"]) + ".csv",
            filetypes=[("CSV files", "*.csv")],
        )
        if not path:
            return
        try:
            export_prescriptions_set(record, path)
        except (OSError, ValueError) as exc:
            messagebox.showerror("Cannot export prescriptions set", str(exc), parent=root)
            return
        status.set(
            f"Exported {len(record['prescriptions'])} prescriptions from {record['name']} with α/β = {record['alpha_beta']:g} Gy."
        )

    def set_enabled_changed(key):
        sync_prescription_tabs(prefer_id=key if enabled_vars[key].get() else None)
        invalidate_result()

    def set_all_checks(value):
        for var in enabled_vars.values():
            var.set(value)
        sync_prescription_tabs()
        invalidate_result()

    def reveal_set(event):
        """Keep a checkbox visible when navigating long lists with the keyboard."""
        widget = event.widget
        top, height = selection_canvas.canvasy(0), selection_canvas.winfo_height()
        y, h = widget.winfo_y(), widget.winfo_height()
        full_height = max(1, check_items.winfo_height())
        if y < top:
            selection_canvas.yview_moveto(y / full_height)
        elif y + h > top + height:
            selection_canvas.yview_moveto((y + h - height) / full_height)

    bind_scrolling(selection_canvas, selection_canvas)
    bind_scrolling(check_items, selection_canvas)

    def refresh_sets(prefer_id=None, enable_new=False):
        previous_checks = {key: var.get() for key, var in enabled_vars.items()}
        old_focus = focused.get()
        for child in check_items.winfo_children():
            child.destroy()
        dataset_cache.clear()
        enabled_vars.clear()
        set_labels.clear()
        records = [store.get_set(record["id"]) for record in store.list_sets()]
        selections.sync(records)
        for full in records:
            key = str(full["id"])
            dataset_cache[key] = full
            checked = previous_checks.get(key, False)
            if enable_new and str(prefer_id) == key:
                checked = True
            if not previous_checks and prefer_id is None:
                checked = full["name"] == PROSTATE_SET_NAME
            var = tk.BooleanVar(value=checked)
            enabled_vars[key] = var
            set_labels[key] = tk.StringVar()
            refresh_set_label(key)
            button = ttk.Checkbutton(
                check_items,
                variable=var,
                command=lambda key=key: set_enabled_changed(key),
                textvariable=set_labels[key],
            )
            button.pack(anchor="w", padx=4, pady=4)
            bind_scrolling(button, selection_canvas)
            button.bind("<FocusIn>", reveal_set)
        sync_prescription_tabs(
            prefer_id=prefer_id if prefer_id is not None else old_focus, refresh=True
        )
        invalidate_result()

    def import_set_csv():
        path = filedialog.askopenfilename(
            parent=root,
            title="Import prescriptions set from CSV",
            filetypes=[
                ("CSV files", "*.csv"),
                ("Tab-separated files", "*.tsv"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        try:
            imported = read_prescriptions_set_csv(path)
        except (OSError, ValueError) as exc:
            messagebox.showerror("Cannot import prescriptions set", str(exc), parent=root)
            return
        edit_dialog(imported=imported)

    def edit_dialog(record=None, imported=None):
        dialog = tk.Toplevel(root)
        dialog.title(
            "Review imported prescriptions set"
            if imported is not None
            else "New prescriptions set"
            if record is None
            else "Edit prescriptions set"
        )
        dialog.geometry("1020x610")
        dialog.minsize(840, 420)
        dialog.transient(root)
        dialog.grab_set()

        body = ttk.Frame(dialog, padding=12)
        body.pack(fill="both", expand=True)
        top = ttk.LabelFrame(
            body, text="Prescriptions set information — all fields required", padding=10
        )
        top.pack(fill="x")
        ttk.Label(top, text="Prescriptions set name").grid(row=0, column=0, sticky="w", padx=5)
        ttk.Label(top, text="α/β (Gy)").grid(row=0, column=1, sticky="w", padx=5)
        draft = record or imported or {}
        set_name = tk.StringVar(value=draft.get("name", ""))
        alpha_beta = tk.StringVar(
            value=f"{draft['alpha_beta']:g}" if draft.get("alpha_beta") is not None else ""
        )
        name_entry = tk.Entry(top, textvariable=set_name, width=48)
        ab_entry = tk.Entry(top, textvariable=alpha_beta, width=14)
        name_entry.grid(row=1, column=0, sticky="we", padx=5, pady=(0, 4))
        ab_entry.grid(row=1, column=1, sticky="w", padx=5, pady=(0, 4))
        top.columnconfigure(0, weight=1)

        prescriptions = ttk.LabelFrame(
            body, text="Radiotherapy prescriptions", padding=8
        )
        prescriptions.pack(fill="both", expand=True, pady=(10, 6))
        reference_choice = tk.StringVar(value="auto")
        ttk.Radiobutton(
            prescriptions,
            text="Automatic reference (first 2 Gy/fraction prescription)",
            variable=reference_choice,
            value="auto",
        ).pack(anchor="w", pady=(0, 6))
        column_headers = ttk.Frame(prescriptions)
        column_headers.pack(fill="x", padx=(0, 16))
        widths = (78, 185, 92, 108, 162, 84)
        for column, heading in enumerate(
            ("Reference", "Prescription name", "D (Gy)", "n (fractions)", "Local tumor control (%)")
        ):
            ttk.Label(column_headers, text=heading).grid(row=0, column=column, sticky="w", padx=3)
        for column, width in enumerate(widths):
            column_headers.columnconfigure(column, minsize=width, weight=int(column == 1))
        ttk.Label(column_headers, text="Optional").grid(row=1, column=4, sticky="w", padx=3)

        scrolling = ttk.Frame(prescriptions)
        scrolling.pack(fill="both", expand=True)
        sc_canvas = tk.Canvas(scrolling, highlightthickness=0)
        scrollbar = ttk.Scrollbar(scrolling, orient="vertical", command=sc_canvas.yview)
        sc_canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        sc_canvas.pack(side="left", fill="both", expand=True)
        rows_parent = ttk.Frame(sc_canvas)
        canvas_id = sc_canvas.create_window((0, 0), window=rows_parent, anchor="nw")
        rows_parent.bind(
            "<Configure>", lambda e: sc_canvas.configure(scrollregion=sc_canvas.bbox("all"))
        )
        sc_canvas.bind("<Configure>", lambda e: sc_canvas.itemconfigure(canvas_id, width=e.width))
        rows = []
        next_row_id = 0

        def remove_row(row):
            if len(rows) <= 2:
                messagebox.showinfo(
                    "Radiotherapy prescriptions",
                    "At least two prescriptions are required.",
                    parent=dialog,
                )
                return
            rows.remove(row)
            if reference_choice.get() == row["id"]:
                reference_choice.set("auto")
            row["frame"].destroy()

        def add_row(name="", D="", n="", control=None):
            nonlocal next_row_id
            row_id = str(next_row_id)
            next_row_id += 1
            frame = ttk.Frame(rows_parent)
            frame.pack(fill="x", pady=3)
            ttk.Radiobutton(frame, variable=reference_choice, value=row_id).grid(
                row=0, column=0, sticky="w", padx=3
            )
            name_var, D_var, n_var = (
                tk.StringVar(value=name),
                tk.StringVar(value=str(D)),
                tk.StringVar(value=str(n)),
            )
            control_var = tk.StringVar(value="" if control is None else str(control))
            e_name = tk.Entry(frame, textvariable=name_var, width=18)
            e_D = tk.Entry(frame, textvariable=D_var, width=8)
            e_n = tk.Entry(frame, textvariable=n_var, width=8)
            e_control = tk.Entry(frame, textvariable=control_var, width=12)
            for column, entry in enumerate((e_name, e_D, e_n, e_control), start=1):
                entry.grid(row=0, column=column, sticky="ew", padx=3)
            for column, width in enumerate(widths):
                frame.columnconfigure(column, minsize=width, weight=int(column == 1))
            row = {
                "id": row_id,
                "frame": frame,
                "values": (name_var, D_var, n_var),
                "entries": (e_name, e_D, e_n),
                "control": control_var,
                "control_entry": e_control,
            }
            ttk.Button(frame, text="Remove", command=lambda: remove_row(row)).grid(
                row=0, column=5, sticky="e", padx=(6, 2)
            )
            rows.append(row)
            sc_canvas.yview_moveto(1.0)

        previous = draft.get("prescriptions")
        if previous:
            chosen_reference = draft.get("reference_prescription")
            reference_key = (Prescription.from_value(chosen_reference).schedule_key
                             if chosen_reference is not None else None)
            for prescription in previous:
                add_row(prescription.name, prescription.total_gy, prescription.fractions,
                        prescription.local_control_pct)
                if prescription.schedule_key == reference_key:
                    reference_choice.set(rows[-1]["id"])
        else:
            add_row()
            add_row()

        actions = ttk.Frame(body)
        actions.pack(fill="x", pady=3)
        ttk.Button(actions, text="+ Add prescription", command=add_row).pack(side="left")

        def save_form():
            valid = True

            def mark(entry, error):
                nonlocal valid
                entry.configure(bg="#ffe9e9" if error else "white")
                if error:
                    valid = False

            mark(name_entry, not set_name.get().strip())
            try:
                ab = float(alpha_beta.get())
                ab_bad = not math.isfinite(ab) or ab <= 0
            except ValueError:
                ab_bad = True
            mark(ab_entry, ab_bad)
            saved_rows = []
            bad_controls = False
            for row in rows:
                variables = row["values"]
                values = [v.get().strip() for v in variables]
                mark(row["entries"][0], not values[0])
                try:
                    total = float(values[1])
                    bad_total = not math.isfinite(total) or total <= 0
                except ValueError:
                    bad_total = True
                try:
                    nf = float(values[2])
                    bad_n = not math.isfinite(nf) or nf <= 0 or not nf.is_integer()
                except ValueError:
                    bad_n = True
                mark(row["entries"][1], bad_total)
                mark(row["entries"][2], bad_n)
                try:
                    control = validated_local_control(row["control"].get())
                    mark(row["control_entry"], False)
                except ValueError:
                    mark(row["control_entry"], True)
                    bad_controls = True
                    control = None
                if valid:
                    saved_rows.append(Prescription(*values, control))
            if not valid:
                messagebox.showerror(
                    "Prescription inputs",
                    "Local tumor control must be a percentage from 0 to 100, or left blank."
                    if bad_controls else "Enter the required names and positive α/β and D values, "
                    "with a positive integer number of fractions.",
                    parent=dialog,
                )
                return
            try:
                new_id = store.save_set(
                    set_name.get(),
                    alpha_beta.get(),
                    saved_rows,
                    set_id=record["id"] if record else None,
                    reference_prescription=next(
                        (
                            values
                            for row, values in zip(rows, saved_rows)
                            if row["id"] == reference_choice.get()
                        ),
                        None,
                    ),
                )
            except (ValueError, KeyError) as exc:
                messagebox.showerror("Cannot save prescriptions set", str(exc), parent=dialog)
                return
            dialog.destroy()
            selections.alpha_beta_text.pop(str(new_id), None)
            refresh_sets(prefer_id=new_id, enable_new=(record is None))

        ttk.Button(actions, text="Cancel", command=dialog.destroy).pack(side="right", padx=4)
        ttk.Button(actions, text="Save prescriptions set", command=save_form).pack(
            side="right", padx=4
        )
        dialog.bind("<Escape>", lambda _: dialog.destroy())
        name_entry.focus_set()

    def edit_selected():
        if focused.get() in dataset_cache:
            edit_dialog(store.get_set(int(focused.get())))
        else:
            messagebox.showinfo("Edit", "Select a saved prescriptions set first.")

    def delete_selected():
        key = focused.get()
        if key not in dataset_cache:
            return
        name = dataset_cache[key]["name"]
        if messagebox.askyesno(
            "Delete prescriptions set",
            f'Permanently delete prescriptions set "{name}" and all its prescriptions?',
            parent=root,
        ):
            store.delete_set(int(key))
            focused.set("")
            refresh_sets()

    def run():
        if not can_run():
            return
        try:
            checked = [
                selections.configured_set(dataset_cache[key])
                for key, enabled in enabled_vars.items()
                if enabled.get()
            ]
            if not checked:
                raise ValueError(
                    "Select at least one prescriptions set before running optimization."
                )
            method = next(
                (key for key, label in method_labels.items() if label == var_method.get()),
                None,
            )
            if method is None:
                raise ValueError("Select a valid RD EQD2 minimization criterion.")
            res = analyze_selected(
                checked,
                float(var_start.get()),
                float(var_end.get()),
                float(var_step.get()),
                method=method,
                bound_pct=float(var_bound_pct.get()),
            )
            results_table.heading(
                "bounds",
                text=f"{res['bound_pct']:g}% {'range' if method == 'range' else 'SD'} bounds (Gy)",
            )
            res["plot_settings"] = dict(plot_settings)
            existing = result_holder.get("analysis_window")
            if existing is not None and existing.winfo_exists():
                existing.destroy()
            result_holder["analysis_window"] = None
            result_holder["result"] = res
            clear_analysis_tabs()
            result_holder["embedded_analysis"] = open_analysis_window(
                root, res, container=analysis_controls, tabs=result_tabs
            )
            result_tabs.select(plotbox)
            for item in results_table.get_children():
                results_table.delete(item)
            results_table.configure(height=min(3, len(res["sets"])))
            boundary_count = 0
            for dataset in res["sets"]:
                r = dataset["result"]
                i = r["idx"]
                boundary = i == 0 or i == len(r["q"]) - 1
                boundary_count += int(boundary)
                objective = {"range": r["spread"], "sd": r["sample_sd"], "sse": r["sse"]}[method][i]
                lo, hi = r["profile_bounds"]
                bounds_text = (
                    f"{lo:.3f}–{hi:.3f}" if lo is not None and hi is not None else "Outside search"
                )
                results_table.insert(
                    "",
                    "end",
                    values=(
                        dataset["name"],
                        len(r["schedules"]),
                        f"{r['signed_ab']:g}",
                        f"{r['q'][i]:.5g}",
                        f"{objective:.3f}",
                        f"{r['mean'][i]:.3f}",
                        f"{r['sample_sd'][i]:.3f}",
                        f"{r['rsd'][i]:.3f}",
                        f"{r['eta'][i]:.3f}" if np.isfinite(r["eta"][i]) else "N/A",
                        f"{r['spread'][i]:.3f}",
                        bounds_text,
                        "BOUNDARY" if boundary else "Interior",
                    ),
                )
            status.set(
                f"{len(res['sets'])} set(s), {sum(len(ds['result']['schedules']) for ds in res['sets'])} prescriptions; "
                f"independently calibrated using "
                f"{OPTIMIZATION_METHODS[method].lower()}; "
                f"{res['bound_pct']:g}% profile bounds. "
                + (
                    f"WARNING: {boundary_count} minimum/minima lie on a grid boundary; expand the interval."
                    if boundary_count
                    else "All minima lie inside the selected grid."
                )
            )
            ax.set_axis_on()
            draw_range_plot(ax, res)
            canvas.draw()
            sync_action_states()
            workspace_tabs.select(results_area)
        except Exception as exc:
            invalidate_result()
            status.set("Optimization failed. Review the inputs and run again.")
            messagebox.showerror("Calculation error", str(exc), parent=root)

    def save():
        if result_holder["result"] is None:
            messagebox.showinfo("Export", "Run optimization first.", parent=root)
            return
        folder = filedialog.askdirectory(title="Choose output folder")
        if folder:
            try:
                export(result_holder["result"], folder)
                messagebox.showinfo("Saved", f"Saved CSV data and graph to {folder}", parent=root)
            except Exception as exc:
                messagebox.showerror("Export error", str(exc), parent=root)

    def image_export_dialog():
        if result_holder["result"] is None:
            messagebox.showinfo("Export image", "Run optimization first.", parent=root)
            return
        dialog = tk.Toplevel(root)
        dialog.title("Export plot image")
        dialog.resizable(False, False)
        dialog.transient(root)
        dialog.grab_set()
        body = ttk.Frame(dialog, padding=16)
        body.pack(fill="both", expand=True)
        fmt_var = tk.StringVar(value=next(iter(IMAGE_FORMATS)))
        ppi_var = tk.StringVar(value="300")
        ttk.Label(body, text="Image type").grid(row=0, column=0, sticky="w", padx=(0, 16), pady=5)
        format_input = ttk.Combobox(
            body, textvariable=fmt_var, state="readonly", width=20, values=list(IMAGE_FORMATS)
        )
        format_input.grid(row=0, column=1, sticky="ew", pady=5)
        ttk.Label(body, text="Resolution (PPI)").grid(
            row=1, column=0, sticky="w", padx=(0, 16), pady=5
        )
        ppi_input = ttk.Entry(body, textvariable=ppi_var, width=22)
        ppi_input.grid(row=1, column=1, sticky="ew", pady=5)
        ttk.Label(
            body,
            text="PDF, EPS and SVG use vector graphics; PPI applies to raster elements.",
            wraplength=370,
            foreground="#555555",
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(8, 10))
        body.columnconfigure(1, weight=1)
        actions = ttk.Frame(body)
        actions.grid(row=3, column=0, columnspan=2, sticky="e", pady=(5, 0))

        def save_image():
            # Validate the resolution before opening the file chooser.
            try:
                ppi_value = float(ppi_var.get())
                if (
                    not math.isfinite(ppi_value)
                    or not ppi_value.is_integer()
                    or not 1 <= ppi_value <= 1200
                ):
                    raise ValueError()
            except (TypeError, ValueError):
                messagebox.showerror(
                    "Invalid PPI", "Enter a whole-number PPI between 1 and 1200.", parent=dialog
                )
                ppi_input.focus_set()
                return
            ext = IMAGE_FORMATS[fmt_var.get()]
            path = filedialog.asksaveasfilename(
                parent=dialog,
                title="Save optimization plot",
                initialfile="optimization_profile" + ext,
                defaultextension=ext,
                filetypes=[(fmt_var.get(), "*" + ext)],
            )
            if not path:
                return
            # The chosen type governs export even when a typed filename has
            # another or no extension. Never save JPEG bits to a .png path.
            destination = Path(path)
            if destination.suffix.lower() != ext:
                if destination.suffix.lower() in IMAGE_EXTENSIONS:
                    destination = destination.with_suffix(ext)
                elif not destination.suffix:
                    destination = destination.with_suffix(ext)
                else:
                    messagebox.showerror(
                        "Invalid file name",
                        f"Use a {ext} extension for the selected image type.",
                        parent=dialog,
                    )
                    return
            try:
                export_image(result_holder["result"], destination, ppi=int(ppi_value))
            except Exception as exc:
                messagebox.showerror("Image export error", str(exc), parent=dialog)
                return
            dialog.destroy()
            messagebox.showinfo("Saved image", f"Saved plot to {destination}", parent=root)

        ttk.Button(actions, text="Cancel", command=dialog.destroy).pack(side="right", padx=(8, 0))
        ttk.Button(actions, text="Save image…", command=save_image).pack(side="right")
        dialog.bind("<Escape>", lambda event: dialog.destroy())
        ppi_input.focus_set()

    def open_analysis():
        if result_holder["result"] is None:
            return
        existing = result_holder.get("analysis_window")
        if existing is not None and existing.winfo_exists():
            existing.lift()
            existing.focus_set()
            return
        result_holder["analysis_window"] = open_analysis_window(root, result_holder["result"])

    analysis_button = ttk.Button(result_toolbar, text="Analysis…", command=open_analysis)
    analysis_button.pack(side="right", padx=(8, 0))
    run_button = ttk.Button(bar, text="Run optimization", style="Accent.TButton", command=run)
    run_button.pack(side="right")
    ttk.Button(result_toolbar, text="Plot settings…", command=plot_settings_dialog).pack(
        side="right", padx=(8, 0)
    )
    export_menu = tk.Menu(root, tearoff=False)
    export_menu.add_command(label="Export image…", command=image_export_dialog)
    export_menu.add_command(label="Export results + graph", command=save)
    export_menu.add_separator()

    def analysis_export(kind, clinical=False):
        view = result_holder.get("embedded_analysis")
        if view is None:
            messagebox.showinfo("Export", "Run optimization first.", parent=root)
        elif kind == "tables":
            view.export_tables()
        elif clinical and not view.has_clinical():
            messagebox.showinfo("Export", "No observed local control for this set.", parent=root)
        else:
            view.export_plot(clinical=clinical)

    export_menu.add_command(label="Export tables…", command=lambda: analysis_export("tables"))
    export_menu.add_command(label="Export dispersion plot…", command=lambda: analysis_export("plot"))
    export_menu.add_command(label="Export clinical evaluation plot…",
                            command=lambda: analysis_export("plot", clinical=True))
    ttk.Menubutton(result_toolbar, text="Export", menu=export_menu).pack(side="right", padx=(8, 0))
    menubar = tk.Menu(root)
    root.configure(menu=menubar)
    file_menu = tk.Menu(menubar, tearoff=False)
    menubar.add_cascade(label="File", menu=file_menu)
    def save_current_plot():
        if result_holder["result"] is None or file_menu.entrycget("Save plot…", "state") == "disabled":
            return
        if result_tabs.select() == str(plotbox):
            image_export_dialog()
        else:
            view = result_holder.get("embedded_analysis")
            if view is not None:
                view.save_plot()

    def print_current_plot():
        if result_holder["result"] is None or file_menu.entrycget("Print plot…", "state") == "disabled":
            return
        if result_tabs.select() != str(plotbox):
            view = result_holder.get("embedded_analysis")
            if view is not None:
                view.print_plot()
            return
        import tempfile

        try:
            path = Path(tempfile.mkdtemp(prefix="rd-dq-print-")) / "optimization.pdf"
            export_image(result_holder["result"], path)
            open_print_preview(path)
        except Exception as exc:
            messagebox.showerror("Print plot", str(exc), parent=root)

    file_menu.add_command(label="Save plot…", accelerator="Ctrl+S", command=save_current_plot)
    file_menu.add_command(label="Print plot…", accelerator="Ctrl+P", command=print_current_plot)
    file_menu.add_separator()
    file_menu.add_cascade(label="Export results", menu=export_menu)
    file_menu.add_separator()
    file_menu.add_command(label="Exit", command=root.destroy)
    set_menu = tk.Menu(menubar, tearoff=False)
    menubar.add_cascade(label="Prescriptions", menu=set_menu)
    set_menu.add_command(label="New prescriptions set", accelerator="Ctrl+N", command=edit_dialog)
    set_menu.add_command(label="Import prescriptions set…", accelerator="Ctrl+O", command=import_set_csv)
    set_menu.add_command(label="Export prescriptions set…", command=export_set_csv)
    set_menu.add_separator()
    set_menu.add_command(label="Edit prescriptions set", command=edit_selected)
    set_menu.add_command(label="Delete prescriptions set", command=delete_selected)
    set_menu.add_separator()
    set_menu.add_command(label="Select all sets", command=lambda: set_all_checks(True))
    set_menu.add_command(label="Clear all sets", command=lambda: set_all_checks(False))
    analysis_menu = tk.Menu(menubar, tearoff=False)
    menubar.add_cascade(label="Analysis", menu=analysis_menu)
    analysis_menu.add_command(label="Run optimization", accelerator="Ctrl+R", command=run)
    analysis_menu.add_command(label="Analysis…", command=open_analysis)
    view_menu = tk.Menu(menubar, tearoff=False)
    menubar.add_cascade(label="View", menu=view_menu)
    view_menu.add_command(label="Plot settings…", command=plot_settings_dialog)
    view_menu.add_separator()
    view_menu.add_command(label="Prescriptions", command=lambda: workspace_tabs.select(prescription_list))
    view_menu.add_command(label="Results", command=lambda: workspace_tabs.select(results_area))
    view_menu.add_separator()

    def select_result_tab(label):
        for page in result_tabs.tabs():
            if result_tabs.tab(page, "text") == label and result_tabs.tab(page, "state") != "disabled":
                workspace_tabs.select(results_area)
                result_tabs.select(page)
                return

    for label in ("Optimization profile", "EQD2 dispersion", "Clinical evaluation", "Robustness"):
        view_menu.add_command(label=label, command=lambda label=label: select_result_tab(label))
    help_menu = tk.Menu(menubar, tearoff=False)
    menubar.add_cascade(label="Help", menu=help_menu)
    help_menu.add_command(label="Parameter guide", command=lambda: messagebox.showinfo(
        "Optimization settings",
        "Bounds use sample EQD2 SD for SSE/SD optimization, or EQD2 range for range optimization.\n\n"
        "Enter α/β for q ≥ 0; q < 0 uses its negative. At q = 0, EQD2 = D.", parent=root
    ))
    help_menu.add_command(label="About RD Dq Optimizer", command=lambda: messagebox.showinfo(
        "RD Dq Optimizer", "RD Dq Optimizer\nResilience–Depletion EQD2 and BED calculations.", parent=root
    ))
    for shortcut, command in (("<Control-n>", edit_dialog), ("<Control-o>", import_set_csv),
                              ("<Control-r>", run), ("<Control-s>", save_current_plot),
                              ("<Control-p>", print_current_plot)):
        root.bind(shortcut, lambda event, command=command: command())
    result_tabs.bind("<<NotebookTabChanged>>", sync_action_states, add="+")
    workspace_tabs.bind("<<NotebookTabChanged>>", sync_action_states, add="+")
    sidebar_initialized = False

    def initialize_sidebar(_event=None):
        nonlocal sidebar_initialized
        if not sidebar_initialized and sidebar.winfo_height() > 100:
            sidebar.sashpos(0, int(sidebar.winfo_height() * 0.50))
            sidebar_initialized = True

    sidebar.bind("<Map>", lambda event: root.after_idle(initialize_sidebar), add="+")
    if sidebar.winfo_ismapped():
        root.after_idle(initialize_sidebar)
    methods.bind("<<ComboboxSelected>>", method_changed)
    for var in (var_start, var_end, var_step, var_bound_pct):
        var.trace_add("write", invalidate_result)
    refresh_sets()
    root.mainloop()


def main():
    parser = argparse.ArgumentParser(
        description="RD EQD2 Optimizer — calibrate Dq from radiotherapy prescriptions"
    )
    parser.add_argument("--batch", action="store_true", help="run from command line, no GUI")
    parser.add_argument(
        "--ab",
        type=float,
        default=None,
        help="positive alpha/beta value for q >= 0 (Gy); q < 0 uses its negative",
    )
    parser.add_argument("--dq-start", type=float, default=DEFAULT_Q_START)
    parser.add_argument("--dq-end", type=float, default=DEFAULT_Q_END)
    parser.add_argument("--dq-step", type=float, default=DEFAULT_Q_STEP)
    parser.add_argument(
        "--bound-pct",
        type=float,
        default=5.0,
        help="percentage above minimum sample EQD2 SD (SSE/SD) or range (range method); default: 5",
    )
    parser.add_argument(
        "--dataset",
        choices=["prostate"],
        default="prostate",
        help="built-in prescriptions set (prostate)",
    )
    parser.add_argument(
        "--csv",
        type=str,
        help="read schedule CSV with name,total_gy,fractions and optional "
        "prescriptions_set,alpha_beta,is_reference,local_control_pct",
    )
    parser.add_argument(
        "--set", dest="set_name", type=str, help="batch-optimize a saved prescriptions set by name"
    )
    parser.add_argument(
        "--db", type=str, help="path to prescriptions database (default: user data folder)"
    )
    parser.add_argument(
        "--method",
        choices=tuple(OPTIMIZATION_METHODS),
        default="sse",
        help="RD-derived EQD2 minimization criterion: SSE (default), sample SD, or range",
    )
    parser.add_argument("--out", default="results", help="output folder")
    parser.add_argument(
        "--plot-title",
        default=None,
        help="custom graph title; default is the RD EQD2 minimization profile for the selected criterion",
    )
    a = parser.parse_args()
    if not a.batch:
        return launch_gui(db_path=a.db)
    if a.csv and a.set_name:
        parser.error("Use either --csv or --set, not both.")
    if a.set_name:
        db = (
            PrescriptionStore(path=a.db, initial_sets=DEFAULT_SETS)
            if a.db
            else PrescriptionStore(initial_sets=DEFAULT_SETS)
        )
        record = db.find_set(a.set_name)
        if record is None:
            parser.error(f"No saved prescriptions set called {a.set_name!r} in the database.")
        data = record["prescriptions"]
        reference_record = record
        ab = record["alpha_beta"] if a.ab is None else a.ab
    else:
        if a.csv:
            imported = read_prescriptions_set_csv(a.csv)
            reference_record = imported
            data = imported["prescriptions"]
            ab = (
                (imported["alpha_beta"] if imported["alpha_beta"] is not None else 1.5)
                if a.ab is None
                else a.ab
            )
        else:
            data = PROSTATE_SET["prescriptions"]
            reference_record = DEFAULT_SETS[PROSTATE_SET_NAME]
            ab = PROSTATE_ALPHA_BETA if a.ab is None else a.ab
    result = analyze(
        ab, a.dq_start, a.dq_end, a.dq_step, data, method=a.method, bound_pct=a.bound_pct
    )
    result["plot_settings"] = normalized_plot_settings({"title": a.plot_title or ""})
    result["reference_prescription"] = reference_schedule(reference_record)
    export(result, a.out)
    i = result["idx"]
    print("Symbolic formula:", EQD2_EXPR)
    print("Optimization method:", OPTIMIZATION_METHODS[a.method])
    print(
        f"input alpha/beta={result['ab']:g} Gy; fitted alpha/beta={result['signed_ab']:g} Gy; Dq={result['q'][i]:.6f} Gy; SSE={result['sse'][i]:.4f} Gy^2; range={result['spread'][i]:.4f} Gy; sample SD={result['sample_sd'][i]:.4f} Gy; mean={result['mean'][i]:.4f} Gy"
    )
    print(
        f"{result['bound_pct']:g}% {'range' if a.method == 'range' else 'SD'} profile bounds:",
        result["profile_bounds"],
    )
    if i in (0, len(result["q"]) - 1):
        print("WARNING: optimum is on grid boundary.")
    for prescription, eq in zip(result["schedules"], result["eqd2"][i]):
        name, total, fx = prescription.schedule_key
        print(f"{name:32} {total:g} Gy/{fx} fx  -> {eq:.3f} Gy EQD2")
    print("Saved results to", a.out)


if __name__ == "__main__":
    main()
