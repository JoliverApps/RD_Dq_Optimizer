# Validation

Signed fixed-ratio RD calibration verified on 6 November 2026. The application
keeps each set's finite nonzero α/β fixed and optimizes only q on its compatible
branch, including the continuous linear-survival boundary at q = 0.

Application SHA-256:

```text
d7066a9be78631ac1447d344004f67fe44cdb44d5827a4b7248888340aa4d267
```

## Completed checks

| Check | Result |
| --- | --- |
| Ruff lint and formatting | Passed for `src` and `tools` |
| Python bytecode compilation | Passed |
| Full source runtime suite without GUI | Passed, all 12 check groups |
| Positive and inverse-shoulder EQD2 | Matched independently integrated hazards to relative tolerance 2 × 10⁻¹³ |
| Signed symbolic expression and numerical mapping | Both branches matched known r and s values |
| q = 0 and one-sided limits | EQD2 = D exactly at zero; nearby values converge from each compatible branch |
| Zero-boundary analysis | BED = D, r = 0, s unavailable; JSON and plots succeed |
| Negative synthetic isoeffective set | SSE, SD, and range recover q = −3 Gy within 2 × 10⁻⁶ Gy at fixed α/β = −5 Gy |
| Signed profile bounds | Negative crossings bracket the optimum and match the specified SD tolerance |
| Mixed-sign sets | Independently optimized and exported on compatible candidate domains; opposite-branch CSV cells are blank |
| Old positive-only SQLite schema | Migrated with set IDs, prescription IDs, metadata, and foreign-key relationships preserved |
| Negative saved ratios | Save, reopen, CSV round trip, and cascading deletion passed |
| Negative analysis and robustness | Passed, including JSON export and dispersion plots |
| α/β = −2 Gy | RD remains available; undefined LQ EQD2 comparisons are marked unavailable |
| Command-line negative-ratio export | Passed at α/β = −2 Gy over q ∈ [−10, 0] |
| Original prostate calibration | Preserved within the existing baseline tolerance |
| GUI interaction | Not run in this environment; no graphical display is available |
| Windows executable build | Not run in this Linux environment |

The prostate baseline remains Dq ≈ 3.54386 Gy, mean EQD2 ≈ 75.81790 Gy, and
RSD ≈ 2.83660%. The runtime suite checks these against the original calibration
with absolute tolerance 2 × 10⁻⁵.

The existing suite also verifies prostate-only initialization, saved-set
preservation, prescription selection, independent α/β drafts, CSV imports,
all three objectives, analysis and robustness tables, CSV/TSV/JSON/LaTeX
exports, image formats, combined-set exports, and plot preferences.

Reproduce the headless checks with:

```text
python -m ruff check src tools
python -m ruff format --check src tools
python -m compileall -q src tools
python src/main.py --self-test signed-validation.json --no-gui
```

The GUI integration checks have been extended to enter, optimize, save, and
analyze a negative ratio. Windows builds still require both source and packaged
executable checks, including GUI interaction, before reporting build success.
