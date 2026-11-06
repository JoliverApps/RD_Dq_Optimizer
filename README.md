# RD EQD2 Optimizer

RD EQD2 Optimizer estimates the Resilience–Depletion (RD) shoulder descriptor Dq from a set of radiotherapy prescriptions by minimizing the dispersion of their equivalent doses in 2 Gy fractions (EQD2). Each prescriptions set has a fixed α/β magnitude.

The application provides prescription management, BED and EQD2 calculations, optimization profiles, dose–dispersion plots, descriptive statistics, robustness analysis, and data and image exports. A prostate set containing eight prescriptions at α/β = 1.5 Gy is available on first launch.

## Installation

### Windows executable

Open `RD_EQD2_Optimizer.exe`. The executable includes its Python runtime and dependencies. Choose a writable folder when saving exports.

### Python application

Install Python 3.11 or 3.12 with pip and Tcl/Tk. Download and extract the repository, open a terminal in its folder, and run:

```text
python -m pip install -r requirements.txt
python rd_eqd2_optimizer.py
```

If you downloaded the standalone `rd_eqd2_optimizer.py` file without the repository, install its dependencies and launch it with:

```text
python -m pip install numpy==2.2.6 matplotlib==3.10.3 sympy==1.14.0
python rd_eqd2_optimizer.py
```

## Run an optimization

1. Select the prescriptions sets to analyze.
2. Choose a set to view its prescriptions. Check the **Optimize** boxes for the prescriptions to include. Each selected set needs at least two prescriptions.
3. Enter **|α/β| (Gy)** for each set. Click **Save α/β** to store the value for future sessions.
4. Set **q start**, **q end**, and **q step**, then choose a minimization criterion.
5. Click **Run optimization**.

Each selected set is optimized independently. Prescription checkboxes control the current analysis; they do not delete saved prescriptions.

### Parameters and search interval

The α/β magnitude must be finite and nonzero. It remains fixed during optimization. The candidate parameter q represents the shoulder displacement, and its fitted value is reported as Dq. The sign of α/β follows q.

| Candidate q | α/β sign |
| --- | --- |
| q ≥ 0 | Positive |
| q < 0 | Negative |

The default interval is −50 to +50 Gy with a step of 0.001 Gy. The optimizer evaluates the full selected interval. Zero is included once when the interval spans zero; at zero, BED and EQD2 equal the physical dose.

Use a smaller step for a finer grid. If the application reports too many grid points, increase the step or narrow the interval.

### Minimization criterion

| Criterion | Quantity minimized |
| --- | --- |
| SSE | Sum of squared deviations of prescription EQD2 values from their mean |
| SD | Sample standard deviation of EQD2 |
| Range | Difference between the highest and lowest EQD2 |

**Near-optimal bounds (%)** defines a tolerance above the minimum. SSE and SD optimization use the sample SD for these bounds; range optimization uses the EQD2 range. The default tolerance is 5%.

## Read the results

The results table reports Dq, the minimized objective, mean EQD2, sample SD, relative standard deviation (RSD), trend measure η, EQD2 range, and profile bounds.

A minimum marked **BOUNDARY** lies at an endpoint of the search interval. A wider interval may help locate an interior minimum. **Outside search** means that the selected interval does not contain both profile-bound crossings.

Click **Analysis…** for:

- A scatter plot of EQD2 against dose per fraction.
- Per-prescription RD BED and EQD2, with LQ comparisons.
- Summary statistics, fitted trends, and RD parameters r and s.
- Leave-one-out and α/β sensitivity analyses.

At Dq = 0, the linear response gives BED = EQD2 = D. The sensitization rate s is displayed as **N/A**.

For α/β = −2 Gy, LQ EQD2 is unavailable because its reference denominator is zero. RD results remain available. For other negative α/β values, LQ extrapolations can produce negative equivalent doses.

### Robustness analysis

Open the **Robustness** tab in the analysis window and click **Run robustness**. Leave-one-out analysis requires at least three prescriptions.

To assess α/β sensitivity, enter comma-separated nonzero magnitudes. Each magnitude is held fixed during its own optimization.

## Manage prescriptions sets

Use **New prescriptions set**, **Edit prescriptions set**, and **Delete prescriptions set** to manage saved data. Each prescription requires a name, a positive total dose in Gy, and a positive integer number of fractions.

Use **Export prescriptions set…** to save the viewed set's checked prescriptions and current α/β as CSV. Use **Import prescriptions set…** to load a CSV file.

The CSV columns are:

| Column | Contents |
| --- | --- |
| `name` | Prescription name |
| `total_gy` | Total dose in Gy |
| `fractions` | Number of fractions |
| `prescriptions_set` | Set name |
| `alpha_beta` | Fixed α/β magnitude in Gy |

The first three columns are required for import. A file must describe one prescriptions set. Prescription names must be unique within that set. The `samples/prostate.csv` file provides an example.

## Export data and plots

- **Export results + graph** saves the optimization grid, optimal prescription values, and an optimization plot.
- **Export image…** saves the optimization plot as PNG, JPEG, or EPS with a selected resolution.
- **Plot settings…** controls titles, labels, fonts, colors, dimensions, and axis limits.
- The analysis window exports tables as CSV, TSV, JSON, or LaTeX, and dispersion plots as PNG, JPEG, EPS, PDF, or SVG.


## Command-line use

Run the prostate set:

```text
python rd_eqd2_optimizer.py --batch --dataset prostate --out results
```

Analyze a CSV file:

```text
python rd_eqd2_optimizer.py --batch --csv prescriptions.csv --ab 1.5 --dq-start 0 --dq-end 50 --dq-step 0.01 --out results
```

Search across positive and negative q:

```text
python rd_eqd2_optimizer.py --batch --csv prescriptions.csv --ab 10 --dq-start=-50 --dq-end=50 --out results
```

Use `--set "Set name"` to select a saved prescriptions set. Use `--method sse`, `--method sd`, or `--method range` to choose the criterion. Use `--db` to select a prescriptions database. Run `python rd_eqd2_optimizer.py --help` for all options.

## Saved data

On Windows, the default files are:

| File | Location |
| --- | --- |
| Prescriptions database | `%USERPROFILE%\.rd_dq_optimizer\prescriptions.sqlite3` |
| Plot preferences | `%USERPROFILE%\.rd_dq_optimizer\plot_settings.json` |
| Windows executable log | `%USERPROFILE%\.rd_dq_optimizer\application.log` |

On other platforms, the data folder is `.rd_dq_optimizer` in the user's home directory. The Python application and Windows executable use the same default database. Back up the database file to retain saved sets.

A different database can be selected with `--db` or the `RD_DQ_DB_PATH` environment variable.

## Build a Windows executable

Install:

- Visual Studio 2022 with **Desktop development with C++** and **MSVC v143**.
- 64-bit Python 3.11 or 3.12 with pip, venv, Tcl/Tk, and the Python launcher.

Open `RD_EQD2_Optimizer.sln`, select **Release | x64**, and choose **Build > Build Solution**. The first build requires internet access to install dependencies. Run the resulting `dist\Release\RD_EQD2_Optimizer.exe`.

To build from Command Prompt, run:

```bat
Build.cmd Build Release
```

If Python is installed in a custom location:

```bat
set "RD_PYTHON=C:\Path\To\Python312\python.exe"
Build.cmd Build Release
```

A Release build produces a single executable. A Debug build produces `dist\Debug\RD_EQD2_Optimizer\RD_EQD2_Optimizer.exe`; distribute its entire containing folder, including `_internal`.

## Troubleshooting

| Problem | Action |
| --- | --- |
| Missing NumPy, Matplotlib, or SymPy | Run `python -m pip install -r requirements.txt` with the interpreter used to launch the application. |
| Tkinter is unavailable | Install Tcl/Tk support for the selected Python interpreter. |
| Too many grid points | Increase q step or narrow the interval. |
| A Windows executable reports a startup error | Check `application.log` in the data folder. |
