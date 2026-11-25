# RD EQD2 Optimizer

RD EQD2 Optimizer is a Python desktop app for comparing fractionation schedules on the Resilience–Depletion (RD) scale. It estimates the shoulder descriptor Dq by minimizing the dispersion of equivalent doses in 2 Gy fractions (EQD2) across selected prescriptions. It also calculates BED, compares RD and LQ results, and plots observed local tumor control against RD EQD2.

The first launch opens a prostate preset with ten prescriptions at α/β = 1.5 Gy. For the model, the calibration procedure, and the outcome analysis, see the [mathematical reference](MATHEMATICS.md).

## Installation

### Desktop downloads

Download the package for your operating system from the [latest GitHub release](https://github.com/JoliverApps/RD_Dq_Optimizer/releases/latest). Each package includes the Python runtime and dependencies.

| Platform | Download | Launch |
| --- | --- | --- |
| Windows x64 | [Portable executable](https://github.com/JoliverApps/RD_Dq_Optimizer/releases/latest/download/RD_EQD2_Optimizer-Windows-x86_64.exe) | Open the downloaded `.exe`. |
| macOS Apple silicon | [App archive](https://github.com/JoliverApps/RD_Dq_Optimizer/releases/latest/download/RD_EQD2_Optimizer-macOS-arm64.zip) | Extract the archive and open `RD_EQD2_Optimizer.app`. |
| macOS Intel | [App archive](https://github.com/JoliverApps/RD_Dq_Optimizer/releases/latest/download/RD_EQD2_Optimizer-macOS-x86_64.zip) | Extract the archive and open `RD_EQD2_Optimizer.app`. |
| Linux x64 | [Executable archive](https://github.com/JoliverApps/RD_Dq_Optimizer/releases/latest/download/RD_EQD2_Optimizer-Linux-x86_64.tar.gz) | Extract the archive and run `./RD_EQD2_Optimizer` in a graphical session. |

The Linux build targets Ubuntu 22.04 and compatible desktop distributions. The Windows executable is unsigned and the macOS downloads are unnotarized. The release includes [SHA-256 checksums](https://github.com/JoliverApps/RD_Dq_Optimizer/releases/latest/download/SHA256SUMS.txt) for all four packages.

### Python application

Install Python 3.11 or 3.12 with pip and Tcl/Tk. Clone or download the repository and run the following from its root.

```sh
python -m pip install .
rd-eqd2-optimizer
```

If you downloaded the standalone `rd_eqd2_optimizer.py` file, install the dependencies and run it directly.

```sh
python -m pip install numpy==2.2.6 matplotlib==3.10.3 scipy==1.17.0 sympy==1.14.0
python rd_eqd2_optimizer.py
```

## Desktop interface

The left column groups prescription sets and optimization settings. Drag the divider between the two boxes to adjust their heights, or the divider beside the workspace to adjust their widths. **New set**, **Edit set**, **Import…**, and **Export…** manage the selected set. All prescription-management commands are in the **Prescriptions** menu. **File** provides saving, printing, and result exports; **Help** includes parameter guidance.

The workspace has two main tabs. **Prescriptions** contains a subtab for each selected prescription set. **Results** contains **Optimization profile**, **EQD2 dispersion**, **Clinical evaluation**, and **Robustness**. Buttons within each group share the same dimensions. Keyboard shortcuts: **Ctrl+N** creates a set, **Ctrl+O** imports one, **Ctrl+R** runs optimization, **Ctrl+S** saves the selected plot, and **Ctrl+P** opens its print preview.

**Run optimization** is disabled in the button and menu when no set is selected, or when any selected set contains fewer than two checked prescriptions. **Analysis…** becomes available only after a successful run. Changing inputs clears the old results and disables analysis until another run succeeds.

## Run an optimization

1. Select **Prostate — 10 prescriptions**, create a set, or import a CSV.
2. In **Prescriptions**, check **Optimize** for at least two prescriptions per selected set.
3. Enter **α/β (Gy)** for each set. Use **Save α/β** to store the value for future sessions.
4. Set the q search interval and step, and choose **SSE**, **SD**, or **Range**.
5. Click **Run optimization**, then use the result tabs to inspect the profile, EQD2 dispersion, clinical evaluation, and robustness. **Analysis…** opens the full analysis in a separate window.

A successful run opens the **Results** tab. Each set is optimized independently. Switching tabs preserves the current prescription selections and α/β entries. The **Optimize** checkboxes control which saved rows take part in the current fit.

With all ten built-in prostate prescriptions, the default search and SSE criterion give Dq ≈ 3.4833 Gy, mean EQD2 ≈ 75.5906 Gy, and sample SD ≈ 2.9625 Gy. The [sample CSV](samples/prostate.csv) contains the same prescription doses and fraction counts.

### Parameters and search interval

Enter a positive finite α/β value for q ≥ 0. The q < 0 branch uses the negative of that entered value. The candidate parameter q represents the shoulder displacement, and its fitted value is reported as Dq. The sign of α/β follows q.

| Candidate q | RD response | α/β of the response |
| --- | --- | --- |
| q > 0 | Standard shoulder | Positive |
| q = 0 | Single-hit linear law | Undefined |
| q < 0 | Inverted shoulder | Negative |

The default interval is −50 to +50 Gy with a step of 0.001 Gy. The grid includes zero once when the interval spans it. At q = 0, BED and EQD2 equal the total physical dose.

Use a smaller step for a finer grid. If the application reports too many grid points, increase the step or narrow the interval.

### Minimization criterion

| Criterion | Quantity minimized |
| --- | --- |
| SSE | Sum of squared deviations of prescription EQD2 values from their mean |
| SD | Sample standard deviation of EQD2 |
| Range | Difference between the highest and lowest EQD2 |

**Near-optimal bounds (%)** defines a tolerance above the minimum. The bounds use sample SD for SSE and SD fits and EQD2 range for range fits. The default tolerance is 5%.

## Save and print

In the main window, **File → Save plot…** saves the selected result plot. **File → Print plot…** opens a PDF in the system viewer; use that viewer’s Print command to choose a printer and print settings. Plot actions are enabled when a valid plot is selected in **Results**.

The separate **Analysis…** window has its own **File** and **Export** menus. Use **Save plot…**, **Save tables…**, or **Print plot…** there. Plot saving and printing follow the selected dispersion or clinical plot. For prescription-value, summary, and robustness tables, use **Save tables…** or the table export commands. Other export formats remain available through **Export**.

## Read the results

The results table reports Dq, the signed α/β for the fitted branch, the minimized objective, mean EQD2, sample SD, relative standard deviation (RSD), trend measure η, EQD2 range, and profile bounds.

A minimum marked **BOUNDARY** lies at a search endpoint. **Outside search** indicates that at least one profile-bound crossing lies beyond the selected interval.

The main window shows four result tabs: **Optimization profile**, **EQD2 dispersion**, **Clinical evaluation**, and **Robustness**. The **Optimized set** dropdown selects the set used by the three analysis tabs. The optimization profile displays all fitted sets.

**Analysis…** opens a separate window with five tabs for the selected optimized set.

| Tab | Contents |
| --- | --- |
| **EQD2 dispersion** | RD and optional LQ EQD2 against dose per fraction, with a reference line and summary statistics |
| **Clinical evaluation** | RD EQD2 against observed local tumor control, with summary statistics and a bounded beta fit when available |
| **Prescription values** | Per-prescription RD and LQ BED and EQD2, observed local control, and dose differences |
| **Summary & trends** | RD parameters, descriptive statistics, trend coefficients, and profile bounds |
| **Robustness** | Leave-one-out recalibration and α/β sensitivity |

At Dq = 0, the sensitization rate s is displayed as **N/A**. At α/β = −2 Gy, the LQ EQD2 reference denominator is zero and the comparison is displayed as **N/A**. With other negative α/β values, LQ extrapolations can produce negative equivalent doses.

### Clinical evaluation

1. In **New prescriptions set** or **Edit prescriptions set**, enter **Local tumor control (%)** on each row with an observed outcome. A blank cell represents missing data. A value of 0 represents observed 0% local control. For CSV import, use `local_control_pct`.
2. Run the optimization with the desired prescriptions. The Dq fit uses the total doses and fraction counts of all checked rows.
3. Select **Clinical evaluation** in the results area, or open **Analysis… → Clinical evaluation**. Each checked row with an outcome contributes one equally weighted point using its RD EQD2 at the fitted Dq.

The plot and summary show means, observed minimum–maximum ranges, a linear trend, and R² where defined. The clinical tab separates the scrollable numerical analysis from the plot with a draggable divider. Basic analysis and beta-regression results appear together in the information pane. A shaded rectangle marks a compact majority subgroup when the [cluster rule](MATHEMATICS.md#clinical-evaluation) is met. The summary compares the enclosing areas on the RD and LQ scales for the same observations when the ratio is defined.

Use **Beta mean** to show the fitted mean curve and **95% beta interval** to overlay the central conditional outcome interval. The dose-dependent fit requires at least four observations with varying EQD2 and varying local control. At constant EQD2, three or more varying outcomes can support an intercept-only fit. Use **Export clinical evaluation plot…** to save the figure.

### Robustness analysis

Open the **Robustness** tab in the analysis window and click **Run robustness**. Leave-one-out analysis requires at least three prescriptions.

To assess α/β sensitivity, enter comma-separated positive reference values. Each reference value is used for q ≥ 0 and its negative is used for q < 0.

## Manage prescription sets

Use **New prescriptions set**, **Edit prescriptions set**, and **Delete prescriptions set** to manage saved data. Each prescription requires a name, a positive total dose in Gy, and a positive integer number of fractions.

The first prescription with 2 Gy per fraction supplies the automatic reference for the dispersion plot. In the prostate preset, this is 78 Gy in 39 fractions. In the set editor, select a row's **Reference** button to override the choice or **Automatic reference** to restore it. The analysis window fills **Reference EQD2 (Gy)** with the reference row's RD EQD2 at the fitted Dq, even if that row is unchecked for optimization. You can also enter a plot reference directly in the analysis window. All EQD2 calculations use 2 Gy as the reference fraction size.

Use **Export prescriptions set…** to save the checked prescriptions in the viewed set as a CSV with the current α/β and any entered outcomes. Keep the reference row checked to include its reference marker in the CSV. Use **Import prescriptions set…** to open a CSV in the editor and review it before saving.

| Column | Contents |
| --- | --- |
| `name` | Prescription name |
| `total_gy` | Total dose in Gy |
| `fractions` | Number of fractions |
| `prescriptions_set` | Set name |
| `alpha_beta` | α/β input for q ≥ 0, in Gy |
| `is_reference` | Optional 1 for the chosen reference row, 0 or blank otherwise; at most one row may be marked |
| `local_control_pct` | Optional observed local tumor control percentage from 0 to 100 |

The first three columns are required. A file holds one prescription set with unique prescription names. When the set name is omitted, the editor uses the filename. Start with the [prostate CSV example](samples/prostate.csv).

## Export data and plots

| Control | Output |
| --- | --- |
| **Export results + graph** | `grid_results.csv`, `optimal_schedules.csv`, and `range.png` |
| **Export image…** | The optimization profile in PDF, JPG, PNG, EPS, or SVG |
| **Export dispersion plot…** and **Export clinical evaluation plot…** | The corresponding analysis figures in the same five formats |
| **Export tables…** | Analysis tables in CSV, TSV, JSON, or LaTeX, with options to include robustness results and dispersion plots |

Use **Plot settings…** to style the optimization profile. Image export accepts a resolution in pixels per inch. PDF, EPS, and SVG preserve vector lines and text.

## Command-line use

Use the `rd-eqd2-optimizer` command after installing the Python project. For the standalone file, use `python rd_eqd2_optimizer.py` in its place.

```sh
rd-eqd2-optimizer --batch --dataset prostate --out results/prostate
rd-eqd2-optimizer --batch --csv samples/prostate.csv --ab 1.5 --dq-start 0 --dq-end 50 --dq-step 0.01 --out results/csv
rd-eqd2-optimizer --batch --set "Prostate — 10 prescriptions" --out results/saved
```

The built-in and saved-set commands use their respective α/β values unless `--ab` overrides them. A CSV uses its `alpha_beta` value when provided and defaults to 1.5 Gy otherwise. Choose `--method sse`, `--method sd`, or `--method range` to set the objective. Batch mode writes the optimization grid, prescription values, and profile image. Use the GUI for clinical evaluation and robustness exports. Run `rd-eqd2-optimizer --help` for all options.

## Saved data

The app stores its data under `%USERPROFILE%\.rd_dq_optimizer` on Windows and `~/.rd_dq_optimizer` on macOS and Linux. The Python application and desktop downloads share these paths for the same user account.

| File | Contents |
| --- | --- |
| `prescriptions.sqlite3` | Saved prescription sets, α/β values, reference choices, and local control |
| `plot_settings.json` | Saved optimization plot preferences |
| `application.log` | Output from windowed builds without a console |

Close the app before copying `prescriptions.sqlite3` as a backup. Use `--db` or the `RD_DQ_DB_PATH` environment variable to select a different database.

The original eight-prescription prostate set is updated in place if its prescription rows match the original preset. Saved α/β and reference choices carry forward. If you edited the rows under the original set name, the ten-prescription preset appears alongside your saved set.

## Build a desktop package

Run the build on the target platform with 64-bit Python 3.11 or 3.12, pip, venv, and Tcl/Tk. Enable a graphical session for the runtime checks.

```text
python tools/build.py Build Release
```

The build installs pinned dependencies in `.venv-build`, runs source and packaged-app checks, and writes the app and build manifest to `dist/Release`. On headless Linux, use `xvfb-run -a python tools/build.py Build Release`. On Windows, `Build.cmd Build Release` also works. Set `RD_PYTHON` to the full path of `python.exe` if the Windows launcher cannot find your interpreter.

Desktop builds on GitHub are **manual only**. Commits, pull requests, merges, and package-version changes do not start a build. Use the standalone Python file to evaluate changes before requesting a desktop build.

To build and publish desktop downloads:

1. For a new release, set a new version in `pyproject.toml` and commit it to `main`. Existing published versions are not overwritten.
2. Open **Actions → Build desktop apps → Run workflow**.
3. Select **main**, then click **Run workflow**.

The [workflow](.github/workflows/desktop-release.yml) builds and tests Windows x64, macOS Apple silicon, macOS Intel, and Linux x64. Publication runs only after all four platform checks pass. Packages and checksums are uploaded to a draft, and the release becomes public only after every upload succeeds. If a build or upload fails, the last published release remains available. A manual run on another branch produces build artifacts without publishing a release.

### Sign local Windows Debug builds

Install a Windows 10 or Windows 11 SDK with SignTool. In PowerShell, open the repository folder and run:

```powershell
py -3.12 tools\debug_signing.py setup
```

Read the certificate trust description and type `TRUST` to confirm. Windows may also ask you to confirm trusting the certificate. The setup creates a one-year RSA code-signing certificate and trusts its public certificate in the current user's Root and TrustedPublisher stores. The non-exportable private key remains in the current user's Personal certificate store; it is not written to the repository. Only the certificate thumbprint is saved in `%LOCALAPPDATA%\RD_EQD2_Optimizer\debug-signing.json`. Repeating setup reuses an existing unexpired certificate.

Build and run the local Debug copy:

```powershell
.\Build.cmd Build Debug
.\dist\Debug\RD_EQD2_Optimizer\RD_EQD2_Optimizer.exe
```

Configured Debug builds sign the executable, verify its Authenticode signature, then run the executable's checks. Release builds do not use the development certificate. The certificate provides trust for local testing under the configured Windows account, not other computers or public distribution. Smart App Control requires a certificate issued by a trusted provider and can still block this locally signed executable.

To stop signing Debug builds, delete the local `debug-signing.json` settings file. To remove the development certificate's trust, run `certmgr.msc` and remove the certificate named **RD EQD2 Optimizer Local Debug** from **Trusted Root Certification Authorities → Certificates**, **Trusted Publishers → Certificates**, and **Personal → Certificates**, matching its thumbprint to the value printed during setup.

## Troubleshooting

| Problem | Action |
| --- | --- |
| Missing NumPy, Matplotlib, SciPy, or SymPy | Run `python -m pip install -r requirements.txt` with the interpreter used to launch the application. |
| Tkinter is unavailable | Install Tcl/Tk support for the selected Python interpreter. |
| Build exits with code −1610612730 (0xA0000006) | Python Install Manager could not launch the requested runtime with automatic installation disabled. Run `py install 3.12`, then build again. |
| Too many grid points | Increase q step or narrow the interval. |
| A Windows executable reports a startup error | Check `application.log` in the data folder. |

RD EQD2 Optimizer is distributed under the [MIT License](LICENSE).
