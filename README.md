# RD EQD2 Optimizer

RD EQD2 Optimizer estimates the Resilience–Depletion (RD) shoulder descriptor Dq from a set of radiotherapy prescriptions by minimizing the dispersion of their equivalent doses in 2 Gy fractions (EQD2). It evaluates standard shoulders, the single-hit linear law, and inverted shoulders across the selected q interval.

The application provides prescription management, BED and EQD2 calculations, optimization profiles, dose–dispersion plots, descriptive statistics, robustness analysis, and data and image exports. A prostate set containing ten prescriptions at α/β = 1.5 Gy is available on first launch.

## Installation

### Desktop downloads

Download the package for your operating system from the [latest GitHub release](https://github.com/JoliverApps/RD_Dq_Optimizer/releases/latest). Each package includes the Python runtime and dependencies.

| Platform | Download | Launch |
| --- | --- | --- |
| Windows x64 | [Portable executable](https://github.com/JoliverApps/RD_Dq_Optimizer/releases/latest/download/RD_EQD2_Optimizer-Windows-x86_64.exe) | Open the downloaded `.exe`. |
| macOS Apple silicon | [App archive](https://github.com/JoliverApps/RD_Dq_Optimizer/releases/latest/download/RD_EQD2_Optimizer-macOS-arm64.zip) | Extract the archive and open `RD_EQD2_Optimizer.app`. |
| macOS Intel | [App archive](https://github.com/JoliverApps/RD_Dq_Optimizer/releases/latest/download/RD_EQD2_Optimizer-macOS-x86_64.zip) | Extract the archive and open `RD_EQD2_Optimizer.app`. |
| Linux x64 | [Executable archive](https://github.com/JoliverApps/RD_Dq_Optimizer/releases/latest/download/RD_EQD2_Optimizer-Linux-x86_64.tar.gz) | Extract the archive and run `./RD_EQD2_Optimizer` in a graphical session. |

The Linux build targets Ubuntu 22.04 and compatible desktop distributions. The public Windows executable has no publisher signature, and the macOS app has no Developer ID notarization. Local security policy can restrict either download.

### Python application

Install Python 3.11 or 3.12 with pip and Tcl/Tk. Clone or download the repository, open a terminal in its folder, and run:

```text
python -m pip install .
rd-eqd2-optimizer
```

If you downloaded the standalone `rd_eqd2_optimizer.py` file without the repository, install its dependencies and launch it with:

```text
python -m pip install numpy==2.2.6 matplotlib==3.10.3 scipy==1.17.0 sympy==1.14.0
python rd_eqd2_optimizer.py
```

## Run an optimization

1. Select the prescriptions sets to analyze.
2. Open a selected set's tab in **Prescriptions in the viewed set**. Check the **Optimize** boxes for the prescriptions to include. Each selected set needs at least two prescriptions.
3. Enter **α/β (Gy)** for each set. Click **Save α/β** to store the value for future sessions.
4. Set **q start**, **q end**, and **q step**, then choose a minimization criterion.
5. Click **Run optimization**.

Each selected set has its own tab, prescription choices, and α/β entry. Switching tabs preserves these choices. Each set is optimized independently. Prescription checkboxes control the current analysis; they do not delete saved prescriptions.

### Parameters and search interval

Enter a positive finite α/β value for q ≥ 0. The q < 0 branch uses the negative of that entered value. The candidate parameter q represents the shoulder displacement, and its fitted value is reported as Dq. The sign of α/β follows q.

| Candidate q | RD response | α/β of the response |
| --- | --- | --- |
| q > 0 | Standard shoulder | Positive |
| q = 0 | Single-hit linear law | Undefined; EQD2 = D |
| q < 0 | Inverted shoulder | Negative |

The default interval is −50 to +50 Gy with a step of 0.001 Gy. The optimizer evaluates the full selected interval. Zero is included once when the interval spans zero; at zero, BED and EQD2 equal the physical dose.

Use a smaller step for a finer grid. If the application reports too many grid points, increase the step or narrow the interval.

See [Mathematical model and calculations](MATHEMATICS.md) for the signed reconstruction, BED and EQD2 equations, numerical evaluation, and optimization definitions.

### Minimization criterion

| Criterion | Quantity minimized |
| --- | --- |
| SSE | Sum of squared deviations of prescription EQD2 values from their mean |
| SD | Sample standard deviation of EQD2 |
| Range | Difference between the highest and lowest EQD2 |

**Near-optimal bounds (%)** defines a tolerance above the minimum. SSE and SD optimization use the sample SD for these bounds; range optimization uses the EQD2 range. The default tolerance is 5%.

## Read the results

The results table reports Dq, the selected α/β reference, the minimized objective, mean EQD2, sample SD, relative standard deviation (RSD), trend measure η, EQD2 range, and profile bounds.

A minimum marked **BOUNDARY** lies at an endpoint of the search interval. A wider interval may help locate an interior minimum. **Outside search** means that the selected interval does not contain both profile-bound crossings.

Click **Analysis…** for:

- A scatter plot of EQD2 against dose per fraction.
- Per-prescription RD BED and EQD2, with LQ comparisons.
- Summary statistics, fitted trends, and RD parameters r and s.
- Leave-one-out and α/β sensitivity analyses.

At Dq = 0, the linear response gives BED = EQD2 = D. The sensitization rate s is displayed as **N/A**.

For α/β = −2 Gy, LQ EQD2 is unavailable because its reference denominator is zero. RD results remain available. For other negative α/β values, LQ extrapolations can produce negative equivalent doses.

### Clinical evaluation

Enter an observed **Local tumor control (%)** for each prescription with available outcome data. Leave unavailable outcomes blank. After optimization, open **Analysis… → Clinical evaluation**. Each selected prescription with an outcome contributes one equally weighted point at its fitted RD EQD2. The summary and plot show means, observed minimum–maximum ranges, and an unweighted linear trend with R² when defined. The enclosing area on the RD scale is also shown relative to the LQ area when the comparison is defined.

The plot shades a subgroup when it contains at least three observations and a strict majority of the points. Along one axis, each gap to its neighboring points must exceed the subgroup width. Along the other axis, the subgroup must span at most 75% of the full observed range. If several subgroups qualify, the app uses the largest and then the one with the smallest relative enclosing area.

The beta regression fits a logit mean against EQD2 and a common precision. Its optional 95% band shows the fitted conditional 2.5th and 97.5th percentiles. A dose-dependent fit requires at least four outcomes and an intercept-only fit at constant EQD2 requires at least three. When any outcome is 0% or 100%, the beta likelihood uses the transformation `(y × (N − 1) + 0.5) / N` on all proportions. The scatter and basic analysis use the original observations. [Beta regression](https://doi.org/10.1080/0266476042000214501) · [Endpoint transformation](https://doi.org/10.1037/1082-989X.11.1.54)

Use **Export clinical evaluation plot…** to save the figure.

### Robustness analysis

Open the **Robustness** tab in the analysis window and click **Run robustness**. Leave-one-out analysis requires at least three prescriptions.

To assess α/β sensitivity, enter comma-separated positive reference values. Each reference value is used for q ≥ 0 and its negative is used for q < 0.

## Manage prescriptions sets

Use **New prescriptions set**, **Edit prescriptions set**, and **Delete prescriptions set** to manage saved data. Each prescription requires a name, a positive total dose in Gy, and a positive integer number of fractions.

The first prescription with 2 Gy per fraction is used automatically as the reference. In the set editor, select a prescription's **Reference** button to override this choice, or select **Automatic reference** to restore it. The saved choice follows that row when its name, dose, or fraction count is edited. The main prescription list marks the reference, and the analysis window fills **Reference EQD2 (Gy)** with its RD EQD2 at the fitted Dq. The reference remains available when unchecked for optimization. Sets without a 2 Gy/fraction prescription or an explicit choice leave the reference blank; it can also be entered directly in the analysis window.

Use **Export prescriptions set…** to save the viewed set's checked prescriptions and current α/β as CSV. Use **Import prescriptions set…** to load a CSV file.

The CSV columns are:

| Column | Contents |
| --- | --- |
| `name` | Prescription name |
| `total_gy` | Total dose in Gy |
| `fractions` | Number of fractions |
| `prescriptions_set` | Set name |
| `alpha_beta` | α/β input for q ≥ 0, in Gy |
| `is_reference` | Optional 1 for the chosen reference row, 0 or blank otherwise; at most one row may be marked |
| `local_control_pct` | Optional observed local tumor control percentage from 0 to 100 |

The first three columns are required for import. A file must describe one prescriptions set. Prescription names must be unique within that set. The `samples/prostate.csv` file provides an example.

## Export data and plots

- **Export results + graph** saves the optimization grid, optimal prescription values, and an optimization plot.
- **Export image…** saves the optimization plot as PDF, JPG, PNG, EPS, or SVG, in that order, with a selected resolution.
- **Plot settings…** controls titles, labels, fonts, colors, dimensions, and vertical axis limits. The horizontal axis spans q start to q end.
- The analysis window exports tables as CSV, TSV, JSON, or LaTeX, and offers the same PDF, JPG, PNG, EPS, and SVG choices for dispersion plots. PDF, EPS, and SVG preserve vector lines and text; PPI controls raster resolution.

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

On other platforms, the data folder is `.rd_dq_optimizer` in the user's home directory. The Python application and packaged builds use the same default database. Back up the database file to retain saved sets.

The original eight-prescription prostate set is updated in place if its prescription rows match the original preset. Saved α/β and reference choices carry forward. If you edited the rows under the original set name, the ten-prescription preset appears alongside your saved set.

A different database can be selected with `--db` or the `RD_DQ_DB_PATH` environment variable.

## Build a desktop package

Run the build on the target platform with 64-bit Python 3.11 or 3.12, pip, venv, and Tcl/Tk. Enable a graphical session for the runtime checks.

```text
python tools/build.py Build Release
```

The build installs pinned dependencies in `.venv-build`, verifies the Python source, creates the native app with PyInstaller, and runs the same checks against the packaged app. Release output is in `dist/Release`. On headless Linux, run the build under `xvfb-run -a`. On Windows, `Build.cmd Build Release` also works. Set `RD_PYTHON` to the full path of `python.exe` if the Windows launcher cannot find your interpreter.

GitHub Actions builds and tests the platform packages on pull requests. Updating the project version in `pyproject.toml` on `main` builds the packages and publishes a GitHub release with SHA-256 checksums once the platform tests pass.

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
