# RD EQD2 Optimizer

This Visual Studio solution builds the supplied RD EQD2 Optimizer into a Windows x64 executable. The application in `src/rd_eqd2_optimizer.py` is identical to the accompanying standalone `rd_eqd2_optimizer.py`. It includes the prescription selection controls, main-page α/β entry, CSV set export/import, RD calculations, SQLite database, analysis window, robustness calculations, plot settings, and export tools.

The **Prostate — 8 prescriptions** set at **α/β = 1.5 Gy** is the only built-in default. Other sets can be created or imported. Previously saved sets remain available; initialization does not delete or overwrite them. Deleted defaults are not recreated on subsequent launches.

## Signed RD calibration

Each prescriptions set retains its entered **finite, nonzero α/β** throughout optimization. Only the candidate shoulder displacement `q` is optimized. Negative α/β values describe the effective inverse-shoulder branch and are supported in the main page, saved sets, CSV imports/exports, analysis, and robustness calculations. Zero α/β and nonfinite inputs are rejected.

With `a = α/β` and `q = r/s`, the root is selected to retain `s > 0`:

| Fixed ratio | Candidate domain | Sensitization rate |
| --- | --- | --- |
| `a > 0` | `q > 0` | `s = (sqrt(1 + 2a/q) − 1)/a` |
| `a < 0` | `q < 0` | `s = (−1 − sqrt(1 + 2a/q))/a` |
| Either sign | `q = 0` | Linear-survival boundary: `r = 0`, `BED = D`, `EQD2 = D`; finite `s` is undetermined. |

The default search interval is −50 to +50 Gy. For each set, the optimizer uses the part compatible with its fixed ratio's sign, including zero when the selected interval spans it. Thus, positive and negative sets can be optimized independently using the same search controls. An interval containing no compatible candidate produces an explanatory error. The application never changes the sign or magnitude of the entered ratio to accommodate a candidate. Both signs of `q` are supported across the model; a single fixed ratio selects one branch.

At zero, β vanishes and the strict α/β ratio is undefined. The entered finite ratio labels the branch approaching this boundary; it is not assigned as the ratio of the exact zero-β law. EQD2 and BED have continuous limits there, which are evaluated directly without division by zero. If an optimum reaches zero, analysis reports `r = 0` and `s = N/A`.

Numerical evaluation uses `expm1` and a low-argument series to avoid cancellation. Existing positive-only SQLite databases are migrated transactionally, retaining set IDs, prescription IDs, and metadata. The LQ EQD2 comparison is reported as unavailable at α/β = −2 Gy because its reference denominator vanishes; RD calculations remain available. For other negative ratios, LQ comparisons are algebraic low-dose extrapolations and may produce negative values. Robustness sensitivity values must retain the fitted branch's sign.

Example with a negative fixed ratio:

```bat
python src/rd_eqd2_optimizer.py --batch --ab=-5 --dq-start=-50 --dq-end=0 --out inverse-shoulder-results
```

## Select prescriptions and edit α/β

1. Check the **prescriptions sets** to optimize in the left panel.
2. Choose the set to view, then click its **Optimize** checkboxes to include or exclude individual prescriptions. **Select all prescriptions** and **Clear prescriptions** apply to the viewed set. At least two checked prescriptions are required per enabled set.
3. Edit **α/β (Gy)** above the prescription list. The current entry is used immediately when you run optimization or export the set. Each set keeps its own entry and prescription choices as you switch between sets.
4. Click **Save α/β** to retain that value for future sessions. Prescription checkboxes are session choices and do not remove saved prescriptions.
5. Click **Run optimization**. Plots, analysis, and result exports use exactly the checked prescriptions and entered α/β for each enabled set.

Use **Export prescriptions set…** to export the viewed set's checked prescriptions and current α/β as an importable CSV. This button works before optimization. The CSV has `name,total_gy,fractions,prescriptions_set,alpha_beta` columns. **Import prescriptions set…** accepts this format and older CSVs with a `tumor` column.

To run only the standalone file, install the dependencies from `requirements.txt`, then run `python rd_eqd2_optimizer.py`.

## Build in Visual Studio

1. Install **Visual Studio 2022** with the **Desktop development with C++** workload and the **MSVC v143** tools. This supplies Visual Studio's Makefile project support.
2. Install **64-bit Python 3.12** from [python.org](https://www.python.org/downloads/windows/), including **pip**, **venv**, **Tcl/Tk**, and the **Python launcher**. Python 3.11 x64 is also supported. Python is needed on the build computer only.
3. Extract this entire ZIP into a writable folder and open **RD_EQD2_Optimizer.sln**.
4. Select **Release | x64**, then choose **Build > Build Solution** (`Ctrl+Shift+B`).
5. Run or distribute **dist\Release\RD_EQD2_Optimizer.exe**.

The first build downloads dependencies and can take several minutes. Later builds reuse the private build environment. An internet connection is required to install the dependencies initially. Wait for the final **Build and executable verification passed** message.

The Release output is one executable with Python, Tcl/Tk, NumPy, Matplotlib, and SymPy bundled inside it. End users can run it without installing Python, Visual Studio, or those libraries. First startup includes extracting its bundled runtime into the user's temporary folder.

## Build without the IDE

Open Command Prompt in this folder:

```bat
Build.cmd Build Release
```

Other supported commands:

```bat
Build.cmd Build Debug
Build.cmd Rebuild Release
Build.cmd Clean Release
Build.cmd Test Release
Build.cmd Run
```

`Test` verifies the source, calculations, exports, and GUI. `Run` opens the source application in the prepared Python environment. `Clean` removes generated output for the chosen configuration. These commands never delete the application's saved prescription database.

If Python is installed in a custom location, set its exact interpreter path before building or starting Visual Studio:

```bat
set "RD_PYTHON=C:\Path\To\Python312\python.exe"
Build.cmd Build Release
```

Alternatively, from a Visual Studio Developer Command Prompt:

```bat
msbuild RD_EQD2_Optimizer.sln /t:Build /p:Configuration=Release /p:Platform=x64
```

## Configurations and files

| Configuration | Executable | Distribution |
| --- | --- | --- |
| Release | `dist\Release\RD_EQD2_Optimizer.exe` | Copy the single executable. |
| Debug | `dist\Debug\RD_EQD2_Optimizer\RD_EQD2_Optimizer.exe` | Keep the entire containing folder, including `_internal`. |

Debug uses a console and a directory of runtime files for easier diagnosis. It preserves the application's batch output. Use `Ctrl+F5` to start a built configuration from Visual Studio. Python source breakpoints require a Python debugger; the solution's default launch runs the packaged program.

This is a Visual Studio Makefile project. Its build commands call PyInstaller to package the standalone Python application. The scientific application has not been translated to C++ or C#. Python files and build files appear in Solution Explorer and remain editable.

## Verification

Every build runs integration checks **before packaging** and **against the resulting executable**. Checks cover the original prostate calibration, SymPy execution, all three minimization methods, SQLite operations, CSV import, prostate-only default initialization, saved-set preservation, independent user-created sets, prescription subsets, per-set α/β drafts, CSV set round trips, analysis and robustness tables, image formats, preference persistence, and GUI/dialog creation. Test databases are temporary and isolated from saved user data.

Verification reports are written to:

```text
.build\Release\source-test.json
.build\Release\executable-test.json
```

A build fails if either check fails. `dist\Release\build-info.json` records the executable checksum, application source checksum, installed dependencies, and check results. Read `VALIDATION.md` for checks completed when this project was prepared.

## Saved data

Existing application storage is retained:

```text
%USERPROFILE%\.rd_dq_optimizer\prescriptions.sqlite3
%USERPROFILE%\.rd_dq_optimizer\plot_settings.json
```

The executable and the standalone Python application use the same default database. Existing prescription sets remain available. The `RD_DQ_DB_PATH` environment variable and the application's `--db` argument can select a different database, as in the application. Export destinations are selected in the application.

If a windowed Release run encounters a Python startup error, it shows an error dialog. Its standard output and error stream are recorded in `%USERPROFILE%\.rd_dq_optimizer\application.log`. The Debug build shows output directly in its console.

## Batch example

Use `--set "Saved set name"` to optimize a saved prescriptions set.


Use the Debug executable to see output in Command Prompt:

```bat
"dist\Debug\RD_EQD2_Optimizer\RD_EQD2_Optimizer.exe" --batch --dataset prostate --out results
```

The Release executable supports the same arguments and exports, with printed output going to its application log. From another program or script, wait for the process to exit before consuming its files.

`samples/prostate.csv` contains the eight prescriptions from the uploaded source. The application remains the authority for its built-in prostate set.

## Dependencies and maintenance

`requirements.txt` pins the application libraries and `requirements-build.txt` pins PyInstaller and its hook package. Transitive dependencies are recorded after each successful build in `build-info.json`. Dependencies retain their respective licenses; no third-party library code is included in this source archive.

After editing the application, select **Build > Rebuild Solution**. To recreate the build environment after changing Python installations, remove the project's `.venv-build` directory and build again. This directory contains build dependencies, not user prescriptions.

The standalone application and `src/rd_eqd2_optimizer.py` contain the same code. The storage API uses `list_sets`, `get_set`, `find_set`, `save_set`, and `delete_set`. Existing database files retain their format. CSV imports accept both current column names and older `tumor` headers.

`pyproject.toml` defines the Python formatting and lint rules. With Ruff installed, run `ruff check src tools` and `ruff format --check src tools` from the project folder.

Reference documentation: [Visual Studio Makefile projects](https://learn.microsoft.com/en-us/cpp/build/reference/creating-a-makefile-project?view=msvc-170), [PyInstaller usage](https://pyinstaller.org/en/stable/usage.html), and [PyInstaller specifications](https://pyinstaller.org/en/stable/spec-files.html).
