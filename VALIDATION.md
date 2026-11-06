# Validation

Code cleanup completed on 30 October 2026. The standalone `rd_eqd2_optimizer.py` and the application in this project contain identical code.

Application SHA-256:

```text
029739b78bc7f77069c44ef235f59e024f947dbeaf7413b5ab72e7201d31e8ed
```

## Cleanup

- Standard Python formatting, sorted imports, and consistent prescriptions-set names.
- Removed unused text parsers, compatibility wrappers, unused plot variables, and the obsolete hidden command-line alias.
- Shared alpha/beta, prescription, and image-resolution validation.
- Shared profile statistics and dispersion evaluation, replacing duplicate optimization code.
- Shared build-environment setup and streamed file hashing.
- Added `pyproject.toml` for formatting and lint rules.

The application remains a single Python file. Prostate remains the only built-in default. Saved database files and older CSV header formats remain supported.

## Completed checks

| Check | Result |
| --- | --- |
| Ruff lint and formatting checks | Passed for the standalone file and project Python files |
| Python 3.9 syntax compatibility | Passed |
| Standalone/project source equality and project file paths | Passed |
| Numerical comparison with the pre-cleanup application | Exact equality across 21 cases covering all three objectives, alpha/beta changes, prescription subsets, identical fraction doses, and boundary minima |
| Analysis and robustness comparison | Exact equality |
| Optimization CSV exports | Byte-identical |
| Profile and dispersion PNG plots | Pixel-identical |
| Saved database created by the previous code | Read correctly by the cleaned storage API |
| Full runtime and GUI integration checks | Passed on Linux with a virtual display |
| Portable build helpers | Environment isolation and file checksums verified |
| Windows executable build | Not run in this Linux environment; the Windows build runs source and executable checks automatically |

Runtime checks cover prostate-only initialization, saved-data preservation, checkbox selection, main-page alpha/beta editing and saving, switching sets, CSV set export/import, detailed analysis, robustness, plot settings, and image exports.

The full prostate set at alpha/beta = 1.5 Gy retains Dq = 3.543859355746318 Gy, mean EQD2 = 75.81790297672867 Gy, and RSD = 2.8365977004593472%.
