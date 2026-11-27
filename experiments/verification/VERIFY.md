# Verifying the Paper A correction against the released code

Article: Rabiei, P.; Arias-Aranda, D.; Stantchev, V. *Appl. Sci.* 2026, 16, 7581.

The Academic Editor asks that the recomputed results in our correction be independently verified against the released v0.5.1 code. This file describes how to do that on your own machine, with nothing but the public repository. The workflow `.github/workflows/verify-correction.yml` runs the same steps on a GitHub-hosted machine; its runs and logs are public under the repository's Actions tab.

What the run does:
- It reruns the original (published) objective with v0.5.1 and checks it reproduces the archived May 2026 runs.
- It reruns every experiment of the article with the corrected objective and checks that it reproduces the results released with v0.5.1, value by value.

Only CPU and wall times are not compared, since they depend on the machine. Everything else must be identical.

## Files in this folder

| File | What it is |
|---|---|
| `VERIFY.md` | these instructions |
| `verify_correction.py` | the comparison script (Python standard library only) |
| `paperA-v0.2.0-archived-manifests.zip` | the archived run files of the published article (May 2026), the reference for step 5 |
| `archived-manifests.sha256` | SHA-256 checksums of the seven files inside the zip |

## Time and machine

- **Full check:** about 6 CPU-hours, as nine processes in parallel. With 8–10 cores it takes 2.5–3 hours, with 4 cores about 6–7 hours. It needs about 1 GB of disk.
- **Quick check (step 4b):** the main 540-run comparison only, about 45 minutes on any recent laptop. It is enough if the full check is not practical, but the full check is better.

Keep the machine awake and plugged in for the whole run. On a Mac, use the `caffeinate` command shown below. On Windows, set the power plan so the computer does not sleep.

Operating system: macOS or Linux work directly. On Windows, use WSL (Ubuntu): open PowerShell as administrator, run `wsl --install`, restart, open "Ubuntu", and follow the Linux steps inside it.

## 1. Install the tools

You need `git` and `uv` (uv installs the right Python by itself).

macOS / Linux / WSL:

```sh
git --version          # if missing: macOS "xcode-select --install"; Ubuntu/WSL "sudo apt install git unzip"
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Close and reopen the terminal, then check that `uv --version` prints a version.

## 2. Get the released code

```sh
mkdir -p ~/pva-verify && cd ~/pva-verify
git clone --branch v0.5.1 https://github.com/presidio-v/presidio-hardened-vol-assign.git code
cd code
git rev-parse 'v0.5.1^{commit}'
```

The last command must print `344b0a37c6603826c3b9e548314fda0e1a73df1e`. If it prints something else, stop: this is not the released code.

(The same code is archived on Zenodo: https://doi.org/10.5281/zenodo.24061353.)

## 3. Install the environment and unpack the reference files

Still in `~/pva-verify/code`:

```sh
uv sync --frozen --python 3.12
.venv/bin/python -c "import numpy, scipy, pymoo; print(numpy.__version__, scipy.__version__, pymoo.__version__)"
```

The second command should print `2.4.4 1.17.1 0.6.2`.

Get this folder (`experiments/verification/` on the `main` branch) next to `code/`:

```sh
cd ~/pva-verify
git clone --depth 1 https://github.com/presidio-v/presidio-hardened-vol-assign.git kit
cp kit/experiments/verification/* .
```

Then unpack the archived runs:

```sh
cd ~/pva-verify
mkdir archive && cd archive
unzip ../paperA-v0.2.0-archived-manifests.zip
shasum -a 256 -c ../archived-manifests.sha256     # on Linux/WSL: sha256sum -c ../archived-manifests.sha256
```

All seven lines must end in `OK`.

## 4a. Full run (recommended)

```sh
cd ~/pva-verify/code
nohup sh experiments/run_correction.sh ~/pva-verify/archive > ~/pva-verify/run.log 2>&1 &
```

On a Mac, use this line instead, which also keeps the machine from sleeping until the run ends:

```sh
nohup caffeinate -i sh experiments/run_correction.sh ~/pva-verify/archive > ~/pva-verify/run.log 2>&1 &
```

You can close the terminal; the run continues.

The run writes progress to `experiments/results/correction/logs/`. To see how far it is:

```sh
tail -n 1 ~/pva-verify/code/experiments/results/correction/logs/*.log
```

It is finished when nine logs end in a line starting with `Done:` and `~/pva-verify/run.log` has stopped growing. The longest one is `h3b.log` (100 samples). Then go to step 5.

## 4b. Quick run (only if the full run is not practical)

```sh
cd ~/pva-verify/code
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
R=experiments/results/correction
.venv/bin/python -m experiments.run_h1_h2_h4 --out $R/repro_directed/h1_h2_h4 > ~/pva-verify/repro.log 2>&1 &
.venv/bin/python -m experiments.run_h1_h2_h4 --ulpp-objective left_out --out $R/left_out/h1_h2_h4 > ~/pva-verify/left_out.log 2>&1 &
wait
```

Both logs end with `Done: 540 runs`. Then go to step 5 and add `--quick` to the command.

## 5. Compare

```sh
cd ~/pva-verify
code/.venv/bin/python -I verify_correction.py --clone code --archive archive          # add --quick after 4b
```

The script prints a report and saves it as `code/verification-report.txt`. Expected result: every line says `IDENTICAL`, and the last line is `RESULT: ALL VALUES REPRODUCED`.

If any line says `differ` or `MISSING`, do not try to fix it. Report it as it is, with the logs; a difference is a result too.

## 6. Report

Send the report `code/verification-report.txt` and the logs (`~/pva-verify/run.log`, or `repro.log` and `left_out.log` after the quick run) to the corresponding author, with the machine, the operating system and the date of the run.

## What the script compares

1. `repro_directed/h1_h2_h4/manifest.csv`, the original objective rerun with v0.5.1, against the archived May 2026 manifest. This shows the released code still produces the published results.
2. All 23 result files of the correction that ship with tag v0.5.1 (run manifests, H1 analyses, instance-aware hypervolume, the tables), against the files your run just wrote. The run overwrites them inside your clone; the script reads the originals from the tag with `git show`.

It compares every value except CPU and wall-clock times. For the two tables built from the corrected runs (`corrected_table3.csv`, `corrected_pvalues.csv`), the CPU column is skipped because it holds the CPU times measured on your machine.

The script uses only the Python standard library and has no network access. You can read it in a couple of minutes.
