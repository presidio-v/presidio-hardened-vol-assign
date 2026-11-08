"""Copy the published CPU times into the corrected H1/H2/H4 manifest.

The corrected ULPP objective changes CPU time by under 3 % per cell on one machine, so the
Paper A correction keeps the published CPU measurements. This writes
manifest_publishedcpu.csv next to the corrected manifest: corrected HV/NNS/SM/MID, published
cpu_time_sec, and the locally measured CPU in cpu_time_sec_measured.

Usage: python -m experiments.correction_cpu_merge <published_manifest> <corrected_manifest>
"""

import sys
from pathlib import Path

import pandas as pd


def main() -> None:
    published, corrected = Path(sys.argv[1]), Path(sys.argv[2])
    p = pd.read_csv(published).set_index("run_id")
    c = pd.read_csv(corrected).set_index("run_id")
    if set(p.index) != set(c.index):
        raise SystemExit("run_id sets differ between the two manifests")
    c["cpu_time_sec_measured"] = c["cpu_time_sec"]
    c["cpu_time_sec"] = p["cpu_time_sec"]
    out = corrected.with_name("manifest_publishedcpu.csv")
    c.reset_index().to_csv(out, index=False)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
