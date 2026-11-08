"""The instance generator must give identical CSVs in every process, whatever the hash seed."""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

_SCRIPT = (
    "import sys; from pathlib import Path; "
    "from experiments.generate_instances import SizeSpec, generate_instance; "
    "d = generate_instance(SizeSpec('p60_c4', 4, 60, 20), seed=7, out_dir=Path(sys.argv[1])); "
    "print(d)"
)


def _digest(directory: Path) -> str:
    h = hashlib.sha256()
    for name in ("people.csv", "centers.csv", "travel.csv"):
        h.update((directory / name).read_bytes())
    return h.hexdigest()


def test_instances_do_not_depend_on_the_hash_seed(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    digests = []
    for hash_seed in ("1", "2"):
        out = tmp_path / hash_seed
        env = {**os.environ, "PYTHONHASHSEED": hash_seed}
        res = subprocess.run(
            [sys.executable, "-c", _SCRIPT, str(out)],
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
            check=True,
            timeout=120,
        )
        digests.append(_digest(Path(res.stdout.strip())))
    assert digests[0] == digests[1]
