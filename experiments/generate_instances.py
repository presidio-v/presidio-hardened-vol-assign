"""Deterministic instance generator for the MDPI experiment plan.

Produces three problem sizes — small (5/150/50), medium (8/225/75), large
(10/300/100) — as people.csv / centers.csv / travel.csv triples under
``experiments/instances/<size>/``. Parameter ranges follow ATRes baseline
intuitions (vulnerability spread across age and disability/injury;
infrastructure damage skewed slightly toward higher values; travel
durations linked to a synthetic distance grid).

Two design choices worth documenting:

1. **Seeded numpy generator.** A single seed (default 42) drives every
   random draw across all three sizes, so the instances are bit-exact
   reproducible and the small/medium/large triples have the same
   demographic flavour modulo size.

2. **Travel duration tied to a distance proxy.** For each (person,
   center) pair we draw a distance in [0, 100] km and convert to minutes
   at 60 km/h, capped at 180 min. Road-condition and hazard categoricals
   are drawn independently with skew toward middling values so that the
   FIS rule base sees representative coverage across all linguistic
   levels.

Usage:
    .venv/bin/python -m experiments.generate_instances [--seed 42] [--out experiments/instances]

Or programmatically:
    from experiments.generate_instances import generate_instance, SIZES
    generate_instance(SIZES["small"], seed=42, out_dir=Path("..."))
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SizeSpec:
    """One problem-size specification."""

    name: str
    n_centers: int
    n_people: int
    n_dir: int


SIZES: dict[str, SizeSpec] = {
    "small": SizeSpec("small", n_centers=5, n_people=150, n_dir=50),
    "medium": SizeSpec("medium", n_centers=8, n_people=225, n_dir=75),
    "large": SizeSpec("large", n_centers=10, n_people=300, n_dir=100),
}


_DISABILITY = ("none", "minor", "severe")
_DISABILITY_P = (0.70, 0.20, 0.10)

_INJURY = ("none", "minor", "moderate", "serious", "life_threatening")
_INJURY_P = (0.40, 0.25, 0.20, 0.10, 0.05)

_LIVING = ("with_support", "alone")
_LIVING_P = (0.65, 0.35)

_RCS = ("clear", "partially_blocked", "blocked")
_RCS_P = (0.45, 0.40, 0.15)

_PHS = ("none", "minor", "moderate", "significant", "extreme")
_PHS_P = (0.30, 0.30, 0.25, 0.10, 0.05)


def _people_frame(spec: SizeSpec, rng: np.random.Generator) -> pd.DataFrame:
    n = spec.n_people
    rows = {
        "person_id": [f"P{i:04d}" for i in range(n)],
        "age": np.round(rng.uniform(5, 90, size=n), 1),
        "disability_status": rng.choice(_DISABILITY, size=n, p=_DISABILITY_P),
        "injury_level": rng.choice(_INJURY, size=n, p=_INJURY_P),
        "living_status": rng.choice(_LIVING, size=n, p=_LIVING_P),
        "idl": np.round(rng.beta(2.5, 2.0, size=n) * 100, 2),  # slight skew high
        "rtr": np.round(rng.uniform(1.0, 48.0, size=n), 2),
    }
    return pd.DataFrame(rows)


def _centers_frame(spec: SizeSpec, rng: np.random.Generator) -> pd.DataFrame:
    n = spec.n_centers
    rows = {
        "center_id": [f"C{i:02d}" for i in range(n)],
        "cor": np.round(rng.uniform(20, 90, size=n), 2),
        "rdr": np.round(rng.uniform(10, 80, size=n), 2),
    }
    return pd.DataFrame(rows)


def _travel_frame(
    people: pd.DataFrame,
    centers: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    pids = people["person_id"].tolist()
    cids = centers["center_id"].tolist()
    n_pairs = len(pids) * len(cids)

    # Synthetic distance proxy: km drawn from a heavy-tailed Beta(1.5, 4)·100
    distance_km = rng.beta(1.5, 4.0, size=n_pairs) * 100.0
    # 60 km/h average → minutes; clip into the FIS universe [0, 180]
    td = np.clip(distance_km, a_min=2.0, a_max=180.0)

    rcs = rng.choice(_RCS, size=n_pairs, p=_RCS_P)
    phs = rng.choice(_PHS, size=n_pairs, p=_PHS_P)

    rows = {
        "person_id": np.repeat(pids, len(cids)),
        "center_id": np.tile(cids, len(pids)),
        "td": np.round(td, 2),
        "rcs": rcs,
        "phs": phs,
    }
    return pd.DataFrame(rows)


def generate_instance(spec: SizeSpec, seed: int, out_dir: Path) -> Path:
    """Write three CSVs for `spec` into `out_dir/<spec.name>/`. Returns the dir."""
    target = out_dir / spec.name
    target.mkdir(parents=True, exist_ok=True)

    # Independent generator per size so adding sizes later doesn't perturb
    # earlier instances. Seed = (base_seed, hash of size name).
    salt = abs(hash(spec.name)) % (2**32 - 1)
    rng = np.random.default_rng(seed=(seed, salt))

    people = _people_frame(spec, rng)
    centers = _centers_frame(spec, rng)
    travel = _travel_frame(people, centers, rng)

    people.to_csv(target / "people.csv", index=False)
    centers.to_csv(target / "centers.csv", index=False)
    travel.to_csv(target / "travel.csv", index=False)
    (target / "INSTANCE.md").write_text(
        f"# Instance: {spec.name}\n\n"
        f"- n_centers: {spec.n_centers}\n"
        f"- n_people: {spec.n_people}\n"
        f"- n_dir:    {spec.n_dir}\n"
        f"- seed:     {seed} (salted with hash('{spec.name}'))\n\n"
        f"Generated by `experiments/generate_instances.py`. "
        f"Reproducible: re-run with the same `--seed` to recover identical CSVs.\n"
    )
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42, help="Base RNG seed (default 42).")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).parent / "instances",
        help="Output directory (default experiments/instances).",
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        choices=sorted(SIZES.keys()),
        default=sorted(SIZES.keys()),
        help="Which sizes to generate (default: all).",
    )
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    for size_name in args.sizes:
        path = generate_instance(SIZES[size_name], seed=args.seed, out_dir=args.out)
        print(f"  {size_name:6s} → {path}")


if __name__ == "__main__":
    main()
