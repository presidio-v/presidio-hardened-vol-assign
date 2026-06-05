"""Generate plausible synthetic humanitarian datasets for the worked example/demo.

Deterministic (seeded). People and centres are placed on a 70 km x 70 km
affected-area grid and distances are Euclidean (clipped to [1, 100] km), so the
distances are internally consistent rather than independent noise. Writes
``people.csv`` + ``centers.csv`` under ``examples/<name>/``.

Run from the repo root:  python examples/generate_examples.py
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
_AREA_KM = 70.0  # side length of the affected area


def generate(name: str, n_people: int, n_centers: int, seed: int) -> tuple[int, int]:
    """Write people.csv + centers.csv for one instance; return (demand, capacity)."""
    rng = np.random.default_rng(seed)

    center_xy = rng.uniform(0, _AREA_KM, size=(n_centers, 2))
    people_xy = rng.uniform(0, _AREA_KM, size=(n_people, 2))
    center_ids = [f"C{j + 1}" for j in range(n_centers)]

    vulnerability = np.clip(rng.normal(5.0, 2.5, n_people), 0, 10)
    mobility = np.clip(rng.normal(5.5, 2.5, n_people), 0, 10)
    group_size = rng.choice([1, 1, 1, 2, 2, 3, 4, 5], size=n_people)

    # Euclidean person->centre distances (km), clipped into the valid range.
    deltas = people_xy[:, None, :] - center_xy[None, :, :]
    distance = np.clip(np.sqrt((deltas**2).sum(axis=-1)), 1.0, 100.0)

    demand = int(group_size.sum())
    base_cap = math.ceil(1.2 * demand / n_centers)
    capacity = base_cap + rng.integers(0, base_cap // 4 + 1, size=n_centers)
    service_level = np.clip(rng.normal(6.5, 2.0, n_centers), 0, 10)
    road = np.clip(rng.normal(6.0, 2.0, n_centers), 0, 10)

    out_dir = ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "centers.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["center_id", "capacity", "service_level", "road_accessibility"])
        for j, cid in enumerate(center_ids):
            w.writerow(
                [cid, int(capacity[j]), round(float(service_level[j]), 1), round(float(road[j]), 1)]
            )

    with (out_dir / "people.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            ["person_id", "vulnerability", "mobility", "group_size"]
            + [f"distance_center_{cid}" for cid in center_ids]
        )
        for i in range(n_people):
            w.writerow(
                [
                    f"P{i + 1}",
                    round(float(vulnerability[i]), 1),
                    round(float(mobility[i]), 1),
                    int(group_size[i]),
                ]
                + [round(float(distance[i, j]), 1) for j in range(n_centers)]
            )

    return demand, int(capacity.sum())


_INSTANCES = [
    ("small", 12, 3, 11),
    ("paper_scale", 150, 5, 42),
]


if __name__ == "__main__":
    for name, n_people, n_centers, seed in _INSTANCES:
        demand, capacity = generate(name, n_people, n_centers, seed)
        print(
            f"{name:12s}: {n_people:3d} people / {n_centers} centres "
            f"| demand={demand} capacity={capacity}"
        )
