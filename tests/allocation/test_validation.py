"""Unit tests for allocation/validation.py — happy path + each error class."""

from __future__ import annotations

import pandas as pd
import pytest

from presidio_vol_assign.allocation.models import AllocationConfig, AllocationSolverType, Weights
from presidio_vol_assign.allocation.validation import (
    load_allocation_problem,
    validate_allocation_config,
)


class TestLoadAllocationProblem:
    def test_happy_path(self, csv_dir):
        p = load_allocation_problem(
            csv_dir / "people.csv",
            csv_dir / "centers.csv",
            csv_dir / "travel.csv",
            n_dir=4,
        )
        assert p.n_people == 8
        assert p.n_centers == 3
        assert len(p.travel) == 24

    def test_missing_file_raises(self, csv_dir):
        with pytest.raises(FileNotFoundError):
            load_allocation_problem(
                csv_dir / "missing.csv",
                csv_dir / "centers.csv",
                csv_dir / "travel.csv",
                n_dir=4,
            )

    def test_n_dir_zero_rejected(self, csv_dir):
        with pytest.raises(ValueError, match="n_dir must be > 0"):
            load_allocation_problem(
                csv_dir / "people.csv",
                csv_dir / "centers.csv",
                csv_dir / "travel.csv",
                n_dir=0,
            )

    def test_n_dir_too_large_rejected(self, csv_dir):
        with pytest.raises(ValueError, match="n_dir must be <"):
            load_allocation_problem(
                csv_dir / "people.csv",
                csv_dir / "centers.csv",
                csv_dir / "travel.csv",
                n_dir=8,  # equals n_people, not strictly less
            )

    def test_missing_travel_pair_rejected(self, csv_dir):
        # Drop one travel row
        df = pd.read_csv(csv_dir / "travel.csv")
        df = df.iloc[1:]
        df.to_csv(csv_dir / "travel.csv", index=False)
        with pytest.raises(ValueError, match="travel CSV is missing"):
            load_allocation_problem(
                csv_dir / "people.csv",
                csv_dir / "centers.csv",
                csv_dir / "travel.csv",
                n_dir=4,
            )

    def test_unknown_disability_value_rejected(self, csv_dir):
        df = pd.read_csv(csv_dir / "people.csv")
        df.loc[0, "disability_status"] = "extreme"  # not a valid enum value
        df.to_csv(csv_dir / "people.csv", index=False)
        with pytest.raises(ValueError, match="disability_status must be one of"):
            load_allocation_problem(
                csv_dir / "people.csv",
                csv_dir / "centers.csv",
                csv_dir / "travel.csv",
                n_dir=4,
            )

    def test_age_out_of_range_rejected(self, csv_dir):
        df = pd.read_csv(csv_dir / "people.csv")
        df.loc[0, "age"] = 200
        df.to_csv(csv_dir / "people.csv", index=False)
        with pytest.raises(ValueError, match="age must be in"):
            load_allocation_problem(
                csv_dir / "people.csv",
                csv_dir / "centers.csv",
                csv_dir / "travel.csv",
                n_dir=4,
            )

    def test_duplicate_person_id_rejected(self, csv_dir):
        df = pd.read_csv(csv_dir / "people.csv")
        df.loc[1, "person_id"] = df.loc[0, "person_id"]
        df.to_csv(csv_dir / "people.csv", index=False)
        with pytest.raises(ValueError, match="duplicate person_id"):
            load_allocation_problem(
                csv_dir / "people.csv",
                csv_dir / "centers.csv",
                csv_dir / "travel.csv",
                n_dir=4,
            )

    def test_travel_unknown_person_rejected(self, csv_dir):
        df = pd.read_csv(csv_dir / "travel.csv")
        df.loc[0, "person_id"] = "P_NOT_REAL"
        df.to_csv(csv_dir / "travel.csv", index=False)
        with pytest.raises(ValueError, match="not present in people.csv"):
            load_allocation_problem(
                csv_dir / "people.csv",
                csv_dir / "centers.csv",
                csv_dir / "travel.csv",
                n_dir=4,
            )


class TestValidateAllocationConfig:
    def test_valid_config_passes(self):
        cfg = AllocationConfig(solver=AllocationSolverType.NSGA3, objectives=4)
        validate_allocation_config(cfg)  # no raise

    def test_bad_objectives_rejected(self):
        cfg = AllocationConfig(solver=AllocationSolverType.NSGA2, objectives=5)
        with pytest.raises(ValueError, match="objectives must be 3 or 4"):
            validate_allocation_config(cfg)

    def test_bad_pop_size_rejected(self):
        cfg = AllocationConfig(solver=AllocationSolverType.NSGA2, pop_size=1)
        with pytest.raises(ValueError, match="pop_size"):
            validate_allocation_config(cfg)

    def test_bad_weight_rejected(self):
        cfg = AllocationConfig(solver=AllocationSolverType.NSGA2, weights=Weights(was=2.5))
        with pytest.raises(ValueError, match="weights.was must be in"):
            validate_allocation_config(cfg)

    def test_zero_divisions_rejected(self):
        cfg = AllocationConfig(solver=AllocationSolverType.NSGA3, nsga3_divisions=0)
        with pytest.raises(ValueError, match="nsga3_divisions"):
            validate_allocation_config(cfg)
