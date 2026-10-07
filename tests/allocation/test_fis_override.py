"""Unit tests for the FIS rule-base override mechanism."""

from __future__ import annotations

import json

import pytest

from presidio_vol_assign.allocation.fis import (
    FIS1_RULES,
    RULE_COUNTS,
    VALID_FIS_NAMES,
    build_fis_with_drops,
    clear_fis_overrides,
    evaluate_fis1_ulpp,
    fis_overrides,
    load_fis_rules_spec,
    set_fis_overrides,
)


class TestRuleConstants:
    def test_rule_counts_match(self):
        assert RULE_COUNTS["fis1"] == len(FIS1_RULES) == 27
        assert RULE_COUNTS["fis2_til"] == 9
        assert RULE_COUNTS["fis2a_trd"] == 9
        assert RULE_COUNTS["fis2b_rpd"] == 3
        assert RULE_COUNTS["fis3"] == 27

    def test_valid_fis_names(self):
        assert VALID_FIS_NAMES == frozenset({"fis1", "fis2_til", "fis2a_trd", "fis2b_rpd", "fis3"})


class TestBuildFISWithDrops:
    def test_no_drops_round_trip(self):
        # An empty drop list should yield a working system
        sys = build_fis_with_drops("fis1", [])
        assert sys is not None

    def test_single_drop_works(self):
        # Drop one rule from FIS1 (still 26 left)
        sys = build_fis_with_drops("fis1", [0])
        assert sys is not None

    @pytest.mark.parametrize(
        "name,bad_indices",
        [
            ("fis1", [99]),
            ("fis1", [-1]),
            ("fis2b_rpd", [3]),  # only 3 rules, max valid is 2
        ],
    )
    def test_out_of_range_rejected(self, name, bad_indices):
        with pytest.raises(ValueError, match="out of range"):
            build_fis_with_drops(name, bad_indices)

    def test_unknown_fis_rejected(self):
        with pytest.raises(ValueError, match="Unknown FIS name"):
            build_fis_with_drops("not_a_fis", [0])

    def test_drop_all_rejected(self):
        # Cannot drop every rule — system would be empty
        with pytest.raises(ValueError, match="at least one must remain"):
            build_fis_with_drops("fis2b_rpd", [0, 1, 2])


class TestOverrideRegistry:
    def teardown_method(self):
        clear_fis_overrides()

    def test_set_and_clear(self):
        sys = build_fis_with_drops("fis1", [0])
        set_fis_overrides({"fis1": sys})
        # Eval still works (output value depends on which rules fire — we only
        # check the call succeeds and produces a number in range)
        v = evaluate_fis1_ulpp(0.5, 50.0, 24.0)
        assert 0.0 <= v <= 100.0
        clear_fis_overrides()
        v2 = evaluate_fis1_ulpp(0.5, 50.0, 24.0)
        assert 0.0 <= v2 <= 100.0

    def test_context_manager_restores(self):
        # baseline
        baseline = evaluate_fis1_ulpp(0.9, 90.0, 4.0)
        with fis_overrides({"fis1": [0]}):
            v_inside = evaluate_fis1_ulpp(0.9, 90.0, 4.0)
            assert 0.0 <= v_inside <= 100.0
        v_after = evaluate_fis1_ulpp(0.9, 90.0, 4.0)
        assert v_after == baseline

    def test_context_manager_restores_on_exception(self):
        baseline = evaluate_fis1_ulpp(0.9, 90.0, 4.0)
        with pytest.raises(RuntimeError):
            with fis_overrides({"fis1": [0]}):
                raise RuntimeError("boom")
        v_after = evaluate_fis1_ulpp(0.9, 90.0, 4.0)
        assert v_after == baseline


class TestLoadFISRulesSpec:
    def test_valid_spec(self, tmp_path):
        spec_path = tmp_path / "spec.json"
        spec_path.write_text(json.dumps({"fis1": [3, 5], "fis3": [12]}))
        spec = load_fis_rules_spec(spec_path)
        assert spec == {"fis1": [3, 5], "fis3": [12]}

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_fis_rules_spec(tmp_path / "missing.json")

    def test_malformed_json(self, tmp_path):
        spec_path = tmp_path / "bad.json"
        spec_path.write_text("not valid json{")
        with pytest.raises(ValueError, match="Invalid JSON"):
            load_fis_rules_spec(spec_path)

    def test_unknown_fis_name(self, tmp_path):
        spec_path = tmp_path / "bad.json"
        spec_path.write_text(json.dumps({"fis_alien": [0]}))
        with pytest.raises(ValueError, match="Unknown FIS name"):
            load_fis_rules_spec(spec_path)

    def test_non_int_indices(self, tmp_path):
        spec_path = tmp_path / "bad.json"
        spec_path.write_text(json.dumps({"fis1": [0.5, 1]}))
        with pytest.raises(ValueError, match="non-negative ints"):
            load_fis_rules_spec(spec_path)

    def test_top_level_must_be_object(self, tmp_path):
        spec_path = tmp_path / "bad.json"
        spec_path.write_text(json.dumps([1, 2, 3]))
        with pytest.raises(ValueError, match="JSON object"):
            load_fis_rules_spec(spec_path)
