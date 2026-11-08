"""String enums render as their value, so they are safe in f-strings, CSV and JSON."""

from __future__ import annotations

import json
from enum import StrEnum

import pytest

from presidio_vol_assign.allocation.models import (
    AllocationSolverType,
    DisabilityStatus,
    HazardLevel,
    InjuryLevel,
    LivingStatus,
    RoadCondition,
)
from presidio_vol_assign.allocation.turbulence import TurbulenceMode
from presidio_vol_assign.models import SkillType, SolverType
from presidio_vol_assign.security import AuditStatus

_ENUMS = (
    AllocationSolverType,
    AuditStatus,
    DisabilityStatus,
    HazardLevel,
    InjuryLevel,
    LivingStatus,
    RoadCondition,
    SkillType,
    SolverType,
    TurbulenceMode,
)


@pytest.mark.parametrize("enum_cls", _ENUMS, ids=lambda c: c.__name__)
def test_members_render_as_value(enum_cls: type[StrEnum]) -> None:
    assert issubclass(enum_cls, StrEnum)
    for member in enum_cls:
        assert str(member) == member.value
        assert f"{member}" == member.value
        assert json.dumps(member) == json.dumps(member.value)
        assert enum_cls(str(member)) is member
