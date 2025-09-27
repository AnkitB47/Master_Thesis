"""Utilities for modeling the Freiberg HP-POX benchmark and derived plants."""

from .configuration import (
    CaseDefinition,
    GeometryProfile,
    GeometrySegment,
    HeatLossModel,
    InletStream,
    OperatingEnvelope,
    load_case_definition,
)
from .pfr import PlugFlowOptions, PlugFlowResult, PlugFlowSolver
from .plasma import PlasmaSurrogateConfig
from .reduction import GAGNNReducer, ReductionConfig
from .compat import reconcile_feed_with_mechanism

__all__ = [
    "CaseDefinition",
    "GeometryProfile",
    "GeometrySegment",
    "HeatLossModel",
    "InletStream",
    "OperatingEnvelope",
    "PlugFlowSolver",
    "PlugFlowOptions",
    "PlugFlowResult",
    "load_case_definition",
    "PlasmaSurrogateConfig",
    "GAGNNReducer",
    "ReductionConfig",
    "reconcile_feed_with_mechanism",
]
