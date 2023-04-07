"""Define main module/ API hierarchy.

Private attributes (starting with '_') should not be accessed lest thou would
risketh the consequences of internal API changes.
"""
from typing import Union as _Alias_Union
from typing_extensions import TypeAlias as _TypeAlias

from jit_env.version import __version__, __version_info__

from jit_env import _core

from jit_env import specs
from jit_env import wrappers
from jit_env import compat

Environment = _core.Environment
Wrapper = _core.Wrapper
StepType = _core.StepType
TimeStep = _core.TimeStep

# Type Annotations/ Alias defined as a type Union to distinguish Variables
# from an Alias: https://github.com/python/mypy/issues/3494
Action: _TypeAlias = _Alias_Union[_core.Action]
State: _TypeAlias = _Alias_Union[_core.State]
Observation: _TypeAlias = _Alias_Union[_core.Observation]

# Helper functions for creating TimeStep namedtuples with default settings.
restart = _core.restart
termination = _core.termination
transition = _core.transition
truncation = _core.truncation
