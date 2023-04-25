"""Define main module/ API hierarchy.

Private attributes (starting with '_') should not be accessed lest thou would
risketh the consequences of internal API changes.
"""
from jit_env.version import __version__, __version_info__

from jit_env import _core
from jit_env._core import (
    Action as Action,
    State as State,
    Observation as Observation
)

from jit_env import specs
from jit_env import wrappers
from jit_env import compat

Environment = _core.Environment
Wrapper = _core.Wrapper
StepType = _core.StepType
TimeStep = _core.TimeStep

# Helper functions for creating TimeStep namedtuples with default settings.
restart = _core.restart
termination = _core.termination
transition = _core.transition
truncation = _core.truncation
