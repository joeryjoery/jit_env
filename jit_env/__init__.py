from jit_env import _core
from jit_env import specs
from jit_env.version import __version__, __version_info__

Environment = _core.Environment
StepType = _core.StepType
TimeStep = _core.TimeStep

# Helper functions for creating TimeStep namedtuples with default settings.
restart = _core.restart
termination = _core.termination
transition = _core.transition
truncation = _core.truncation
