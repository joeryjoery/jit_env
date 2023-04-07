"""Implements Wrappers as API hooks to third party libraries.

Supported third party libraries are:
 - dm_env

The recommended usage is as follows:

```python
env = MyJitEnvEnvironment(...)

if out := make_deepmind_wrapper() is None:
    raise ModuleNotFoundError()

to_dm, spec_converter = out

my_dm_env = to_dm(env)
```
"""
from __future__ import annotations as _annotations
import typing as _typing

import jax as _jax

from jit_env import _core
from jit_env import specs as _specs


def make_deepmind_wrapper() -> None | tuple[type, _typing.Callable]:
    """If dm_env can be imported, return an Environment and Spec converter

    If the library can not be loaded, this function suppresses the error.

    Returns:
        None if dm_env cannot be imported. Otherwise it returns a tuple with;
        1)  A dm_env.Environment that is initialized with a jit_env.Environment
            and a jax Pseudo RNG Key.
        2)  A function that converts jit_env.specs to compatible dm_env.specs.
    """
    try:
        import dm_env
        from dm_env import specs as dm_specs
    except ModuleNotFoundError:
        return None

    def specs_to_dm_specs(spec: _specs.Spec) -> dm_specs.Array:
        pass  # TODO

    class ToDeepmindEnv(dm_env.Environment):
        """A dm_env.Environment that wraps a jit_env.Environment.

        This Environment class is not functionally pure in the step and
        reset functions as it requires maintaining a class state for the
        Environment State and the Random Key.
        """

        def __init__(
                self,
                env: _core.Environment,
                rng: _jax.random.KeyArray = _jax.random.PRNGKey(0)
        ):
            self.env = env
            self.rng = rng

            self._state = None

        def reset(self) -> dm_env.TimeStep:
            self.rng, key = _jax.random.split(self.rng)
            self._state, step = self.env.reset(key)
            return dm_env.restart(step.observation)

        def step(self, action) -> dm_env.TimeStep:
            self._state, step = self.env.step(self._state, action)
            if step.last():
                # Discount computation should be handled in `env`.
                return dm_env.truncation(
                    step.reward, step.observation, step.discount
                )
            return dm_env.transition(
                step.reward, step.observation, step.discount
            )

        def observation_spec(self):
            return specs_to_dm_specs(self.env.observation_spec())

        def action_spec(self):
            return specs_to_dm_specs(self.env.action_spec())

        def reward_spec(self):
            return specs_to_dm_specs(self.env.reward_spec())

        def discount_spec(self):
            return specs_to_dm_specs(self.env.discount_spec())

    return ToDeepmindEnv, specs_to_dm_specs


def make_gymnasium_wrapper() -> None | tuple[_typing.Callable, type]:
    pass  # TODO
