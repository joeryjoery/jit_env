"""Implements Wrappers as API hooks to third party libraries.

Supported third party libraries are:
 - dm_env

The recommended usage is as follows::

    env = MyJitEnvEnvironment(...)

    if out := make_deepmind_wrapper() is not None:
        to_dm, spec_converter = out
        my_dm_env = to_dm(env)
"""
from __future__ import annotations as _annotations
import typing as _typing

import jax as _jax

import jaxtyping as _jxtype

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
    except ModuleNotFoundError:  # pragma: no cover
        return None

    def specs_to_dm_specs(spec: _specs.Spec) -> _jxtype.PyTree[dm_specs.Array]:
        """Convert a compatible `jit_env` spec into a dm_env spec (tree)."""

        if isinstance(spec, _specs.DiscreteArray):
            int_shape = _jax.numpy.shape(spec.num_values)
            if len(int_shape):
                return dm_specs.BoundedArray(
                    shape=int_shape,
                    dtype=spec.dtype,
                    minimum=0,
                    maximum=spec.num_values,
                    name=spec.name or None
                )
            else:
                return dm_specs.DiscreteArray(
                    num_values=int(spec.num_values),
                    dtype=spec.dtype,
                    name=spec.name or None,
                )
        elif isinstance(spec, _specs.BoundedArray):
            return dm_specs.BoundedArray(
                shape=spec.shape,
                dtype=spec.dtype,
                minimum=spec.minimum,
                maximum=spec.maximum,
                name=spec.name or None,
            )
        elif isinstance(spec, _specs.Array):
            return dm_specs.Array(
                shape=spec.shape,
                dtype=spec.dtype,
                name=spec.name or None,
            )
        else:
            return _jax.tree_map(specs_to_dm_specs, _specs.unpack_spec(spec))

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

            # Precompile dm_env compatible environment specs.
            env_spec = _specs.make_environment_spec(env)
            self._env_spec = _jax.tree_map(specs_to_dm_specs, env_spec)

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
            return self._env_spec.observations

        def action_spec(self):
            return self._env_spec.actions

        def reward_spec(self):
            return self._env_spec.rewards

        def discount_spec(self):
            return self._env_spec.discounts

    return ToDeepmindEnv, specs_to_dm_specs


def make_gymnasium_wrapper() -> None | tuple[_typing.Callable, type]:
    ...  # pragma: no cover  # TODO
