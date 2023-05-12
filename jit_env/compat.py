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
from collections.abc import Mapping, Sequence

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


def make_gymnasium_wrapper() -> None | tuple[type, _typing.Callable]:
    try:
        import gymnasium as gym
        import numpy as np
    except ModuleNotFoundError:  # pragma: no cover
        return None

    def specs_to_gym_space(spec: _specs.Spec) -> gym.Space:
        """Convert a compatible `jit_env` spec into a gym space.

        Note that jax DTypes convert to numpy DTypes in NDArray creation.
        """
        if isinstance(spec, _specs.DiscreteArray):
            if not spec.shape:
                return gym.spaces.Discrete(
                    int(spec.num_values)
                )
            else:
                return gym.spaces.MultiDiscrete(
                    np.asarray(spec.num_values, spec.dtype),
                    spec.dtype
                )

        elif isinstance(spec, _specs.BoundedArray):
            return gym.spaces.Box(
                np.asarray(spec.minimum, spec.dtype),
                np.asarray(spec.maximum, spec.dtype),
                spec.shape,
                spec.dtype
            )
        elif isinstance(spec, _specs.Array):
            with np.errstate(invalid='ignore'):  # suppresses `int(np.inf)`
                return gym.spaces.Box(
                    np.broadcast_to(-np.inf, spec.shape).astype(spec.dtype),
                    np.broadcast_to(np.inf, spec.shape).astype(spec.dtype),
                    spec.shape,
                    spec.dtype
                )
        else:
            if isinstance(spec, _specs.Tree):
                unpacked = spec.as_spec_struct()
                out = _jax.tree_map(
                    specs_to_gym_space, unpacked,
                    is_leaf=lambda z: z is not unpacked
                )
            else:
                out = _jax.tree_map(specs_to_gym_space, spec)

            if isinstance(out, Mapping):
                return gym.spaces.Dict(dict(out))
            if isinstance(out, Sequence):
                return gym.spaces.Tuple(tuple(out))

        raise NotImplementedError(
            f"Conversion of {spec.__class__.__name__} is not supported!"
        )

    class ToGym(gym.Env[_core.Observation, _core.Action]):

        def __init__(
                self,
                env: _core.Environment,
                seed: int = 0,
        ):
            super().__init__()

            self.env = env

            self.rng: _jax.random.KeyArray = _jax.random.PRNGKey(seed)
            self.env_state, _ = env.reset(self.rng)

            self.metadata.update({
                "name": str(env), "render_modes": ["human", "rgb_array"]
            })

        @property
        def action_space(self):
            return specs_to_gym_space(
                self.env.action_spec()
            )

        @property
        def observation_space(self):
            return specs_to_gym_space(
                self.env.observation_spec()
            )

        def _seed(self, seed: int = 0):
            """Set RNG seed (or use 0)"""
            self.rng = _jax.random.PRNGKey(seed)

        def step(
                self, action: _core.Action
        ) -> tuple[
            _core.Observation, gym.core.SupportsFloat, bool, bool, dict
        ]:
            """Step environment, follow new step API"""
            self.env_state, step = self.env.step(self.env_state, action)

            return (
                step.observation,
                np.asarray(step.reward),
                bool(step.last()),
                bool(step.last()),
                step.extras or {}
            )

        def reset(
                self,
                *,
                seed: int | None = None,
                options: dict | None = None
        ) -> tuple[_core.Observation, dict]:
            """Reset environment, update parameters and seed if provided"""
            if seed is not None:
                self._seed(seed)

            self.rng, reset_key = _jax.random.split(self.rng)
            self.env_state, step = self.env.reset(reset_key)

            return step.observation, (step.extras or {})

        def render(
                self, mode: str = "human"
        ) -> gym.core.RenderFrame | list[gym.core.RenderFrame] | None:
            return self.env.render(self.env_state)  # pragma: no cover

    return ToGym, specs_to_gym_space
