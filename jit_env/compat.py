"""Implements Wrappers as API hooks to third party libraries"""
from __future__ import annotations
from typing import Callable

import jax

import jit_env
from jit_env import specs


def make_deepmind_wrapper() -> None | tuple[Callable, type]:
    try:
        import dm_env
        from dm_env import specs as dm_specs
    except ModuleNotFoundError:
        return None

    def specs_to_dm_specs(spec: specs.Spec) -> dm_specs.Array:
        pass  # TODO

    class ToDeepmindEnv(dm_env.Environment):

        def __init__(
                self,
                env: jit_env.Environment,
                rng: jax.random.KeyArray = jax.random.PRNGKey(0),
                **stage_kwargs
        ):
            self.env = env
            self.rng = rng

            self._state = None

            self._reset_fun = jax.jit(env.reset, **stage_kwargs)
            self._step_fun = jax.jit(env.step, **stage_kwargs)

        def reset(self) -> dm_env.TimeStep:
            self.rng, key = jax.random.split(self.rng)
            self._state, step = self._reset_fun(key)
            return dm_env.restart(step.observation)

        def step(self, action) -> dm_env.TimeStep:
            self._state, step = self._step_fun(self._state, action)
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
            pass

        def reward_spec(self):
            pass

        def discount_spec(self):
            pass

    return specs_to_dm_specs, ToDeepmindEnv


def make_gymnasium_wrapper() -> None | tuple[Callable, type]:
    pass  # TODO
