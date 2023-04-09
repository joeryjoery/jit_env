from __future__ import annotations
from typing import Any, NamedTuple

import pytest

import jax

import jit_env
from jit_env import specs


class DummyState(NamedTuple):
    key: jax.random.KeyArray


class DummyEnv(jit_env.Environment):

    def reset(
            self,
            key: jax.random.KeyArray
    ) -> tuple[jit_env.State, jit_env.TimeStep]:
        return DummyState(key=key), jit_env.restart(0.0)

    def step(
            self,
            state: jit_env.State,
            action: jit_env.Action
    ) -> tuple[jit_env.State, jit_env.TimeStep]:
        return state, jit_env.transition(1.0, 1.0, 1.0)

    def reward_spec(self) -> specs.Spec:
        return specs.BoundedArray((), float, 0.0, 1.0, 'reward')

    def discount_spec(self) -> specs.Spec:
        return specs.BoundedArray((), float, 0.0, 1.0, 'discount')

    def observation_spec(self) -> specs.Spec:
        return specs.Array((), float, 'observation')

    def action_spec(self) -> specs.Spec:
        return specs.Array((), float, 'action')

    def render(self, state: jit_env.State) -> Any:
        pass


@pytest.fixture
def dummy_env():
    return DummyEnv()
