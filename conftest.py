from __future__ import annotations
from typing import NamedTuple

import pytest

import jax

from jaxtyping import PRNGKeyArray

import jit_env
from jit_env import specs


class DummyState(NamedTuple):
    key: PRNGKeyArray


class DummyEnv(jit_env.Environment):

    def __init__(self):
        super().__init__(renderer=lambda *_: jax.numpy.ones(()))

    def reset(
            self,
            key: PRNGKeyArray,
            *,
            options: jit_env.EnvOptions = None
    ) -> tuple[DummyState, jit_env.TimeStep]:
        return DummyState(key=key), jit_env.restart(jax.numpy.zeros(()))

    def step(
            self,
            state: DummyState,
            action: jit_env.Action
    ) -> tuple[DummyState, jit_env.TimeStep]:
        if action is None:
            return state, jit_env.termination(*jax.numpy.ones((2,)))
        return state, jit_env.transition(*jax.numpy.ones((3,)))

    def reward_spec(self) -> specs.Spec:
        return specs.BoundedArray((), float, 0.0, 1.0, 'reward')

    def discount_spec(self) -> specs.Spec:
        return specs.BoundedArray((), float, 0.0, 1.0, 'discount')

    def observation_spec(self) -> specs.Spec:
        return specs.Array((), float, 'observation')

    def action_spec(self) -> specs.Spec:
        return specs.Array((), float, 'action')


@pytest.fixture
def dummy_env():
    return DummyEnv()
