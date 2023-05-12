from __future__ import annotations
from typing import TYPE_CHECKING
from dataclasses import replace

import jax
import jax.numpy as jnp

import jit_env
from jit_env import specs

import jaxtyping as jxtype


if TYPE_CHECKING:  # pragma: no cover
    # See: https://github.com/python/mypy/issues/6239
    from dataclasses import dataclass
else:
    from chex import dataclass


@dataclass(frozen=True)
class MyState:
    key: jax.random.KeyArray
    count: jxtype.Int32[jxtype.Array, '']


class CountingEnv(jit_env.Environment):
    """Implement a simple reference Environment that sums Action ints.

    The environment accumulates all integer actions until the given maximum
    is exceeded, then it terminates with reward and discount = 0.0. Otherwise,
    it always gives reward and discount = 1.0.
    """

    def __init__(self, maximum: int | jxtype.Integer[jxtype.Array, '']):
        super().__init__()
        self.maximum = maximum

    def reset(
            self,
            key: jax.random.KeyArray
    ) -> tuple[MyState, jit_env.TimeStep]:
        state = MyState(key=key, count=jnp.zeros((), jnp.int32))
        return state, jit_env.restart(state.count, shape=())

    def step(
            self,
            state: MyState,
            action: jit_env.Action
    ) -> tuple[MyState, jit_env.TimeStep]:
        state = replace(state, count=state.count + action)

        step = jax.lax.cond(
            state.count < self.maximum,
            lambda: jit_env.transition(1.0, state.count, 1.0),
            lambda: jit_env.termination(1.0, state.count, shape=())
        )

        return state, step

    def reward_spec(self) -> specs.Spec:
        return specs.BoundedArray(
            (), jnp.float32, jnp.zeros(()), jnp.ones(()), 'reward'
        )

    def discount_spec(self) -> specs.Spec:
        return specs.BoundedArray(
            (), jnp.float32, jnp.zeros(()), jnp.ones(()), 'discount'
        )

    def observation_spec(self) -> specs.Spec:
        return specs.Array((), jnp.int32, 'observation')

    def action_spec(self) -> specs.Spec:
        return specs.DiscreteArray(self.maximum, jnp.int32, 'action')
