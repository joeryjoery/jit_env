from __future__ import annotations
from typing import Any

import jax

from jit_env import _core
from jit_env import specs


class VmapWrapper(_core.Wrapper):

    def reset(
            self, key: jax.random.KeyArray
    ) -> tuple[_core.State, _core.TimeStep]:
        return jax.vmap(self.env.reset)(key)

    def step(
            self, state: _core.State, action: _core.Action
    ) -> tuple[_core.State, _core.TimeStep]:
        return jax.vmap(self.env.step)(state, action)

    def render(self, state: _core.State) -> Any:
        state_0 = jax.tree_map(lambda x: x.at[0].get(), state)
        return super().render(state_0)


class BatchSpecMixin:
    """Provide a fixed-size batched environment-spec as a MixIn"""
    env: _core.Environment

    def __init__(self, env: _core.Environment, num: int, *args, **kwargs):
        super().__init__(env, *args, **kwargs)
        self.num = num

    def action_spec(self) -> specs.Spec:
        if self.num > 0:
            return specs.Batched(self.env.action_spec(), num=self.num)
        return self.env.action_spec()

    def observation_spec(self) -> specs.Spec:
        if self.num > 0:
            return specs.Batched(self.env.observation_spec(), num=self.num)
        return self.env.observation_spec()

    def reward_spec(self) -> specs.Batched | specs.Array:
        spec = self.env.reward_spec()
        if isinstance(spec, specs.Array) or \
                isinstance(spec, specs.BoundedArray):
            return spec.replace(shape=(self.num, *spec.shape))

        return specs.Batched(self.env.reward_spec(), num=self.num)

    def discount_spec(self) -> specs.Batched | specs.BoundedArray:
        spec = self.env.discount_spec()
        if isinstance(spec, specs.BoundedArray):
            return spec.replace(shape=(self.num, *spec.shape))

        return specs.Batched(self.env.discount_spec(), num=self.num)


class Tile(BatchSpecMixin, VmapWrapper):
    """VmapWrapper but with a *fixed* batch-size and single-key reset"""

    def reset(
            self, 
            key: jax.random.KeyArray
    ) -> tuple[_core.State, _core.TimeStep]:
        return super().reset(jax.random.split(key, num=self.num))


