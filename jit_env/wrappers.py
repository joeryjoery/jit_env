"""Module with reference wrappers for Jax Transforms on Environments.

"""
from __future__ import annotations as _annotations
import typing as _typing
import dataclasses as _dataclasses

import jax as _jax

from jit_env import _core
from jit_env import specs as _specs


class Jit(_core.Wrapper):
    """Wrapper to jax.jit compile the environment step and reset functions."""

    def __init__(
            self,
            env: _core.Environment,
            **jit_kwargs: _typing.Any
    ):
        super().__init__(env)

        self._step_fun = _jax.jit(env.step, **jit_kwargs)
        self._reset_fun = _jax.jit(env.reset, **jit_kwargs)

    def reset(
            self,
            key: _jax.random.KeyArray
    ) -> tuple[_core.State, _core.TimeStep]:
        return self._reset_fun(key)

    def step(
            self,
            state: _core.State,
            action: _core.Action
    ) -> tuple[_core.State, _core.TimeStep]:
        return self._step_fun(state, action)


class Vmap(_core.Wrapper):
    """Wrapper to batch over the environment with unspecified size.

    Since `vmap` is ambiguous in the batch-size, we do not modify the
    environment `specs`. These should thus be interpreted as a `slice` of
    the batch dimensions.
    """

    def reset(
            self, key: _jax.random.KeyArray
    ) -> tuple[_core.State, _core.TimeStep]:
        return _jax.vmap(self.env.reset)(key)

    def step(
            self, state: _core.State, action: _core.Action
    ) -> tuple[_core.State, _core.TimeStep]:
        return _jax.vmap(self.env.step)(state, action)

    def render(self, state: _core.State) -> _typing.Any:
        state_0 = _jax.tree_map(lambda x: x.at[0].get(), state)
        return super().render(state_0)


class BatchSpecMixin:
    """Provide a fixed-size batched environment-spec as a MixIn."""
    env: _core.Environment

    def __init__(
            self,
            env: _core.Environment,
            num: int,
            *args: _typing.Any,
            **kwargs: _typing.Any
    ):
        # Cooperative Multiple Inheritance doesn't annotate well with mypy.
        # MyPy only sees that super() calls object and fails to see the
        # context that the MixIn could be used in.
        # See: https://github.com/python/mypy/issues/4001
        super().__init__(env, *args, **kwargs)  # type: ignore
        self.num = num

    def action_spec(self) -> _specs.Spec:
        if self.num > 0:
            return _specs.Batched(self.env.action_spec(), num=self.num)
        return self.env.action_spec()

    def observation_spec(self) -> _specs.Spec:
        if self.num > 0:
            return _specs.Batched(self.env.observation_spec(), num=self.num)
        return self.env.observation_spec()

    def reward_spec(self) -> _specs.Batched | _specs.Array:
        spec = self.env.reward_spec()
        if isinstance(spec, _specs.Array) or \
                isinstance(spec, _specs.BoundedArray):
            return spec.replace(shape=(self.num, *spec.shape))

        return _specs.Batched(self.env.reward_spec(), num=self.num)

    def discount_spec(self) -> _specs.Batched | _specs.BoundedArray:
        spec = self.env.discount_spec()
        if isinstance(spec, _specs.BoundedArray):
            return spec.replace(shape=(self.num, *spec.shape))

        return _specs.Batched(self.env.discount_spec(), num=self.num)


class Tile(BatchSpecMixin, Vmap):
    """Wrapper to batch over the environment with fixed size."""

    def reset(
            self, 
            key: _jax.random.KeyArray
    ) -> tuple[_core.State, _core.TimeStep]:
        return super().reset(_jax.random.split(key, num=self.num))


class ResetMixin:
    env: _core.Environment

    def _auto_reset(
            self,
            state: _core.State,
            step: _core.TimeStep
    ) -> tuple[_core.State, _core.TimeStep]:

        key, _ = _jax.random.split(state.key)
        state, reset_timestep = self.env.reset(key)

        # Replace observation with reset observation.
        # Do not modify the rewards, discount, or extras.
        timestep = _dataclasses.replace(
            step,
            observation=reset_timestep.observation,
            step_type=reset_timestep.step_type,
        )

        return state, timestep

    @staticmethod
    def _identity(x: _typing.Any) -> _typing.Any:
        """Helper method defined to prevent recompilations of `lambda`s."""
        return x


class AutoReset(ResetMixin, _core.Wrapper):
    """Resets Environments at TimeStep.last().

    WARNING: do not wrap `Vmap` on this Wrapper. This will lead to both step
    and reset functions being traced on every `step` call.
    Instead opt to use `AutoResetVmap`.
    """

    def step(
            self,
            state: _core.State,
            action: _core.Action
    ) -> tuple[_core.State, _core.TimeStep]:
        state, timestep = self.env.step(state, action)

        state, timestep = _jax.lax.cond(
            timestep.last(),
            self._auto_reset,
            self._identity,
            state,
            timestep,
        )

        return state, timestep


class VmapAutoReset(ResetMixin, Vmap):
    """Combines Vmap and AutoReset to optimize branch tracing in env.step.

    The difference between AutoReset and VmapAutoReset is that VmapAutoReset
    splits up the calls to `step` and `reset` into two distinct calls.

    We use `jax.vmap` for the conventional Environment step and use
    `jax.lax.map` for the Environment reset. This is more efficient as mixing
    reset and step may result in multi-program multi-data calls which `vmap`
    is not optimized for (single-program multi-data). `lax.map` handles this
    scenario more efficiently.
    """

    def step(
            self,
            state: _core.State,
            action: _core.Action
    ) -> tuple[_core.State, _core.TimeStep]:
        """Variant of AutoReset that splits up the calls to step and reset."""

        # Batched computation using `Vmap`.
        state, timestep = super().step(state, action)

        # Map heterogeneous computation (non-parallelizable).
        state, timestep = _jax.lax.map(
            lambda args: self._maybe_reset(*args), (state, timestep)
        )
        return state, timestep

    def _maybe_reset(
        self, state: _core.State, step: _core.TimeStep
    ) -> tuple[_core.State, _core.TimeStep]:
        return _jax.lax.cond(
            step.last(),
            self._auto_reset,
            lambda *x: x,
            state,
            step,
        )


class TileAutoReset(Tile, VmapAutoReset):
    """Variant of VmapAutoReset with a fixed batch size.

    This class mixes in an override to `env.reset` and to all `specs`.

    Note that this mixed in class inherits from `Vmap` multiple times, the
    class method resolution order still gives precedence to Tile over Vmap
    when calling the `reset` and `spec` functions (which is all we need).

    The shortened MRO is given by:
    [TileAutoReset, Tile, BatchSpecMixin, VmapAutoReset, ResetMixin, Vmap, ...]
    """
