"""Module with reference wrappers for Jax Transforms on Environments.

This Module implements Wrappers for:
 - Staging of the environment step and reset functions with jax.jit.
 - Batching of the environment step and reset functions with jax.vmap.
 - Automatically Resetting Environments upon `StepType.LAST`.
 - Optimized Wrappers for combining Batching and AutoResetting.
"""
from __future__ import annotations as _annotations
import typing as _typing

import jax as _jax

import jit_env
from jit_env import _core
from jit_env import specs as _specs


class Jit(_core.Wrapper):
    """Wrapper to jax.jit compile the environment step and reset functions."""

    def __init__(
            self,
            env: _core.Environment,
            **jit_kwargs: _typing.Any
    ):
        """Initializes the wrapper by staging `env.step` and `env.reset`.

        Args:
            env:
                The environment to accelerate using jax.jit.
            **jit_kwargs:
                keyword arguments passed to jax.jit for both `env.step` and
                `env.reset`. Defer to the jax documentation for up-to-date
                documentation on the keyword arguments.
        """

        super().__init__(env)

        self._jit_kwargs = jit_kwargs
        self._step_fun = _jax.jit(env.step, **jit_kwargs)
        self._reset_fun = _jax.jit(env.reset, **jit_kwargs)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(" \
               f"env={repr(self.env)}," \
               f"jit_kwargs={self._jit_kwargs})"

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

    def __init__(self, env: jit_env.Environment, **step_vmap_kwargs):
        """Initializes the transforming `env.step` and `env.reset` with vmap.

        The step function may receive additional keyword arguments to modify
        behaviour. Unlike Jit, these kwargs are not used for transforming
        the reset function. This may be useful for holding either a state or
        an action constant using `in_axes`, which is not symmetrically
        compatible with `reset`.

        Args:
            env:
                The environment to batch using jax.vmap.
            **step_vmap_kwargs:
                keyword arguments passed to jax.vmap only for `env.step`.
                Defer to the jax documentation for up-to-date documentation
                on the keyword arguments.
        """
        super().__init__(env)

        self._step_vmap_kwargs = step_vmap_kwargs

        self._step_fun = _jax.vmap(env.step, **step_vmap_kwargs)
        self._reset_fun = _jax.vmap(env.reset)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(" \
               f"env={repr(self.env)}," \
               f"step_vmap_kwargs={self._step_vmap_kwargs})"

    def reset(
            self,
            key: _jax.random.KeyArray  # (Batch, dim_key)
    ) -> tuple[_core.State, _core.TimeStep]:
        return self._reset_fun(key)

    def step(
            self,
            state: _core.State,  # Tree[(Batch, dim_leaf), ...]
            action: _core.Action  # Tree[(Batch, dim_leaf), ...]
    ) -> tuple[_core.State, _core.TimeStep]:
        return self._step_fun(state, action)

    def render(self, state: _core.State) -> _typing.Any:
        """Generate a pixel-observation based on the 0'th state slice. """
        state_0 = _jax.tree_map(lambda x: x.at[0].get(), state)
        return super().render(state_0)


class StopGradient(_core.Wrapper):
    """Wrapper to cancel out all Env-dependent gradients."""

    def step(
            self,
            state: _core.State,
            action: _core.Action
    ) -> tuple[_core.State, _core.TimeStep]:
        return _jax.tree_map(
            _jax.lax.stop_gradient, super().step(state, action)
        )


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
        """Use Cooperative Multiple Inheritance to initialize the mixed class.

        This class simultaneously initializes the fixed size that the
        Environment is batched over.

        Args:
            env: The base Environment to compose with
            num: The batch-size to be mapped over.
            *args: Cooperative Multiple Inheritance Variadic Args
            **kwargs: Cooperative Multiple Inheritance Keyword Args
        """
        if not num > 0:
            raise ValueError(
                "Cannot batch over an empty dimension! Arg: num > 0!"
            )

        # Cooperative Multiple Inheritance doesn't annotate well with mypy.
        # MyPy only sees that super() calls object and fails to see the
        # context that the MixIn could be used in.
        # See: https://github.com/python/mypy/issues/4001
        super().__init__(env, *args, **kwargs)  # type: ignore
        self.num = num

    def action_spec(self) -> _specs.Batched:
        """Compose the base env.action_spec() inside a Batched Spec."""
        return _specs.Batched(self.env.action_spec(), num=self.num)

    def observation_spec(self) -> _specs.Batched:
        """Compose the base env.observation_spec() inside a Batched Spec."""
        return _specs.Batched(self.env.observation_spec(), num=self.num)

    def reward_spec(self) -> _specs.Batched | _specs.Array:
        """Either reshape or compose env.reward_spec() to a batch.

        If the env.reward_spec() is a Array type, which allows for reshaping,
        the spec is reshaped. Otherwise it is composed with Batched.
        """
        spec = self.env.reward_spec()
        if isinstance(spec, _specs.Array):
            return _specs.reshape_spec(spec, prepend=(self.num,))

        return _specs.Batched(self.env.reward_spec(), num=self.num)

    def discount_spec(self) -> _specs.Batched | _specs.BoundedArray:
        """Either reshape or compose env.discount_spec() to a batch.

        If the env.discount_spec() is a Array type, which allows for reshaping,
        the spec is reshaped. Otherwise it is composed with Batched.
        """
        spec = self.env.discount_spec()
        if isinstance(spec, _specs.BoundedArray):
            # TODO: Not compatibile with Array (should not be valid anyway).
            return _specs.reshape_spec(spec, prepend=(self.num,))

        return _specs.Batched(self.env.discount_spec(), num=self.num)


class Tile(BatchSpecMixin, Vmap):
    """Wrapper to batch over the environment with fixed size.

    This Wrapper is identical to `Vmap`, but it requires only a single
    Pseudo RNG Key to the reset function, which it splits up into a fixed
    size Key Array. As a result, the environment-spec can be properly
    specified, which is done by the `BatchSpecMixin`.

    The constructor is called first through `BatchSpecMixin` which
    simultaneously requires the user to define the batch-size as `num`.
    """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(" \
               f"env={repr(self.env)}," \
               f"num={self.num}," \
               f"step_vmap_kwargs={self._step_vmap_kwargs})"

    def reset(
            self,
            key: _jax.random.KeyArray
    ) -> tuple[_core.State, _core.TimeStep]:
        return super().reset(_jax.random.split(key, num=self.num))


class ResetMixin:
    """Mixin to provide helper functions for resetting Environments"""
    env: _core.Environment

    def _auto_reset(
            self,
            state: _core.State,
            step: _core.TimeStep
    ) -> tuple[_core.State, _core.TimeStep]:
        """Helper method to reset the environment and modify TimeStep.

        Upon termination, this method ensures that the reward, discount,
        and extras of the terminal `TimeStep` are not overwritten. Only
        the `observation` and `step_type` are modified at a terminal state.

        This method also ensures that the Pseudo RNG Key carried by the
        environment state is properly updated.

        Args:
            state:
                The terminal State from a previous call to `env.step`.
            step:
                The TimeStep object from a previous cal to `env.step`.
                The step_type attribute is expected to be StepType.LAST.

        Returns:
            A tuple of `State` and `TimeStep` at indices;

            1) The `State` object is expected to carry a chain of `key` values
               in order to internally maintain the random state.
            2) The TimeStep object will have step_type set to `StepType.FIRST`
               and observation generated by env.reset. The discount, reward,
               and extras fields are copied from the `step` argument.
        """

        key, _ = _jax.random.split(state.key)
        state, reset_timestep = self.env.reset(key)

        timestep = _core.TimeStep(
            step_type=reset_timestep.step_type,  # Overwrite step
            reward=step.reward,
            discount=step.discount,
            observation=reset_timestep.observation,  # Overwrite step
            extras=step.extras
        )

        return state, timestep

    @staticmethod
    def _identity(*x: _typing.Any) -> _typing.Any:
        """Helper method defined to prevent recompilations of `lambda`s.

        Args:
            x: Some value

        Returns:
            The same value `x`.
        """
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
        """Calls the environment step function and reset function as needed.

        If the environment returned a `TimeStep` with `StepType.LAST`, then
        the `reset` function is called again to restart the `State` and
        `TimeStep` objects. Note that the resetted `TimeStep` object
        maintains the reward, discount, and extras field of the terminated
        `TimeStep`. Only the `step_type` and `observation` fields correspond
        to the restarted Environment.

        Args:
            state:
                A State carried through `step` and created by `reset`.
            action:
                A Jax-compatible data-structure adhering to self.action_spec().

        Returns:
            A tuple of `State` and `TimeStep` at indices;

            1)  The `State` object is expected to carry a chain of `key` values
                in order to internally maintain the random state.
            2)  The TimeStep object will always have step_type set to
                `StepType.MID` since `StepType.LAST` is reset internally
                through a call to `env.reset`. The reward, discount, and
                extras fields of a terminal step are carried to the resetted
                step. Only the observation and step_type correspond to the
                resetted State.
        """
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
        # Batched homogenous computation using `Vmap`.
        state, timestep = super().step(state, action)

        # Map heterogeneous computation (non-parallelizable).
        state, timestep = _jax.lax.map(self._maybe_reset, (state, timestep))

        return state, timestep

    def _maybe_reset(
            self, args: tuple[_core.State, _core.TimeStep]
    ) -> tuple[_core.State, _core.TimeStep]:
        """Helper method defined to prevent recompilations of `lambda`s."""
        state, step = args
        return _jax.lax.cond(
            step.last(),
            self._auto_reset,
            self._identity,
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

    The mixed in functions by evalation order are:
    [(), (reset), (*_spec), (step, _maybe_reset), (_identity, _auto_reset),
    (reset, step, render)]
    """
