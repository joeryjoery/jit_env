"""This module defines the fundamental logic for defining RL-Environment IO.

This module defines:
    - The key Types/ Datastructures concerning Agent-Environment Input-Output.
    - An interface for defining Environment Logic
    - An interface for composing Environment Logic
    - Helper functions to initialize Types communicated to the Agent.
"""
from __future__ import annotations
import abc

from typing import (
    Any, TYPE_CHECKING, TypeVar, Generic, Sequence,
    Protocol, Callable
)
from dataclasses import field

if TYPE_CHECKING:  # pragma: no cover
    # See: https://github.com/python/mypy/issues/6239
    from dataclasses import dataclass
else:
    from chex import dataclass

from jaxtyping import Array, ArrayLike, PyTree, Num, Int8, Bool, PRNGKeyArray

import jax.numpy as jnp

from jit_env import specs


# Define Environment IO Types.

class StateProtocol(Protocol):
    """Environment states should always carry a PRNGKey for predictable IO.

    For example, if we were to save states in a replay buffer (for whatever
    reason) we require all dependent data to be contained in State.
    This would not be the case if `key` were part of the Environment
    `step` function, as this could be changed arbitrarily by the caller.
    In turn, this leads to un-reproducible behaviour.
    """
    key: PRNGKeyArray


# The following should all be valid Jax types
Action = TypeVar("Action")

Observation = TypeVar("Observation")
StepT = TypeVar("StepT", bound=Int8[Array, ''])
RewardT = TypeVar("RewardT", bound=PyTree[ArrayLike])
DiscountT = TypeVar("DiscountT", bound=PyTree[ArrayLike])

State = TypeVar("State", bound=StateProtocol)


class StepType:
    """Defines the status of a `TimeStep` within a sequence."""
    FIRST: Int8[Array, ''] = jnp.array(0, jnp.int8)
    MID: Int8[Array, ''] = jnp.array(1, jnp.int8)
    LAST: Int8[Array, ''] = jnp.array(2, jnp.int8)


@dataclass(init=True, repr=True, eq=True, frozen=True)
class TimeStep(Generic[Observation, RewardT, DiscountT, StepT]):
    """Defines the datastructure that is communicated to the Agent.

    While dm_env utilizes a NamedTuple, we opted for a mappable dataclass
    to allow for modular transformations using `dataclasses.replace`, which
    is a private method for `NamedTuple` types. To avoid mutability confusion
    this dataclass is made `frozen`, fields can only be "modified" by creating
    new objects of this Type.

    Like in dm_env, `env.reset` will generate a `TimeStep` with step_type set
    to `StepType.FIRST`. If `env.step` terminates the episode, step_type is
    set to `StepType.LAST`. Otherwise, step_type is set to `StepType.MID`.

    Attributes:
        step_type:
            A StepType value that indicates episode boundaries.
        reward:
            A (tree of) numerical value(s) to be optimized for by an Agent.
        discount:
            A (tree of) float(s) in [0., 1.] to scale cumulative returns.
        observation:
            Generic Jax-Compatible data-structure that the agent observes to
            compute new actions.
        extras:
            Optional data that is typically not communicated to the Agent
            but allows the user to track certain loss-metrics, accuracies,
            or other information. This can be stored in a Replay Buffer for
            training or for generally monitoring Agent behaviour.
            This field is excluded from object comparisons.
    """
    step_type: StepT
    reward: RewardT
    discount: DiscountT
    observation: Observation
    extras: dict[str, Any] | None = field(default=None, compare=False)

    def first(self) -> Bool[Array, '']:
        """Proxy function to check if step was generated by `reset`."""
        return self.step_type == StepType.FIRST

    def mid(self) -> Bool[Array, '']:
        """Proxy function to check if step was generated during an episode."""
        return self.step_type == StepType.MID

    def last(self) -> Bool[Array, '']:
        """Proxy function to check if step was terminated."""
        return self.step_type == StepType.LAST


# Define Environment and Wrapper

class Environment(
    Generic[State, Action, Observation, RewardT, DiscountT],
    metaclass=abc.ABCMeta
):
    """Interface for defining Environment logic for RL-Agents. """

    def __init__(self, *, renderer: Callable[[State], Any] | None = None):
        """
        Initializes the Environment by optionally providing a renderer.

        The renderer should be separated from the Environment logic itself
        to reduce coupling. The Agent's behaviour should not depend
        on rendered images, this is to provide a visualization for the user.

        Args:
            renderer:
                A callable from States to a visually rendered object of State.
                The visuals are for the user, and not the agent. If renderer
                is None, then calling Environment.render will raise a
                NotImplementedError.
        """
        self._renderer = renderer

    def __str__(self) -> str:
        """Returns a minimal representation of the Environment Structure."""
        return self.__class__.__name__

    def __repr__(self) -> str:
        """Returns a complete informative representation of self.

        Opposed to `str`, the class can be reconstructed from the
        `repr` information.
        """
        return self.__str__()

    @property
    def unwrapped(self) -> Environment[
        State, Action, Observation, RewardT, DiscountT
    ]:
        """Helper function to support unpacking Composite Environments."""
        return self

    @abc.abstractmethod
    def reset(
            self, 
            key: PRNGKeyArray
    ) -> tuple[
        State, TimeStep[Observation, RewardT, DiscountT, Int8[Array, '']]
    ]:
        """Starts a new episode as a functionally pure transformation.

        Args:
            key: Pseudo RNG Key to initialize `State` with.

        Returns:
            A tuple of `State` and `TimeStep` at indices;

            1) The `State` object is expected to carry a chain of `key` values
               in order to internally maintain the random state.
            2) The TimeStep object will have step_type set to `StepType.FIRST`
               and discount cast to a structure of `1.0` values.
        """

    @abc.abstractmethod
    def step(self, state: State, action: Action) -> tuple[
        State, TimeStep[Observation, RewardT, DiscountT, Int8[Array, '']]
    ]:
        """Updates the environment according to the given state and action.

        If the environment already returned a `TimeStep` with `StepType.LAST`
        then subsequent calls to `step` will still execute but remain stuck
        inside the absorbing terminal state.

        Args:
            state:
                A State carried through `step` and created by `reset`.
            action:
                A Jax-compatible data-structure adhering to self.action_spec().

        Returns:
            A tuple of `State` and `TimeStep` at indices;

            1)  The `State` object is expected to carry a chain of `key` values
                in order to internally maintain the random state.
            2)  The TimeStep object will have step_type set to `StepType.MID`
                if the environment continues, otherwise it is `StepType.LAST`.
                Depending on termination the discount will be set to either
                step.discount or zero.
        """

    @abc.abstractmethod
    def reward_spec(self) -> specs.Spec:
        """Describes the reward returned by the environment before it exists.

        Typically, this can be a single float.

        Returns:
            A specs.Spec type indicating the reward data-structure.
        """

    @abc.abstractmethod
    def discount_spec(self) -> specs.Spec:
        """Describes the discount returned by the environment before it exists.

        Typically, this can be a single float between [0.0, 1.0]. It is also
        typical that the spec has the same dimensionality as the rewards, or
        that it can be broadcast along the rewards.

        Returns:
            A specs.Spec type indicating the discount data-structure.
        """

    @abc.abstractmethod
    def observation_spec(self) -> specs.Spec:
        """Describes the observations of this environment before it exists.

        The observation can be a general (composite) data-structure of Jax
        compatible leaf types.

        Returns:
            A specs.Spec type indicating the observation data-structure.
        """

    @abc.abstractmethod
    def action_spec(self) -> specs.Spec:
        """Describes the action modality of this environment before it exists.

        The action can be a general (composite) data-structure of Jax
        compatible leaf types.

        Returns:
            A specs.Spec type indicating the action data-structure.
        """

    def close(self):
        """Frees any resources used by the environment.

        Implement this method for an environment backed by an external process.

        This method can be used directly::

            env = Env(...)
            # Use env.
            env.close()

        or via a context manager::

            with Env(...) as env:
              # Use env.
        """

    def render(self, state: State) -> Any:
        """Generate a pixel-observation based on the given state.

        To support rendering environments, this should be provided within
        the Environment's constructor. Otherwise, calling this function
        will raise a NotImplementedError.

        Args:
            state: A state object to generate a visualization of.

        Raises:
            NotImplementedError:
                If no renderer is provided to Environment.
        """
        if self._renderer is None:
            raise NotImplementedError(  # pragma: no cover
                "Render Function not Implemented"
            )
        return self._renderer(state)

    def __enter__(self) -> Environment:
        """Allows the environment to be used in a with-statement context."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Allows the environment to be used in a with-statement context."""
        del exc_type, exc_value, traceback  # Unused.
        self.close()


class Wrapper(
    Environment[State, Action, Observation, RewardT, DiscountT],
    Generic[State, Action, Observation, RewardT, DiscountT],
    metaclass=abc.ABCMeta
):
    """Interface for Composing Environment logic for RL-Agents. """

    def __init__(
            self,
            env: Environment[State, Action, Observation, RewardT, DiscountT],
            *args, **kwargs
    ):
        """Initializes the Composite Environment with a base Environment.

        The `env` attribute can be accessed for reading out attributes, but it
        should not be modified.

        Args:
            env: The base environment to extend with functionality.

            args: See Environment
            kwargs: See Environment
        """
        super().__init__(*args, **kwargs)
        self._env = env

    def __str__(self) -> str:
        """Returns a recursive composition structure of all Wrappers."""
        return f"{self.__class__.__name__}({str(self.env)})"

    def __repr__(self) -> str:
        """Returns a recursive complete informative representation of self.

        Opposed to `str`, the Wrapped class hierarchy can be
        reconstructed from the `repr` information.
        """
        return f"{self.__class__.__name__}(env={repr(self.env)})"

    @property
    def env(self):
        """Helper function to unpack Composite Environments by one level."""
        return self._env

    @property
    def unwrapped(self) -> Environment[
        State, Action, Observation, RewardT, DiscountT
    ]:
        """Helper function to unpack Composite Environments to the base."""
        return self.env.unwrapped

    def reset(self, key: PRNGKeyArray) -> tuple[
        State, TimeStep[Observation, RewardT, DiscountT, Int8[Array, '']]
    ]:
        return self.env.reset(key)

    def step(self, state: State, action: Action) -> tuple[
        State, TimeStep[Observation, RewardT, DiscountT, Int8[Array, '']]
    ]:
        return self.env.step(state, action)

    def reward_spec(self) -> specs.Spec:
        return self.env.reward_spec()

    def discount_spec(self) -> specs.Spec:
        return self.env.discount_spec()

    def observation_spec(self) -> specs.Spec:
        return self.env.observation_spec()

    def action_spec(self) -> specs.Spec:
        return self.env.action_spec()

    def render(self, state: State) -> Any:
        return self.env.render(state)


# Define helpers to instantiate TimeStep objects at Environment boundaries
# Note that these helpers are not expected to conform to any Spec.

def restart(
        observation: Observation,
        extras: dict | None = None,
        shape: int | Sequence[int] = (),
        dtype: Any = float
) -> TimeStep[
    Observation,
    PyTree[Num[Array, '...']],
    PyTree[Num[Array, '...']],
    Int8[Array, '']
]:
    """Returns a `TimeStep` with `step_type` set to `StepType.FIRST`.

    Unlike dm_env the reward and discount are not `None` to prevent array
    shape inconsistencies when scanning over environments.
    """
    return TimeStep(
        step_type=StepType.FIRST,
        reward=jnp.zeros(shape, dtype=dtype),
        discount=jnp.ones(shape, dtype=float),
        observation=observation,
        extras=extras,
    )


def transition(
        reward: PyTree[Num[Array, '...']],
        observation: Observation,
        discount: PyTree[Num[Array, '...']] | None = None,
        extras: dict | None = None,
        shape: int | Sequence[int] = ()
) -> TimeStep[
    Observation,
    PyTree[Num[Array, '...']],
    PyTree[Num[Array, '...']],
    Int8[Array, '']
]:
    """Returns a `TimeStep` with `step_type` set to `StepType.MID`. """
    return TimeStep(
        step_type=StepType.MID,
        reward=reward,
        discount=(jnp.ones(shape, float) if discount is None else discount),
        observation=observation,
        extras=extras,
    )


def termination(
        reward: PyTree[Num[Array, '...']],
        observation: Observation,
        extras: dict | None = None,
        shape: int | Sequence[int] = ()
) -> TimeStep[
    Observation,
    PyTree[Num[Array, '...']],
    PyTree[Num[Array, '...']],
    Int8[Array, '']
]:
    """Returns a `TimeStep` with `step_type` set to `StepType.LAST`. """
    return TimeStep(
        step_type=StepType.LAST,
        reward=reward,
        discount=jnp.zeros(shape, dtype=float),
        observation=observation,
        extras=extras,
    )


def truncation(
        reward: PyTree[Num[Array, '...']],
        observation: Observation,
        discount: PyTree[Num[Array, '...']] | None = None,
        extras: dict | None = None,
        shape: int | Sequence[int] = ()
) -> TimeStep[
    Observation,
    PyTree[Num[Array, '...']],
    PyTree[Num[Array, '...']],
    Int8[Array, '']
]:
    """Alternative to `termination` that does not set `discount` to zero. """
    return TimeStep(
        step_type=StepType.LAST,
        reward=reward,
        discount=(jnp.ones(shape, float) if discount is None else discount),
        observation=observation,
        extras=extras,
    )
