# TODO: Update Documentation
from __future__ import annotations
import abc

from typing import Any, TYPE_CHECKING, TypeVar, Generic, Sequence, Protocol

if TYPE_CHECKING:  # https://github.com/python/mypy/issues/6239
    from dataclasses import dataclass
else:
    from chex import dataclass

from jaxtyping import Bool, Array, PyTree, Num, Int8

import jax
import jax.numpy as jnp

from jit_env import specs


# Define Environment IO Types.

class StateProtocol(Protocol):
    key: jax.random.KeyArray


# The following should all be valid Jax types
Observation = TypeVar("Observation")
State = TypeVar("State", bound="StateProtocol")
Action = TypeVar("Action")


class StepType(jnp.int8):
    """Defines the status of a `TimeStep` within a sequence."""
    FIRST = jnp.array(0, jnp.int8)
    MID = jnp.array(1, jnp.int8)
    LAST = jnp.array(2, jnp.int8)


@dataclass
class TimeStep(Generic[Observation]):
    step_type: Int8[Array, '']
    reward: PyTree[Num[Array, '...']]
    discount: PyTree[Num[Array, '...']]
    observation: Observation
    extras: dict | None = None

    def first(self) -> Bool[Array, '']:
        return self.step_type == StepType.FIRST

    def mid(self) -> Bool[Array, '']:
        return self.step_type == StepType.MID

    def last(self) -> Bool[Array, '']:
        return self.step_type == StepType.LAST


# Define Environment and Wrapper

class Environment(metaclass=abc.ABCMeta):
    """Abstract base class for Python RL environments.
    Observations and valid actions are described with `Array` specs, defined in
    the `specs` module.
    """

    def __str__(self) -> str:
        return self.__class__.__name__

    @property
    def unwrapped(self) -> Environment:
        return self

    @abc.abstractmethod
    def reset(self, key: jax.random.KeyArray) -> tuple[State, TimeStep]:
        """Starts a new sequence and returns the first `TimeStep` of this sequence.
        Returns:
          A `TimeStep` namedtuple containing:
            step_type: A `StepType` of `FIRST`.
            reward: `None`, indicating the reward is undefined.
            discount: `None`, indicating the discount is undefined.
            observation: A NumPy array, or a nested dict, list or tuple of arrays.
              Scalar values that can be cast to NumPy arrays (e.g. Python floats)
              are also valid in place of a scalar array. Must conform to the
              specification returned by `observation_spec()`.
        """

    @abc.abstractmethod
    def step(self, state: State, action: Action) -> tuple[State, TimeStep]:
        """Updates the environment according to the action and returns a `TimeStep`.
        If the environment returned a `TimeStep` with `StepType.LAST` at the
        previous step, this call to `step` will start a new sequence and `action`
        will be ignored.
        This method will also start a new sequence if called after the environment
        has been constructed and `reset` has not been called. Again, in this case
        `action` will be ignored.
        Args:
          state: A State
          action: A NumPy array, or a nested dict, list or tuple of arrays
            corresponding to `action_spec()`.
        Returns:
          A `TimeStep` namedtuple containing:
            step_type: A `StepType` value.
            reward: Reward at this timestep, or None if step_type is
              `StepType.FIRST`. Must conform to the specification returned by
              `reward_spec()`.
            discount: A discount in the range [0, 1], or None if step_type is
              `StepType.FIRST`. Must conform to the specification returned by
              `discount_spec()`.
            observation: A NumPy array, or a nested dict, list or tuple of arrays.
              Scalar values that can be cast to NumPy arrays (e.g. Python floats)
              are also valid in place of a scalar array. Must conform to the
              specification returned by `observation_spec()`.
        """

    @abc.abstractmethod
    def reward_spec(self) -> specs.Spec:
        """Describes the reward returned by the environment.
        Typically, this can be a single float.
        Returns:
          An `Array` spec, or a nested dict, list or tuple of `Array` specs.
        """

    @abc.abstractmethod
    def discount_spec(self) -> specs.Spec:
        """Describes the discount returned by the environment.
        Typically, this can be a single float between 0 and 1.
        Returns:
          An `Array` spec, or a nested dict, list or tuple of `Array` specs.
        """

    @abc.abstractmethod
    def observation_spec(self) -> specs.Spec:
        """Defines the observations provided by the environment.
        May use a subclass of `specs.Array` that specifies additional properties
        such as min and max bounds on the values.
        Returns:
          An `Array` spec, or a nested dict, list or tuple of `Array` specs.
        """

    @abc.abstractmethod
    def action_spec(self) -> specs.Spec:
        """Defines the actions that should be provided to `step`.
        May use a subclass of `specs.Array` that specifies additional properties
        such as min and max bounds on the values.
        Returns:
          An `Array` spec, or a nested dict, list or tuple of `Array` specs.
        """

    def close(self):
        """Frees any resources used by the environment.
        Implement this method for an environment backed by an external process.
        This method can be used directly
        ```python
        env = Env(...)
        # Use env.
        env.close()
        ```
        or via a context manager
        ```python
        with Env(...) as env:
          # Use env.
        ```
        """
        pass

    def render(self, state: State) -> Any:
        """Generate a pixel-observation based on the given state. """
        raise NotImplementedError("Render Function not Implemented")

    def __enter__(self):
        """Allows the environment to be used in a with-statement context."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Allows the environment to be used in a with-statement context."""
        del exc_type, exc_value, traceback  # Unused.
        self.close()


class Wrapper(Environment, Generic[State], metaclass=abc.ABCMeta):

    def __init__(self, env: Environment):
        super().__init__()
        self.env = env

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({str(self.env)})"

    def __getattr__(self, item: str) -> Any:
        if item == "__setstate__":
            raise AttributeError(item)
        return getattr(self.env, item)

    @property
    def unwrapped(self) -> Environment:
        return self.env.unwrapped

    def reset(self, key: jax.random.KeyArray) -> tuple[State, TimeStep]:
        return self.env.reset(key)

    def step(self, state: State, action: Action) -> tuple[State, TimeStep]:
        return self.env.step(state, action)

    def reward_spec(self) -> specs.Spec:
        return self.env.reward_spec()

    def discount_spec(self) -> specs.Spec:
        return self.env.discount_spec()

    def observation_spec(self) -> specs.Spec:
        return self.env.observation_spec()

    def action_spec(self) -> specs.Spec:
        return self.env.action_spec()


# Define helpers to instantiate TimeStep objects at Environment boundaries

def restart(
    observation: Observation,
    extras: dict | None = None,
    shape: int | Sequence[int] = ()
) -> TimeStep:
    """Returns a `TimeStep` with `step_type` set to `StepType.FIRST`. """
    return TimeStep(
        step_type=StepType.FIRST,
        reward=jnp.zeros(shape, dtype=float),
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
) -> TimeStep:
    """Returns a `TimeStep` with `step_type` set to `StepType.MID`. """
    discount = jnp.ones(shape, jnp.float32) if discount is None else discount
    return TimeStep(
        step_type=StepType.MID,
        reward=reward,
        discount=discount,
        observation=observation,
        extras=extras,
    )


def termination(
    reward: PyTree[Num[Array, '...']],
    observation: Observation,
    extras: dict | None = None,
    shape: int | Sequence[int] = ()
) -> TimeStep:
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
) -> TimeStep:
    """Alternative to `termination` that does not set `discount` to zero. """
    discount = jnp.ones(shape, jnp.float32) if discount is None else discount
    return TimeStep(
        step_type=StepType.LAST,
        reward=reward,
        discount=discount,
        observation=observation,
        extras=extras,
    )
