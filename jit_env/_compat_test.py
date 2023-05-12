from __future__ import annotations
from collections import OrderedDict
import typing

import gymnasium as gym

import dm_env

import jaxtyping
import pytest

import chex

from dm_env import specs as dm_specs

import numpy as np

import jax
from jax import numpy as jnp

import jit_env
from jit_env import specs as jit_specs


@pytest.fixture
def to_dm_wrapper(dummy_env: jit_env.Environment):
    out = jit_env.compat.make_deepmind_wrapper()
    assert out is not None, "Cannot run dm_env test with module missing!"
    return out[0]


@pytest.fixture
def to_dm_spec(dummy_env: jit_env.Environment):
    out = jit_env.compat.make_deepmind_wrapper()
    assert out is not None, "Cannot run dm_env test with module missing!"
    return out[1]


@pytest.fixture
def to_gym_wrapper(dummy_env: jit_env.Environment):
    out = jit_env.compat.make_gymnasium_wrapper()
    assert out is not None, "Cannot run gym test with module missing!"
    return out[0]


@pytest.fixture
def to_gym_space(dummy_env: jit_env.Environment):
    out = jit_env.compat.make_gymnasium_wrapper()
    assert out is not None, "Cannot run gym test with module missing!"
    return out[1]


class TestDMEnvConversion:

    @pytest.mark.usefixtures('to_dm_spec')
    @pytest.mark.parametrize(
        'in_spec, dm_spec', [
            (
                    jit_env.specs.Tuple(
                        jit_specs.DiscreteArray(3, name='number'),
                        jit_specs.BoundedArray((), jnp.float32, 0.0, 1.0),
                        jit_specs.Array((), jnp.float32)
                    ),
                    (
                            dm_specs.DiscreteArray(3, name='number'),
                            dm_specs.BoundedArray((), np.float32, 0.0, 1.0),
                            dm_specs.Array((), np.float32)
                    )
            ),
            (
                    jit_env.specs.Dict(
                        a=jit_specs.DiscreteArray(3, name='number'),
                        b=jit_specs.Array((), jnp.float32)
                    ),
                    {
                        'a': dm_specs.DiscreteArray(3, name='number'),
                        'b': dm_specs.Array((), np.float32)
                    }
            ),
            (
                    jit_specs.DiscreteArray(
                        jnp.ones((3,), jnp.int32), name='num'
                    ),
                    dm_specs.BoundedArray(
                        (3,), np.int32, 0, np.ones(3, np.int32), name='num'
                    )
            )
        ]
    )
    def test_spec_conversion(
            self,
            to_dm_spec: typing.Callable[
                [jit_specs.Spec], jaxtyping.PyTree[dm_specs.Array]
            ],
            in_spec: jit_specs.Spec,
            dm_spec: jaxtyping.PyTree[dm_specs.Array]
    ):
        out_spec = to_dm_spec(in_spec)

        _ = jax.tree_map(lambda a, b: type(a) == type(b), out_spec, dm_spec)
        _ = jax.tree_map(lambda a, b: a.name == b.name, out_spec, dm_spec)
        _ = jax.tree_map(lambda a, b: a.shape == b.shape, out_spec, dm_spec)
        _ = jax.tree_map(lambda a, b: a.dtype == b.dtype, out_spec, dm_spec)

        samples = jax.tree_map(lambda s: s.generate_value(), out_spec)
        dm_samples = jax.tree_map(lambda s: s.generate_value(), dm_spec)

        chex.assert_trees_all_equal(samples, dm_samples, ignore_nones=True)
        chex.assert_trees_all_equal_shapes_and_dtypes(
            samples, dm_samples, ignore_nones=True
        )

    @pytest.mark.usefixtures('dummy_env')
    @pytest.mark.usefixtures('to_dm_wrapper')
    def test_wrapper(
            self,
            dummy_env: jit_env.Environment,
            to_dm_wrapper: type
    ):
        my_dm_env = to_dm_wrapper(dummy_env, jax.random.PRNGKey(0))
        env_spec = jit_specs.make_environment_spec(my_dm_env)

        step: dm_env.TimeStep = my_dm_env.reset()

        assert step.first()
        env_spec.observations.validate(step.observation)

        assert step.reward is None
        assert step.discount is None

        action = my_dm_env.action_spec().generate_value()
        env_spec.actions.validate(action)

        step = my_dm_env.step(action)

        assert step.mid()
        env_spec.observations.validate(step.observation)
        env_spec.rewards.validate(step.reward)
        env_spec.discounts.validate(step.discount)

        step = my_dm_env.step(None)

        assert step.last()
        env_spec.observations.validate(step.observation)
        env_spec.rewards.validate(step.reward)
        env_spec.discounts.validate(step.discount)


class TestGymEnvConversion:

    def test_wrong_spec(
            self, to_gym_space: typing.Callable[[jit_specs.Spec], gym.Space]
    ):
        with pytest.raises(NotImplementedError):
            _ = to_gym_space(None)  # type: ignore

    @pytest.mark.usefixtures('to_gym_space')
    @pytest.mark.parametrize(
        'in_spec, gym_space', [
            (
                    jit_env.specs.Tuple(
                        jit_specs.DiscreteArray(3),
                        jit_specs.BoundedArray((), jnp.float32, 0.0, 1.0),
                        jit_specs.Array((), jnp.float32)
                    ),
                    gym.spaces.Tuple((
                            gym.spaces.Discrete(3),
                            gym.spaces.Box(0.0, 1.0, (), np.float32),
                            gym.spaces.Box(
                                float('-inf'), float('inf'), (), np.float32
                            )
                    ))
            ),
            (
                    jit_env.specs.Tree(
                        leaves=[
                            jit_specs.DiscreteArray(3),
                            jit_specs.Array((), jnp.float32)
                        ],
                        structure=jax.tree_util.tree_structure(
                            OrderedDict(a=0, b=0)
                        )
                    ),
                    gym.spaces.Dict({
                        'a': gym.spaces.Discrete(3),
                        'b': gym.spaces.Box(
                            float('-inf'), float('inf'), (), np.float32
                        )
                    })
            ),
            (
                    jit_specs.DiscreteArray(jnp.ones((3,), int)),
                    gym.spaces.MultiDiscrete(np.ones(3, int))
            ),
            (
                    jit_env.specs.Tree(
                        leaves=[
                            jit_specs.Tuple(
                                jit_specs.DiscreteArray(1),
                                jit_env.specs.Tree(
                                    leaves=[
                                        jit_specs.DiscreteArray(2)
                                    ],
                                    structure=jax.tree_util.tree_structure(
                                        OrderedDict(b=0)
                                    ),
                                    name='inner'
                                )
                            )
                        ],
                        structure=jax.tree_util.tree_structure(
                            OrderedDict(a=0)
                        ),
                        name='outer'
                    ),
                    gym.spaces.Dict({
                        'a': gym.spaces.Tuple((
                                gym.spaces.Discrete(1),
                                gym.spaces.Dict({'b': gym.spaces.Discrete(2)})
                        ))
                    })
            )
        ]
    )
    def test_spec_conversion(
            self,
            to_gym_space: typing.Callable[
                [jit_specs.Spec], gym.Space
            ],
            in_spec: jit_specs.Spec,
            gym_space: gym.Space
    ):
        """

        Note that Gymnasium.spaces.Dict implements an collections.OrderedDict
        at version==0.28.0. Until this is changed, this results in us
        opting for using `Tree` to compare to the gym `Dict` over our
        version of `Dict`. We might change this test in the future if this is
        refactored in Gymnasium.

        Another note is that the `dtype` assertions are disabled as Gymnasium
        does not allow free control over all dtype specifications, and `jax`
        disables double precision by default. Since this is more of a
        disparity between numpy and jax, we simply suppress this explicit
        check and only validate if we can validly convert between the two.
        """
        out_space = to_gym_space(in_spec)

        _ = jax.tree_map(lambda a, b: type(a) == type(b), out_space, gym_space)
        _ = jax.tree_map(lambda a, b: a.shape == b.shape, out_space, gym_space)
        _ = jax.tree_map(lambda a, b: a.dtype == b.dtype, out_space, gym_space)

        samples = in_spec.generate_value()
        gym_converted = out_space.sample()
        gym_samples = gym_space.sample()

        # Numpy uses 64-bits for default precision, jax uses 32-bits.
        chex.assert_trees_all_equal_shapes(  # Differs in Type Accuracy
            samples, gym_converted, ignore_nones=True
        )
        chex.assert_trees_all_equal_shapes(  # Differs in Type Accuracy
            samples, gym_samples, ignore_nones=True
        )
        # Note: The following line is important to test, but the current API
        # of gymnasium does not allow setting explicit dtypes for `Discrete`.
        # TODO: Comment out following line when dtype API compatible.
        # chex.assert_trees_all_equal_shapes_and_dtypes(
        #     gym_samples, gym_converted, ignore_nones=True
        # )

        # Check if dtype can be accurately promoted/ demoted/ converted.
        chex.assert_trees_all_equal_comparator(
            lambda a, b: np.can_cast(a.dtype, b.dtype, casting='same_kind'),
            lambda a, b: f'DType conversion of {a.dtype} does '
                         f'not return a subdtype of {b.dtype}',
            samples, gym_samples,
            ignore_nones=True
        )

    @pytest.mark.usefixtures('dummy_env')
    @pytest.mark.usefixtures('to_gym_wrapper')
    def test_wrapper(
            self,
            dummy_env: jit_env.Environment,
            to_gym_wrapper: type
    ):
        my_gym_env: gym.Env = to_gym_wrapper(dummy_env, 0)

        env_spec = jit_specs.make_environment_spec(dummy_env)

        env_spec.actions.validate(my_gym_env.action_space.sample())
        env_spec.observations.validate(my_gym_env.observation_space.sample())

        observation, info = my_gym_env.reset(seed=123)
        env_spec.observations.validate(observation)

        action = my_gym_env.action_space.sample() * 0.0
        env_spec.actions.validate(action)

        observation, reward, last, stopped, info = my_gym_env.step(action)

        assert not last
        assert not stopped

        env_spec.observations.validate(observation)
        env_spec.rewards.validate(reward)

        observation, reward, last, stopped, info = my_gym_env.step(None)

        assert last

        env_spec.observations.validate(observation)
        env_spec.rewards.validate(reward)
