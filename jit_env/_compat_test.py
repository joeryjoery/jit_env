from __future__ import annotations
import typing

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
