from __future__ import annotations

import jax.random
import jaxtyping
import pytest

import chex

from jax import numpy as jnp
from jax.typing import ArrayLike

import jit_env


@pytest.mark.usefixtures('dummy_env')
def test_environment_staging(dummy_env: jit_env.Environment):
    """Test compatibility with different kinds of jax acceleration options."""

    @chex.all_variants
    def step(
            s: jit_env.State,
            a: jit_env.Action
    ) -> tuple[jit_env.State, jit_env.TimeStep]:
        return dummy_env.step(s, a)

    @chex.all_variants
    def reset(
            key: jax.random.KeyArray
    ) -> tuple[jit_env.State, jit_env.TimeStep]:
        return dummy_env.reset(key)

    action = dummy_env.action_spec().generate_value()

    state, step = dummy_env.reset(jax.random.PRNGKey(0))
    assert hasattr(state, 'key'), "State has no PRNGKey attribute!"
    assert step.first(), "Reset did not set TimeStep.step = StepType.FIRST"

    state, step = dummy_env.step(state, action)
    assert hasattr(state, 'key'), "State has no PRNGKey attribute!"
    assert not step.first(), "Reset did not update TimeStep.step."


@pytest.mark.parametrize(
    'step_a, step_b, expected',
    [
        (jit_env.restart(2.), jit_env.termination(1., 2.), False),
        (jit_env.truncation(1., 2., 0.), jit_env.termination(1., 2.), True)
    ]
)
def test_timestep_comparison(
        step_a: jit_env.TimeStep,
        step_b: jit_env.TimeStep,
        expected: bool
):
    assert (step_a == step_b) == expected


class TestTimeStepHelpers:
    """These tests mostly copy those in `dm_env._test_environment.py`."""

    @pytest.mark.parametrize(
        'observation, shape', [(-1, ()), ([2, 3], (1, 2))]
    )
    def test_restart(
            self,
            observation: ArrayLike,
            shape: tuple[int, ...]
    ):
        step: jit_env.TimeStep = jit_env.restart(observation, shape=shape)

        assert step.first(), \
            "Call to `restart` did not assign StepType.FIRST!"

        assert step.observation == observation, \
            "Call to `restart` did not correctly store `observation`."

        assert jnp.equal(step.reward, jnp.zeros(shape, jnp.float32)).all(), \
            f"Call to `restart` produced incorrect rewards: " \
            f"R: {step.reward} Vs. E: {jnp.zeros(shape, jnp.float32)}"
        assert jnp.equal(step.discount, jnp.ones(shape, jnp.float32)).all(), \
            f"Call to `restart` produced incorrect discounts: " \
            f"R:{step.discount} Vs. E:{jnp.ones(shape, jnp.float32)}"

        assert (jnp.shape(step.reward) == shape), \
            "Call to `restart` produced incorrectly shaped rewards: " \
            f"R: {jnp.shape(step.reward)} Vs. E: {shape}"
        assert (jnp.shape(step.discount) == shape), \
            "Call to `restart` produced incorrectly shaped discounts: " \
            f"R: {jnp.shape(step.discount)} Vs. E: {shape}"

    @pytest.mark.parametrize(
        'reward, observation, discount',
        [(2.0, -1., 1.0), (0., (2., 3.), 0.)]
    )
    def test_transition(
            self,
            observation: ArrayLike,
            reward: ArrayLike,
            discount: ArrayLike
    ):
        step: jit_env.TimeStep = jit_env.transition(
            reward=reward, observation=observation, discount=discount
        )
        assert step.mid(), \
            "Call to `transition` did not assign StepType.MID!"

        assert step.observation == observation, \
            "Call to `transition` did not correctly store `observation`."

        assert jnp.equal(step.reward, jnp.asarray(reward)), \
            f"Call to `transition` produced incorrect rewards: " \
            f"R: {step.reward} Vs. E: {jnp.asarray(reward)}"
        assert jnp.equal(step.discount, jnp.asarray(discount)), \
            f"Call to `transition` produced incorrect discounts: " \
            f"R:{step.discount} Vs. E:{jnp.asarray(discount)}"

        assert (jnp.shape(step.reward) == jnp.shape(reward)), \
            "Call to `transition` produced incorrectly shaped rewards: " \
            f"R: {jnp.shape(step.reward)} Vs. E: {jnp.shape(reward)}"
        assert (jnp.shape(step.discount) == jnp.shape(discount)), \
            "Call to `transition` produced incorrectly shaped discounts: " \
            f"R: {jnp.shape(step.discount)} Vs. E: {jnp.shape(discount)}"

    @pytest.mark.parametrize(
        'reward, observation, shape',
        [(-1., 2.0, ()), (0., (2., 3.), (1, 2))]
    )
    def test_termination(
            self,
            reward: ArrayLike,
            observation: ArrayLike,
            shape: tuple[int, ...]
    ):
        step = jit_env.termination(reward, observation, shape=shape)

        assert step.last(), \
            "Call to `termination` did not assign StepType.LAST!"

        assert step.observation == observation, \
            "Call to `termination` did not correctly store `observation`."

        assert jnp.equal(step.reward, jnp.asarray(reward)).all(), \
            f"Call to `termination` produced incorrect rewards: " \
            f"R: {step.reward} Vs. E: {jnp.asarray(reward)}"
        assert jnp.equal(step.discount, jnp.zeros(shape)).all(), \
            f"Call to `termination` produced incorrect discounts: " \
            f"R:{step.discount} Vs. E:{jnp.zeros(shape)}"

        assert (jnp.shape(step.reward) == jnp.shape(reward)), \
            "Call to `termination` produced incorrectly shaped rewards: " \
            f"R: {jnp.shape(step.reward)} Vs. E: {jnp.shape(reward)}"
        assert (jnp.shape(step.discount) == shape), \
            "Call to `termination` produced incorrectly shaped discounts: " \
            f"R: {jnp.shape(step.discount)} Vs. E: {shape}"

    @pytest.mark.parametrize(
        'reward, observation, discount',
        [(-1., 2.0, 1.0), (0., (2., 3.), 0.)]
    )
    def test_truncation(
            self,
            reward: ArrayLike,
            observation: ArrayLike,
            discount: ArrayLike
    ):
        step = jit_env.truncation(reward, observation, discount)

        assert step.last(), \
            "Call to `truncation` did not assign StepType.LAST!"

        assert step.observation == observation, \
            "Call to `truncation` did not correctly store `observation`."

        assert jnp.equal(step.reward, jnp.asarray(reward)), \
            f"Call to `truncation` produced incorrect rewards: " \
            f"R: {step.reward} Vs. E: {jnp.asarray(reward)}"
        assert jnp.equal(step.discount, jnp.asarray(discount)), \
            f"Call to `truncation` produced incorrect discounts: " \
            f"R:{step.discount} Vs. E:{jnp.asarray(discount)}"

        assert (jnp.shape(step.reward) == jnp.shape(reward)), \
            "Call to `truncation` produced incorrectly shaped rewards: " \
            f"R: {jnp.shape(step.reward)} Vs. E: {jnp.shape(reward)}"
        assert (jnp.shape(step.discount) == jnp.shape(discount)), \
            "Call to `truncation` produced incorrectly shaped discounts: " \
            f"R: {jnp.shape(step.discount)} Vs. E: {jnp.shape(discount)}"

    @pytest.mark.parametrize(
        'step_type, is_first, is_mid, is_last',
        [
            (jit_env.StepType.FIRST, True, False, False),
            (jit_env.StepType.MID, False, True, False),
            (jit_env.StepType.LAST, False, False, True)
        ]
    )
    def test_step_type_helpers(
            self,
            step_type: jaxtyping.Int8[jaxtyping.Array, ''],
            is_first: bool,
            is_mid: bool,
            is_last: bool
    ):
        step = jit_env.TimeStep(
            reward=None, discount=None, observation=None, step_type=step_type
        )

        assert is_first == step.first()
        assert is_mid == step.mid()
        assert is_last == step.last()
