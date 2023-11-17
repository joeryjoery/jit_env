from __future__ import annotations

import jax

import jit_env
import counting_env


def test_env_integration(max_value: int = 5):
    env = counting_env.CountingEnv(max_value)

    spec = jit_env.specs.make_environment_spec(env)

    state, step = env.reset(jax.random.key(0))

    assert step.first()
    spec.observations.validate(step.observation)
    spec.rewards.validate(step.reward)
    spec.discounts.validate(step.discount)

    action = env.action_spec().generate_value()
    state, step = env.step(state, action)

    assert step.mid()
    spec.actions.validate(action)
    spec.observations.validate(step.observation)
    spec.rewards.validate(step.reward)
    spec.discounts.validate(step.discount)

    action = jax.numpy.ones_like(action) * (max_value - 1)

    # Repeat max-value action 2 times to stay within action bounds.
    state, step = env.step(state, action)
    state, step = env.step(state, action)

    assert step.last()
    spec.actions.validate(action)
    spec.observations.validate(step.observation)
    spec.rewards.validate(step.reward)
    spec.discounts.validate(step.discount)
