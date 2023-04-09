import pytest

import jax
from jax import numpy as jnp
from jax import tree_util

import jit_env
from jit_env import wrappers


@pytest.mark.usefixtures('dummy_env')
@pytest.mark.parametrize('num', [1, 2])
def test_tile(dummy_env: jit_env.Environment, num: int):
    state, step = dummy_env.reset(jax.random.PRNGKey(0))

    action = dummy_env.action_spec().generate_value()
    new_state, new_step = dummy_env.step(state, action)

    tiled_env = wrappers.Tile(dummy_env, num=num)
    states, steps = tiled_env.reset(jax.random.PRNGKey(0))

    actions = tiled_env.action_spec().generate_value()
    new_states, new_steps = tiled_env.step(states, actions)

    matching_init_states = tree_util.tree_map(
        lambda a, b: jnp.shape(a) == (num, *jnp.shape(b)),
        states, state
    )
    matching_init_steps = tree_util.tree_map(
        lambda a, b: jnp.shape(a) == (num, *jnp.shape(b)),
        steps, step
    )
    matching_states = tree_util.tree_map(
        lambda a, b: jnp.shape(a) == (num, *jnp.shape(b)),
        new_states, new_state
    )
    matching_steps = tree_util.tree_map(
        lambda a, b: jnp.shape(a) == (num, *jnp.shape(b)),
        new_steps, new_step
    )

    assert all(tree_util.tree_leaves(matching_init_states)), \
        "Mismatching Initial State dimensions!"
    assert all(tree_util.tree_leaves(matching_init_steps)), \
        "Mismatching Initial TimeStep dimensions!"
    assert all(tree_util.tree_leaves(matching_states)), \
        "Mismatching State dimensions!"
    assert all(tree_util.tree_leaves(matching_steps)), \
        "Mismatching TimeStep dimensions!"
