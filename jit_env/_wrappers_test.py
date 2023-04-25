from dataclasses import replace

import pytest

import chex

import jax
from jax import numpy as jnp

import jit_env
from jit_env import wrappers, specs


class TestRepr:

    def test_empty(self, dummy_env: jit_env.Environment):
        wrapped: jit_env.Wrapper = jit_env.Wrapper(dummy_env)

        assert str(wrapped) == f'{wrapped.__class__.__name__}(' \
                               f'{dummy_env.__class__.__name__})'
        assert repr(wrapped) == f'{wrapped.__class__.__name__}(' \
                                f'env={dummy_env.__class__.__name__})'

    @pytest.mark.usefixtures('dummy_env')
    def test_jit(self, dummy_env: jit_env.Environment):
        wrapped = wrappers.Jit(dummy_env)

        assert str(wrapped) == f'{wrapped.__class__.__name__}(' \
                               f'{dummy_env.__class__.__name__})'
        assert repr(wrapped) == f'{wrapped.__class__.__name__}(' \
                                f'env={dummy_env.__class__.__name__},' \
                                f'jit_kwargs={{}})'

    @pytest.mark.usefixtures('dummy_env')
    def test_vmap(self, dummy_env: jit_env.Environment):
        wrapped = wrappers.Vmap(dummy_env)

        assert str(wrapped) == f'{wrapped.__class__.__name__}(' \
                               f'{dummy_env.__class__.__name__})'
        assert repr(wrapped) == f'{wrapped.__class__.__name__}(' \
                                f'env={dummy_env.__class__.__name__},' \
                                f'step_vmap_kwargs={{}})'

    @pytest.mark.usefixtures('dummy_env')
    def test_tile(self, dummy_env: jit_env.Environment, num: int = 2):
        wrapped = wrappers.Tile(dummy_env, num=num, in_axes=(0, None))

        assert str(wrapped) == f'{wrapped.__class__.__name__}(' \
                               f'{dummy_env.__class__.__name__})'
        assert repr(wrapped) == f'{wrapped.__class__.__name__}(' \
                                f'env={dummy_env.__class__.__name__},' \
                                f'num={num},' \
                                f'step_vmap_kwargs={{\'in_axes\': (0, None)}})'


def test_unwrap(dummy_env: jit_env.Environment):
    wrapped = wrappers.Jit(dummy_env)
    doubly_wrapped = wrappers.Vmap(wrapped)

    assert wrapped.unwrapped is dummy_env
    assert doubly_wrapped.unwrapped is dummy_env
    assert dummy_env.unwrapped is dummy_env

    assert wrapped.env is dummy_env
    assert doubly_wrapped.env is wrapped


@pytest.mark.usefixtures('dummy_env')
def test_jit(
        dummy_env: jit_env.Environment[
            jit_env.State, jax.Array, jit_env.Observation
        ]
):
    jitted = wrappers.Jit(dummy_env)

    # For type checker
    jit_state: jit_env.State

    # Reset logic
    state, step = dummy_env.reset(jax.random.PRNGKey(0))
    jit_state, jit_step = jitted.reset(jax.random.PRNGKey(0))

    chex.assert_trees_all_equal(state, jit_state, ignore_nones=True)
    chex.assert_trees_all_equal(step, jit_step, ignore_nones=True)
    chex.assert_trees_all_equal_shapes_and_dtypes(
        state, jit_state, ignore_nones=True
    )
    chex.assert_trees_all_equal_shapes_and_dtypes(
        (step,), (jit_step,), ignore_nones=True
    )

    # Step logic
    state, step = dummy_env.step(state, jnp.zeros(()))
    jit_state, jit_step = jitted.step(jit_state, jnp.zeros(()))

    chex.assert_trees_all_equal(state, jit_state, ignore_nones=True)
    chex.assert_trees_all_equal(step, jit_step, ignore_nones=True)
    chex.assert_trees_all_equal_shapes_and_dtypes(
        state, jit_state, ignore_nones=True
    )
    chex.assert_trees_all_equal_shapes_and_dtypes(
        (step,), (jit_step,), ignore_nones=True
    )


@pytest.mark.usefixtures('dummy_env')
def test_stopgrad(dummy_env: jit_env.Environment):
    class ScaleRewardWrapper(jit_env.Wrapper):

        def step(
                self,
                s: jit_env.State,
                a: jit_env.Action
        ) -> tuple[jit_env.State, jit_env.TimeStep]:
            s, t = super().step(s, a)
            # y = constant * x + x
            # dy/dx = constant + 1
            t = replace(t, reward=(t.reward * a + a))
            return s, t

    scale_reward = ScaleRewardWrapper(dummy_env)
    stopped = wrappers.StopGradient(scale_reward)

    jacfun = jax.jacrev(
        lambda *a: scale_reward.step(*a)[1].reward,
        argnums=1
    )
    stop_jacfun = jax.jacrev(
        lambda *a: stopped.step(*a)[1].reward,
        argnums=1
    )

    action = jnp.ones_like(dummy_env.action_spec().generate_value())
    action = action * 10.0

    state, _ = dummy_env.reset(jax.random.PRNGKey(0))
    _, reference_step = dummy_env.step(state, action)

    jac = jacfun(state, action)
    jac_zero = stop_jacfun(state, action)

    # Computation:
    # y = constant * x + x
    # dy/dx = constant + 1
    assert (jac == (reference_step.reward + 1.0)).all()
    assert (jac_zero == 0.0).all()


@pytest.mark.usefixtures('dummy_env')
def test_autoreset(dummy_env: jit_env.Environment):
    env = wrappers.AutoReset(dummy_env)

    state, step = env.reset(jax.random.PRNGKey(0))
    ref_state, ref_step = dummy_env.reset(jax.random.PRNGKey(0))

    assert step.first()
    assert ref_step.first()

    chex.assert_trees_all_equal(step, ref_step, ignore_nones=True)
    chex.assert_trees_all_equal_shapes_and_dtypes(
        (step,), (ref_step,), ignore_nones=True
    )

    for _ in range(5):
        state, step = env.step(state, 0.0)
        ref_state, ref_step = dummy_env.step(ref_state, 0.0)

        assert step.mid()
        assert ref_step.mid()

        chex.assert_trees_all_equal(step, ref_step, ignore_nones=True)
        chex.assert_trees_all_equal_shapes_and_dtypes(
            (step,), (ref_step,), ignore_nones=True
        )

    state, step = env.step(state, None)
    ref_state, ref_step = dummy_env.step(ref_state, None)

    assert step.first()
    assert ref_step.last()

    assert jnp.all(step.reward == 1.0)
    assert jnp.all(step.reward == ref_step.reward)

    assert jnp.all(step.discount == 0.0)
    assert jnp.all(step.discount == ref_step.discount)

    # AutoReset resets observation to zero, ref_step has 1.0 from `step`.
    assert jnp.all(step.observation == 0.0)
    assert jnp.all(step.observation != ref_step.observation)


class TestVmap:

    @pytest.mark.usefixtures('dummy_env')
    def test_env(
            self,
            dummy_env: jit_env.Environment[
                jit_env.State, jax.Array, jit_env.Observation
            ],
            batch_size: int = 5
    ):
        key = jax.random.PRNGKey(0)
        batched = wrappers.Vmap(dummy_env)

        # For type checker
        states: jit_env.State

        # Reset logic
        state, step = dummy_env.reset(key)
        states, steps = batched.reset(jax.random.split(key, num=batch_size))

        chex.assert_tree_shape_prefix(
            (states, steps), (batch_size,), ignore_nones=True
        )

        sliced = jax.tree_map(lambda x: x.at[0].get(), (states, steps))
        chex.assert_trees_all_equal_shapes_and_dtypes(
            sliced, (state, step), ignore_nones=True
        )

        # Step logic
        state, step = dummy_env.step(state, jnp.zeros(()))
        states, steps = batched.step(states, jnp.zeros((batch_size,)))

        chex.assert_tree_shape_prefix(
            (states, steps), (batch_size,), ignore_nones=True
        )

        sliced = jax.tree_map(lambda x: x.at[0].get(), (states, steps))
        chex.assert_trees_all_equal_shapes_and_dtypes(
            sliced, (state, step), ignore_nones=True
        )

    @pytest.mark.usefixtures('dummy_env')
    def test_render(
            self,
            dummy_env: jit_env.Environment[
                jit_env.State, jit_env.Action, jit_env.Observation
            ],
            batch_size: int = 5
    ):
        key = jax.random.PRNGKey(0)
        batched = wrappers.Vmap(dummy_env)

        # For type checker
        states: jit_env.State

        state, _ = dummy_env.reset(key)
        states, _ = batched.reset(jax.random.split(key, num=batch_size))

        single_render = dummy_env.render(state)
        batch_render = batched.render(states)

        chex.assert_trees_all_equal(
            single_render, batch_render, ignore_nones=True
        )
        chex.assert_trees_all_equal_shapes_and_dtypes(
            single_render, batch_render, ignore_nones=True
        )

    @pytest.mark.usefixtures('dummy_env')
    def test_wrongly_wrapped_autoreset(
            self,
            dummy_env: jit_env.Environment[
                jit_env.State, jit_env.Action, jit_env.Observation
            ]
    ):
        vmap_first = wrappers.AutoReset(wrappers.Vmap(dummy_env))
        vmap_last = wrappers.Vmap(wrappers.AutoReset(dummy_env))

        keys = jax.random.split(jax.random.PRNGKey(0), num=5)
        first, _ = vmap_first.reset(keys)
        last: jit_env.State = vmap_last.reset(keys)[0]

        with pytest.raises(TypeError):
            # lax.cond in AutoReset will receive incompatible Array of bools
            vmap_first.step(first, jnp.zeros((5,)))

        # Array of bools can be handled per element by vmapping last.
        vmap_last.step(last, jnp.zeros((5,)))

    @pytest.mark.usefixtures('dummy_env')
    def test_autoreset(
            self,
            dummy_env: jit_env.Environment[
                jit_env.State, jit_env.Action, jit_env.Observation
            ],
            num: int = 2
    ):
        # Doubly or single-wrapping should both work.
        # Singly is preferred as it is faster, this is not explicitly tested.
        doubly_wrapped = wrappers.Vmap(
            wrappers.AutoReset(dummy_env), in_axes=(0, None)
        )
        singly_wrapped = wrappers.VmapAutoReset(dummy_env, in_axes=(0, None))
        keys = jax.random.split(jax.random.PRNGKey(0), num=num)

        # For type checker
        doubly_out: tuple[jit_env.State, jit_env.TimeStep]
        singly_out: tuple[jit_env.State, jit_env.TimeStep]

        doubly_out = doubly_wrapped.reset(keys)
        singly_out = singly_wrapped.reset(keys)

        chex.assert_trees_all_equal(doubly_out, singly_out, ignore_nones=True)
        chex.assert_trees_all_equal_shapes_and_dtypes(
            singly_out, doubly_out, ignore_nones=True
        )

        a = jnp.zeros(())  # Action is held constant across batch
        for _ in range(5):
            doubly_out = doubly_wrapped.step(doubly_out[0], a)
            singly_out = singly_wrapped.step(singly_out[0], a)

            chex.assert_trees_all_equal(
                doubly_out, singly_out, ignore_nones=True
            )
            chex.assert_trees_all_equal_shapes_and_dtypes(
                singly_out, doubly_out, ignore_nones=True
            )

        doubly_out = doubly_wrapped.step(doubly_out[0], None)
        singly_out = singly_wrapped.step(singly_out[0], None)

        chex.assert_trees_all_equal(
            doubly_out, singly_out, ignore_nones=True
        )
        chex.assert_trees_all_equal_shapes_and_dtypes(
            singly_out, doubly_out, ignore_nones=True
        )


class TestTile:

    @pytest.mark.usefixtures('dummy_env')
    def test_empty(self, dummy_env: jit_env.Environment):
        with pytest.raises(ValueError):
            wrappers.Tile(dummy_env, 0)

    @pytest.mark.usefixtures('dummy_env')
    def test_spec(self, dummy_env: jit_env.Environment, num: int = 2):
        tiled_env = wrappers.Tile(dummy_env, num=num)

        spec = specs.make_environment_spec(dummy_env)
        batch_spec = specs.make_environment_spec(tiled_env)

        samples = jax.tree_map(lambda s: s.generate_value(), spec)
        batch = jax.tree_map(lambda s: s.generate_value(), batch_spec)

        chex.assert_tree_shape_prefix(batch, (num,), ignore_nones=True)

        for i in range(num):
            sliced = jax.tree_map(lambda x: x.at[i].get(), batch)
            chex.assert_trees_all_equal_shapes_and_dtypes(
                sliced, samples, ignore_nones=True
            )

    @pytest.mark.usefixtures('dummy_env')
    def test_modified_spec(self, dummy_env: jit_env.Environment, num: int = 2):
        # Test to cover all behaviour of reward_spec and discount_spec.
        env_spec = specs.make_environment_spec(dummy_env)

        dummy_env.reward_spec = lambda *_: specs.Tuple(  # type: ignore
            env_spec.rewards
        )
        dummy_env.discount_spec = lambda *_: specs.Tuple(  # type: ignore
            env_spec.discounts
        )

        tiled = wrappers.Tile(dummy_env, num=num)
        batch_spec = specs.make_environment_spec(tiled)

        assert isinstance(batch_spec.rewards, specs.Batched)
        assert isinstance(batch_spec.discounts, specs.Batched)

        assert batch_spec.rewards.num == num
        assert batch_spec.discounts.num == num

    @pytest.mark.usefixtures('dummy_env')
    def test_tile(self, dummy_env: jit_env.Environment, num: int = 2):
        # Test distinguishing feature of Tile Vs. Vmap.
        tiled_env = wrappers.Tile(dummy_env, num=num)
        states, steps = tiled_env.reset(jax.random.PRNGKey(0))  # type: ignore

        chex.assert_tree_shape_prefix(
            (states, steps), (num,), ignore_nones=True
        )

    @pytest.mark.usefixtures('dummy_env')
    def test_autoreset(
            self,
            dummy_env: jit_env.Environment[
                jit_env.State, jit_env.Action, jit_env.Observation
            ],
            num: int = 2
    ):
        # Tile and Vmap are equivalent aside from the `reset` call.
        vmapped = wrappers.VmapAutoReset(dummy_env, in_axes=(0, None))
        tiled = wrappers.TileAutoReset(dummy_env, num=num, in_axes=(0, None))

        keys = jax.random.split(jax.random.PRNGKey(0), num=num)

        # For type checker
        vmap_out: tuple[jit_env.State, jit_env.TimeStep]
        tile_out: tuple[jit_env.State, jit_env.TimeStep]

        vmap_out = vmapped.reset(keys)
        tile_out = tiled.reset(jax.random.PRNGKey(0))

        chex.assert_trees_all_equal(vmap_out, tile_out, ignore_nones=True)
        chex.assert_trees_all_equal_shapes_and_dtypes(
            tile_out, vmap_out, ignore_nones=True
        )

        a = jnp.zeros(())  # Action is held constant across batch
        for _ in range(5):
            vmap_out = vmapped.step(vmap_out[0], a)
            tile_out = tiled.step(tile_out[0], a)

            chex.assert_trees_all_equal(
                vmap_out, tile_out, ignore_nones=True
            )
            chex.assert_trees_all_equal_shapes_and_dtypes(
                tile_out, vmap_out, ignore_nones=True
            )

        vmap_out = vmapped.step(vmap_out[0], None)
        tile_out = tiled.step(tile_out[0], None)

        chex.assert_trees_all_equal(
            vmap_out, tile_out, ignore_nones=True
        )
        chex.assert_trees_all_equal_shapes_and_dtypes(
            tile_out, vmap_out, ignore_nones=True
        )
