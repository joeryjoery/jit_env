from __future__ import annotations
import pytest
import typing
import pickle

import chex

import jax
from jax import numpy as jnp

from jit_env import specs

INT_SIZE: int = 3
BATCH_SIZE: int = 2

VEC_MIN: float = 0.0
VEC_MAX: float = 1.0

SCALAR_SHAPE: tuple = ()
INT_SHAPE: tuple = ()
VECTOR_SHAPE: tuple[int] = (2,)
MATRIX_SHAPE: tuple[int, int] = (2, 2)


class DummyStruct(typing.NamedTuple):
    my_spec: specs.Spec
    other_spec: specs.Spec


@pytest.fixture
def scalar_spec() -> specs.Array:
    return specs.Array(SCALAR_SHAPE, dtype=jnp.float32, name='scalar')


@pytest.fixture
def matrix_spec() -> specs.Array:
    return specs.Array(MATRIX_SHAPE, dtype=jnp.float32, name='matrix')


@pytest.fixture
def bounded_vector_spec() -> specs.BoundedArray:
    return specs.BoundedArray(
        VECTOR_SHAPE, dtype=jnp.float32,
        minimum=VEC_MIN, maximum=VEC_MAX, name='vector'
    )


@pytest.fixture
def int_spec() -> specs.DiscreteArray:
    return specs.DiscreteArray(INT_SIZE, name='int')


@pytest.fixture
def tuple_spec(bounded_vector_spec, matrix_spec) -> specs.Tuple:
    return specs.Tuple(bounded_vector_spec, matrix_spec, name='tuple')


@pytest.fixture
def dict_spec(int_spec, tuple_spec) -> specs.Dict:
    return specs.Dict(my_int=int_spec, my_tuple=tuple_spec, name='dict')


@pytest.fixture
def batch_spec(dict_spec) -> specs.Batched:
    return specs.Batched(dict_spec, num=BATCH_SIZE, name='batched')


@pytest.fixture
def tree_spec(batch_spec, scalar_spec) -> specs.Tree:
    dummy = DummyStruct(my_spec=batch_spec, other_spec=scalar_spec)
    return specs.Tree(*jax.tree_util.tree_flatten(dummy), name='tree')


def test_illegal_array(matrix_spec):
    class VargsDummy(specs.Array):
        def __init__(self, *args):
            super().__init__(*args)

    class KwargsDummy(specs.Array):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

    v_arr = VargsDummy((), jnp.int32)
    k_arr = KwargsDummy(shape=(), dtype=jnp.int32)

    with pytest.raises(TypeError):
        v_arr.replace(shape=(1,), dtype=jnp.float32)

    with pytest.raises(TypeError):
        k_arr.replace(shape=(1,), dtype=jnp.float32)


@pytest.mark.usefixtures('dummy_env')
def test_environment_spec(dummy_env):
    env_spec = specs.make_environment_spec(dummy_env)

    r = dummy_env.reward_spec()
    o = dummy_env.observation_spec()
    d = dummy_env.discount_spec()
    a = dummy_env.action_spec()

    def check_spec(spec, reference):
        assert spec.shape == reference.shape
        assert spec.dtype == reference.dtype
        assert spec.name == reference.name

    _ = jax.tree_map(check_spec, env_spec.rewards, r)
    _ = jax.tree_map(check_spec, env_spec.observations, o)
    _ = jax.tree_map(check_spec, env_spec.discounts, d)
    _ = jax.tree_map(check_spec, env_spec.actions, a)


class TestTree:

    def test_serialize(self, tree_spec):
        reconstructed = pickle.loads(pickle.dumps(tree_spec))

        assert isinstance(reconstructed, tree_spec.__class__)

        chex.assert_trees_all_equal(
            tree_spec.generate_value(),
            reconstructed.generate_value()
        )
        chex.assert_trees_all_equal_shapes_and_dtypes(
            tree_spec.generate_value(),
            reconstructed.generate_value()
        )

        def check_leaf(leaf_a, leaf_b):
            assert isinstance(leaf_b, leaf_a.__class__)

            assert leaf_a.dtype == leaf_b.dtype
            assert leaf_a.shape == leaf_b.shape
            assert leaf_a.name == leaf_b.name

        spec_tree = specs.unpack_spec(tree_spec)
        reconstructed_tree = pickle.loads(pickle.dumps(spec_tree))

        _ = jax.tree_map(check_leaf, spec_tree, reconstructed_tree)

    def test_validate(self, tree_spec: specs.Tree):
        out = tree_spec.generate_value()

        tree_spec.validate(out)
        with pytest.raises(TypeError):
            # Wrong Pytree structure (batch, batch) should be (batch, scalar)
            # Error should be raise internally by `jnp.asarray(value)`.
            tree_spec.validate(
                DummyStruct(my_spec=out.my_spec, other_spec=out.my_spec)
            )

    def test_substructs(self, tree_spec: specs.Tree):
        out = tree_spec.generate_value()

        spec_struct = tree_spec.as_spec_struct()

        spec_struct.my_spec.validate(out.my_spec)
        spec_struct.other_spec.validate(out.other_spec)

        out_sub = jax.tree_map(lambda s: s.generate_value(), spec_struct)

        chex.assert_trees_all_equal(out, out_sub)
        chex.assert_trees_all_equal_shapes_and_dtypes(out, out_sub)

    def test_tree_map(self, tree_spec: specs.Tree):
        out = tree_spec.generate_value()

        out_map = jax.tree_map(lambda s: s.generate_value(), tree_spec)
        out_map = out_map.as_spec_struct()

        chex.assert_trees_all_equal(out, out_map)
        chex.assert_trees_all_equal_shapes_and_dtypes(out, out_map)

    def test_repr(self, int_spec):
        my_tuple = specs.Tuple(int_spec, int_spec, name='tuple')
        r = repr(my_tuple)
        s = f'{specs.Tuple.__name__}(name={my_tuple.name},' \
            f'tree=(\'{repr(int_spec)}\', \'{repr(int_spec)}\'))'

        assert r == s

    def test_empty_dict(self):
        with pytest.raises(ValueError):
            _ = specs.Dict()

    def test_empty_tuple(self):
        with pytest.raises(ValueError):
            _ = specs.Tuple()

    def test_replace(self, tree_spec):
        with pytest.raises(RuntimeError):
            tree_spec.replace(key_does_not_matter=None)


class TestBatched:

    def test_serialize(self, batch_spec):
        reconstructed = pickle.loads(pickle.dumps(batch_spec))

        assert isinstance(reconstructed, batch_spec.__class__)

        chex.assert_trees_all_equal(
            batch_spec.generate_value(),
            reconstructed.generate_value()
        )
        chex.assert_trees_all_equal_shapes_and_dtypes(
            batch_spec.generate_value(),
            reconstructed.generate_value()
        )

        def check_leaf(leaf_a, leaf_b):
            assert isinstance(leaf_b, leaf_a.__class__)

            assert leaf_a.dtype == leaf_b.dtype
            assert leaf_a.shape == leaf_b.shape
            assert leaf_a.name == leaf_b.name

        spec_tree = specs.unpack_spec(batch_spec)
        reconstructed_tree = pickle.loads(pickle.dumps(spec_tree))

        _ = jax.tree_map(check_leaf, spec_tree, reconstructed_tree)

    def test_validate(self, batch_spec: specs.Batched):
        out_dict = batch_spec.generate_value()

        # Should run fine
        _ = batch_spec.validate(out_dict)

        with pytest.raises(ValueError):
            wrongly_sized = jnp.arange(BATCH_SIZE - 1)
            _ = batch_spec.validate(out_dict | {'my_int': wrongly_sized})

    def test_shapes(self, batch_spec: specs.Batched):
        out_dict = batch_spec.generate_value()
        my_ints, (my_vec, my_mat) = out_dict['my_int'], out_dict['my_tuple']

        assert jnp.issubdtype(my_ints.dtype, jnp.integer)

        assert my_ints.shape == (BATCH_SIZE, *INT_SHAPE)
        assert my_vec.shape == (BATCH_SIZE, *VECTOR_SHAPE)
        assert my_mat.shape == (BATCH_SIZE, *MATRIX_SHAPE)

    def test_empty_batch(self, int_spec):
        with pytest.raises(ValueError):
            _ = specs.Batched(int_spec, 0)

    def test_struct(self, batch_spec: specs.Batched):
        out_dict = batch_spec.generate_value()

        batched_specs = batch_spec.as_spec_struct()
        new_dict = jax.tree_map(
            lambda s: s.generate_value(), batched_specs
        )

        chex.assert_trees_all_equal(out_dict, new_dict)
        chex.assert_trees_all_equal_shapes_and_dtypes(out_dict, new_dict)


class TestArray:

    def test_serialize(self, scalar_spec):
        reconstructed = pickle.loads(pickle.dumps(scalar_spec))

        assert isinstance(reconstructed, scalar_spec.__class__)

        assert scalar_spec.dtype == reconstructed.dtype
        assert scalar_spec.shape == reconstructed.shape
        assert scalar_spec.name == reconstructed.name

        chex.assert_trees_all_equal(
            scalar_spec.generate_value(),
            reconstructed.generate_value()
        )
        chex.assert_trees_all_equal_shapes_and_dtypes(
            scalar_spec.generate_value(),
            reconstructed.generate_value()
        )

    def test_slots(self, scalar_spec):
        with pytest.raises(AttributeError):
            scalar_spec.new_attr = 5

    def test_generate(self, matrix_spec):
        val = matrix_spec.generate_value()

        assert val.shape == matrix_spec.shape
        assert val.dtype == matrix_spec.dtype

    def test_validate(self, matrix_spec):
        val = matrix_spec.generate_value()
        _ = matrix_spec.validate(val)

        with pytest.raises(ValueError):
            # Reshape error
            matrix_spec.validate(val.reshape(1, *val.shape))

        with pytest.raises(ValueError):
            # Dtype error
            matrix_spec.validate(val.astype(int))

    def test_repr(self, matrix_spec):
        v = matrix_spec.generate_value()

        r = repr(matrix_spec)
        s = f"{specs.Array.__name__}(name={matrix_spec.name}," \
            f"shape={v.shape},dtype={v.dtype})"

        assert r == s

    def test_replace(self, scalar_spec, matrix_spec):
        new_mat_spec = scalar_spec.replace(shape=matrix_spec.shape)
        assert new_mat_spec.shape == matrix_spec.shape

        with pytest.raises(TypeError):
            # Key doesn't exist
            scalar_spec.replace(wrong_key=None)


class TestBounded:

    def test_serialize(self, bounded_vector_spec):
        reconstructed = pickle.loads(pickle.dumps(bounded_vector_spec))

        assert isinstance(reconstructed, bounded_vector_spec.__class__)

        assert bounded_vector_spec.dtype == reconstructed.dtype
        assert bounded_vector_spec.shape == reconstructed.shape
        assert bounded_vector_spec.name == reconstructed.name

        chex.assert_trees_all_equal(
            bounded_vector_spec.generate_value(),
            reconstructed.generate_value()
        )
        chex.assert_trees_all_equal_shapes_and_dtypes(
            bounded_vector_spec.generate_value(),
            reconstructed.generate_value()
        )

    def test_wrong_bounds(self):
        with pytest.raises(ValueError):
            _ = specs.BoundedArray((), jnp.float32, minimum=1, maximum=0)

    def test_wrong_shape(self):
        with pytest.raises(ValueError):
            # Non-broadcastable
            _ = specs.BoundedArray(
                shape=(2, 2, 2), dtype=jnp.float32,
                minimum=jnp.zeros((3, 1, 3)), maximum=jnp.ones((2,))
            )
        with pytest.raises(ValueError):
            # Non-broadcastable
            _ = specs.BoundedArray(
                shape=(2, 2, 2), dtype=jnp.float32,
                minimum=jnp.zeros((2,)), maximum=jnp.ones((3, 1, 3))
            )

    def test_spec(self, bounded_vector_spec):
        my_vec = bounded_vector_spec.generate_value()
        _ = bounded_vector_spec.validate(my_vec)

        with pytest.raises(Exception, match='ValueError: Value exceeded'):
            # Exceed max
            bounded_vector_spec.validate(my_vec + VEC_MAX + 1.0)

        with pytest.raises(Exception, match='ValueError: Value exceeded'):
            # Exceed min
            bounded_vector_spec.validate(my_vec - 1.0)

        with pytest.raises(ValueError):
            # Reshape error
            bounded_vector_spec.validate(my_vec.reshape(1, *my_vec.shape))

        with pytest.raises(ValueError):
            # Dtype error
            bounded_vector_spec.validate(my_vec.astype(int))

        bounded_vector_spec.validate(my_vec * 0 + VEC_MIN)
        bounded_vector_spec.validate(my_vec * 0 + VEC_MAX)


class TestDiscrete:

    def test_serialize(self, int_spec):
        reconstructed = pickle.loads(pickle.dumps(int_spec))

        assert isinstance(reconstructed, int_spec.__class__)

        assert int_spec.dtype == reconstructed.dtype
        assert int_spec.shape == reconstructed.shape
        assert int_spec.name == reconstructed.name

        chex.assert_trees_all_equal(
            int_spec.generate_value(),
            reconstructed.generate_value()
        )
        chex.assert_trees_all_equal_shapes_and_dtypes(
            int_spec.generate_value(),
            reconstructed.generate_value()
        )

    def test_wrong_num_values(self):
        with pytest.raises(ValueError):
            _ = specs.DiscreteArray(0)

    def test_wrong_num_value_dtype(self):
        with pytest.raises(ValueError):
            _ = specs.DiscreteArray(2.0, jnp.float32)  # type: ignore

    def test_wrong_type(self):
        with pytest.raises(ValueError):
            _ = specs.DiscreteArray(5, dtype=jnp.float32)

    def test_spec(self, int_spec):
        my_int = int_spec.generate_value()
        _ = int_spec.validate(my_int)

        with pytest.raises(Exception, match='ValueError: Value exceeded'):
            # Exceed max
            int_spec.validate(my_int + INT_SIZE)

        with pytest.raises(Exception, match='ValueError: Value exceeded'):
            # Exceed min
            int_spec.validate(my_int - 1)

        with pytest.raises(ValueError):
            # Reshape error
            int_spec.validate(my_int.reshape(1, *int_spec.shape))

        with pytest.raises(ValueError):
            # Dtype error
            int_spec.validate(my_int.astype(float))

        int_spec.validate(my_int + INT_SIZE - 1)
        int_spec.validate(0)


class TestReshape:

    def test_discrete(self, int_spec):
        appended = specs.reshape_spec(int_spec, append=(BATCH_SIZE,))
        prepended = specs.reshape_spec(int_spec, prepend=(BATCH_SIZE,))
        both = specs.reshape_spec(int_spec, (BATCH_SIZE,), (BATCH_SIZE,))

        back = appended.generate_value()
        front = prepended.generate_value()
        both = both.generate_value()

        assert back.shape == (BATCH_SIZE,)
        assert front.shape == (BATCH_SIZE,)
        assert both.shape == (BATCH_SIZE, BATCH_SIZE)

    def test_array(self, matrix_spec):
        appended = specs.reshape_spec(matrix_spec, append=(BATCH_SIZE,))
        prepended = specs.reshape_spec(matrix_spec, prepend=(BATCH_SIZE,))
        both = specs.reshape_spec(matrix_spec, (BATCH_SIZE,), (BATCH_SIZE,))

        back = appended.generate_value()
        front = prepended.generate_value()
        both = both.generate_value()

        assert back.shape == (*MATRIX_SHAPE, BATCH_SIZE,)
        assert front.shape == (BATCH_SIZE, *MATRIX_SHAPE)
        assert both.shape == (BATCH_SIZE, *MATRIX_SHAPE, BATCH_SIZE)

    def test_composite(self, tree_spec):
        with pytest.raises(NotImplementedError):
            specs.reshape_spec(tree_spec, prepend=(BATCH_SIZE,))

        tree_struct = specs.unpack_spec(tree_spec)

        appended = jax.tree_map(
            lambda s: specs.reshape_spec(s, append=(BATCH_SIZE,)),
            tree_struct
        )
        prepended = jax.tree_map(
            lambda s: specs.reshape_spec(s, prepend=(BATCH_SIZE,)),
            tree_struct
        )
        both = jax.tree_map(
            lambda s: specs.reshape_spec(s, (BATCH_SIZE,), (BATCH_SIZE,)),
            tree_struct
        )

        back = jax.tree_map(lambda s: s.generate_value(), appended)
        front = jax.tree_map(lambda s: s.generate_value(), prepended)
        both = jax.tree_map(lambda s: s.generate_value(), both)

        back_bools = jax.tree_map(lambda s: s.shape[-1] == BATCH_SIZE, back)
        front_bools = jax.tree_map(lambda s: s.shape[0] == BATCH_SIZE, front)
        both_bools = jax.tree_map(
            lambda s: (s.shape[0], s.shape[1]) == (BATCH_SIZE, BATCH_SIZE),
            both
        )

        assert all(jax.tree_util.tree_leaves(back_bools))
        assert all(jax.tree_util.tree_leaves(front_bools))
        assert all(jax.tree_util.tree_leaves(both_bools))
