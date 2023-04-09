from __future__ import annotations
import pytest
import typing

import chex

import jax
from jax import numpy as jnp

import jit_env
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


@pytest.mark.skip
def test_tree_spec(tree_spec: specs.Tree):
    my_batch, my_scalar = tree_spec.generate_value()

    tree_spec.validate((my_batch, my_scalar))
    with pytest.raises(ValueError):
        # Wrong Pytree structure
        tree_spec.validate((my_scalar, my_batch))

    spec_struct = tree_spec.as_spec_struct()
    spec_struct[0].validate(my_batch)
    spec_struct[1].validate(my_scalar)


class TestBatched:

    @pytest.mark.skip
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

    def test_struct(self, batch_spec: specs.Batched):
        out_dict = batch_spec.generate_value()

        batched_specs = batch_spec.as_spec_struct()
        new_dict = jax.tree_map(
            lambda s: s.generate_value(), batched_specs
        )

        chex.assert_trees_all_equal(out_dict, new_dict)
        chex.assert_trees_all_equal_shapes(out_dict, new_dict)
        chex.assert_trees_all_equal_dtypes(out_dict, new_dict)


class TestArray:

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

    def test_wrong_bounds(self):
        with pytest.raises(ValueError):
            _ = specs.BoundedArray((), jnp.float32, minimum=1, maximum=0)

    def test_wrong_shape(self):
        with pytest.raises(ValueError):
            # Non-broadcastable
            _ = specs.BoundedArray(
                shape=(5, 5, 5), dtype=jnp.float32,
                minimum=jnp.zeros((3, 1, 3)), maximum=jnp.ones((1, 3, 1))
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

        appended = specs.reshape_spec(int_spec, append=(BATCH_SIZE, ))
        prepended = specs.reshape_spec(int_spec, prepend=(BATCH_SIZE, ))
        both = specs.reshape_spec(int_spec, (BATCH_SIZE, ), (BATCH_SIZE, ))

        back = appended.generate_value()
        front = prepended.generate_value()
        both = both.generate_value()

        assert back.shape == (BATCH_SIZE, )
        assert front.shape == (BATCH_SIZE, )
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
        assert both.shape == (BATCH_SIZE, *MATRIX_SHAPE,  BATCH_SIZE)

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