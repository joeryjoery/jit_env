"""Collection of data-structure Specification Types.

These classes aim to communicate what data-structures are communicated
between an Agent and the Environment before these structures exist.

Warning: do not import the classes from this module separately! Instead
import this module as is to prevent confusion with name clashes from e.g.,
typing.Tuple or different spec type implementations by third parties.

In other words, the proper usage of this module is::

    from jit_env import specs
    a = specs.Array(...)  # Proper namespace


Incorrect usage::

    from jit_env.specs import Array
    a = Array(...)  # Can cause namespace clashes

The basic Spec types defined in this Module were adapted from:
 - Jumanji: https://github.com/instadeepai/jumanji

Our implementation of `validate` and `generate_value` are compatible with
typical `jax` transformations like `jax.vmap` or `jax.jit`.
"""
from __future__ import annotations as _annotations
import abc as _abc
import functools as _functools
import inspect as _inspect
import typing as _typing
import typing_extensions as _type_ext

import jaxtyping as _jxtype

import jax as _jax
from jax import numpy as _jnp
from jax import tree_util as _tree_util

import jit_env

_T = _typing.TypeVar("_T")


class Spec(_typing.Generic[_T], metaclass=_abc.ABCMeta):
    """Interface for defining datastructure Specifications.

    This type is meant to 'Specify' a datastructure's shapes, types, and
    general structure without needing to instantiate said structure.

    We copy the `__slots__` implementation from `dm_env` with read-only
    properties to enforce that `specs` cannot be modified at runtime.

    Notes
        Specs are not composable with jax transformations.
    """
    __slots__ = ('_name',)

    def __init__(self, name: str = ""):
        """Initialize a Spec by requiring an explicit naming.

        Args:
            name: Explicit string name for the datastructure specification.
        """
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def __repr__(self) -> str:
        """Return a string representation of the full Specification.

        All __slot__ attributes are collected within the Spec's inheritance
        chain and string concatenated top-to-bottom : left-to-right.

        In other words, this method gives: 'Name(slots_top, slots_lower, ...)'
        """
        # iterator over ((slot_self_a, ...), (slot_super_a, ...), ...)
        slots = (getattr(cls, '__slots__', ()) for cls in type(self).__mro__)

        reverse_ordered, seen = list(), set()
        for s in slots:
            if s and (s not in seen):
                seen.add(s)
                reverse_ordered += list(s)[::-1]

        slot_values = ','.join(
            [f'{attr}={getattr(self, attr)}'.lstrip('_')
             for attr in reverse_ordered[::-1]]
        )

        return f"{self.__class__.__name__}({slot_values})"

    def _fail_validation(self, message: str):
        """Helper function for creating jax transform-compatible errors."""

        def _raiser(*_):
            raise ValueError(message + f' for spec "{self.name}"')

        # ValueError is directly raised outside jax.jit/ jax.vmap/ etc.
        # Otherwise, it is raised through the XLA dispather.
        _jax.debug.callback(_raiser)

    @_abc.abstractmethod
    def validate(self, value: _T) -> _T:
        """Checks if a value conforms to spec. """

    @_abc.abstractmethod
    def generate_value(self) -> _T:
        """Generate a value which conforms to this spec."""

    @_abc.abstractmethod
    def replace(self, **kwargs: _typing.Any) -> Spec:
        """Returns a new copy of `self` with specified attributes replaced """


class PrimitiveSpec(Spec[_T], _typing.Generic[_T], metaclass=_abc.ABCMeta):
    """Distinguishes `leaf` Spec Types from structured/ composite Specs.

    In other words, these Specs do not build Specs from other Specs.
    """
    __slots__ = ('_shape', '_dtype')

    def __init__(
            self,
            shape: _typing.Sequence[int],
            dtype: _typing.Any,  # TODO: jax.typing.DTypeLike
            name: str = ""
    ):
        """Initializes a new `PrimitiveSpec` spec.

        Args:
            shape: A Sequence of integers specifying the array shape.
            dtype: A jax compatible dtype specification for the array.
            name: Explicit string name for the array specification.
        """
        super().__init__(name)
        self._shape = tuple(shape)
        self._dtype = _jnp.zeros((), dtype).dtype  # get uniform `dtype` type

    @property
    def shape(self) -> int | _typing.Sequence[int]:
        return self._shape

    @property
    def dtype(self) -> _typing.Any:
        return self._dtype


class CompositeSpec(Spec[_T], _typing.Generic[_T], metaclass=_abc.ABCMeta):
    """Prevent Spec Types that compose other Spec Types from using replace.

    Since multiple Specs may be composed with similar namespaces it becomes
    ambiguous and difficult to specify which objects should be replaced.
    Furthermore, in deeply nested composite Specs it will not alway be clear
    whether an object needs to be shallow or deep-copied in order to not
    mutate the original Spec.

    Instead, it is safer and more clear to modify any particular leaf Specs
    and rebuild the composite Spec using the constructor.
    """
    __slots__ = ()

    @_abc.abstractmethod
    def as_spec_struct(self) -> _jxtype.PyTree[PrimitiveSpec]:
        """Return the tree of primitive Spec types as is."""

    def replace(self, **kwargs: _typing.Any) -> Spec:
        """Composite Specs cannot be unambiguously replaced.

        Create new CompositeSpec types by calling the class constructor.

        Raises:
            RuntimeError Always.
        """
        raise RuntimeError(
            "Composite Specs cannot be unambiguously mutated, "
            "create a new Spec through the Constructor instead."
        )


class Array(PrimitiveSpec):
    """Describes a Jax Array Specification."""
    __slots__ = ()

    def validate(
            self,
            value: _jxtype.ArrayLike
    ) -> _jxtype.ArrayLike:
        value = _jnp.asarray(value)

        if value.shape != self.shape:
            self._fail_validation(
                f"Expected shape {self.shape} but found {value.shape}"
            )
        if value.dtype != self.dtype:
            self._fail_validation(
                f"Expected dtype {self.dtype} but found {value.dtype}"
            )
        return value

    def generate_value(self) -> _jxtype.Num[_jxtype.Array, '...']:
        """Generate a value which conforms to this spec."""
        return _jnp.zeros(self.shape, self.dtype)

    def __reduce__(
            self
    ) -> tuple[_typing.Type[Array], tuple[_typing.Any, ...]]:
        """Specify how to de-serialize (unpickle) this class Type."""
        return Array, (self._shape, self._dtype, self._name)

    def _get_constructor_kwargs(self) -> _typing.Any:
        """Returns constructor kwargs for instantiating a copy of self."""
        params = _inspect.signature(
            _functools.partial(self.__class__.__init__, self)
        ).parameters
        kinds = {value.kind for value in params.values()}
        if _inspect.Parameter.VAR_POSITIONAL in kinds:
            raise TypeError(
                f"{self.__class__.__module__}.{self.__class__.__name__} "
                f"types do not accept *args."
            )
        elif _inspect.Parameter.VAR_KEYWORD in kinds:
            raise TypeError(
                f"{self.__class__.__module__}.{self.__class__.__name__} "
                f"types do not accept **kwargs."
            )
        return {name: getattr(self, name) for name in params.keys()}

    def replace(self, **kwargs: _typing.Any) -> _type_ext.Self:
        all_kwargs = self._get_constructor_kwargs()
        all_kwargs.update(kwargs)
        return self.__class__(**all_kwargs)


class BoundedArray(Array):
    """A variant of Array that imposes explicit minimum and maximum bounds.

    These bounds annotate (not enforced) a l1 Box constraint.
    """
    __slots__ = ('_minimum', '_maximum')

    def __init__(
            self,
            shape: _typing.Sequence[int],
            dtype: _typing.Any,  # TODO: jax.typing.DTypeLike
            minimum: int | float | _jxtype.Num[_jxtype.Array, '...'],
            maximum: int | float | _jxtype.Num[_jxtype.Array, '...'],
            name: str = ""
    ):
        """Initializes a new `BoundedArray` spec.

        Args:
            shape: A Sequence of integers specifying the array shape.
            dtype: A jax compatible dtype specification for the array.
            minimum: A numerical value/ array of values for the lower bound
            maximum: A numerical value/ array of values for the upper bound
            name: Explicit string name for the array specification.

        Raises:
            ValueError:
                if `minimum` or `maximum` are not compatible with the given
                `shape`, or if `minimum` is larger than `maximum`.
        """
        super().__init__(shape, dtype, name)
        minimum = _jnp.asarray(minimum, dtype)
        maximum = _jnp.asarray(maximum, dtype)
        try:
            bcast_minimum = _jnp.broadcast_to(minimum, shape=shape)
        except ValueError as e:
            raise ValueError(
                "`minimum` is incompatible with `shape`"
            ) from e
        try:
            bcast_maximum = _jnp.broadcast_to(maximum, shape=shape)
        except ValueError as e:
            raise ValueError(
                "`maximum` is incompatible with `shape`"
            ) from e

        if _jnp.any(bcast_minimum > bcast_maximum):
            raise ValueError(
                f"All values in `minimum` must be less "
                f"than or equal to their corresponding "
                f"value in `maximum`, got: \n\t"
                f"minimum={str(minimum)}\n\tmaximum={str(maximum)}"
            )

        self._minimum = minimum
        self._maximum = maximum

    @property
    def minimum(self) -> _jxtype.Num[_jxtype.Array, '...']:
        return self._minimum

    @property
    def maximum(self) -> _jxtype.Num[_jxtype.Array, '...']:
        return self._maximum

    def validate(
            self,
            value: _jxtype.ArrayLike
    ) -> _jxtype.ArrayLike:
        """Check if value falls inbetween the specified bounds."""
        value = super().validate(value)

        _ = _jax.lax.cond(
            (value < self.minimum).any(),
            lambda: self._fail_validation(
                f'Value exceeded minimum: {self.minimum} > {value}'
            ),
            lambda: None
        )

        _ = _jax.lax.cond(
            (value > self.maximum).any(),
            lambda: self._fail_validation(
                f'Value exceeded maximum: {value} > {self.maximum}'
            ),
            lambda: None,
        )

        return value

    def generate_value(self) -> _jxtype.Num[_jxtype.Array, '...']:
        """Generate a jax array of the minima which conforms to this shape."""
        return _jnp.full(
            shape=self.shape, fill_value=self.minimum, dtype=self.dtype
        )

    def __reduce__(
            self
    ) -> tuple[_typing.Type[BoundedArray], tuple[_typing.Any, ...]]:
        """Specify how to de-serialize (unpickle) this class Type."""
        return BoundedArray, (self._shape, self._dtype,
                              self._minimum, self._maximum, self._name)


class DiscreteArray(BoundedArray):
    """A variant of BoundedArray with strict integer values and minimum=0."""
    __slots__ = ()

    def __init__(
            self,
            num_values: int | _typing.Sequence[int] | _jxtype.Integer[
                _jxtype.Array, '...'
            ],
            dtype: _typing.Any = _jnp.int32,  # TODO: jax.typing.DTypeLike
            name: str = ""
    ):
        """Initializes a new `DiscreteArray` spec.

        Args:
            num_values: The number of integers this spec should support.
            dtype: An int-like jax dtype for the array.
            name: Explicit string name for the array specification.

        Raises:
            ValueError:
                if `num_values` is not positive or the given `dtype` is not a
                jax-supported integer type.
        """
        if not _jnp.issubdtype(dtype, _jnp.integer):
            raise ValueError(f"`dtype` must be integer, got {dtype}.")

        num_values = _jnp.asarray(num_values, dtype)
        if not (num_values > 0).all():
            raise ValueError(
                f"`num_values` may only contain positive integers, "
                f"got {num_values}."
            )

        super().__init__(
            shape=_jnp.shape(num_values),
            dtype=dtype,
            minimum=0,
            maximum=num_values - 1,
            name=name
        )

    @property
    def num_values(self) -> _jxtype.Integer[_jxtype.Array, '...']:
        return self.maximum + 1

    def __reduce__(
            self
    ) -> tuple[_typing.Type[DiscreteArray], tuple[_typing.Any, ...]]:
        """Specify how to de-serialize (unpickle) this class Type."""
        return DiscreteArray, (self._maximum+1, self._dtype, self._name)


@_tree_util.register_pytree_node_class
class Tree(CompositeSpec):
    """A Compositional Data Spec that behaves like a Jax PyTree.

    This Spec is created from other specs by providing the leaf specs and
    a desired PyTree structure.
    """
    __slots__ = ('_leave_specs', '_treedef')

    def __init__(
            self,
            leaves: _typing.Sequence[Spec],
            structure: _tree_util.PyTreeDef,
            name: str = ""
    ):
        """Initializes a new `Tree` spec.

        Args:
            leaves:
                A sequence of `Spec` objects to treat as the flattened leaves
                of the given `structure`.
            structure:
                The Jax PyTree defintion to unflatten `leaves`.
            name:
                Explicit string name for the Tree specification.
        """
        super().__init__(name=name)
        self._leave_specs = leaves
        self._treedef = structure

    def __repr__(self) -> str:
        """Override base `repr` by mapping `repr` over all leave-specs."""
        leave_strs = _jax.tree_map(repr, self.leave_specs)
        tree_str = str(_tree_util.tree_unflatten(self.treedef, leave_strs))
        return f'{self.__class__.__name__}(name={self.name},tree={tree_str})'

    @property
    def leave_specs(self) -> _typing.Sequence[Spec]:
        return self._leave_specs

    @property
    def treedef(self) -> _tree_util.PyTreeDef:
        return self._treedef

    def tree_flatten(
            self
    ) -> tuple[_typing.Sequence[Spec], tuple[_tree_util.PyTreeDef, str]]:
        """Return the leaves and all data needed to reinitialize this class"""
        return self.leave_specs, (self.treedef, self.name)

    @classmethod
    def tree_unflatten(
            cls: _typing.Type[Tree],
            aux: tuple[_tree_util.PyTreeDef, str],
            children: _typing.Sequence[Spec]
    ) -> Tree:
        """Reinitialize the class from the leaves and the requisite data"""
        return cls(children, *aux)

    def as_spec_struct(self) -> _jxtype.PyTree[Spec]:
        """Return the unflattend Spec PyTree as is."""
        return _tree_util.tree_unflatten(self.treedef, self.leave_specs)

    def validate(
            self,
            value: _jxtype.PyTree[_jxtype.ArrayLike | None]
    ) -> _jxtype.PyTree[_jxtype.ArrayLike | None]:
        return _jax.tree_map(
            lambda s, v: s.validate(v),
            self.as_spec_struct(), value
        )

    def generate_value(
            self
    ) -> _jxtype.PyTree[_jxtype.Num[_jxtype.Array, '...'] | None]:
        values = [s.generate_value() for s in self.leave_specs]
        return _tree_util.tree_unflatten(self.treedef, values)


class Tuple(Tree):
    """Overrides Tree as a Tuple PyTree Structure."""
    __slots__ = ()

    def __init__(self, *var_specs: Spec, name: str = ""):
        """Initializes a new `Tuple` spec.

        Args:
            var_specs:
                A sequence of `Spec` objects to treat as a Spec tuple.
            name:
                Explicit string name for the Tree specification.

        Raises:
            ValueError: If no specs are provided.
        """
        if not var_specs:
            raise ValueError("Cannot initialize an empty Spec")

        super().__init__(
            list(var_specs),
            _tree_util.tree_structure(var_specs),
            name
        )


class Dict(Tree):
    """Overrides Tree as a Dictionary PyTree Structure."""
    __slots__ = ()

    def __init__(
            self,
            dict_spec: dict[str, Spec] | None = None,
            name: str = "",
            **kwarg_specs: Spec
    ):
        """Initializes a new `Dict` spec.

        Args:
            dict_spec:
                A mapping from strings to `Spec` objects to treat as a Spec
                dictionary. Optionally, one can leaf this argument as None
                and specify all specs inside the **kwarg_specs argument.
            name:
                Explicit string name for the Tree specification.
            kwarg_specs:
                Explicit keyword arguments to also be interpreted as part of
                the Spec Tree.

        Raises:
            ValueError: If the union of all provided specs are empty.
        """
        if dict_spec is None:
            dict_spec = {}

        full_spec = dict_spec | kwarg_specs

        if not full_spec:
            raise ValueError("Cannot initialize an empty Spec")

        super().__init__(
            list(full_spec.values()),
            _tree_util.tree_structure(full_spec),
            name=name
        )


class Batched(CompositeSpec, _typing.Generic[_T]):
    """A generic CompositeSpec that prepends a fixed Batch dimension.

    This Spec type essentially wraps the base spec with a call to jax.vmap
    inside `validate` and `generate_value`
    """
    __slots__ = ('_num', '_spec')

    def __init__(self, spec: Spec, num: int, name: str = ""):
        """Initializes a new `Batched` spec.

        Args:
            spec: The Specification to batch over.
            num: The batch-size to prepend `spec` with.
            name: Explicit string name for the Tree specification.

        Raises:
            ValueError: If `num` is not positive.
        """
        super().__init__(name)
        if not num > 0:
            raise ValueError("Cannot Batch over empty dimensions! num > 0!")

        self._spec = spec
        self._num = num

    @property
    def spec(self) -> Spec:
        return self._spec

    @property
    def num(self) -> int:
        return self._num

    def as_spec_struct(self) -> _jxtype.PyTree[PrimitiveSpec]:
        """Unpack and prepend batch-size to all leaf-specs.

        Warning, to prevent unnecesarily unpacking nested structures, this
        method only unpacks CompositeSpec types by one level, and "re-batches"
        any CompositeSpecs 1 level down.

        To do this recursively, see `unpack_spec`.
        """
        base_spec = self.spec
        if isinstance(base_spec, CompositeSpec):
            base_spec = base_spec.as_spec_struct()

        def reshape(s: Spec):
            if isinstance(s, PrimitiveSpec):
                return reshape_spec(s, prepend=(self.num,))
            return Batched(s, self.num)

        return _jax.tree_map(reshape, base_spec)

    def validate(self, value: _T) -> _T:
        # jax.vmap fails as error-raising is non-homogenous.
        return _jax.lax.map(self.spec.validate, value)

    def generate_value(self) -> _T:
        base_value = self.spec.generate_value()
        return _jax.vmap(lambda x: base_value)(_jnp.arange(self.num))


# Utility functions for handling Spec types

@_typing.overload
def reshape_spec(
        spec: DiscreteArray,
        prepend: _typing.Sequence[int] = (),
        append: _typing.Sequence[int] = ()
) -> DiscreteArray:
    ...  # pragma: no cover


@_typing.overload
def reshape_spec(
        spec: BoundedArray,
        prepend: _typing.Sequence[int] = (),
        append: _typing.Sequence[int] = ()
) -> BoundedArray:
    ...  # pragma: no cover


@_typing.overload
def reshape_spec(
        spec: Array,
        prepend: _typing.Sequence[int] = (),
        append: _typing.Sequence[int] = ()
) -> Array:
    ...  # pragma: no cover


@_typing.overload
def reshape_spec(
        spec: Spec,
        prepend: _typing.Sequence[int] = (),
        append: _typing.Sequence[int] = ()
) -> Spec:
    ...  # pragma: no cover


def reshape_spec(
        spec: Spec,
        prepend: _typing.Sequence[int] = (),
        append: _typing.Sequence[int] = ()
) -> Spec:
    """Utility function to modularly reshape a Spec type.

    This is useful for wrapping Environments with `jax.vmap` like transforms.

    Args:
        spec: The base spec to reshape
        prepend: A sequence of integers to add before spec.shape
        append: A sequence of integers to add after spec.shape

    Returns:
        The reshaped spec.

    Raises:
        NotImplementedError:
            if spec is not of type Array.
    """
    if isinstance(spec, DiscreteArray):
        batched_num = _jnp.broadcast_to(
            spec.num_values, (*prepend, *spec.num_values.shape, *append)
        )
        return spec.replace(num_values=batched_num)
    elif isinstance(spec, Array):
        return spec.replace(
            shape=(*prepend, *spec.shape, *append)  # type: ignore
        )
    else:
        raise NotImplementedError(
            f"Spec of type: {type(spec)} has no implemented reshape rule."
        )


def unpack_spec(
        spec: Spec
) -> Spec:
    """Recursively unpack composite specs to a tree of Primitive Specs."""
    if isinstance(spec, CompositeSpec):
        return _jax.tree_map(unpack_spec, spec.as_spec_struct())
    return spec


def make_environment_spec(env: jit_env.Environment):
    return EnvironmentSpec(
        observations=env.observation_spec(),
        actions=env.action_spec(),
        rewards=env.reward_spec(),
        discounts=env.discount_spec()
    )


class EnvironmentSpec(_typing.NamedTuple):
    """Type to collect all specs of an Environment in one place."""
    observations: Spec
    actions: Spec
    rewards: Spec
    discounts: Spec
