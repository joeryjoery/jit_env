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


_T = _typing.TypeVar("_T")


class Spec(_abc.ABC, _typing.Generic[_T]):
    """Interface for defining datastructure Specifications.

    This type is meant to 'Specify' a datastructure's shapes, types, and
    general structure without needing to instantiate said structure.
    """

    def __init__(self, name: str = ""):
        """Initialize a Spec by requiring an explicit naming.

        Args:
            name: Explicit string name for the datastructure specification.
        """
        self.name = name

    def __str__(self) -> str:
        """Return a string representation of the full Specification."""
        return f"{self.__class__.__name__}(name={self.name})"

    def _fail_validation(self, message: str):
        """Helper function to distinguish validation between specs."""
        raise ValueError(message + f' for spec "{self.name}"')

    @_abc.abstractmethod
    def validate(self, value: _T) -> _T:
        """Checks if a value conforms to spec. """

    @_abc.abstractmethod
    def generate_value(self) -> _T:
        """Generate a value which conforms to this spec."""

    @_abc.abstractmethod
    def replace(self, **kwargs: _typing.Any) -> Spec:
        """Returns a new copy of `self` with specified attributes replaced """


class CompositeSpec(Spec[_T], _typing.Generic[_T], _abc.ABC):
    """Prevent Spec Types that compose other Spec Types from using replace.

    Since multiple Specs may be composed with similar namespaces it becomes
    ambiguous and difficult to specify which objects should be replaced.
    Furthermore, in deeply nested composite Specs it will not alway be clear
    whether an object needs to be shallow or deep-copied in order to not
    mutate the original Spec.

    Instead, it is safer and more clear to modify any particular leaf Specs
    and rebuild the composite Spec using the constructor.
    """

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


class Array(Spec):
    """Describes a Jax Array Specification."""

    def __init__(
            self,
            shape: _typing.Sequence[int],
            dtype: _typing.Any,  # TODO: jax.typing.DTypeLike
            name: str = ""
    ):
        """Initializes a new `Array` spec.

        Args:
            shape: A Sequence of integers specifying the array shape.
            dtype: A jax compatible dtype specification for the array.
            name: Explicit string name for the array specification.
        """
        super().__init__(name)
        self._shape = tuple(shape)
        self._dtype = dtype

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(shape={str(self.shape)}, " \
               f"dtype={str(self.dtype)}, name={str(self.name)})"

    @property
    def shape(self) -> tuple:
        """Returns a `tuple` specifying the array shape."""
        return self._shape

    @property
    def dtype(self) -> _typing.Any:  # TODO: jax.typing.DTypeLike
        """Returns a jax numpy dtype specifying the array dtype."""
        return self._dtype

    def validate(
            self, 
            value: _jxtype.Num[_jxtype.Array, '...']
    ) -> _jxtype.Num[_jxtype.Array, '...']:
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
    """A variant of Array that imposes explicit minimum and maximum bounds."""

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

    def __str__(self) -> str:
        return (
            f"BoundedArray(shape={str(self.shape)}, "
            f"dtype={str(self.dtype)}, "
            f"name={str(self.name)}, "
            f"minimum={str(self.minimum)}, "
            f"maximum={str(self.maximum)})"
        )

    @property
    def minimum(self) -> _jxtype.Num[_jxtype.Array, '...']:
        """Returns a Jax array specifying the minimum bounds (inclusive)."""
        return self._minimum

    @property
    def maximum(self) -> _jxtype.Num[_jxtype.Array, '...']:
        """Returns a Jax array specifying the maximum bounds (inclusive)."""
        return self._maximum

    def validate(
            self, 
            value: _jxtype.Num[_jxtype.Array, '...']
    ) -> _jxtype.Num[_jxtype.Array, '...']:
        value = super().validate(value)
        if (value < self.minimum).any() or (value > self.maximum).any():
            self._fail_validation(
                "Values were not all within bounds "
                f"{self.minimum} <= {value} <= {self.maximum}"
            )
        return value

    def generate_value(self) -> _jxtype.Num[_jxtype.Array, '...']:
        """Generate a jax array of the minima which conforms to this shape."""
        return _jnp.ones(shape=self.shape, dtype=self.dtype) * self.minimum


class DiscreteArray(BoundedArray):
    """A variant of BoundedArray with integer values and a maximum."""

    def __init__(
            self, 
            num_values: int,
            dtype: _typing.Any = _jnp.int32,  # TODO: jax.typing.DTypeLike
            name: str = ""
    ):
        """Initializes a new `BoundedArray` spec.

        Args:
            num_values: The number of integers this spec should support.
            dtype: An int-like jax dtype for the array.
            name: Explicit string name for the array specification.

        Raises:
            ValueError:
                if `num_values` is not positive or the given `dtype` is not a
                jax-supported integer type.
        """
        if (not num_values > 0) or (not _jnp.issubdtype(
                type(num_values), _jnp.integer)):
            raise ValueError(
                f"`num_values` must be a positive integer, got {num_values}."
            )

        if not _jnp.issubdtype(dtype, _jnp.integer):
            raise ValueError(f"`dtype` must be integer, got {dtype}.")

        num_values = int(num_values)
        maximum = num_values - 1

        super().__init__((), dtype, minimum=0, maximum=maximum, name=name)
        self._num_values = num_values

    def __str__(self) -> str:
        return (
            f"DiscreteArray(shape={str(self.shape)}, "
            f"dtype={str(self.dtype)}, "
            f"name={str(self.name)}, minimum={str(self.minimum)},"
            f" maximum={str(self.maximum)}, "
            f"num_values={str(self.num_values)})"
        )

    @property
    def num_values(self) -> int:
        """Returns the number of items."""
        return self._num_values


class MultiDiscreteArray(BoundedArray):
    """Generalizes DiscreteArray to higher dimensions."""

    def __init__(
        self,
        num_values: _typing.Sequence[int] | _jxtype.Integer[
            _jxtype.Array, '...'],
        dtype: _typing.Any = _jnp.int32,
        name: str = ""
    ):
        """Initializes a new `BoundedArray` spec.

        Args:
            num_values:
                The number of integers this spec should support across
                distinct action dimensions.
            dtype:
                An int-like jax dtype for the array.
            name:
                Explicit string name for the array specification.

        Raises:
            ValueError:
                if `num_values` are not all positive or the given `dtype`
                is not a jax-supported integer type.
        """
        num_values = _jnp.asarray(num_values, dtype)
        if (not (num_values > 0).all()) or (not _jnp.issubdtype(
                num_values.dtype, _jnp.integer)):
            raise ValueError(
                f"`num_values` must be an array of positive integers, got {num_values}."
            )

        if not _jnp.issubdtype(dtype, _jnp.integer):
            raise ValueError(f"`dtype` must be integer, got {dtype}.")

        num_values = num_values
        maximum = num_values - 1
        super().__init__(
            shape=num_values.shape,
            dtype=dtype,
            minimum=_jnp.zeros_like(num_values),
            maximum=maximum,
            name=name,
        )
        self._num_values = num_values

    @property
    def num_values(self) -> _jxtype.Integer[_jxtype.Array, '...']:
        """Returns the number of possible values across all dimensions."""
        return self._num_values

    def __repr__(self) -> str:
        return (
            f"MultiDiscreteArray("
            f"shape={repr(self.shape)}, "
            f"dtype={repr(self.dtype)}, "
            f"name={repr(self.name)}, "
            f"minimum={repr(self.minimum)}, "
            f"maximum={repr(self.maximum)}, "
            f"num_values={repr(self.num_values)})"
        )


@_tree_util.register_pytree_node_class
class Tree(CompositeSpec):
    """A Compositional Data Spec that behaves like a Jax PyTree.

    This Spec is created from other specs by providing the leaf specs and
    a desired PyTree structure.
    """

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
        self.leave_specs = leaves
        self.treedef = structure

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

    @property
    def spec_struct(self) -> _jxtype.PyTree[Spec]:
        """Return the unflattend Spec PyTree as is."""
        return _tree_util.tree_unflatten(self.treedef, self.leave_specs)

    def validate(
            self,
            value: _jxtype.PyTree[_jxtype.Num[_jxtype.Array, '...'] | None]
    ) -> _jxtype.PyTree[_jxtype.Num[_jxtype.Array, '...'] | None]:
        as_leaves = _tree_util.tree_leaves(value)
        leaves = _jax.tree_map(
            lambda v, s: s.validate(v),
            as_leaves, self.leave_specs
        )
        return _tree_util.tree_unflatten(self.treedef, leaves)

    def generate_value(
            self
    ) -> _jxtype.PyTree[_jxtype.Num[_jxtype.Array, '...'] | None]:
        values = [s.generate_value() for s in self.leave_specs]
        return _tree_util.tree_unflatten(self.treedef, values)

    def __str__(self) -> str:
        leave_strs = _jax.tree_map(str, self.leave_specs)
        tree_str = str(_tree_util.tree_unflatten(self.treedef, leave_strs))
        return f'{self.__class__.__name__}(name={self.name}, {tree_str})'


class Tuple(Tree):
    """Overrides Tree as a Tuple PyTree Structure."""

    def __init__(self, *var_specs: Spec, name: str = ""):
        """Initializes a new `Tuple` spec.

        Args:
            var_specs:
                A sequence of `Spec` objects to treat as a Spec tuple.
            name:
                Explicit string name for the Tree specification.
        """
        super().__init__(
            list(var_specs),
            _tree_util.tree_structure(var_specs),
            name
        )


class Dict(Tree):
    """Overrides Tree as a Dictionary PyTree Structure."""

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

        self._defaults = {'name': name, 'dict_spec': dict_spec} | kwarg_specs
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

        self.spec = spec
        self.num = num

    def __str__(self) -> str:
        return f'{self.__class__.__name__}(name={self.name}, ' \
               f'spec={str(self.spec)}, num={self.num})'

    def validate(self, value: _T) -> _T:
        return _jax.vmap(self.spec.validate)(value)

    def generate_value(self) -> _T:
        base_value = self.spec.generate_value()
        return _jax.vmap(lambda x: base_value)(_jnp.arange(self.num))


class EnvironmentSpec(_typing.NamedTuple):
    """Type to collect all specs of an Environment in one place."""
    observations: Spec
    actions: Spec
    rewards: Spec
    discounts: Spec
