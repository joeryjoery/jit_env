"""Collection of data-structure Specification Types.

These classes aim to communicate what data-structures are communicated
between an Agent and the Environment before these structures exist.

Warning: do not import the classes from this module separately! Instead
import this module as is to prevent confusion with name clashes from e.g.,
typing.Tuple or different spec type implementations by third parties.

In other words, the proper usage of this module is:

```
from jit_env import specs

a = specs.Array(...)  # Proper namespace
```

Incorrect usage:

```
from jit_env.specs import Array

a = Array(...)  # Can cause namespace clashes
```
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

    def __init__(self, name: str = ""):
        self.name = name

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"

    @_abc.abstractmethod
    def validate(self, value: _T) -> _T:
        """Checks if a value conforms to spec. """

    @_abc.abstractmethod
    def generate_value(self) -> _T:
        """Generate a value which conforms to this spec."""

    @_abc.abstractmethod
    def replace(self, **kwargs: _typing.Any) -> Spec:
        """Returns a new copy of `self` with specified attributes replaced """


class CompositeSpec(Spec, _abc.ABC):
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
        raise RuntimeError(
            "Composite Specs cannot be unambiguously mutated, "
            "create a new Spec through the Constructor instead."
        )


class Array(Spec):

    def __init__(
            self,
            shape: _typing.Sequence[int],
            dtype: _jnp.dtype,
            name: str = ""
    ):
        """Initializes a new `Array` spec.

        Args:
            shape: an iterable specifying the array shape.
            dtype: jax numpy dtype or string specifying the array dtype.
            name: string containing a semantic name for this spec.
        """
        super().__init__(name)
        self._shape = tuple(shape)
        self._dtype = dtype

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(shape={repr(self.shape)}, " \
               f"dtype={repr(self.dtype)}, name={repr(self.name)})"

    @property
    def shape(self) -> tuple:
        """Returns a `tuple` specifying the array shape."""
        return self._shape

    @property
    def dtype(self) -> _jnp.dtype:
        """Returns a jax numpy dtype specifying the array dtype."""
        return self._dtype

    def _fail_validation(self, message: str) -> None:
        if self.name:
            message += f" for spec {self.name}."
        else:
            message += "."
        raise ValueError(message)

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
        return _jnp.zeros(shape=self.shape, dtype=self.dtype)

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
        # Note that we assume direct correspondence between the names
        # of constructor arguments and attributes.
        return {name: getattr(self, name) for name in params.keys()}

    def replace(self, **kwargs: _typing.Any) -> _type_ext.Self:
        all_kwargs = self._get_constructor_kwargs()
        all_kwargs.update(kwargs)
        return self.__class__(**all_kwargs)


class BoundedArray(Array):

    def __init__(
        self,
        shape: _typing.Sequence[int],
        dtype: _jnp.dtype,
        minimum: int | float | _jxtype.Num[_jxtype.Array, '...'],
        maximum: int | float | _jxtype.Num[_jxtype.Array, '...'],
        name: str = ""
    ):
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
                f"minimum={repr(minimum)}\n\tmaximum={repr(maximum)}"
            )

        self._minimum = minimum
        self._maximum = maximum

    def __repr__(self) -> str:
        return (
            f"BoundedArray(shape={repr(self.shape)}, "
            f"dtype={repr(self.dtype)}, "
            f"name={repr(self.name)}, "
            f"minimum={repr(self.minimum)}, "
            f"maximum={repr(self.maximum)})"
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

    def __init__(
            self, 
            num_values: int,
            dtype: _typing.Any = _jnp.int32,  # TODO: jax.typing.DTypeLike
            name: str = ""
    ):
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

    def __repr__(self) -> str:
        return (
            f"DiscreteArray(shape={repr(self.shape)}, "
            f"dtype={repr(self.dtype)}, "
            f"name={repr(self.name)}, minimum={repr(self.minimum)},"
            f" maximum={repr(self.maximum)}, "
            f"num_values={repr(self.num_values)})"
        )

    @property
    def num_values(self) -> int:
        """Returns the number of items."""
        return self._num_values


@_tree_util.register_pytree_node_class
class Tree(CompositeSpec):

    def __init__(
            self,
            leaves: _typing.Sequence[Spec],
            structure: _tree_util.PyTreeDef,
            name: str = ""
    ):
        super().__init__(name=name)
        self.leave_specs = leaves
        self.treedef = structure

    def tree_flatten(
            self
    ) -> tuple[_typing.Sequence[Spec], tuple[_tree_util.PyTreeDef, str]]:
        # jax.tree hook
        return self.leave_specs, (self.treedef, self.name)

    @classmethod
    def tree_unflatten(
            cls: _typing.Type[Tree],
            aux: tuple[_tree_util.PyTreeDef, str],
            children: _typing.Sequence[Spec]
    ) -> Tree:
        # jax.tree_hook
        return cls(children, *aux)

    @property
    def spec_struct(self):
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

    def __repr__(self) -> str:
        leave_reprs = _jax.tree_map(repr, self.leave_specs)
        tree_repr = repr(_tree_util.tree_unflatten(self.treedef, leave_reprs))
        return f'{self.__class__.__name__}(name={self.name}, {tree_repr})'


class Tuple(Tree):

    def __init__(self, *var_specs: Spec, name: str = ""):
        super().__init__(
            list(var_specs),
            _tree_util.tree_structure(var_specs),
            name
        )


class Dict(Tree):

    def __init__(
            self,
            dict_spec: dict[str, Spec] | None = None,
            name: str = "",
            **kwarg_specs: Spec
    ):
        if dict_spec is None:
            dict_spec = {}

        self._defaults = {'name': name, 'dict_spec': dict_spec} | kwarg_specs
        full_spec = (dict_spec | kwarg_specs)
        super().__init__(
            list(full_spec.values()),
            _tree_util.tree_structure(full_spec),
            name=name
        )


class Batched(CompositeSpec, _typing.Generic[_T]):

    def __init__(self, spec: Spec, num: int, name: str = ""):
        super().__init__(name)
        self.spec = spec
        self.num = num

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(name={self.name}, ' \
               f'spec={repr(self.spec)}, num={self.num})'

    def validate(self, value: _T) -> _T:
        return _jax.vmap(self.spec.validate)(value)

    def generate_value(self) -> _T:
        base_value = self.spec.generate_value()
        return _jax.vmap(lambda x: base_value)(_jnp.arange(self.num))


class EnvironmentSpec(_typing.NamedTuple):
    observations: Spec
    actions: Spec
    rewards: Spec
    discounts: Spec
