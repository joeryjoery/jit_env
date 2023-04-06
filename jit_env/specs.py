"""Collection of Specification types: indicates data shapes to an Agent

Warning: do not import the classes from this module separately! Instead
import this module as is to prevent confusion with name clashes from e.g.,
typing.Tuple

Note: Conventionally, one can also use nested specs using built-in tuple
      dict, list, etc. instead of ever really needing any of the specs
      defined in this file. The specs defined here are only intended for
      type explicitness and utility and may not port well to e.g.,
      the dm_env API.
"""
from __future__ import annotations
import abc
from typing import Any, TypeVar, Sequence, Type, Generic, NamedTuple, Iterable

from jaxtyping import Num, Array

from jax import numpy as jnp
from jax import tree_map, vmap, tree_util

from jaxtyping import PyTree

T = TypeVar("T")


class Spec(abc.ABC, Generic[T]):

    def __init__(self, name: str = ""):
        self.name = name

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={repr(self.name)})"

    @abc.abstractmethod
    def validate(self, value: T) -> T:
        """Checks if a value conforms to spec. """

    @abc.abstractmethod
    def generate_value(self) -> T:
        """Generate a value which conforms to this spec."""

    @abc.abstractmethod
    def replace(self, **kwargs: Any) -> Spec:
        """Returns a new copy of `self` with specified attributes replaced """


class Array(Spec):

    def __init__(
            self,
            shape: Sequence[int],
            dtype: jnp.dtype | type,
            name: str = ""
    ):
        """Initializes a new `Array` spec.

        Args:
            shape: an iterable specifying the array shape.
            dtype: jax numpy dtype or string specifying the array dtype.
            name: string containing a semantic name for the corresponding array. Defaults to `''`.
        """
        super().__init__(name)
        self._shape = tuple(shape)
        self._dtype = get_valid_dtype(dtype)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(shape={repr(self.shape)}, " \
               f"dtype={repr(self.dtype)}, name={repr(self.name)})"

    def __reduce__(self) -> Any:
        """To allow pickle to serialize the spec."""
        return Array, (self._shape, self._dtype, self.name)

    @property
    def shape(self) -> tuple:
        """Returns a `tuple` specifying the array shape."""
        return self._shape

    @property
    def dtype(self) -> jnp.dtype:
        """Returns a jax numpy dtype specifying the array dtype."""
        return self._dtype

    def _fail_validation(self, message: str) -> None:
        if self.name:
            message += f" for spec {self.name}."
        else:
            message += "."
        raise ValueError(message)

    def validate(self, value: Num[Array, '...']) -> Num[Array, '...']:
        value = jnp.asarray(value)
        if value.shape != self.shape:
            self._fail_validation(
                f"Expected shape {self.shape} but found {value.shape}"
            )
        if value.dtype != self.dtype:
            self._fail_validation(
                f"Expected dtype {self.dtype} but found {value.dtype}"
            )
        return value

    def generate_value(self) -> Num[Array, '...']:
        """Generate a value which conforms to this spec."""
        return jnp.zeros(shape=self.shape, dtype=self.dtype)

    def _get_constructor_kwargs(self) -> Dict[str, Any]:
        """Returns constructor kwargs for instantiating a new copy of this spec."""
        # Get the names and kinds of the constructor parameters.
        params = inspect.signature(
            functools.partial(type(self).__init__, self)
        ).parameters
        # __init__ must not accept *args or **kwargs, since otherwise we won't be
        # able to infer what the corresponding attribute names are.
        kinds = {value.kind for value in params.values()}
        if inspect.Parameter.VAR_POSITIONAL in kinds:
            raise TypeError("specs.Array subclasses must not accept *args.")
        elif inspect.Parameter.VAR_KEYWORD in kinds:
            raise TypeError("specs.Array subclasses must not accept **kwargs.")
        # Note that we assume direct correspondence between the names of constructor
        # arguments and attributes.
        return {name: getattr(self, name) for name in params.keys()}

    def replace(self, **kwargs: Any) -> Array:
        all_kwargs = self._get_constructor_kwargs()
        all_kwargs.update(kwargs)
        return self.__class__(**all_kwargs)


class BoundedArray(Array):

    def __init__(
        self,
        shape: Sequence[int],
        dtype: jnp.dtype | type,
        minimum: float | int | Sequence[float | int],
        maximum: float | int | Sequence[float | int],
        name: str = "",
    ):
        super().__init__(shape, dtype, name)
        minimum = jnp.asarray(minimum, dtype)
        maximum = jnp.asarray(maximum, dtype)
        try:
            bcast_minimum = jnp.broadcast_to(minimum, shape=shape)
        except ValueError as jnp_exception:
            raise ValueError(
                "`minimum` is incompatible with `shape`"
            ) from jnp_exception
        try:
            bcast_maximum = jnp.broadcast_to(maximum, shape=shape)
        except ValueError as jnp_exception:
            raise ValueError(
                "`maximum` is incompatible with `shape`"
            ) from jnp_exception

        if jnp.any(bcast_minimum > bcast_maximum):
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

    def __reduce__(self) -> Any:
        """To allow pickle to serialize the spec."""
        return BoundedArray, (
            self._shape,
            self._dtype,
            self._minimum,
            self._maximum,
            self._name,
        )

    @property
    def minimum(self) -> Num[Array, '...']:
        """Returns a Jax array specifying the minimum bounds (inclusive)."""
        return self._minimum

    @property
    def maximum(self) -> Num[Array, '...']:
        """Returns a Jax array specifying the maximum bounds (inclusive)."""
        return self._maximum

    def validate(self, value: Num[Array, '...']) -> Num[Array, '...']:
        value = super().validate(value)
        if (value < self.minimum).any() or (value > self.maximum).any():
            self._fail_validation(
                "Values were not all within bounds "
                f"{repr(self.minimum)} <= {repr(value)} <= {repr(self.maximum)}"
            )
        return value

    def generate_value(self) -> Num[Array, '...']:
        """Generate a jax array of the minima which conforms to this shape."""
        return jnp.ones(shape=self.shape, dtype=self.dtype) * self.minimum


class DiscreteArray(BoundedArray):

    def __init__(
        self, num_values: int, dtype: jnp.dtype | type = jnp.int32, name: str = ""
    ):

        if num_values <= 0 or not jnp.issubdtype(type(num_values), jnp.integer):
            raise ValueError(
                f"`num_values` must be a positive integer, got {num_values}."
            )

        if not jnp.issubdtype(dtype, jnp.integer):
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

    def __reduce__(self) -> Any:
        """To allow pickle to serialize the spec."""
        return DiscreteArray, (self._num_values, self._dtype, self._name)

    @property
    def num_values(self) -> int:
        """Returns the number of items."""
        return self._num_values


@tree_util.register_pytree_node_class
class Tree(Spec):

    def __init__(
            self,
            leaves: Sequence[Spec],
            structure: tree_util.PyTreeDef,
            name: str = ""
    ):
        super().__init__(name=name)
        self.leave_specs = leaves
        self.treedef = structure

    def tree_flatten(
            self
    ) -> tuple[Sequence[Spec], tuple[tree_util.PyTreeDef, str]]:
        # jax.tree hook
        return self.leave_specs, (self.treedef, self.name)

    @classmethod
    def tree_unflatten(
            cls: Type[Tree],
            aux: tuple[tree_util.PyTreeDef, str],
            children: Sequence[Spec]
    ) -> Tree:
        # jax.tree_hook
        return cls(children, *aux)

    @property
    def spec_struct(self):
        return tree_util.tree_unflatten(self.treedef, self.leave_specs)

    def validate(
            self,
            value: PyTree[Num[Array, '...'] | None]
    ) -> PyTree[Num[Array, '...'] | None]:
        as_leaves = tree_util.tree_leaves(value)
        leaves = tree_map(
            lambda v, s: s.validate(v),
            as_leaves, self.leave_specs
        )
        return tree_util.tree_unflatten(self.treedef, leaves)

    def generate_value(self) -> PyTree[Num[Array, '...'] | None]:
        values = [s.generate_value() for s in self.leave_specs]
        return tree_util.tree_unflatten(self.treedef, values)

    def __repr__(self) -> str:
        leave_reprs = tree_map(repr, self.leave_specs)
        tree_repr = repr(tree_util.tree_unflatten(self.treedef, leave_reprs))
        return f'{self.__class__.__name__}(name={self.name}, {tree_repr})'

    def replace(self, **kwargs: Any) -> Tree:
        old_kwargs = {
            'leaves': self.leave_specs,
            'structure': self.treedef,
            'name': self._name
        }
        return self.__class__(**(old_kwargs | kwargs))


class Tuple(Tree):

    def __init__(self, *var_specs: Spec, name: str = ""):
        if (len(var_specs) == 1) and isinstance(var_specs[0], tuple):
            var_specs, = var_specs

        super().__init__(
            list(var_specs),
            tree_util.tree_structure(var_specs),
            name
        )

    def replace(self, *var_specs: Spec, name: str = "") -> Tree:
        return self.__class__(*var_specs, name=name)


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
        super().__init__(
            list((dict_spec | kwarg_specs).values()),
            tree_util.tree_structure(dict_spec | kwarg_specs),
            name=name
        )

    def replace(self, **kwargs: Any) -> Tree:
        return self.__class__(self._defaults | kwargs)


class Batched(Spec):

    def __init__(self, spec: Spec, num: int, name: str = ""):
        super().__init__(name)
        self.spec = spec
        self.num = num

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(name={self.name}, ' \
               f'spec={repr(self.spec)}, num={self.num})'

    def validate(self, value: T) -> T:
        return vmap(self.spec.validate)(value)

    def generate_value(self) -> T:
        base_value = self.spec.generate_value()
        return vmap(lambda x: base_value)(jnp.arange(self.num))

    def replace(self, **kwargs: Any) -> Batched:
        arguments = {
            'action_spec': kwargs.pop('action_spec', default=self.spec),
            'num': kwargs.pop('num', default=self.num),
            'name': kwargs.pop('name', default=self.name)
        }
        return type(self)(**arguments)


class EnvironmentSpec(NamedTuple):
    observations: Spec
    actions: Spec
    rewards: Spec
    discounts: Spec
