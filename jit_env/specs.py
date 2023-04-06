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
from typing import Any, TypeVar, Sequence, Type, Generic

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


# TODO: Implement Array, BoundedArray, and DiscreteArray


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
