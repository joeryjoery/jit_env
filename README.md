![Run Tox Tests](https://github.com/joeryjoery/jit_env/actions/workflows/tests.yml/badge.svg)
![Python Version](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11-blue)
![Package Version](https://img.shields.io/badge/jit__env-0.1.3-blue)
# `jit_env`: A Jax Compatible RL Environment API
`jit_env` is a library that aims to adhere closely to the `dm_env` interface
while allowing for `jax` transformations inside Environment implementations
and defining clear type annotations.

Like `dm_env` our API consists of the main components:

- `jit_env.Environment`: An abstract base class for RL environments.
- `jit_env.TimeStep`: A container class representing the outputs of the environment on each time step (transition).
- `jit_env.specs`: A module containing primitives that are used to describe the format of the actions consumed by an environment, as well as the observations, rewards, and discounts it returns.

This is extended with the components:
- `jit_env.Wrapper`: An interface built on top of Environment that allows modular transformations of the base Environment.
- `jit_env.Action, jit_env.Observation, jit_env.State`: Explicit types that concern Agent-Environment IO.
- `jit_env.compat`: A Module containing API hooks to other Agent-Environment interfaces like `dm_env` or `gymnasium`.
- `jit_env.wrappers`: A Module containing a few generally useful implementations for `Wrapper` (that simultaneously serves as a reference).

Note that this module is only an interface and does not implement any
Environments itself. The implementations in `examples` serve to illustrate the syntax.
For a more thorough review of the semantics, please refer to the [dm-env](https://github.com/deepmind/dm_env/blob/master/dm_env/specs.py) 
wiki and compare our implementation of `jit_env.Environment` with `dm_env.Environment` and the conversion as given in [`compat.py`](https://github.com/joeryjoery/jit_env/blob/main/jit_env/compat.py).

## Installation
`jit_env` can be installed with:

`python -m pip install jit-env`

You can also install it directly from our GitHub repository using pip:

`python -m pip install git+git://github.com/joeryjoery/jit_env.git`

or alternatively by checking out a local copy of our repository and running:

`python -m pip install /path/to/local/jit_env/`

## The Big Difference with `dm_env`
The main difference between this API and the standard `dm_env` API is
that our definition of `jit_env.Environment` is functionally pure.
This allows the the logic to e.g., be batched over or accelerated 
using `jax.vmap` or `jax.jit`. 

On top of that, we extend the `specs` logic of what `dm_env` provides.
The `specs` module defines primitive for how the Agent interacts with 
the Environment. We explicitly implement additional specs that are 
compatible with `jax` based `PyTree` objects.
This allows for tree-based operations on the `spec` objects themselves, 
which in turn gives some added flexibility in designing desired 
state-action spaces.

Some other modified behaviours include: 
 - `restart` providing a reference value for reward and discount in place of `None`
 - `StepType` is no longer an `enum` type as `jax.jit` would type convert `enum` types to jax primitives anyway. It remains a namespace for defining episode boundaries.
 - `TimeStep` is now a frozen `chex.dataclass` to allow usage of `replace` within the public API (which is private for `NamedTuple`).
 - `TimeStep` carries an additional `extras` field to carry optional data (metrics) not shown to the agent.
 - all helper `restart`, `transition`, etc., now take a `shape` value to generate the reference `reward` or `discount` fields.

### Why `jit_env`
I developed this module to have a reliable Environment backend that is less subject
to refactoring changes as other libraries while providing free compatibility to both `jax` 
transforms as well as any other popular type of Agent-Environment interface. 

The hope is that this library will not require much maintenance/ alterations 
(aside from some type-hint updates) after an official 1.0.0 release. 

## Cite us
If you are a particularly nice person and this work was useful to you, you can
cite this repository as:
```bibtex
@misc{jit_env_2023,
  author={Joery A. de Vries},
  title={jit\_env: A Jax interface for reinforcement learning environments},
  year={2023},
  url={http://github.com/joeryjoery/jit_env}
}
```

## References
This library was heavily inspired by the following libraries:
- dm-env: [https://github.com/deepmind/dm_env](https://github.com/deepmind/dm_env)
- jumanji: [https://github.com/instadeepai/jumanji](https://github.com/instadeepai/jumanji)
- gymnax: [https://github.com/RobertTLange/gymnax](https://github.com/RobertTLange/gymnax)
