# `jit_env`: A Jax Compatible RL Environment API
`jit_env` is a library that aims to adhere closely to the `dm_env` interface
while allowing for `Jax` transformations inside Environment implementations
and defining clear type annotations.

Like `dm_env` our API consists of the main components:

- `jit_env.Environment`: An abstract base class for RL environments.
- `jit_env.TimeStep`: A container class representing the outputs of the environment on each time step (transition).
- `jit_env.specs`: A module containing primitives that are used to describe the format of the actions consumed by an environment, as well as the observations, rewards, and discounts it returns.
- TODO: Testing is not yet implemented.

This is extended with the components:
- `jit_env.Wrapper`: An interface built on top of Environment that allows modular transformations of the base Environment.
- `jit_env.Action, jit_env.Observation, jit_env.State`: Explicit types that concern Agent-Environment IO.
- `jit_env.compat`: A Module containing API hooks to other Agent-Environment interfaces like `dm_env` or `gymnasium`.
- `jit_env.wrappers`: A Module containing a few generally useful implementations for `Wrapper` (that simultaneously serves as a reference).

Note that this module is only an interface and does not implement any
Environments itself.

## The Big Difference
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

## Installation
`jit_env` can be installed with: (FUTURE NOT YET ON PYPI)

`python -m pip install jit-env`

You can also install it directly from our GitHub repository using pip:

`python -m pip install git+git://github.com/joeryjoery/jit_env.git`

or alternatively by checking out a local copy of our repository and running:

`python -m pip install /path/to/local/jit_env/`

## Cite us
If you are a particularly nice person and this work was useful to you, you can
cite this repository as:
```
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


