import os
from setuptools import setup, find_packages


def _parse_requirements(path: str) -> list[str]:
    """Returns content of given requirements file."""
    with open(os.path.join(path)) as file:
        lst = [line.rstrip() for line in file
               if not (line.isspace() or line.startswith("#"))]
    return lst


_pwd = os.path.dirname(os.path.abspath(__file__))
_req = _parse_requirements(os.path.join(
    _pwd, './', 'requirements.txt'))

_d = {}
with open('jit_env/version.py') as f:
    exec(f.read(), _d)

try:
    __version__ = _d['__version__']
except KeyError as e:
    raise RuntimeError('Module versioning not found!') from e


setup(
    name='jit_env',
    version=__version__,
    description='A Jax interface for Reinforcement Learning environments.',
    author='Joery A. de Vries',
    author_email="J.A.deVries@tudelft.nl",
    keywords='reinforcement-learning python machine learning jax',
    packages=find_packages(exclude=['examples']),
    python_requires='>=3.7',
    install_requires=_req,
    tests_require=['pytest']
)
