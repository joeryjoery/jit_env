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
_req_dev = _parse_requirements(os.path.join(
    _pwd, './', 'requirements_dev.txt'))

_d = {}
with open('jit_env/version.py') as f:
    exec(f.read(), _d)

try:
    __version__ = _d['__version__']
except KeyError as e:
    raise RuntimeError('Module versioning not found!') from e

with open('README.md') as f:
    long_desc = f.read()


setup(
    name='jit_env',
    version=__version__,
    description='A Jax interface for Reinforcement Learning environments.',
    long_description=long_desc,
    long_description_content_type='text/markdown',
    author='Joery A. de Vries',
    author_email="J.A.deVries@tudelft.nl",
    keywords='reinforcement-learning python machine learning jax',
    packages=find_packages(),
    package_data={'jit_env': ['py.typed']},
    url='https://github.com/joeryjoery/jit_env',
    license='MIT',
    python_requires='>=3.9',
    install_requires=_req,
    extras_require={'dev': _req_dev},
    classifiers=[
        'Operating System :: OS Independent',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Software Development :: Testing :: Mocking',
        'Topic :: Software Development :: Testing :: Unit'
    ]
)
