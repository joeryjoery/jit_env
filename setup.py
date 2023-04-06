from importlib import util
from setuptools import find_packages
from setuptools import setup


def get_version():
    spec = util.spec_from_file_location('version', 'jit_env/version.py')
    mod = util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.__version__


setup(
    name='jit_env',
    version=get_version(),
    description='A Jax interface for Reinforcement Learning environments.',
    author='Joery de Vries',
    keywords='reinforcement-learning python machine learning',
    packages=find_packages(exclude=['examples']),
    python_requires='>=3.7',
    install_requires=[
        'jax',
    ],
    tests_require=[
        'pytest',
    ],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: POSIX :: Linux',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: MacOS :: MacOS X',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)