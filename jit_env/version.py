"""Module Versioning. This is directly updated to `setup.py`."""


def _version_as_tuple(version_str: str) -> tuple[int, ...]:
    return tuple(int(i) for i in version_str.split(".") if i.isdigit())


__version__: str = '0.1.3'
__version_info__: tuple[int, ...] = _version_as_tuple(__version__)
