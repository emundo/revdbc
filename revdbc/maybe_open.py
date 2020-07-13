import os
from typing import cast, Any, AnyStr, Generic, IO, Union

PathLike = Union[str, bytes, os.PathLike]
Openable = Union[PathLike, IO[Any]]


class NullContextManager(Generic[AnyStr]):
    def __init__(self, resource: IO[AnyStr]):
        self._resource: IO[AnyStr] = resource

    def __enter__(self) -> IO[AnyStr]:
        return self._resource

    def __exit__(self, *args: Any) -> None:
        pass


def maybe_open(obj: Openable, mode: str = "r") -> Union[NullContextManager[Any], IO[Any]]:
    """
    Tries to open `obj` as a file. If that attempt fails, assumes that `obj` already is an opened file.

    Args:
        obj: Either some representation of a path to a file or an opened file-like object.
        mode: The mode to open the file with, in case `obj` needs to be opened. Refer to :func:`open` for
            details.

    Returns:
        A context manager which returns either the newly opened file-like object or the original `obj`. Closes
        the file when exiting the context manager if it was opened.

    Raises:
        OSError: upon failure in case `obj` had to be opened.
    """

    try:
        return open(os.fspath(cast(PathLike, obj)), mode)
    except TypeError:
        return NullContextManager(cast(IO[Any], obj))
