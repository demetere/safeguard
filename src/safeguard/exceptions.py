from typing import TypeVar

from .helper.base_container import BaseContainer

_CONTAINER_TYPE = TypeVar('_CONTAINER_TYPE', bound=BaseContainer, covariant=True)


class UnwrapFailedError(Exception):
    """Raised when a safeguard can not be unwrapped into a meaningful value."""

    __slots__ = ('halted_container',)

    def __init__(self, container: _CONTAINER_TYPE, message: str) -> None:
        """
        Saves halted safeguard in the inner state.

        So, this safeguard can later be unpacked from this exception
        and used as a regular value.
        """
        super().__init__(message)
        self.halted_container = container

