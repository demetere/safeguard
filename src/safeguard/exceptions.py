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


class IncorrectCallableException(Exception):
    """Raised when a callable is not a function or a method."""

    def __init__(self, function_name: str, expected: str, got: str) -> None:
        origin_function = function_name.replace("async_", "")
        recommended_function = f"{got}_{origin_function}" if got == "async" else origin_function
        super().__init__(
            "Function '{0}' expected to get a {1} function, but got a {2} function. (maybe call {3})".format(
                function_name,
                expected,
                got,
                recommended_function
            ),
        )


class IncorrectActionCalledException(Exception):
    """Raised when an action is not a function or a method."""

    def __init__(self, action_name: str, expected: str, got: str) -> None:
        origin_function = action_name.replace("async_", "")
        recommended_function = f"{got}_{origin_function}" if got == "async" else origin_function
        super().__init__(
            "Action '{0}' is expecting {1} chain, instead got {2} chain. (maybe call {3})".format(
                action_name,
                expected,
                got,
                recommended_function
            ),
        )
