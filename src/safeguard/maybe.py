from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import (
    Any,
    Awaitable,
    Callable,
    Generic,
    NoReturn,
    Optional,
    ParamSpec,
    Type,
    TypeVar,
    final, override,
)

from .exceptions import UnwrapFailedError
from .helper.base_container import BaseContainer

_T = TypeVar("_T", covariant=True)  # Success type
_U = TypeVar("_U")
_P = ParamSpec("_P")
_R = TypeVar("_R")
_TBE = TypeVar("_TBE", bound=BaseException)


class Maybe(
    BaseContainer,
    Generic[_T],
    metaclass=ABCMeta,
):
    """A safeguard object that may or may not contain a value."""

    def is_some(self) -> bool:
        """
        Return whether the safeguard is a `Some`.

        Returns:
            bool: True if the safeguard is a `Some`, False otherwise.

        Examples:
            >>> Some(1).is_some()
            True
            >>> Nothing().is_some()
            False
        """
        return isinstance(self, Some)

    def is_nothing(self) -> bool:
        """
        Return whether the safeguard is a `Nothing`.

        Returns:
            bool: True if the safeguard is a `Nothing`, False otherwise.

        Examples:
            >>> Some(1).is_nothing()
            False
            >>> Nothing().is_nothing()
            True
        """
        return not self.is_some()

    @abstractmethod
    def unwrap(self) -> _T:
        """
        Retrieve the value if present, otherwise raise an exception.

        Returns:
            _T: The contained value if present.

        Raises:
            UnwrapFailedError: If the safeguard is a `Nothing`.

        Examples:
            >>> Some(1).unwrap()
            1
            >>> Nothing().unwrap()
            Traceback (most recent call last):
            ...
            UnwrapFailedError: Called `Maybe.unwrap()` on a `Nothing` value
        """

    @abstractmethod
    def unwrap_or(self, default: _U) -> _T | _U:
        """
        Retrieve the value if present, otherwise return a default value.

        Args:
            default (_U): The default value to return if the safeguard is a `Nothing`.

        Returns:
            _T | _U: The contained value if present, otherwise the default value.

        Examples:
            >>> Some(1).unwrap_or(2)
            1
            >>> Nothing().unwrap_or(2)
            2
        """

    @abstractmethod
    def unwrap_or_else(self, op: Callable[[], _U]) -> _T | _U:
        """
        Retrieve the value if present, otherwise compute a default value.

        Args:
            op (Callable[[], _U]): A function that computes a default value.

        Returns:
            _T | _U: The contained value if present, otherwise the result of the function.

        Examples:
            >>> Some(1).unwrap_or_else(lambda: 2)
            1
            >>> Nothing().unwrap_or_else(lambda: 2)
            2
        """

    @abstractmethod
    def unwrap_or_raise(self, e: Type[_TBE]) -> _T | NoReturn:
        """
        Retrieve the value if present, otherwise raise a specified exception.

        Args:
            e (Type[_TBE]): The exception to raise if the safeguard is a `Nothing`.

        Returns:
            _T: The contained value if present.

        Raises:
            _TBE: The specified exception if the safeguard is a `Nothing`.

        Examples:
            >>> Some(1).unwrap_or_raise(Exception)
            1
            >>> Nothing().unwrap_or_raise(Exception)
            Traceback (most recent call last):
            ...
            Exception
        """

    @abstractmethod
    def map(self, op: Callable[[_T], _U]) -> Maybe[_U]:
        """
        Apply a function to the value if present.

        Args:
            op (Callable[[_T], _U]): The function to apply to the value.

        Returns:
            Maybe[_U]: A new `Maybe` safeguard with the mapped value.

        Examples:
            >>> Some(1).map(lambda x: x + 1).map(lambda x: x * x)
            Some(4)
            >>> Nothing().map(lambda x: x + 1).map(lambda x: x * x)
            Nothing()
        """

    @abstractmethod
    def apply_or(self, op: Callable[[_T], _U], default: _R) -> _U | _R:
        """
        Apply a function to the value if present, otherwise return a default value.

        Args:
            op (Callable[[_T], _U]): The function to apply to the value.
            default (_R): The default value to return if the safeguard is a `Nothing`.

        Returns:
            _U | _R: The result of the function if the value is present, otherwise the default value.

        Examples:
            >>> Some(1).apply_or(lambda x: x + 1, 0)
            2
            >>> Nothing().apply_or(lambda x: x + 1, 0)
            0
        """

    @abstractmethod
    def or_else(self, op: Callable[[], Maybe[_T]]) -> Maybe[_T]:
        """
        Retrieve the safeguard if it has a value, otherwise call a function to get another safeguard.

        Args:
            op (Callable[[], Maybe[_T]]): The function to call if the safeguard is a `Nothing`.

        Returns:
            Maybe[_T]: The current safeguard if it has a value, otherwise the result of the function.

        Examples:
            >>> Some(1).or_else(lambda: Some(2))
            Some(1)
            >>> Nothing().or_else(lambda: Some(2))
            Some(2)
        """

    @abstractmethod
    def filter(self, predicate: Callable[[_T], bool]) -> Maybe[_T]:
        """
        Retrieve the safeguard if the value satisfies a predicate, otherwise return an empty safeguard.

        Args:
            predicate (Callable[[_T], bool]): The predicate to test the value.

        Returns:
            Maybe[_T]: The current safeguard if the value satisfies the predicate, otherwise `Nothing`.

        Examples:
            >>> Some(1).filter(lambda x: x > 0)
            Some(1)
            >>> Some(1).filter(lambda x: x < 0)
            Nothing()
            >>> Nothing().filter(lambda x: x > 0)
            Nothing()
        """

    @classmethod
    def of(cls, value: _T | None) -> Maybe[_T]:
        """
        Create a `Maybe` from a value, returning `Nothing` if the value is `None`.

        Args:
            value (_T | None): The value to wrap in a `Maybe`.

        Returns:
            Maybe[_T]: A `Some` if the value is not `None`, otherwise `Nothing`.

        Examples:
            >>> Maybe.of(1)
            Some(1)
            >>> Maybe.of(None)
            Nothing()
        """
        return Some(value) if value is not None else Nothing()


@final
class Some(Maybe[_T]):
    """A safeguard that indicates the presence of a value."""
    __slots__ = ()
    _value: _T

    def unwrap(self) -> _T:
        """Return the value from the safeguard."""
        return self._value

    def unwrap_or(self, default: _U) -> _T:
        """Return the value from the safeguard."""
        return self._value

    def unwrap_or_else(self, op: Callable[[], _U]) -> _T | _U:
        """Return the value from the safeguard."""
        return self._value

    def unwrap_or_raise(self, e: Type[_TBE]) -> _T:
        """Return the value from the safeguard."""
        return self._value

    def map(self, op: Callable[[_T], _U]) -> Some[_U]:
        """Map the value from the safeguard to another value."""
        return Some(op(self._value))

    def apply_or(self, op: Callable[[_T], _U], default: _R) -> _U:
        """Map the value from the safeguard to another value."""
        return op(self._value)

    def or_else(self, op: Callable[[], Maybe[_T]]) -> Maybe[_T]:
        """Return the current `Maybe` if it is `Some`, otherwise call `op` and return its result."""
        return self

    def filter(self, predicate: Callable[[_T], bool]) -> Maybe[_T]:
        """Return the current `Maybe` if the inner value satisfies the predicate, otherwise return `Nothing`."""
        return self if predicate(self._value) else Nothing()


@final
class Nothing(Maybe[Any]):
    """A safeguard that indicates the absence of a value."""
    __slots__ = ()

    _value: None
    _instance: Optional['Nothing'] = None

    def __new__(cls, *args: Any, **kwargs: Any) -> 'Nothing':
        if cls._instance is None:
            cls._instance = object.__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        super().__init__(None)

    @override
    def __eq__(self, other: Any) -> bool:
        return isinstance(other, Nothing)

    @override
    def __repr__(self) -> str:
        return "Nothing()"

    @override
    def __hash__(self) -> int:
        # A large random number is used here to avoid a hash collision with
        # something else since there is no real inner value for us to hash.
        return hash((False, 982006445019657274590041599673))

    def unwrap(self) -> NoReturn:
        """Raise an `UnwrapFailedError`."""
        raise UnwrapFailedError(self, "Called `Maybe.unwrap()` on a `Nothing` value")

    def unwrap_or(self, default: _U) -> _U:
        """Return `default`."""
        return default

    def unwrap_or_else(self, op: Callable[[], _U]) -> _T | _U:
        """Return the result of calling `op`."""
        return op()

    def unwrap_or_raise(self, e: Type[_TBE]) -> NoReturn:
        """Raise the specified exception."""
        raise e()

    def map(self, op: Callable[[_T], _U]) -> Nothing:
        """Return self because there is no value to map."""
        return self

    def apply_or(self, op: Callable[[_T], _U], default: _R) -> _R:
        """Return `default` because there is no value to map."""
        return default

    def or_else(self, op: Callable[[], Maybe[_T]]) -> Maybe[_T]:
        """Return the result of calling `op`."""
        return op()

    def filter(self, predicate: Callable[[_T], bool]) -> Maybe[_T]:
        """Return self because there is no value to filter."""
        return self


def maybe(f: Callable[_P, _R]) -> Callable[_P, Maybe[_R]]:
    """
    Wrap a function that may return `None`.

    This decorator converts the return value of the function into a `Maybe` object,
    wrapping non-`None` values in a `Some` and `None` values in a `Nothing`.

    Args:
        f (Callable[_P, _R]): The function to wrap.

    Returns:
        Callable[_P, Maybe[_R]]: The wrapped function.

    Examples:
        >>> @maybe
        ... def divide(a: int, b: int) -> float | None:
        ...     if b == 0:
        ...         return None
        ...     return a / b
        >>> divide(1, 0)
        Nothing()
        >>> divide(4, 2)
        Some(2.0)
    """

    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> Maybe[_R]:
        return Maybe.of(f(*args, **kwargs))

    return wrapper


def async_maybe(f: Callable[_P, Awaitable[_R]]) -> Callable[_P, Awaitable[Maybe[_R]]]:
    """
    Wrap an async function that may return `None`.

    This decorator converts the return value of the async function into a `Maybe` object,
    wrapping non-`None` values in a `Some` and `None` values in a `Nothing`.

    Args:
        f (Callable[_P, Awaitable[_R]]): The async function to wrap.

    Returns:
        Callable[_P, Awaitable[Maybe[_R]]]: The wrapped async function.

    Examples:
        >>> import asyncio
        ...
        >>> @async_maybe
        ... async def divide(a: int, b: int) -> float | None:
        ...     if b == 0:
        ...         return None
        ...     return a / b
        ...
        >>> async def nothing_():
        ...     return await divide(1, 0)
        >>> async def some_():
        ...     return await divide(4, 2)
        >>> asyncio.run(nothing_())
        Nothing()
        >>> asyncio.run(some_())
        Some(2.0)
    """

    async def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> Maybe[_R]:
        return Maybe.of(await f(*args, **kwargs))

    return wrapper
