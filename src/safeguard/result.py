from __future__ import annotations

import functools
import inspect
import typing
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import (
    Any,
    AsyncGenerator,
    Awaitable,
    Callable,
    Coroutine,
    Generator,
    Generic,
    Iterator,
    List,
    NoReturn,
    Optional,
    ParamSpec,
    Self,
    Tuple,
    Type,
    TypeAlias,
    TypeVar,
    Union,
    final,
    overload,
)

from .exceptions import IncorrectActionCalledException, IncorrectCallableException, UnwrapFailedError
from .helper.base_container import BaseContainer

_T = TypeVar("_T", covariant=True)  # Success type
_E = TypeVar("_E", covariant=True)  # Error type
_U = TypeVar("_U")
_F = TypeVar("_F")
_P = ParamSpec("_P")
_R = TypeVar("_R")
_TBE = TypeVar("_TBE", bound=BaseException)


@dataclass(frozen=True, slots=True, eq=False)
class Transformation:
    """
    Represents a transformation to be applied to a `Result` object.

    Attributes:
        method (str): The name of the method to be called.
        func (Callable[..., Any]): The function to be applied.
    """
    method: str
    func: Callable[..., Any]


def _action(
    func: Callable[..., Any],
) -> Callable[..., Any]:
    """Decorator to mark a method as an action for `Result`.

    This ensures that if there are any pending transformations in the chain,
    they are executed before the method is called.
    """

    @functools.wraps(func)
    def wrapper(self: Result[_T, _E], *args: _P.args, **kwargs: _P.kwargs) -> Any | NoReturn:
        result: Result[_T, _E] = self
        if result._chain:
            result = result.execute()

        if self is result:
            return func(result, *args, **kwargs)
        return getattr(result, func.__name__)(*args, **kwargs)

    return wrapper


def _async_action(
    func: Callable[..., Coroutine[Any, Any, Any]]
) -> Callable[..., Coroutine[Any, Any, Any]]:
    """Decorator to mark an async method as an action for `Result`.

    This ensures that if there are any pending transformations in the chain,
    they are executed before the method is called.
    """

    @functools.wraps(func)
    async def async_wrapper(self: Result[_T, _E], *args: _P.args, **kwargs: _P.kwargs) -> Any:
        result: Result[_T, _E] = self
        if result._chain:
            result = await result.async_execute()

        if self is result:
            return await func(result, *args, **kwargs)
        return await getattr(result, func.__name__)(*args, **kwargs)

    return async_wrapper


def _validate_op(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to validate the operation type (sync/async) consistency.
    """

    @functools.wraps(func)
    def wrapper(self: Result[_T, _E], op: Callable[..., Any], *args: _P.args, **kwargs: _P.kwargs) -> Any | NoReturn:
        if (inspect.iscoroutinefunction(func) or 'async' in func.__name__) and not inspect.iscoroutinefunction(op):
            raise IncorrectCallableException(func.__name__, expected="async", got="sync")

        if not inspect.iscoroutinefunction(func) and 'async' not in func.__name__ and inspect.iscoroutinefunction(op):
            raise IncorrectCallableException(func.__name__, expected="sync", got="async")

        return func(self, op, *args, **kwargs)

    return wrapper


def _chain_compatibility(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to ensure chain compatibility between sync and async methods.
    """

    @functools.wraps(func)
    def wrapper(self: Result[_T, _E], *args: _P.args, **kwargs: _P.kwargs) -> Any | NoReturn:
        if self._is_async_chain() and not inspect.iscoroutinefunction(func):
            raise IncorrectActionCalledException(func.__name__, expected="sync", got="async")
        return func(self, *args, **kwargs)

    return wrapper


class Result(BaseContainer, Generic[_T, _E], metaclass=ABCMeta):
    """
    Result safeguard representing either success (`Ok`) or failure (`Err`).

    This safeguard is used to model computations that can fail, similar to
    the Result type in languages like Rust.

    But there is a touch of Python, we are introducing lazy transformations. We
    create chain of `Transformations` and execute them only when we need call `Action` on it.
    Those ideas are taken from `Spark`.

    Methods:
        - is_ok: Check if the result is `Ok`.
        - is_err: Check if the result is `Err`.
        - unwrap: Get the value if `Ok`, or raise an error if `Err`.
        - async_unwrap: Get the value if `Ok`, or raise an error if `Err`.
        - unwrap_err: Get the error if `Err`, or raise an error if `Ok`.
        - async_unwrap_err: Get the error if `Err`, or raise an error if `Ok`.
        - unwrap_or: Get the value if `Ok`, or return a default value if `Err`.
        - async_unwrap_or: Get the value if `Ok`, or return a default value if `Err`.
        - unwrap_or_raise: Get the value if `Ok`, or raise a specified exception if `Err`.
        - async_unwrap_or_raise: Get the value if `Ok`, or raise a specified exception if `Err`.
        - unwrap_or_else: Get the value if `Ok`, or compute a default value if `Err`.
        - async_unwrap_or_else: Get the value if `Ok`, or compute a default value if `Err`.
        - map: Apply a function to the value if `Ok`.
        - async_map: Apply an async function to the value if `Ok`.
        - map_err: Apply a function to the error if `Err`.
        - async_map_err: Apply an async function to the error if `Err`.
        - apply_or: Apply a function to the value if `Ok`, or return a default value if `Err`.
        - async_apply_or: Apply an async function to the value if `Ok`, or return a default value if `Err`.
        - and_then: Apply a function to the value if `Ok`, otherwise return self if `Err`.
        - async_and_then: Apply an async function to the value if `Ok`, otherwise return self if `Err`.
        - or_else: Return self if `Ok`, otherwise apply a function and return the result.
        - async_or_else: Return self if `Ok`, otherwise apply an async function and return the result.
    """
    __slots__ = ('_chain',)
    _value: _T
    _chain: List[Transformation]

    def __init__(self, value: _T) -> None:
        """Initialize empty chain of transformations."""
        super().__init__(value)
        object.__setattr__(self, '_chain', [])

    def is_ok(self) -> bool:
        """
        Return `True` if the safeguard is `Ok`, `False` otherwise.

        Usage:
        >>> Ok(1).is_ok()
        True
        >>> Err(1).is_ok()
        False
        """
        return isinstance(self, Ok)

    def is_err(self) -> bool:
        """
        Return `True` if the safeguard is `Err`, `False` otherwise.

        Usage:
        >>> Ok(1).is_err()
        False
        >>> Err(1).is_err()
        True
        """
        return not self.is_ok()

    def _is_async_chain(self) -> bool:
        """Check if the chain of transformations contains async functions."""
        return any(
            inspect.iscoroutinefunction(transformation.func)
            for transformation in self._chain
        )

    @abstractmethod
    def unwrap(self) -> _T | NoReturn:
        """Get value from successful safeguard or raise an exception for failed one.
        
        This method is an `Action` so it will call `exceute()` on the chain of `Transformations`
        if there is any `Transformation` in the chain.

        Usage:
        >>> Ok(1).unwrap()
        1
        >>> Err(1).unwrap()
        Traceback (most recent call last):
        ...
        UnwrapFailedError: Called `Result.unwrap()` on an `Err` value
        """

    @abstractmethod
    async def async_unwrap(self) -> _T | NoReturn:
        """Get value from successful safeguard or raise an exception for failed one.

        This method is an `Action` so it will call `exceute()` on the chain of `Transformations`
        if there is any `Transformation` in the chain.

        Usage:
        TODO
        """

    @abstractmethod
    def unwrap_err(self) -> _E | NoReturn:
        """Get error from failed safeguard or raise an exception for successful one.
        
        This method is an `Action` so it will call `exceute()` on the chain of `Transformations`
        if there is any `Transformation` in the chain.

        Usage:
        >>> Ok(1).unwrap_err()
        Traceback (most recent call last):
        ...
        UnwrapFailedError: Called `Result.unwrap_err()` on an `Ok` value
        >>> Err(1).unwrap_err()
        1
        """

    @abstractmethod
    async def async_unwrap_err(self) -> _E | NoReturn:
        """Get error from failed safeguard or raise an exception for successful one.

        This method is an `Action` so it will call `exceute()` on the chain of `Transformations`
        if there is any `Transformation` in the chain.

        Usage:
        TODO
        """

    @abstractmethod
    def unwrap_or(self, _default: _U) -> _T | _U:
        """Get value from successful safeguard or return default for failed one.
        
        This method is an `Action` so it will call `exceute()` on the chain of `Transformations`
        if there is any `Transformation` in the chain.

        Usage:
        >>> Ok(1).unwrap_or(2)
        1
        >>> Err(1).unwrap_or(2)
        2
        """

    @abstractmethod
    async def async_unwrap_or(self, _default: _U) -> _T | _U:
        """Get value from successful safeguard or return default for failed one.

        This method is an `Action` so it will call `exceute()` on the chain of `Transformations`
        if there is any `Transformation` in the chain.

        Usage:
        TODO
        """

    @abstractmethod
    def unwrap_or_raise(self, e: Type[_TBE] | None = None) -> _T | NoReturn:
        """Get value from successful safeguard or raise an exception for failed one.
        
        This method is an `Action` so it will call `exceute()` on the chain of `Transformations`
        if there is any `Transformation` in the chain.

        Usage:
        >>> Ok(1).unwrap_or_raise(Exception)
        1
        >>> Err(1).unwrap_or_raise(Exception)
        Traceback (most recent call last):
        ...
        Exception
        >>> Err(Exception()).unwrap_or_raise()
        Traceback (most recent call last):
        ...
        Exception
        """

    @abstractmethod
    async def async_unwrap_or_raise(self, e: Type[_TBE] | None = None) -> _T | NoReturn:
        """Get value from successful safeguard or raise an exception for failed one.

        This method is an `Action` so it will call `exceute()` on the chain of `Transformations`
        if there is any `Transformation` in the chain.

        Usage:
        TODO
        """

    @abstractmethod
    def unwrap_or_else(self, op: Callable[[], _U]) -> _T | _U:
        """Get value from successful safeguard or compute a default for failed one.
        
        This method is an `Action` so it will call `exceute()` on the chain of `Transformations`
        if there is any `Transformation` in the chain.

        Usage:
        TODO
        """

    @abstractmethod
    async def async_unwrap_or_else(self, op: Callable[[], _U | Awaitable[_U]]) -> _T | _U:
        """Get value from successful safeguard or compute a default for failed one.

        This method is an `Action` so it will call `exceute()` on the chain of `Transformations`
        if there is any `Transformation` in the chain.

        Usage:
        TODO

        """

    @abstractmethod
    def apply_or(self, op: Callable[[_T], _U], default: _R) -> _U | _R:
        """Apply the operation to the value of a successful safeguard or return the default value.

        This method creates `Action` and adds calls execute on it.

        Usage:
        >>> Ok(1).apply_or(lambda x: x * x, 0)
        1
        >>> Err(1).apply_or(lambda x: x * x, 0)
        0
        """

    @abstractmethod
    async def async_apply_or(self, op: Callable[[_T], Awaitable[_U]], default: _R) -> _U | _R:
        """Apply the operation to the value of a successful safeguard or return the default value.

        This method creates `Action` and adds calls execute on it.

        # TODO
        """

    @_validate_op
    def map(self, op: Callable[[_T], _U]) -> Self:
        """Map the value of a successful safeguard to a new value. For failed containers, return self.

        This method creates a new `Transformation` and adds it to the chain.

        Usage:
        >>> Ok(1).map(lambda x: x + 1).map(str).execute()
        Ok('2')
        >>> Err(1).map(lambda x: x + 1).map(str).execute()
        Err(1)
        """
        self._chain.append(Transformation('_map', op))
        return self

    @abstractmethod
    def _map(self, op: Callable[[_T], _U]) -> Result[_U, _E]:
        """Apply Transformation to the value of the safeguard."""

    @_validate_op
    def async_map(self, op: Callable[[_T], Awaitable[_U]]) -> Self:
        """Map the value of a successful safeguard to a new value. For failed containers, return self.

        This method creates a new `Transformation` and adds it to the chain.

        Usage:
        >>> Ok(1).async_map(lambda x: x + 1).async_map(str).execute()
        Ok('2')
        >>> Err(1).async_map(lambda x: x + 1).async_map(str).execute()
        Err(1)
        """
        self._chain.append(Transformation('_async_map', op))
        return self

    @abstractmethod
    async def _async_map(self, op: Callable[[_T], Awaitable[_U]]) -> Result[_U, _E]:
        """Apply Transformation to the value of the safeguard."""

    @_validate_op
    def map_err(self, op: Callable[[_E], _R | Awaitable[_R]]) -> Self:
        """Map the error of a failed safeguard to a new value. For successful containers, do nothing.
        
        This method creates a new `Transformation` and adds it to the chain.

        Usage:
        >>> Ok(1).map_err(lambda x: x + 1).execute()
        Ok(1)
        >>> Err(1).map_err(lambda x: x + 1).execute()
        Err(2)
        """
        self._chain.append(Transformation('_map_err', op))
        return self

    @abstractmethod
    def _map_err(self, op: Callable[[_E], _R]) -> Result[_T, _R]:
        """Apply Transformation to the error of the safeguard."""

    @_validate_op
    def async_map_err(self, op: Callable[[_E], Awaitable[_R]]) -> Self:
        """Map the error of a failed safeguard to a new value. For successful containers, do nothing.

        This method creates a new `Transformation` and adds it to the chain.

        Usage:
        >>> Ok(1).async_map_err(lambda x: x + 1).execute()
        Ok(1)
        >>> Err(1).async_map_err(lambda x: x + 1).execute()
        Err(2)
        """
        self._chain.append(Transformation('_async_map_err', op))
        return self

    @abstractmethod
    async def _async_map_err(self, op: Callable[[_E], Awaitable[_R]]) -> Result[_T, _R]:
        """Apply Transformation to the error of the safeguard."""

    @_validate_op
    def and_then(self, op: Callable[[_T], Result[_U, _E]]) -> Self:
        """Do nothing if failed, otherwise apply the operation to the value.

        This method creates `Transformation` and adds it to the chain.

        Usage:
        >>> Ok(1).and_then(lambda x: Ok(x + 1))
        Ok(2)
        >>> Err(1).and_then(lambda x: Ok(x + 1))
        Err(1)
        """
        self._chain.append(Transformation('_and_then', op))
        return self

    @abstractmethod
    def _and_then(self, op: Callable[[_T], Result[_U, _E]]) -> Result[_U | _T, _E]:
        """Apply the operation to the value of the safeguard."""

    @_validate_op
    def async_and_then(self, op: Callable[[_T], Awaitable[Result[_U, _E]]]) -> Self:
        """Do nothing if failed, otherwise apply the operation to the value.

        This method creates `Transformation` and adds it to the chain.

        Usage:
        >>> Ok(1).and_then(lambda x: Ok(x + 1))
        Ok(2)
        >>> Err(1).and_then(lambda x: Ok(x + 1))
        Err(1)
        """
        self._chain.append(Transformation('_async_and_then', op))
        return self

    @abstractmethod
    async def _async_and_then(self, op: Callable[[_T], Awaitable[Result[_U, _E]]]) -> Result[_U | _T, _E]:
        """Apply the operation to the value of the safeguard."""

    @_validate_op
    def or_else(self, op: Callable[[], Result[_T, _E]]) -> Self:
        """Do nothing if successful, otherwise apply the operation and return the result.

        This method creates `Transformation` and adds it to the chain.

        Usage:
        >>> Ok(1).or_else(lambda: Ok(2))
        Ok(1)
        >>> Err(1).or_else(lambda: Ok(2))
        Ok(2)
        """
        self._chain.append(Transformation('_or_else', op))
        return self

    @abstractmethod
    def _or_else(self, op: Callable[[], Result[_T, _E]]) -> Result[_T, _E]:
        """Apply the operation to the safeguard."""

    @_validate_op
    def async_or_else(self, op: Callable[[], Awaitable[Result[_T, _E]]]) -> Self:
        """Do nothing if successful, otherwise apply the operation and return the result.

        This method creates `Transformation` and adds it to the chain.

        Usage:
        >>> Ok(1).or_else(lambda: Ok(2))
        Ok(1)
        >>> Err(1).or_else(lambda: Ok(2))
        Ok(2)
        """
        self._chain.append(Transformation('_async_or_else', op))
        return self

    @abstractmethod
    async def _async_or_else(self, op: Callable[[], Awaitable[Result[_T, _E]]]) -> Result[_T, _E]:
        """Apply the operation to the safeguard."""

    @_validate_op
    def inspect(self, op: Callable[[_T], Any]) -> Self:
        """Calls a function with the contained value if `Ok`. Returns the original result.

        Usage:
        TODO
        """
        self._chain.append(Transformation('_inspect', op))
        return self

    @abstractmethod
    def _inspect(self, op: Callable[[_T], Any]) -> Result[_T, _E]:
        """Calls a function with the contained value if `Ok`. Returns the original result."""

    @_validate_op
    def async_inspect(self, op: Callable[[_T], Awaitable[Any]]) -> Self:
        """Calls a function with the contained value if `Ok`. Returns the original result.

        Usage:
        TODO
        """
        self._chain.append(Transformation('_async_inspect', op))
        return self

    @abstractmethod
    async def _async_inspect(self, op: Callable[[_T], Awaitable[Any]]) -> Result[_T, _E]:
        """Calls a function with the contained value if `Ok`. Returns the original result."""

    @_validate_op
    def inspect_err(self, op: Callable[[_E], Any]) -> Self:
        """Calls a function with the contained value if `Err`. Returns the original result.

        Usage:
        TODO:

        """
        self._chain.append(Transformation('_inspect_err', op))
        return self

    @abstractmethod
    def _inspect_err(self, op: Callable[[_E], Any]) -> Result[_T, _E]:
        """Calls a function with the contained value if `Err`. Returns the original result."""

    @_validate_op
    def async_inspect_err(self, op: Callable[[_E], Awaitable[Any]]) -> Self:
        """Calls a function with the contained value if `Err`. Returns the original result.

        Usage:
        TODO
        """
        self._chain.append(Transformation('_async_inspect_err', op))
        return self

    @abstractmethod
    async def _async_inspect_err(self, op: Callable[[_E], Awaitable[Any]]) -> Result[_T, _E]:
        """Calls a function with the contained value if `Err`. Returns the original result."""

    @_chain_compatibility
    def execute(self) -> Result[_T, _E]:
        """Execute the chain of operations and return the final result.
        """
        result: Result[_T, _E] = self
        for transformation in self._chain:
            method = getattr(result, transformation.method)
            result = method(transformation.func)
        return result

    async def async_execute(self) -> Result[_T, _E]:
        """Execute the chain of operations and return the final result.
        """
        result: Result[_T, _E] = self
        for transformation in self._chain:
            method = getattr(result, transformation.method)
            if inspect.iscoroutinefunction(method):
                result = await method(transformation.func)
            else:
                result = method(transformation.func)
        return result


ResultE: TypeAlias = Result[_T, Exception]


@final
class Ok(Result[_T, Any]):
    """
    A value that indicates success and which stores arbitrary data for the return value.

    Methods:
        - unwrap: Return the value.
        - unwrap_err: Raise `UnwrapFailedError`.
        - unwrap_or: Return the value.
        - unwrap_or_raise: Return the value.
        - unwrap_or_else: Return the value.
        - map: Apply a function to the value.
        - async_map: Apply an async function to the value.
        - map_err: Return self.
        - apply_or: Apply a function to the value.
        - or_else: Return self.
        - and_then: Apply a function to the value.
    """
    __slots__ = ()
    _chain: List[Transformation]
    _value: _T

    def __iter__(self) -> Iterator[_T]:
        yield self._value

    @_chain_compatibility
    @_action
    def unwrap(self) -> _T:
        """Return the value from the safeguard."""
        return self._value

    @_async_action
    async def async_unwrap(self) -> _T:
        """Return the value from the safeguard."""
        return self._value

    @_chain_compatibility
    @_action
    def unwrap_err(self) -> NoReturn:
        """Raise `UnwrapFailedError`."""
        raise UnwrapFailedError(self, "Called `Result.unwrap_err()` on an `Ok` value")

    @_async_action
    async def async_unwrap_err(self) -> NoReturn:
        """Raise `UnwrapFailedError`."""
        raise UnwrapFailedError(self, "Called `Result.async_unwrap_err()` on an `Ok` value")

    @_chain_compatibility
    @_action
    def unwrap_or(self, _default: _U) -> _T:
        """Return the value from the safeguard."""
        return self._value

    @_async_action
    async def async_unwrap_or(self, _default: _U) -> _T:
        return self._value

    @_chain_compatibility
    @_action
    def unwrap_or_raise(self, e: Type[_TBE] | None = None) -> _T:
        """Return the value from the safeguard."""
        return self._value

    @_async_action
    async def async_unwrap_or_raise(self, e: Type[_TBE] | None = None) -> _T:
        """Return the value from the safeguard."""
        return self._value

    @_chain_compatibility
    @_validate_op
    @_action
    def unwrap_or_else(self, op: Callable[[], _U]) -> _T:
        """Return the value from the safeguard."""
        return self._value

    @_async_action
    async def async_unwrap_or_else(self, op: Callable[[], _U]) -> _T:
        """Return the value from the safeguard."""
        return self._value

    @_chain_compatibility
    @_validate_op
    @_action
    def apply_or(self, op: Callable[[_T], _U], default: _R) -> _U:
        """Return the result of applying `op` to the value."""
        return op(self._value)

    @_async_action
    @typing.no_type_check  # TODO: Remove this and typehint properly
    async def async_apply_or(self, op: Callable[[_T], _U | Awaitable[_U]], default: _R) -> _U:
        """Return the result of applying `op` to the value."""
        if inspect.iscoroutinefunction(op):
            return await op(self._value)
        return op(self._value)

    def _map(self, op: Callable[[_T], _U]) -> Ok[_U]:
        """Return the result of applying `op` to the value."""
        return Ok(op(self._value))

    async def _async_map(self, op: Callable[[_T], Awaitable[_U]]) -> Ok[_U]:
        """Return the result of applying `op` to the value."""
        return Ok(await op(self._value))

    def _map_err(self, op: Callable[[_U], _R]) -> Ok[_T]:
        """Return self."""
        return self

    async def _async_map_err(self, op: Callable[[_U], Awaitable[_R]]) -> Ok[_T]:
        """Return self."""
        return self

    def _and_then(self, op: Callable[[_T], Result[_U, _E]]) -> Result[_U, _E]:
        """Return the result of applying `op` to the value."""
        return op(self._value)

    async def _async_and_then(self, op: Callable[[_T], Awaitable[Result[_U, _E]]]) -> Result[_U, _E]:
        """Return the result of applying `op` to the value."""
        return await op(self._value)

    def _or_else(self, op: Callable[[], Result[_T, _E]]) -> Ok[_T]:
        """Return self."""
        return self

    async def _async_or_else(self, op: Callable[[], Awaitable[Result[_T, _E]]]) -> Ok[_T]:
        """Return self."""
        return self

    def _inspect(self, op: Callable[[_T], Any]) -> Ok[_T]:
        """Call the function with the value. Return self."""
        op(self._value)
        return self

    async def _async_inspect(self, op: Callable[[_T], Awaitable[Any]]) -> Ok[_T]:
        """Call the function with the value. Return self."""
        await op(self._value)
        return self

    def _inspect_err(self, op: Callable[[_E], Any]) -> Ok[_T]:
        """Return self."""
        return self

    async def _async_inspect_err(self, op: Callable[[_E], Awaitable[Any]]) -> Ok[_T]:
        """Return self."""
        return self


@final
class Err(Result[Any, _E]):
    """
    A value that indicates failure and which stores arbitrary data for the error value.

    Methods:
        - unwrap: Raise `UnwrapFailedError`.
        - unwrap_err: Return the error value.
        - unwrap_or: Return the default value.
        - unwrap_or_raise: Raise the specified exception.
        - unwrap_or_else: Return the result of the operation.
        - map: Return self.
        - async_map: Return self.
        - map_err: Apply a function to the error.
        - apply_or: Return the default value.
        - or_else: Return the result of the operation.
        - and_then: Return self.
    """
    __slots__ = ()
    _chain: List[Transformation]
    _value: _E

    def __iter__(self) -> Iterator[_T]:
        def _iter() -> Iterator[NoReturn]:
            # Exception will be raised when the iterator is advanced, not when it's created
            raise DoException(self)
            yield  # This yield will never be reached, but is necessary to create a generator

        return _iter()

    @_chain_compatibility
    @_action
    def unwrap(self) -> NoReturn:
        """Raise `UnwrapFailedError`."""
        raise UnwrapFailedError(self, "Called `Result.unwrap()` on an `Err` value")

    @_async_action
    async def async_unwrap(self) -> NoReturn:
        """Raise `UnwrapFailedError`."""
        raise UnwrapFailedError(self, "Called `Result.unwrap_async()` on an `Err` value")

    @_chain_compatibility
    @_action
    def unwrap_err(self) -> _E:
        """Return the error value."""
        return self._value

    @_async_action
    async def async_unwrap_err(self) -> _E:
        """Return the error value."""
        return self._value

    @_chain_compatibility
    @_action
    def unwrap_or(self, _default: _U) -> _U:
        """Return the default value."""
        return _default

    @_async_action
    async def async_unwrap_or(self, _default: _U) -> _U:
        """Return the default value."""
        return _default

    @_chain_compatibility
    @_action
    def unwrap_or_raise(self, e: Type[_TBE] | None = None) -> NoReturn:
        """Raise the specified exception."""
        if e is not None:
            raise e
        if isinstance(self._value, Exception):
            raise self._value
        raise TypeError(f"Expected an exception, got {self._value!r}")

    @_async_action
    async def async_unwrap_or_raise(self, e: Type[_TBE] | None = None) -> NoReturn:
        """Raise the specified exception."""
        if e:
            raise e
        if isinstance(self._value, Exception):
            raise self._value
        raise TypeError(f"Expected an exception, got {self._value!r}")

    @_chain_compatibility
    @_validate_op
    @_action
    def unwrap_or_else(self, op: Callable[[], _U]) -> _U:
        """Return the result of the operation."""
        return op()

    @_async_action
    @typing.no_type_check  # TODO: Remove this and typehint properly
    async def async_unwrap_or_else(self, op: Callable[[], _U | Awaitable[_U]]) -> _U:
        """Return the result of the operation."""
        if inspect.iscoroutinefunction(op):
            return await op()
        return op()

    @_chain_compatibility
    @_validate_op
    @_action
    def apply_or(self, op: Callable[[_T], _U], default: _R) -> _R:
        """Return the default value."""
        return default

    @_async_action
    async def async_apply_or(self, op: Callable[[_T], Awaitable[_U]], default: _R) -> _R:
        """Return the default value."""
        return default

    def _map(self, op: Callable[[_T], _U]) -> Err[_E]:
        """Return self."""
        return self

    async def _async_map(self, op: Callable[[_T], Awaitable[_U]]) -> Err[_E]:
        """Return self."""
        return self

    def _map_err(self, op: Callable[[_E], _R]) -> Err[_R]:
        """Return the result of applying `op` to the error."""
        return Err(op(self._value))

    async def _async_map_err(self, op: Callable[[_E], Awaitable[_R]]) -> Err[_R]:
        """Return the result of applying `op` to the error."""
        return Err(await op(self._value))

    def _and_then(self, op: Callable[[_T], Result[_U, _E]]) -> Err[_E]:
        """Return self."""
        return self

    async def _async_and_then(self, op: Callable[[_T], Awaitable[Result[_U, _E]]]) -> Err[_E]:
        """Return self."""
        return self

    def _or_else(self, op: Callable[[], Result[_T, _E]]) -> Result[_T, _E]:
        """Return the result of the operation."""
        return op()

    async def _async_or_else(self, op: Callable[[], Awaitable[Result[_T, _E]]]) -> Result[_T, _E]:
        """Return the result of the operation."""
        return await op()

    def _inspect(self, op: Callable[[_T], Any]) -> Err[_E]:
        """Return self."""
        return self

    async def _async_inspect(self, op: Callable[[_T], Awaitable[Any]]) -> Err[_E]:
        """Return self."""
        return self

    def _inspect_err(self, op: Callable[[_E], Any]) -> Err[_E]:
        """Call the function with the error. Return self."""
        op(self._value)
        return self

    async def _async_inspect_err(self, op: Callable[[_E], Awaitable[Any]]) -> Err[_E]:
        """Call the function with the error. Return self."""
        await op(self._value)
        return self


class DoException(Exception):
    """
    This is used to signal to `do()` that the safeguard is an `Err`,
    which short-circuits the generator and returns that Err.
    Using this exception for control flow in `do()` allows us
    to simulate `and_then()` in the Err case: namely, we don't call `op`,
    we just return `self` (the Err).
    """

    def __init__(self, err: Err[_E]) -> None:
        self.err = err


@overload
def safe(
    function: Callable[_P, _R] | Callable[_P, Generator[_R, Any, Any]]
) -> Callable[_P, Result[_R, Exception]] | Callable[_P, Generator[Result[_R, Exception], Any, Any]]:
    ...


@overload
def safe(
    *,
    exceptions: Tuple[Type[Exception], ...]
) -> Callable[
    [Callable[_P, _R] | Callable[_P, Generator[_R, Any, Any]]],
    Callable[_P, Result[_R, Exception]] | Callable[_P, Generator[Result[_R, Exception], Any, Any]]
]:
    ...


def safe(
    function: Optional[Callable[_P, _R]] | Optional[Callable[_P, Generator[_R, Any, Any]]] = None,
    *,
    exceptions: Optional[Tuple[Type[Exception], ...]] = None
) -> Union[
    Callable[_P, Result[_R, Exception]] | Callable[_P, Generator[Result[_R, Exception], Any, Any]],
    Callable[
        [Callable[_P, _R] | Callable[_P, Generator[_R, Any, Any]]],
        Callable[_P, Result[_R, Exception]] | Callable[_P, Generator[Result[_R, Exception], Any, Any]]
    ]
]:
    """Decorator to convert exception-throwing function to ``Result`` safeguard.

        Should be used with care, since it only catches ``Exception`` subclasses.
        It does not catch ``BaseException`` subclasses.

        If you need to mark ``async`` function as ``safe``,
        use :func:`safeguard.result.async_safe` instead.
        This decorator only works with sync functions.

        Usage:
        >>> @safe
        ... def func(a: int) -> float:
        ...     return 1 / a
        ...
        >>> assert func(1) == Ok(1.0)
        >>> assert func(0) == Err(ZeroDivisionError)

        >>> @safe(exceptions=(ZeroDivisionError,))
        ... def func(a: int) -> float:
        ...     return 1 / a
        ...
        >>> assert func(1) == Ok(1.0)
        >>> assert func(0) == Err(ZeroDivisionError)
    """

    def decorator(
        function_: Callable[_P, _R] | Callable[_P, Generator[_R, Any, Any]],
        exceptions_: Tuple[Type[_TBE], ...]
    ) -> Callable[_P, Result[_R, _TBE]] | Callable[_P, Generator[Result[_R, _TBE], Any, Any]]:
        """
        Decorator to turn a function or generator into one that returns a ``Result``.
        """

        @functools.wraps(function_)
        def wrapper_function(*args: _P.args, **kwargs: _P.kwargs) -> Result[_R, _TBE]:
            assert inspect.isfunction(function_)
            try:
                return Ok(function_(*args, **kwargs))
            except exceptions_ as exc:
                return Err(exc)

        @functools.wraps(function_)
        def wrapper_generator(*args: _P.args, **kwargs: _P.kwargs) -> Generator[Result[_R, _TBE], Any, Any]:
            assert inspect.isgeneratorfunction(function_)
            try:
                for value in function_(*args, **kwargs):
                    yield Ok(value)
            except exceptions_ as exc:
                yield Err(exc)

        if inspect.isgeneratorfunction(function_):
            return wrapper_generator
        return wrapper_function

    if callable(function):
        return decorator(function, exceptions_=(Exception,))
    if isinstance(exceptions, tuple):
        return lambda f_: decorator(f_, exceptions)

    raise TypeError("safe() requires either a function or a tuple of exceptions")


@overload
def async_safe(
    function: Callable[_P, Awaitable[_R]] | Callable[_P, AsyncGenerator[_R, Any]]
) -> Callable[_P, Awaitable[Result[_R, Exception]]] | Callable[_P, AsyncGenerator[Result[_R, Exception], Any]]:
    ...


@overload
def async_safe(
    *,
    exceptions: Tuple[Type[Exception], ...]
) -> Callable[
    [Callable[_P, Awaitable[_R]] | Callable[_P, AsyncGenerator[_R, Any]]],
    Callable[_P, Awaitable[Result[_R, Exception]]] | Callable[_P, AsyncGenerator[Result[_R, Exception], Any]]
]:
    ...


def async_safe(
    function: Optional[Callable[_P, Awaitable[_R]]] | Optional[Callable[_P, AsyncGenerator[_R, Any]]] = None,
    *,
    exceptions: Optional[Tuple[Type[Exception], ...]] = None
) -> Union[
    Callable[_P, Awaitable[Result[_R, Exception]]] | Callable[_P, AsyncGenerator[Result[_R, Exception], Any]],
    Callable[
        [Callable[_P, Awaitable[_R]] | Callable[_P, AsyncGenerator[_R, Any]]],
        Callable[_P, Awaitable[Result[_R, Exception]]] | Callable[_P, AsyncGenerator[Result[_R, Exception], Any]]
    ]
]:
    """
    Decorator to convert exception-throwing async function or async generator to ``Result`` safeguard.

    Should be used with care, since it only catches ``Exception`` subclasses.
    It does not catch ``BaseException`` subclasses.

    Usage:
    >>> import asyncio
    ...
    >>> @async_safe
    ... async def func(a: int) -> float:
    ...     return 1 / a
    ...
    >>> async def check_without_parameters() -> None:
    ...     assert await func(1) == Ok(1.0)
    ...     assert await func(0) == Err(ZeroDivisionError)
    ...
    >>> @async_safe(exceptions=(ZeroDivisionError,))
    ... async def func(a: int) -> float:
    ...     return 1 / a
    ...
    >>> async def check_with_parameters() -> None:
    ...     assert await func(1) == Ok(1.0)
    ...     assert await func(0) == Err(ZeroDivisionError)
    ...
    >>> asyncio.run(check_without_parameters())
    >>> asyncio.run(check_with_parameters())
    """

    def decorator(
        function_: Callable[_P, Awaitable[_R]] | Callable[_P, AsyncGenerator[_R, Any]],
        exceptions_: Tuple[Type[Exception], ...]
    ) -> Callable[_P, Awaitable[Result[_R, Exception]]] | Callable[_P, AsyncGenerator[Result[_R, Exception], Any]]:
        """Decorator to turn an async function or async generator into one that returns a ``Result``.
        """

        @functools.wraps(function_)
        async def wrapper_function(*args: _P.args, **kwargs: _P.kwargs) -> Result[_R, Exception]:
            assert inspect.iscoroutinefunction(function_)
            try:
                return Ok(await function_(*args, **kwargs))
            except exceptions_ as exc:
                return Err(exc)

        @functools.wraps(function_)
        async def wrapper_generator(*args: _P.args, **kwargs: _P.kwargs) -> AsyncGenerator[Result[_R, Exception], Any]:
            assert inspect.isasyncgenfunction(function_)
            try:
                async for value in function_(*args, **kwargs):
                    yield Ok(value)
            except exceptions_ as exc:
                yield Err(exc)

        if inspect.isasyncgenfunction(function_):
            return wrapper_generator
        return wrapper_function

    if callable(function):
        return decorator(function, exceptions_=(Exception,))
    if isinstance(exceptions, tuple):
        return lambda f_: decorator(f_, exceptions)

    raise TypeError("async_safe() requires either a function or a tuple of exceptions")


@typing.no_type_check  # TODO: Remove this and typehint properly
def do(gen: Generator[Result[_T, _E], None, None]) -> Result[_T, _E]:
    """Do notation for Result (syntactic sugar for sequence of `and_then()` calls).


    Usage:

    ``` rust
    // This is similar to
    use do_notation::m;
    let final_result = m! {
        x <- Ok("hello");
        y <- Ok(True);
        Ok(len(x) + int(y) + 0.5)
    };
    ```

    ``` python
    final_result: Result[float, int] = do(
            Ok(len(x) + int(y) + 0.5)
            for x in Ok("hello")
            for y in Ok(True)
        )
    ```

    NOTE: If you exclude the type annotation e.g. `Result[float, int]`
    your type checker might be unable to infer the return type.
    To avoid an error, you might need to help it with the type hint.
    """
    try:
        return next(gen)
    except DoException as e:
        out: Err[_E] = e.err  # type: ignore
        return out
    except TypeError as te:
        # Turn this into a more helpful error message.
        # Python has strange rules involving turning generators involving `await`
        # into async generators, so we want to make sure to help the user clearly.
        if "'async_generator' object is not an iterator" in str(te):
            raise TypeError(
                "Got async_generator but expected generator."
                "See the section on do notation in the README."
            )
        raise te


async def do_async(
    gen: Union[Generator[Result[_T, _E], None, None], AsyncGenerator[Result[_T, _E], None]]
) -> Result[_T, _E]:
    """Async version of do. Example:

    ``` python
    final_result: Result[float, int] = await do_async(
        Ok(len(x) + int(y) + z)
            for x in await get_async_result_1()
            for y in await get_async_result_2()
            for z in get_sync_result_3()
        )
    ```

    NOTE: Python makes generators async in a counter-intuitive way.

    ``` python
    # This is a regular generator:
        async def foo(): ...
        do(Ok(1) for x in await foo())
    ```

    ``` python
    # But this is an async generator:
        async def foo(): ...
        async def bar(): ...
        do(
            Ok(1)
            for x in await foo()
            for y in await bar()
        )
    ```

    We let users try to use regular `do()`, which works in some cases
    of awaiting async values. If we hit a case like above, we raise
    an exception telling the user to use `do_async()` instead.
    See `do()`.

    However, for better usability, it's better for `do_async()` to also accept
    regular generators, as you get in the first case:

    ``` python
    async def foo(): ...
        do(Ok(1) for x in await foo())
    ```

    Furthermore, neither mypy nor pyright can infer that the second case is
    actually an async generator, so we cannot annotate `do_async()`
    as accepting only an async generator. This is additional motivation
    to accept either.
    """
    try:
        if isinstance(gen, AsyncGenerator):
            return await gen.__anext__()
        else:
            return next(gen)
    except DoException as e:
        out: Err[_E] = e.err  # type: ignore
        return out
