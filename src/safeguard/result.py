from __future__ import annotations

import functools
import inspect
from abc import ABCMeta, abstractmethod
from typing import (
    Any,
    AsyncGenerator,
    Awaitable,
    Callable,
    Generator,
    Generic,
    Iterator,
    NoReturn,
    Optional, ParamSpec,
    Tuple, Type,
    TypeVar,
    Union,
    cast, final,
)

from .exceptions import UnwrapFailedError
from .helper.base_container import BaseContainer

_T = TypeVar("_T", covariant=True)  # Success type
_E = TypeVar("_E", covariant=True)  # Error type
_U = TypeVar("_U")
_F = TypeVar("_F")
_P = ParamSpec("_P")
_R = TypeVar("_R")
_TBE = TypeVar("_TBE", bound=BaseException)


class Result(BaseContainer, Generic[_T, _E], metaclass=ABCMeta):
    """
    Result safeguard representing either success (`Ok`) or failure (`Err`).

    This safeguard is used to model computations that can fail, similar to
    the Result type in languages like Rust.

    Methods:
        - is_ok: Check if the result is `Ok`.
        - is_err: Check if the result is `Err`.
        - unwrap: Get the value if `Ok`, or raise an error if `Err`.
        - unwrap_err: Get the error if `Err`, or raise an error if `Ok`.
        - unwrap_or: Get the value if `Ok`, or return a default value if `Err`.
        - unwrap_or_raise: Get the value if `Ok`, or raise a specified exception if `Err`.
        - unwrap_or_else: Get the value if `Ok`, or compute a default value if `Err`.
        - map: Apply a function to the value if `Ok`.
        - map_async: Apply an async function to the value if `Ok`.
        - map_err: Apply a function to the error if `Err`.
        - apply_or: Apply a function to the value if `Ok`, or return a default value if `Err`.
        - or_else: Return self if `Ok`, otherwise apply a function and return the result.
        - and_then: Apply a function to the value if `Ok`, otherwise return self if `Err`.
    """
    __slots__ = ()

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

    @abstractmethod
    def unwrap(self) -> _T | NoReturn:
        """
        Get value from successful safeguard or raise an exception for failed one.

        Usage:
        >>> Ok(1).unwrap()
        1
        >>> Err(1).unwrap()
        Traceback (most recent call last):
        ...
        UnwrapFailedError: Called `Result.unwrap()` on an `Err` value
        """

    @abstractmethod
    def unwrap_err(self) -> _E | NoReturn:
        """
        Get error from failed safeguard or raise an exception for successful one.

        Usage:
        >>> Ok(1).unwrap_err()
        Traceback (most recent call last):
        ...
        UnwrapFailedError: Called `Result.unwrap_err()` on an `Ok` value
        >>> Err(1).unwrap_err()
        1
        """

    @abstractmethod
    def unwrap_or(self, _default: _U) -> _T | _U:
        """
        Get value from successful safeguard or return default for failed one.

        Usage:
        >>> Ok(1).unwrap_or(2)
        1
        >>> Err(1).unwrap_or(2)
        2
        """

    @abstractmethod
    def unwrap_or_raise(self, e: Type[_TBE]) -> _T | NoReturn:
        """
        Get value from successful safeguard or raise an exception for failed one.

        Usage:
        >>> Ok(1).unwrap_or_raise(Exception)
        1
        >>> Err(1).unwrap_or_raise(Exception)
        Traceback (most recent call last):
        ...
        Exception
        """

    @abstractmethod
    def unwrap_or_else(self, op: Callable[[], _U]) -> _T | _U:
        """
        Get value from successful safeguard or compute a default for failed one.

        Usage:
        >>> Ok(1).unwrap_or_else(lambda: 2)
        1
        >>> Err(1).unwrap_or_else(lambda: 2)
        2
        """

    @abstractmethod
    def map(self, op: Callable[[_T], _U]) -> Ok[_U, _E] | Err[_T, _E]:
        """
        Map the value of a successful safeguard to a new value. For failed containers, return self.

        Usage:
        >>> Ok(1).map(lambda x: x + 1).map(str)
        Ok('2')
        >>> Err(1).map(lambda x: x + 1).map(str)
        Err(1)
        """

    @abstractmethod
    async def map_async(self, op: Callable[[_T], Awaitable[_U]]) -> Ok[_U, _E] | Err[_T, _E]:
        """
        Map the value of a successful async safeguard to a new value. For failed containers, return self.

        Usage:
        >>> import asyncio
        >>> async def add_one(x: int) -> int:
        ...     await asyncio.sleep(0.1)
        ...     return x + 1
        ...
        >>> async def main() -> Result[int, Exception]:
        ...     return await (await Ok(1).map_async(add_one)).map_async(add_one)
        ...
        >>> asyncio.run(main())
        """

    @abstractmethod
    def map_err(self, op: Callable[[_E], _R]) -> Ok[_T, _E] | Err[_T, _R]:
        """
        Map the error of a failed safeguard to a new value. For successful containers, return self.

        Usage:
        >>> Ok(1).map_err(lambda x: x + 1)
        Ok(1)
        >>> Err(1).map_err(lambda x: x + 1)
        Err(2)
        """

    @abstractmethod
    def apply_or(self, op: Callable[[_T], _U], default: _R) -> _U | _R:
        """
        Apply the operation to the value of a successful safeguard or return the default value.

        Usage:
        >>> Ok(1).apply_or(lambda x: x * x, 0)
        1
        >>> Err(1).apply_or(lambda x: x * x, 0)
        0
        """

    @abstractmethod
    def or_else(self, op: Callable[[], Result[_T, _E]]) -> Ok[_T, _E] | Result[_T, _E]:
        """
        Return self if successful, otherwise apply the operation and return the result.

        Usage:
        >>> Ok(1).or_else(lambda: Ok(2))
        Ok(1)
        >>> Err(1).or_else(lambda: Ok(2))
        Ok(2)
        """

    @abstractmethod
    def and_then(self, op: Callable[[_T], Result[_U, _E]]) -> Result[_U, _E] | Err[_T, _E]:
        """
        Return self if failed, otherwise apply the operation to the value.

        Usage:
        >>> Ok(1).and_then(lambda x: Ok(x + 1))
        Ok(2)
        >>> Err(1).and_then(lambda x: Ok(x + 1))
        Err(1)
        """


@final
class Ok(Result[_T, _E]):
    """
    A value that indicates success and which stores arbitrary data for the return value.

    Methods:
        - unwrap: Return the value.
        - unwrap_err: Raise `UnwrapFailedError`.
        - unwrap_or: Return the value.
        - unwrap_or_raise: Return the value.
        - unwrap_or_else: Return the value.
        - map: Apply a function to the value.
        - map_async: Apply an async function to the value.
        - map_err: Return self.
        - apply_or: Apply a function to the value.
        - or_else: Return self.
        - and_then: Apply a function to the value.
    """
    __slots__ = ()
    _value: _T

    def __iter__(self) -> Iterator[_T]:
        yield self._value

    def unwrap(self) -> _T:
        """Return the value from the safeguard."""
        return self._value

    def unwrap_err(self) -> NoReturn:
        """Raise `UnwrapFailedError`."""
        raise UnwrapFailedError(self, "Called `Result.unwrap_err()` on an `Ok` value")

    def unwrap_or(self, _default: _U) -> _T:
        """Return the value from the safeguard."""
        return self._value

    def unwrap_or_raise(self, e: Type[_TBE]) -> _T:
        """Return the value from the safeguard."""
        return self._value

    def unwrap_or_else(self, op: Callable[[], _U]) -> _T:
        """Return the value from the safeguard."""
        return self._value

    def map(self, op: Callable[[_T], _U]) -> Ok[_U, _E]:
        """Return the result of applying `op` to the value."""
        return Ok(op(self._value))

    async def map_async(self, op: Callable[[_T], Awaitable[_U]]) -> Ok[_U, _E]:
        """Return the result of applying `op` to the value."""
        return Ok(await op(self._value))

    def map_err(self, op: Callable[[_U], _R]) -> Ok[_T, _E]:
        """Return self."""
        return self

    def apply_or(self, op: Callable[[_T], _U], default: _R) -> _U:
        """Return the result of applying `op` to the value."""
        return op(self._value)

    def or_else(self, op: Callable[[], Result[_T, _E]]) -> Ok[_T, _E]:
        """Return self."""
        return self

    def and_then(self, op: Callable[[_T], Result[_U, _E]]) -> Result[_U, _E]:
        """Return the result of applying `op` to the value."""
        return op(self._value)


@final
class Err(Result[_T, _E]):
    """
    A value that indicates failure and which stores arbitrary data for the error value.

    Methods:
        - unwrap: Raise `UnwrapFailedError`.
        - unwrap_err: Return the error value.
        - unwrap_or: Return the default value.
        - unwrap_or_raise: Raise the specified exception.
        - unwrap_or_else: Return the result of the operation.
        - map: Return self.
        - map_async: Return self.
        - map_err: Apply a function to the error.
        - apply_or: Return the default value.
        - or_else: Return the result of the operation.
        - and_then: Return self.
    """
    __slots__ = ()
    _value: _E

    def __iter__(self) -> Iterator[_T]:
        def _iter() -> Iterator[NoReturn]:
            # Exception will be raised when the iterator is advanced, not when it's created
            raise DoException(self)
            yield  # This yield will never be reached, but is necessary to create a generator

        return _iter()

    def unwrap(self) -> NoReturn:
        """Raise `UnwrapFailedError`."""
        raise UnwrapFailedError(self, "Called `Result.unwrap()` on an `Err` value")

    def unwrap_err(self) -> _E:
        """Return the error value."""
        return self._value

    def unwrap_or(self, _default: _U) -> _U:
        """Return the default value."""
        return _default

    def unwrap_or_raise(self, e: Type[_TBE]) -> NoReturn:
        """Raise the specified exception."""
        raise e

    def unwrap_or_else(self, op: Callable[[], _U]) -> _U:
        """Return the result of the operation."""
        return op()

    def map(self, op: Callable[[_T], _U]) -> Err[_T, _E]:
        """Return self."""
        return self

    async def map_async(self, op: Callable[[_T], Awaitable[_U]]) -> Err[_T, _E]:
        """Return self."""
        return self

    def map_err(self, op: Callable[[_E], _R]) -> Err[_T, _R]:
        """Return the result of applying `op` to the error."""
        return Err(op(self._value))

    def apply_or(self, op: Callable[[_T], _U], default: _R) -> _R:
        """Return the default value."""
        return default

    def or_else(self, op: Callable[[], Result[_T, _E]]) -> Result[_T, _E]:
        """Return the result of the operation."""
        return op()

    def and_then(self, op: Callable[[_T], Result[_U, _E]]) -> Err[_T, _E]:
        """Return self."""
        return self


class DoException(Exception):
    """
    This is used to signal to `do()` that the safeguard is an `Err`,
    which short-circuits the generator and returns that Err.
    Using this exception for control flow in `do()` allows us
    to simulate `and_then()` in the Err case: namely, we don't call `op`,
    we just return `self` (the Err).
    """

    def __init__(self, err: Err[_T, _E]) -> None:
        self.err = err


# TODO: Maybe overload this correctly?

def safe(
    function: Optional[Callable[_P, _R]] | Optional[Callable[_P, Generator[_R, Any, Any]]] = None,
    *,
    exceptions: Optional[Tuple[Type[_TBE], ...]] = None
) -> Union[
    Callable[_P, Result[_R, _TBE]] | Callable[_P, Generator[Result[_R, _TBE], Any, Any]],
    Callable[
        [Callable[_P, _R] | Callable[_P, Generator[_R, Any, Any]]],
        Callable[_P, Result[_R, _TBE]] | Callable[_P, Generator[Result[_R, _TBE], Any, Any]]
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
        e: Type[_TBE] = cast(Type[_TBE], Exception)  # Mypy can't infer this properly
        return decorator(function, exceptions_=(e,))
    if isinstance(exceptions, tuple):
        return lambda f_: decorator(f_, exceptions)

    raise TypeError("safe() requires either a function or a tuple of exceptions")


def async_safe(
    function: Optional[Callable[_P, Awaitable[_R]]] | Optional[Callable[_P, AsyncGenerator[_R, Any]]] = None,
    *,
    exceptions: Optional[Tuple[Type[_TBE], ...]] = None
) -> Union[
    Callable[_P, Awaitable[Result[_R, _TBE]]] | Callable[_P, AsyncGenerator[Result[_R, _TBE], Any]],
    Callable[
        [Callable[_P, Awaitable[_R]] | Callable[_P, AsyncGenerator[_R, Any]]],
        Callable[_P, Awaitable[Result[_R, _TBE]]] | Callable[_P, AsyncGenerator[Result[_R, _TBE], Any]]
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
        exceptions_: Tuple[Type[_TBE], ...]
    ) -> Callable[_P, Awaitable[Result[_R, _TBE]]] | Callable[_P, AsyncGenerator[Result[_R, _TBE], Any]]:
        """Decorator to turn an async function or async generator into one that returns a ``Result``.
        """

        @functools.wraps(function_)
        async def wrapper_function(*args: _P.args, **kwargs: _P.kwargs) -> Result[_R, _TBE]:
            assert inspect.iscoroutinefunction(function_)
            try:
                return Ok(await function_(*args, **kwargs))
            except exceptions_ as exc:
                return Err(exc)

        @functools.wraps(function_)
        async def wrapper_generator(*args: _P.args, **kwargs: _P.kwargs) -> AsyncGenerator[Result[_R, _TBE], Any]:
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
        e: Type[_TBE] = cast(Type[_TBE], Exception)  # Mypy can't infer this properly
        return decorator(function, exceptions_=(e,))
    if isinstance(exceptions, tuple):
        return lambda f_: decorator(f_, exceptions)

    raise TypeError("async_safe() requires either a function or a tuple of exceptions")


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
