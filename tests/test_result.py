from __future__ import annotations

import copy
from typing import Callable

import pytest

from safeguard.exceptions import UnwrapFailedError
from safeguard.result import Err, Ok, Result, async_safe, safe


def test_ok_factories() -> None:
    instance = Ok(1)
    assert instance._value == 1
    assert instance.is_ok() is True


def test_err_factories() -> None:
    instance = Err(2)
    assert instance._value == 2
    assert instance.is_err() is True


def test_immutability() -> None:
    """
    Ok and Err are immutable.
    """
    o = Ok(1)
    n = Err(2)
    with pytest.raises(AttributeError):
        o._value = 2  # type: ignore[attr-defined]
    with pytest.raises(AttributeError):
        n._value = 2  # type: ignore[attr-defined]

    with pytest.raises(AttributeError):
        del o._value  # type: ignore[attr-defined]
    with pytest.raises(AttributeError):
        del n._value  # type: ignore[attr-defined]


def test_copy() -> None:
    o = Ok(1)
    n = Err(2)
    assert copy.copy(o) is o
    assert copy.copy(n) is n
    assert copy.deepcopy(o) is o
    assert copy.deepcopy(n) is n


def test_eq() -> None:
    assert Ok(1) == Ok(1)
    assert Err(1) == Err(1)
    assert Ok(1) != Err(1)
    assert Ok(1) != Ok(2)
    assert Err(1) != Err(2)
    assert not (Ok(1) != Ok(1))
    assert Ok(1) != "abc"
    assert Ok("0") != Ok(0)


def test_hash() -> None:
    assert len({Ok(1), Err("2"), Ok(1), Err("2")}) == 2
    assert len({Ok(1), Ok(2)}) == 2
    assert len({Ok("a"), Err("a")}) == 2


def test_repr() -> None:
    """
    ``repr()`` returns valid code if the wrapped value's ``repr()`` does as well.
    """
    o = Ok(123)
    n = Err(-1)

    assert repr(o) == "Ok(123)"
    assert o == eval(repr(o))

    assert repr(n) == "Err(-1)"
    assert n == eval(repr(n))


def test_ok() -> None:
    res = Ok('haha')
    assert res.is_ok() is True
    assert res.is_err() is False


def test_err() -> None:
    res = Err(':(')
    assert res.is_ok() is False
    assert res.is_err() is True


def test_err_value_is_exception() -> None:
    res = Err(ValueError("Some Error"))
    assert res.is_ok() is False
    assert res.is_err() is True

    with pytest.raises(UnwrapFailedError):
        res.unwrap()


def test_unwrap() -> None:
    o = Ok('yay')
    n = Err('nay')
    assert o.unwrap() == 'yay'
    with pytest.raises(UnwrapFailedError):
        n.unwrap()


def test_unwrap_err() -> None:
    o = Ok('yay')
    n = Err('nay')
    with pytest.raises(UnwrapFailedError):
        o.unwrap_err()
    assert n.unwrap_err() == 'nay'


def test_unwrap_or() -> None:
    o = Ok('yay')
    n = Err('nay')
    assert o.unwrap_or('some_default') == 'yay'
    assert n.unwrap_or('another_default') == 'another_default'


def test_unwrap_or_else() -> None:
    o = Ok('yay')
    n = Err('nay')
    assert o.unwrap_or_else(lambda: 'ERR') == 'yay'
    assert n.unwrap_or_else(lambda: 'ERR') == 'ERR'


def test_unwrap_or_raise() -> None:
    o = Ok('yay')
    n = Err('nay')
    assert o.unwrap_or_raise(ValueError) == 'yay'
    with pytest.raises(ValueError) as exc_info:
        n.unwrap_or_raise(ValueError)


def test_map() -> None:
    o = Ok('yay')
    n = Err('nay')
    assert o.map(str.upper) == Ok('YAY')
    assert n.map(str.upper) == Err('nay')

    num = Ok(3)
    errnum = Err(2)
    assert num.map(str) == Ok('3')
    assert errnum.map(str) == Err(2)


def test_map_err() -> None:
    o = Ok('yay')
    n = Err('nay')
    assert o.map_err(str.upper) == Ok('yay')
    assert n.map_err(str.upper) == Err('NAY')


def test_and_then() -> None:
    assert Ok(2).and_then(sq).and_then(sq) == Ok(16)
    assert Ok(2).and_then(sq).and_then(to_err) == Err(4)
    assert Ok(2).and_then(to_err).and_then(sq) == Err(2)
    assert Err(3).and_then(sq).and_then(sq) == Err(3)

    assert Ok(2).and_then(sq_lambda).and_then(sq_lambda) == Ok(16)
    assert Ok(2).and_then(sq_lambda).and_then(to_err_lambda) == Err(4)
    assert Ok(2).and_then(to_err_lambda).and_then(sq_lambda) == Err(2)
    assert Err(3).and_then(sq_lambda).and_then(sq_lambda) == Err(3)


@pytest.mark.asyncio
async def test_map_async() -> None:
    async def str_upper_async(s: str) -> str:
        return s.upper()

    async def str_async(x: int) -> str:
        return str(x)

    o = Ok('yay')
    n = Err('nay')
    assert (await o.map_async(str_upper_async)) == Ok('YAY')
    assert (await n.map_async(str_upper_async)) == Err('nay')

    num = Ok(3)
    errnum = Err(2)
    assert (await num.map_async(str_async)) == Ok('3')
    assert (await errnum.map_async(str_async)) == Err(2)


def test_apply_or() -> None:
    assert Ok(2).apply_or(lambda x: x * 2, 3) == 4
    assert Err(3).apply_or(lambda x: x * 2, 3) == 3


def test_or_else() -> None:
    assert Ok(2).or_else(lambda: Ok(3)) == Ok(2)
    assert Ok(2).or_else(lambda: Err(3)) == Ok(2)
    assert Err(3).or_else(lambda: Ok(3)) == Ok(3)
    assert Err(3).or_else(lambda: Err(3)) == Err(3)


def test_slots() -> None:
    """
    Ok and Err have slots, so assigning arbitrary attributes fails.
    """
    o = Ok('yay')
    n = Err('nay')
    with pytest.raises(AttributeError):
        o.some_arbitrary_attribute = 1  # type: ignore[attr-defined]
    with pytest.raises(AttributeError):
        n.some_arbitrary_attribute = 1  # type: ignore[attr-defined]


def test_safe() -> None:
    """
    ``safe()`` turns functions into ones that return a ``Result``.
    """

    @safe(exceptions=(ValueError,))
    def good_1(value: int) -> int:
        return value

    @safe
    def good_2(value: int) -> int:
        return value

    good_1_result = good_1(123)
    good_2_result = good_2(123)

    assert isinstance(good_1_result, Ok)
    assert good_1_result.unwrap() == 123
    assert isinstance(good_2_result, Ok)
    assert good_2_result.unwrap() == 123

    @safe(exceptions=(ValueError,))
    def bad_1(value: int) -> int:
        raise ValueError

    @safe
    def bad_2(value: int) -> int:
        raise ValueError

    bad_1_result = bad_1(123)
    bad_2_result = bad_2(123)

    assert isinstance(bad_1_result, Err)
    assert isinstance(bad_1_result.unwrap_err(), ValueError)
    assert isinstance(bad_2_result, Err)
    assert isinstance(bad_2_result.unwrap_err(), ValueError)


def test_invalid_safe_usage() -> None:
    with pytest.raises(TypeError):
        @safe(ValueError, IndexError)
        def f() -> int:
            return 1


def test_safe_with_generator() -> None:
    """
    ``safe()`` works with generators.
    """

    @safe(exceptions=(ValueError,))
    def random_generator_1():
        yield 1
        yield 2
        yield 3

    for i in random_generator_1():
        assert i in (Ok(1), Ok(2), Ok(3))

    @safe
    def random_generator_2():
        yield 1
        yield 2
        yield 3

    for i in random_generator_2():
        assert i in (Ok(1), Ok(2), Ok(3))


def test_safe_with_generator_and_exception() -> None:
    """
    ``safe()`` works with generators that raise exceptions.
    """

    @safe(exceptions=(ValueError,))
    def random_generator_1():
        yield 1
        yield 2
        raise ValueError

    result = [i for i in random_generator_1()]
    assert result[:2] == [Ok(1), Ok(2)]
    assert result[-1].is_err()
    assert isinstance(result[-1].unwrap_err(), ValueError)

    @safe
    def random_generator_2():
        yield 1
        yield 2
        raise ValueError

    result = [i for i in random_generator_2()]
    assert result[:2] == [Ok(1), Ok(2)]
    assert result[-1].is_err()
    assert isinstance(result[-1].unwrap_err(), ValueError)


def test_safe_invalid_usage() -> None:
    """
    Invalid use of ``safe()`` raises reasonable errors.
    """
    message = "requires one or more exception types"

    with pytest.raises(TypeError):
        @safe()  # No exception types specified
        def f() -> int:
            return 1

    with pytest.raises(TypeError):
        @safe("not an exception type")  # type: ignore[arg-type]
        def g() -> int:
            return 1


def test_safe_async_invalid_usage() -> None:
    """
    Invalid use of ``safe()`` raises reasonable errors.
    """
    with pytest.raises(TypeError):
        @async_safe()  # No exception types specified
        async def f() -> int:
            return 1

    with pytest.raises(TypeError):
        @async_safe("not an exception type")  # type: ignore[arg-type]
        async def g() -> int:
            return 1


def test_safe_type_checking() -> None:
    """
    The ``safe()`` is a signature-preserving decorator.
    """

    @safe(exceptions=(ValueError,))
    def f(a: int) -> int:
        return a

    res: Result[int, ValueError]
    res = f(123)  # No mypy error here.
    assert res.unwrap() == 123


@pytest.mark.asyncio
async def test_async_safe() -> None:
    """
    ``async_safe()`` turns functions into ones that return a ``Result``.
    """

    @async_safe(exceptions=(ValueError, IndexError))
    async def good(value: int) -> int:
        return value

    @async_safe(exceptions=(ValueError, IndexError))
    async def bad(value: int) -> int:
        raise ValueError

    good_result = await good(123)
    bad_result = await bad(123)

    assert isinstance(good_result, Ok)
    assert good_result.unwrap() == 123
    assert isinstance(bad_result, Err)
    assert isinstance(bad_result.unwrap_err(), ValueError)

    @async_safe
    async def good_no_exceptions(value: int) -> int:
        return value

    @async_safe
    async def bad_no_exceptions(value: int) -> int:
        raise ValueError

    good_result = await good_no_exceptions(123)
    bad_result = await bad_no_exceptions(123)

    assert isinstance(good_result, Ok)
    assert good_result.unwrap() == 123
    assert isinstance(bad_result, Err)
    assert isinstance(bad_result.unwrap_err(), ValueError)


@pytest.mark.asyncio
async def test_async_safe_with_generator() -> None:
    """
    ``async_safe()`` works with async generators.
    """

    @async_safe(exceptions=(ValueError,))
    async def random_generator_1():
        yield 1
        yield 2
        yield 3

    async for i in random_generator_1():
        assert i in (Ok(1), Ok(2), Ok(3))

    @async_safe
    async def random_generator_2():
        yield 1
        yield 2
        yield 3

    async for i in random_generator_2():
        assert i in (Ok(1), Ok(2), Ok(3))


@pytest.mark.asyncio
async def test_async_safe_with_generator_and_exception() -> None:
    """
    ``async_safe()`` works with async generators that raise exceptions.
    """

    @async_safe(exceptions=(ValueError,))
    async def random_generator():
        yield 1
        yield 2
        raise ValueError

    result = [i async for i in random_generator()]
    assert result[:2] == [Ok(1), Ok(2)]
    assert result[-1].is_err()
    assert isinstance(result[-1].unwrap_err(), ValueError)

    @async_safe
    async def random_generator():
        yield 1
        yield 2
        raise ValueError

    result = [i async for i in random_generator()]
    assert result[:2] == [Ok(1), Ok(2)]
    assert result[-1].is_err()
    assert isinstance(result[-1].unwrap_err(), ValueError)


def sq(i: int) -> Result[int, int]:
    return Ok(i * i)


async def sq_async(i: int) -> Result[int, int]:
    return Ok(i * i)


def to_err(i: int) -> Result[int, int]:
    return Err(i)


async def to_err_async(i: int) -> Result[int, int]:
    return Err(i)


# Lambda versions of the same functions, just for test/type coverage
sq_lambda: Callable[[int], Result[int, int]] = lambda i: Ok(i * i)
to_err_lambda: Callable[[int], Result[int, int]] = lambda i: Err(i)
