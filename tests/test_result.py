from __future__ import annotations

import copy
from typing import Callable

import pytest

from safeguard.exceptions import IncorrectActionCalledException, IncorrectCallableException, UnwrapFailedError
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


@pytest.mark.asyncio
async def test_unwrap_async() -> None:
    o = Ok('yay')
    n = Err('nay')
    assert await o.async_unwrap() == 'yay'
    with pytest.raises(UnwrapFailedError):
        await n.async_unwrap()


def test_unwrap_err() -> None:
    o = Ok('yay')
    n = Err('nay')
    with pytest.raises(UnwrapFailedError):
        o.unwrap_err()
    assert n.unwrap_err() == 'nay'


@pytest.mark.asyncio
async def test_async_unwrap_err() -> None:
    o = Ok('yay')
    n = Err('nay')
    with pytest.raises(UnwrapFailedError):
        await o.async_unwrap_err()
    assert await n.async_unwrap_err() == 'nay'


def test_unwrap_or() -> None:
    o = Ok('yay')
    n = Err('nay')
    assert o.unwrap_or('some_default') == 'yay'
    assert n.unwrap_or('another_default') == 'another_default'


@pytest.mark.asyncio
async def test_async_unwrap_or() -> None:
    o = Ok('yay')
    n = Err('nay')
    assert await o.async_unwrap_or('some_default') == 'yay'
    assert await n.async_unwrap_or('another_default') == 'another_default'


def test_unwrap_or_raise() -> None:
    o = Ok('yay')
    n = Err('nay')
    assert o.unwrap_or_raise(ValueError) == 'yay'
    with pytest.raises(ValueError) as exc_info:
        n.unwrap_or_raise(ValueError)

    with pytest.raises(TypeError):
        n.unwrap_or_raise()

    n = Err(ValueError("Some Error"))
    with pytest.raises(ValueError) as exc_info:
        n.unwrap_or_raise()


@pytest.mark.asyncio
async def test_async_unwrap_or_raise() -> None:
    o = Ok('yay')
    n = Err('nay')
    assert await o.async_unwrap_or_raise(ValueError) == 'yay'
    with pytest.raises(ValueError) as exc_info:
        await n.async_unwrap_or_raise(ValueError)

    with pytest.raises(TypeError):
        await n.async_unwrap_or_raise()

    n = Err(ValueError("Some Error"))
    with pytest.raises(ValueError) as exc_info:
        await n.async_unwrap_or_raise()


def test_unwrap_or_else() -> None:
    o = Ok('yay')
    n = Err('nay')
    assert o.unwrap_or_else(lambda: 'ERR') == 'yay'
    assert n.unwrap_or_else(lambda: 'ERR') == 'ERR'


@pytest.mark.asyncio
async def test_async_unwrap_or_else() -> None:
    o = Ok('yay')
    n = Err('nay')
    assert await o.async_unwrap_or_else(lambda: 'ERR') == 'yay'
    assert await n.async_unwrap_or_else(lambda: 'ERR') == 'ERR'

    async def async_fn() -> str:
        return 'ERR'

    assert await n.async_unwrap_or_else(async_fn) == 'ERR'


def test_apply_or() -> None:
    assert Ok(2).apply_or(lambda x: x * 2, 3) == 4
    assert Err(3).apply_or(lambda x: x * 2, 3) == 3


@pytest.mark.asyncio
async def test_async_apply_or() -> None:
    async def double_async(x: int) -> int:
        return x * 2

    assert await Ok(2).async_apply_or(double_async, 3) == 4
    assert await Err(3).async_apply_or(double_async, 3) == 3

    assert await Ok(2).async_map(double_async).async_apply_or(str, '3') == '4'
    assert await Err(3).async_map(double_async).async_apply_or(str, '3') == '3'


def test_map() -> None:
    num = Ok(3)
    errnum = Err(2)
    assert num.map(str).unwrap() == '3'
    assert errnum.map(str).unwrap_err() == 2


@pytest.mark.asyncio
async def test_async_map() -> None:
    async def str_upper_async(s: str) -> str:
        return s.upper()

    async def str_async(x: int) -> str:
        return str(x)

    o = Ok('yay')
    n = Err('nay')
    assert await o.async_map(str_upper_async).async_unwrap() == 'YAY'
    assert await n.async_map(str_upper_async).async_unwrap_err() == 'nay'

    num = Ok(3)
    errnum = Err(2)
    assert await num.async_map(str_async).async_unwrap() == '3'
    assert await errnum.async_map(str_async).async_unwrap_err() == 2


@pytest.mark.asyncio
async def test_async_map_mix() -> None:
    async def plus_one(x: int) -> int:
        return x + 1

    assert (
               await Ok(1)
               .map(lambda x: x + 1)
               .async_map(plus_one)
               .map(lambda x: x + 1)
               .async_map(plus_one)
               .async_unwrap()
           ) == 5


def test_map_err() -> None:
    o = Ok(3)
    n = Err(2)
    assert o.map_err(str).unwrap() == 3
    assert n.map_err(str).unwrap_err() == '2'


@pytest.mark.asyncio
async def test_async_map_err() -> None:
    async def str_upper_async(s: str) -> str:
        return s.upper()

    async def str_async(x: int) -> str:
        return str(x)

    o = Ok('yay')
    n = Err('nay')
    assert await o.async_map_err(str_upper_async).async_unwrap() == 'yay'
    assert await n.async_map_err(str_upper_async).async_unwrap_err() == 'NAY'

    num = Ok(3)
    errnum = Err(2)
    assert await num.async_map_err(str_async).async_unwrap() == 3
    assert await errnum.async_map_err(str_async).async_unwrap_err() == '2'


@pytest.mark.asyncio
async def test_async_map_err_mix() -> None:
    async def plus_one(x: int) -> int:
        return x + 1

    assert (
               await Err(1)
               .map_err(lambda x: x + 1)
               .async_map_err(plus_one)
               .map_err(lambda x: x + 1)
               .async_map_err(plus_one)
               .async_unwrap_err()
           ) == 5


def test_and_then() -> None:
    assert Ok(2).and_then(sq).and_then(sq).unwrap() == 16
    assert Ok(2).and_then(sq).and_then(to_err).unwrap_err() == 4
    assert Ok(2).and_then(to_err).and_then(sq).unwrap_err() == 2
    assert Err(3).and_then(sq).and_then(sq).unwrap_err() == 3

    assert Ok(2).and_then(sq_lambda).and_then(sq_lambda).unwrap() == 16
    assert Ok(2).and_then(sq_lambda).and_then(to_err_lambda).unwrap_err() == 4
    assert Ok(2).and_then(to_err_lambda).and_then(sq_lambda).unwrap_err() == 2
    assert Err(3).and_then(sq_lambda).and_then(sq_lambda).unwrap_err() == 3


@pytest.mark.asyncio
async def test_async_and_then() -> None:
    assert await Ok(2).async_and_then(sq_async).async_and_then(sq_async).async_unwrap() == 16
    assert await Ok(2).async_and_then(sq_async).async_and_then(to_err_async).async_unwrap_err() == 4
    assert await Ok(2).async_and_then(to_err_async).async_and_then(sq_async).async_unwrap_err() == 2
    assert await Err(3).async_and_then(sq_async).async_and_then(sq_async).async_unwrap_err() == 3

    assert await Ok(2).async_and_then(sq_async).async_and_then(sq_async).async_unwrap() == 16
    assert await Ok(2).async_and_then(sq_async).async_and_then(to_err_async).async_unwrap_err() == 4
    assert await Ok(2).async_and_then(to_err_async).async_and_then(sq_async).async_unwrap_err() == 2
    assert await Err(3).async_and_then(sq_async).async_and_then(sq_async).async_unwrap_err() == 3


@pytest.mark.asyncio
async def test_async_and_then_mix() -> None:
    assert (
               await Ok(2)
               .and_then(sq)
               .async_and_then(sq_async)
               .and_then(sq)
               .async_and_then(sq_async)
               .async_unwrap()
           ) == 65536


def test_or_else() -> None:
    assert Ok(2).or_else(lambda: Ok(3)).unwrap() == 2
    assert Ok(2).or_else(lambda: Err(3)).unwrap() == 2
    assert Err(3).or_else(lambda: Ok(3)).unwrap() == 3
    assert Err(3).or_else(lambda: Err(3)).unwrap_err() == 3


@pytest.mark.asyncio
async def test_async_or_else() -> None:
    async def return_ok() -> Result[int, int]:
        return Ok(3)

    async def return_err() -> Result[int, int]:
        return Err(3)

    assert await Ok(2).async_or_else(return_ok).async_unwrap() == 2
    assert await Ok(2).async_or_else(return_err).async_unwrap() == 2
    assert await Err(3).async_or_else(return_ok).async_unwrap() == 3
    assert await Err(3).async_or_else(return_err).async_unwrap_err() == 3


@pytest.mark.asyncio
async def test_async_or_else_mix() -> None:
    async def return_err_async() -> Result[int, int]:
        return Err(2)

    assert (
               await Err(2)
               .or_else(lambda: Err(4))
               .async_or_else(return_err_async)
               .or_else(lambda: Err(4))
               .async_or_else(return_err_async)
               .async_unwrap_err()
           ) == 2


def test_inspect() -> None:
    oks: list[int] = []
    add_to_oks: Callable[[int], None] = lambda x: oks.append(x)

    assert Ok(2).inspect(add_to_oks).execute() == Ok(2)
    assert Err("e").inspect(add_to_oks).execute() == Err("e")
    assert oks == [2]


@pytest.mark.asyncio
async def test_async_inspect() -> None:
    oks: list[int] = []

    async def add_to_oks_async(x: int) -> None:
        oks.append(x)

    assert await Ok(2).async_inspect(add_to_oks_async).async_execute() == Ok(2)
    assert await Err("e").async_inspect(add_to_oks_async).async_execute() == Err("e")
    assert oks == [2]


def test_inspect_err() -> None:
    errs: list[str] = []
    add_to_errs: Callable[[str], None] = lambda x: errs.append(x)

    assert Ok(2).inspect_err(add_to_errs).execute() == Ok(2)
    assert Err("e").inspect_err(add_to_errs).execute() == Err("e")
    assert errs == ["e"]


@pytest.mark.asyncio
async def test_async_inspect_err() -> None:
    errs: list[str] = []

    async def add_to_errs_async(x: str) -> None:
        errs.append(x)

    assert await Ok(2).async_inspect_err(add_to_errs_async).async_execute() == Ok(2)
    assert await Err("e").async_inspect_err(add_to_errs_async).async_execute() == Err("e")
    assert errs == ["e"]


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


def test_is_chain_async() -> None:
    o = Ok(1).map(lambda x: x + 1).async_map(sq_async)
    assert o._is_async_chain() is True

    o = Ok(1).map(lambda x: x + 1).map(sq)
    assert o._is_async_chain() is False


def test_validate_op_decorator() -> None:
    async def f(x: int) -> int:
        return x

    def g(x: int) -> int:
        return x

    with pytest.raises(IncorrectCallableException) as exc:
        Ok(1).map(f)
    msg = exc.value.args[0]
    assert msg == "Function 'map' expected to get a sync function, but got a async function. (maybe call async_map)"

    with pytest.raises(IncorrectCallableException) as exc:
        Ok(1).async_map(g)
    msg = exc.value.args[0]
    assert msg == "Function 'async_map' expected to get a async function, but got a sync function. (maybe call map)"


def test_chain_compatibility_decorator() -> None:
    with pytest.raises(IncorrectActionCalledException) as exc:
        Ok(1).map(lambda x: x + 1).async_map(sq_async).unwrap()
    msg = exc.value.args[0]
    assert msg == "Action 'unwrap' is expecting sync chain, instead got async chain. (maybe call async_unwrap)"


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
