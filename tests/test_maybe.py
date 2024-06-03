from __future__ import annotations

import copy

import pytest

from safeguard.exceptions import UnwrapFailedError
from safeguard.helper.immutable import ImmutableStateError
from safeguard.maybe import Maybe, Nothing, Some, async_maybe, maybe


def test_some_factories() -> None:
    instance = Some(1)
    assert instance._value == 1
    assert instance.is_some() is True


def test_nothing_factory() -> None:
    instance = Nothing()
    assert instance.is_nothing() is True


def test_immutability() -> None:
    """
    Some and Nothing are immutable.
    """
    o = Some(1)
    n = Nothing()
    with pytest.raises(ImmutableStateError):
        o._value = 2  # type: ignore[attr-defined]
    with pytest.raises(ImmutableStateError):
        n._value = 2  # type: ignore[attr-defined]

    with pytest.raises(ImmutableStateError):
        del o._value  # type: ignore[attr-defined]
    with pytest.raises(ImmutableStateError):
        del n._value


def test_copy() -> None:
    o = Some(1)
    n = Nothing()
    assert copy.copy(o) is o
    assert copy.copy(n) is n
    assert copy.deepcopy(o) is o
    assert copy.deepcopy(n) is n


def test_eq() -> None:
    assert Some(1) == Some(1)
    assert Nothing() == Nothing()
    assert Some(1) != Nothing()
    assert Some(1) != Some(2)
    assert not (Some(1) != Some(1))
    assert Some(1) != "abc"
    assert Some("0") != Some(0)


def test_hash() -> None:
    assert len({Some(1), Nothing(), Some(1), Nothing()}) == 2
    assert len({Some(1), Some(2)}) == 2
    assert len({Some("a"), Nothing()}) == 2


def test_repr() -> None:
    """
    ``repr()`` returns valid code if the wrapped value's ``repr()`` does as well.
    """
    o = Some(123)
    n = Nothing()

    assert repr(o) == "Some(123)"
    assert o == eval(repr(o))

    assert repr(n) == "Nothing()"
    assert n == eval(repr(n))


def test_some() -> None:
    res = Some('haha')
    assert res.is_some() is True
    assert res.is_nothing() is False


def test_nothing() -> None:
    res = Nothing()
    assert res.is_some() is False
    assert res.is_nothing() is True


def test_unwrap() -> None:
    o = Some('yay')
    n = Nothing()
    assert o.unwrap() == 'yay'
    with pytest.raises(UnwrapFailedError):
        n.unwrap()


def test_unwrap_or() -> None:
    o = Some('yay')
    n = Nothing()
    assert o.unwrap_or('some_default') == 'yay'
    assert n.unwrap_or('another_default') == 'another_default'


def test_unwrap_or_else() -> None:
    o = Some('yay')
    n = Nothing()
    assert o.unwrap_or_else(str.upper) == 'yay'
    assert n.unwrap_or_else(lambda: 'default') == 'default'


def test_unwrap_or_raise() -> None:
    o = Some('yay')
    n = Nothing()
    assert o.unwrap_or_raise(ValueError) == 'yay'
    with pytest.raises(ValueError) as exc_info:
        n.unwrap_or_raise(ValueError)
    assert exc_info.value.args == ()


def test_map() -> None:
    o = Some('yay')
    n = Nothing()
    assert o.map(str.upper).unwrap() == 'YAY'
    assert n.map(str.upper).is_nothing()


def test_apply_or() -> None:
    o = Some('yay')
    n = Nothing()
    assert o.apply_or(str.upper, 'hay') == 'YAY'
    assert n.apply_or(str.upper, 'hay') == 'hay'


def test_or_else() -> None:
    assert Some(2).or_else(lambda: Some(3)).unwrap() == 2
    assert Some(2).or_else(lambda: Nothing()).unwrap() == 2
    assert Nothing().or_else(lambda: Some(3)).unwrap() == 3
    assert Nothing().or_else(lambda: Nothing()).is_nothing()


def test_filter() -> None:
    assert Some(2).filter(lambda x: x > 0).unwrap() == 2
    assert Some(2).filter(lambda x: x < 0).is_nothing()
    assert Nothing().filter(lambda x: x > 0).is_nothing()


def test_as_maybe_return_some() -> None:
    @maybe
    def f() -> int:
        return 1

    assert f() == Some(1)


def test_as_maybe_return_nothing() -> None:
    @maybe
    def f() -> int | None:
        return None

    assert f() == Nothing()


@pytest.mark.asyncio
async def test_as_async_maybe_return_some() -> None:
    @async_maybe
    async def f() -> int:
        return 1

    assert await f() == Some(1)


@pytest.mark.asyncio
async def test_as_async_maybe_return_nothing() -> None:
    @async_maybe
    async def f() -> int | None:
        return None

    assert await f() == Nothing()


def test_from_optional_return_some() -> None:
    assert maybe(lambda: 1)() == Some(1)


def test_from_optional_return_nothing() -> None:
    assert maybe(lambda: None)() == Nothing()


@pytest.mark.asyncio
async def test_async_from_optional_return_some() -> None:
    async def async_fn() -> int:
        return 1

    assert await async_maybe(async_fn)() == Some(1)


@pytest.mark.asyncio
async def test_async_from_optional_return_nothing() -> None:
    async def async_fn() -> int | None:
        return None

    assert await async_maybe(async_fn)() == Nothing()


def test_slots() -> None:
    """
    Some and Nothing have slots, so assigning arbitrary attributes fails.
    """
    o = Some('yay')
    n = Nothing()
    with pytest.raises(AttributeError):
        o.some_arbitrary_attribute = 1  # type: ignore[attr-defined]
    with pytest.raises(AttributeError):
        n.some_arbitrary_attribute = 1  # type: ignore[attr-defined]


def sq(i: int) -> Maybe[int]:
    return Some(i * i)


def to_nothing(_: int) -> Maybe[int]:
    return Nothing()
