from __future__ import annotations

from safeguard.maybe import Maybe, Nothing, Some
from safeguard.result import Err, Ok, Result


def test_pattern_matching_on_ok_type() -> None:
    """
    Pattern matching on ``Ok()`` matches the contained value.
    """
    o: Result[str, int] = Ok("yay")
    match o:
        case Ok(value):
            reached = True

    assert value == "yay"
    assert reached


def test_pattern_matching_on_err_type() -> None:
    """
    Pattern matching on ``Err()`` matches the contained value.
    """
    n: Result[int, str] = Err("nay")
    match n:
        case Err(value):
            reached = True

    assert value == "nay"
    assert reached


def test_pattern_matching_on_some_type() -> None:
    """
    Pattern matching on ``Some()`` matches the contained value.
    """
    o: Maybe[str] = Some("yay")
    match o:
        case Some(value):
            reached = True

    assert value == "yay"
    assert reached


def test_pattern_matching_on_nothing_type() -> None:
    """
    Pattern matching on ``Err()`` matches the contained value.
    """
    n: Maybe[int] = Nothing()
    match n:
        case Nothing():
            reached = True

    assert reached
