from abc import ABCMeta
from typing import Any

from .immutable import Immutable


class BaseContainer(Immutable, metaclass=ABCMeta):
    """Utility class to provide all needed magic methods to the context."""

    __slots__ = ('_value',)
    __match_args__ = ('_value',)
    _value: Any

    def __init__(self, value: Any) -> None:
        """
        Wraps the given value in the Container.

        'value' is any arbitrary value of any type including functions.
        """
        object.__setattr__(self, '_value', value)  # noqa: WPS609

    def __repr__(self) -> str:
        """Used to display details of object."""
        return '{0}({1})'.format(
            self.__class__.__qualname__.strip('_'),
            str(self._value),
        )

    def __eq__(self, other: Any) -> bool:
        return type(self) is type(other) and self._value == other._value

    def __ne__(self, other: Any) -> bool:
        return not (self == other)

    def __hash__(self) -> int:
        return hash((True, self._value,))
