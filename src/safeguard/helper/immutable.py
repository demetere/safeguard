from typing import Any, Dict, NoReturn

from typing_extensions import Self


class ImmutableStateError(AttributeError):
    """
    Raised when a safeguard is forced to be mutated.

    It is a sublclass of ``AttributeError`` for two reasons:

    1. It seems kinda reasonable to expect ``AttributeError``
       on attribute modification
    2. It is used inside ``typing.py`` this way,
       we do have several typing features that requires that behaviour

    See: https://github.com/dry-python/returns/issues/394
    """


class Immutable:
    """
    Helper type for objects that should be immutable.

    When applied, each instance becomes immutable.
    Nothing can be added or deleted from it.

    .. code:: pycon
      :force:

      >>> from safeguard.helper.immutable import Immutable
      >>> class MyModel(Immutable):
      ...     ...

      >>> model = MyModel()
      >>> model.prop = 1
      Traceback (most recent call last):
         ...
      returns.helper.exceptions.ImmutableStateError

    See :class:`returns.helper.safeguard.BaseContainer` for examples.

    """  # noqa: RST307

    __slots__ = ()

    def __copy__(self) -> Self:
        """Returns itself."""
        return self

    def __deepcopy__(self, memo: Dict[Any, Any]) -> Self:
        """Returns itself."""
        return self

    def __setattr__(self, attr_name: str, attr_value: Any) -> NoReturn:
        """Makes inner state of the containers immutable for modification."""
        raise ImmutableStateError()

    def __delattr__(self, attr_name: str) -> NoReturn:
        """Makes inner state of the containers immutable for deletion."""
        raise ImmutableStateError()
