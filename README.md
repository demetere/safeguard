# safeguard

This is a simple library for building happy path in python, this looks like a Rust's `Result` type.
This library is inspired by two other libraries that are [returns](https://github.com/dry-python/returns)
and [result](https://github.com/rustedpy/result).

## TODO

- update README to provide more examples
- generate Documentation
- add more tests
- add tox for testing
- add CI/CD

## Installation

Latest release:

``` sh
$ pip install safeguard
```

Latest GitHub `main` branch version:

``` sh
$ pip install git+https://github.com/demetere/safeguard
```

## TL/DR

The main purpose of this library is to provide a way to build happy path without thinking about exceptions. Because of
that we have two main Containers `Maybe` and `Result`.

### `Maybe`

`Maybe` is a way to work with optional values. It can be either `Some(value)` or `Nothing`.
`Some` is a class encapsulating an arbitrary value. `Maybe[T]` is a generic type alias
for `typing.Union[Some[T], None]`. Lets imagine having function for division

``` python
def divide(a: int, b: int) -> Optional[int]:
    if b == 0:
        return None
    return a // b
```

We want to do some operation on the result but we dont know if it returned None or not. We can do it like this:

``` python
def upstream_function():
    result = divide(10, 0)
    if result is None:
        return None
    return result * 2
```

or we can use `Maybe` like this:

``` python
from safeguard.maybe import Maybe, Some, Nothing, maybe

def divide(a: int, b: int) -> Maybe[int]:
    if b == 0:
        return Nothing
    return Some(a // b)
    
def upstream_function():
    result = divide(10, 0)
    return result.map(lambda x: x * 2)
```

What maybe does in this context is that it will return `Nothing` if the result is `None` and `Some` if the result is
not `None`.
This way we can chain operations on the result without thinking about handling `None` values. Alternatively we can
use decorators to use Maybe in a more elegant way:

``` python
from safeguard.maybe import maybe

@maybe
def divide(a: int, b: int) -> int:
    if b == 0:
        return None
    return a // b
    
def upstream_function():
    return divide(10, 0).map(lambda x: x * 2)
```

This will automatically handle None values and return `Nothing` if the result is `None`.

### Result

The idea is that a result value can be either `Ok(value)` or
`Err(error)`, with a way to differentiate between the two. `Ok` and
`Err` are both classes inherited from `Result[T, E]`.
We can use `Result` to handle errors in a more elegant way. Lets imagine having function for fetching user by email:

``` python
def get_user_by_email(email: str) -> Tuple[Optional[User], Optional[str]]:
    """
    Return the user instance or an error message.
    """
    if not user_exists(email):
        return None, 'User does not exist'
    if not user_active(email):
        return None, 'User is inactive'
    user = get_user(email)
    return user, None

user, reason = get_user_by_email('ueli@example.com')
if user is None:
    raise RuntimeError('Could not fetch user: %s' % reason)
else:
    do_something(user)
```

We can refactor this code to use `Result` like this:

``` python
from safeguard.result import Ok, Err, Result

def get_user_by_email(email: str) -> Result[User, str]:
    """
    Return the user instance or an error message.
    """
    if not user_exists(email):
        return Err('User does not exist')
    if not user_active(email):
        return Err('User is inactive')
    user = get_user(email)
    return Ok(user)

user_result = get_user_by_email(email)
if isinstance(user_result, Ok): # or `user_result.is_ok()`
    # type(user_result.unwrap()) == User
    do_something(user_result.unwrap())
else: # or `elif user_result.is_err()`
    # type(user_result.unwrap_err()) == str
    raise RuntimeError('Could not fetch user: %s' % user_result.unwrap_err())
```

If you're using python version `3.10` or later, you can use the
elegant `match` statement as well:

``` python
from result import Result, Ok, Err

def divide(a: int, b: int) -> Result[int, str]:
    if b == 0:
        return Err("Cannot divide by zero")
    return Ok(a // b)

values = [(10, 0), (10, 5)]
for a, b in values:
    match divide(a, b):
        case Ok(value):
            print(f"{a} // {b} == {value}")
        case Err(e):
            print(e)
```

## Contributing

Everyone is welcome to contribute.

You just need to fork the repository, run `poetry install` so you can have the same environment,
make your changes and create a pull request. We will review your changes and merge them if they are good.

## Related Projects

The inspiration was taken from following libraries, some of the ideas and code fragments are from their codebase:

- [returns](https://github.com/dry-python/returns)
- [result](https://github.com/rustedpy/result)

## License

MIT License