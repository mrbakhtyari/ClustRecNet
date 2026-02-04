import warnings
from functools import wraps
from typing import Callable

def ignore_warnings(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"the number of connected components.*",
                category=UserWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message=r"Graph is not fully connected.*",
                category=UserWarning,
            )
            return func(*args, **kwargs)
    return wrapper
