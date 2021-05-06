from collections.abc import Callable
from time import time
from functools import wraps


def time_it(message: str, log_function: Callable):
    def real_decorator(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            start = time()
            val = function(*args, **kwargs)
            log_function(f"{message}: {time() - start}")
            return val

        return wrapper
    return real_decorator
