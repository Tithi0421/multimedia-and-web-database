from time import time
from functools import wraps

def timed(f):
    @wraps(f)
    def wrapper(*argv, **kwargs):
        start = time()
        return_val = f(*argv, **kwargs)
        print(str(f.__name__) + " executed in " + str(time() - start))
        return return_val
    return wrapper