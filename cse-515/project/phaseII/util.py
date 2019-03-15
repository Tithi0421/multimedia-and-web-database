from time import time
from functools import wraps

##
# Add as decorator to any function to time how long the function takes to execute.
#
# Decorates must come after static declarations. For example:
#   @staticmethod
#   @timed
#   def foo():
#
def timed(f):
    @wraps(f)
    def wrapper(*argv, **kwargs):
        start = time()
        return_val = f(*argv, **kwargs)
        print(str(f.__name__) + " executed in " + str(time() - start))
        return return_val
    return wrapper