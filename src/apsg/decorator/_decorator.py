import numpy as np

##################
#   DECORATORS   #
##################


def ensure_arguments(*datatypes):
    def decorator(method):
        def wrapper(self, *args, **kwargs):
            nargs = list(args)
            ok = []
            for ix, cls in enumerate(datatypes):
                try:
                    nargs[ix] = cls(nargs[ix])
                    ok.append(True)
                except Exception:
                    ok.append(False)
            if all(ok):
                return method(self, *nargs, **kwargs)
            else:
                raise TypeError(
                    f'Unsupported arguments for {method.__name__}. \
                      Must be {" or ".join([dt.__name__ for dt in datatypes])}'
                )

        return wrapper

    return decorator


def ensure_first_arg_same(method):
    def arg_check(self, *args):
        nargs = list(args)
        cls = type(self)
        if np.asarray(args[0]).shape == cls.__shape__:
            nargs[0] = cls(args[0])
            return method(self, *nargs)
        raise TypeError(
            f"Unsupported argument for {method.__name__}. Expecting {cls.__name__}"
        )

    return arg_check
