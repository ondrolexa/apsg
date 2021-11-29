import sys
import numpy as np

from apsg.helpers._helper import is_like_vec3, is_like_matrix3

##################
#   DECORATORS   #
##################

########## factory could be used #############


def ensure_first_arg(*datatypes):
    def decorator(method):
        def wrapper(arg, *args, **kwargs):
            nargs = list(args)
            ok = False
            for clsname in datatypes:
                cls = getattr(sys.modules[__name__], clsname)
                try:
                    nargs[0] = cls(nargs[0])
                    ok = True
                except:
                    pass
            if ok:
                return method(*nargs, **kwargs)
            else:
                raise TypeError(
                    f'Unsupported arguments for {method.__name__}. Must be one of {" ".join(datatypes)}'
                )

        return wrapper

    return decorator


################################################


def ensure_first_arg_same(method):
    def arg_check(self, *args):
        nargs = list(args)
        cls = type(self)
        if np.asarray(args[0]).shape == cls.__shape__:
            nargs[0] = cls(args[0])
            return method(self, *nargs)
        raise TypeError(f'Unsupported argument for {method.__name__}. Expecting {cls.__name__}')

    return arg_check

################################################
#  Following are not used....
################################################

def ensure_two_args_same(method):
    def arg_check(self, *args):
        nargs = list(args)
        if len(nargs) == 2:
            if np.asarray(args[0]).shape == cls.__shape__ and np.asarray(args[1]).shape == cls.__shape__:
                nargs[0] = type(self)(args[0])
                nargs[1] = type(self)(args[1])
                return method(self, *nargs)
        raise TypeError(f'Unsupported arguments for {method.__name__}.')

    return arg_check


def ensure_one_arg_vector_or_numeric(method):
    def arg_check(self, *args):
        nargs = list(args)
        if is_like_vec3(nargs[0]):
            nargs[0] = type(self)(args[0])
            return method(self, *nargs)
        elif type(args[0]) in (float, int) or issubclass(type(args[0]), np.number):
            return method(self, *nargs)
        raise TypeError(f"unsupported argument for {method.__name__}")

    return arg_check


def ensure_one_arg_matrix_or_numeric(method):
    def arg_check(self, *args):
        nargs = list(args)
        if is_like_matrix3(nargs[0]):
            nargs[0] = type(self)(args[0])
            return method(self, *nargs)
        elif type(args[0]) in (float, int) or issubclass(type(args[0]), np.number):
            return method(self, *nargs)
        raise TypeError(f"unsupported argument for {method.__name__}")

    return arg_check


def ensure_one_arg_matrix_vector_or_numeric(method):
    def arg_check(self, *args):
        nargs = list(args)
        if is_like_matrix3(nargs[0]):
            nargs[0] = type(self)(args[0])
            return method(self, *nargs)
        elif is_like_vec3(nargs[0]):
            nargs[0] = type(self)(args[0])
            return method(self, *nargs)
        elif type(args[0]) in (float, int) or issubclass(type(args[0]), np.number):
            return method(self, *nargs)
        raise TypeError(f"unsupported argument {args[0]} for {method.__name__}")

    return arg_check
