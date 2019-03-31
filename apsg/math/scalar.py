# -*- coding: utf-8 -*-


import typing


__all__ = ("Scalar",)


Scalar = typing.NewType("Scalar", float)
"""Simple type alias for better semantics."""
