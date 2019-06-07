"""
We define algebra for angle type as follow.

Operations:

- The angles addition: `a + b = c`, where `a`, `b`, `c`, are angles.
- The angles scaling: `s * a = b`, where `a`, `b` are angles and `s` is scalar.
- The zero element.
- The opposite element. 

For conversion between radians and degrees use the standard library module 
functions `math.degrees` and `math.radians`.

Two angles are equal iff thez have the same size and units.

"""


import math


class Angle:
    """
    An angle measures the amount of turn.
    """

    def __init__(self, value, units):
        """
        units = 'degrees' | 'radians'
        """
        self.value = value
        self.units = units

    def __add__(self, other) -> 'Angle':
        # pre-conditions
        if self.units != other.units:
            raise Exception(f"The units `{self.units}` and `{other.units}` doesn't matches!")

        return self.__class__(self.value + other.value, self.units)

    def __sub__(self, other) -> 'Angle':
        return -self + other

    def __neq__(self) -> 'Angle':
        return self.__class__(-self.value, self.units)

    def __eq__(self, other) -> bool:
        return isinstance(other, self.__class__) \
            and self.units == other.units \
            and self.value == other.value

    def __hash__(self, other) -> int:
        return hash((self.__class__.__name__, self.value, self.units))

    def __abs__(self) -> float:
        return abs(self.value)

    @property
    def is_negative(self):
        return self.value < 0

    @property
    def is_positive(self):
        return not self.is_negative
    


class Turn(Angle):
    """
    An angle in as the ratio of full angle.

    It is useful if you want to express rotation as:
    
    Examples:
 
      - Turn(0.5) = 180 degrees
      - Turn(1.0) = 360 degrees
      - Turn(1.5) = 540 degrees
      - ...
    
    """
    def __init__(self, value):
        super().__init__(value=value, units="turn")

    @property
    def degree(self):
        return

    @property
    def radian(self):
        return


class Radian(Angle):
    """
    An angle in radians.
    """

    def __init__(self, value):
        super().__init__(value=value, units="radian")

    @property
    def turn(self):
        return 

    @property
    def degree(self):
        return Degree(self.value * (180/math.pi))


class Degree(Angle):
    """
    An angle in degrees.
    """

    def __init__(self, value):
        super().__init__(value=value, units="degree")

    def turn(self):
        return

    def radian(self):
        return Radian(self.value)
