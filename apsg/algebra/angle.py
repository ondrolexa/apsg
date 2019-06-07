"""
The algebra of computer graphics.

We define algebra for angle type as follow.

a `+` b, where a, b are angles.
s `*` a, where a is angle and s is scalar.
-a is opposite item to a 

For conversion between radians and degrees use the standard library module 
functions `math.degrees` and `math.radians`.

"""


import math


class Angle:
    """
    The angle is ...
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


class Turn(Angle):
    """
    The angle in as the ration of full angle.

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
    The angle in radians.
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
    The angle in degrees.
    """

    def __init__(self, value):
        super().__init__(value=value, units="degree")

    def turn(self):
        return

    def radian(self):
        return Radian(self.value)
