from shape import Shape, ColorMixin, MaterialMixin
from typing import Union, Optional
import math

class ComplexShape(Shape, ColorMixin, MaterialMixin):
    def __init__(self, color: str, material: str):
        # Using super() with explicit MRO handling
        super().__init__()
        ColorMixin.__init__(self, color)
        MaterialMixin.__init__(self, material)
        
    def __str__(self):
        return f"Complex shape with {self.color} color and {self._material} material"

class HybridCircle(ComplexShape):
    def __init__(self, radius: Union[int, float], color: str, material: str):
        super().__init__(color, material)
        self._radius = radius
        self._cached_area: Optional[float] = None

    @property
    def radius(self):
        return self._radius

    @radius.setter
    def radius(self, value):
        self._radius = value
        self._cached_area = None  # Invalidate cache

    def area(self):
        if self._cached_area is None:
            self._cached_area = math.pi * self._radius ** 2
        return self._cached_area

    def perimeter(self):
        return 2 * math.pi * self._radius

    def mass(self):
        return self.area() * self._density