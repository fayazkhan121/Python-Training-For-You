from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def area(self):
        pass

    @abstractmethod
    def perimeter(self):
        pass

class ColorMixin:
    def __init__(self, color="black"):
        self._color = color
    
    @property
    def color(self):
        return self._color

    @color.setter
    def color(self, value):
        self._color = value

class MaterialMixin:
    def __init__(self, material="default"):
        self._material = material
        self._density = self._get_density()

    def _get_density(self):
        densities = {
            "steel": 7.85,
            "aluminum": 2.7,
            "plastic": 1.2
        }
        return densities.get(self._material, 1.0)