"""2D vector and axis-aligned bounding box."""
import math
from dataclasses import dataclass
from typing import Tuple


@dataclass
class Vector2:
    """2D vector for positions and velocities."""
    x: float = 0.0
    y: float = 0.0

    def __add__(self, other: 'Vector2') -> 'Vector2':
        return Vector2(self.x + other.x, self.y + other.y)

    def __sub__(self, other: 'Vector2') -> 'Vector2':
        return Vector2(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> 'Vector2':
        return Vector2(self.x * scalar, self.y * scalar)

    def __rmul__(self, scalar: float) -> 'Vector2':
        return self * scalar

    def magnitude(self) -> float:
        return math.sqrt(self.x ** 2 + self.y ** 2)

    def normalized(self) -> 'Vector2':
        mag = self.magnitude()
        if mag == 0:
            return Vector2(0, 0)
        return Vector2(self.x / mag, self.y / mag)

    def copy(self) -> 'Vector2':
        return Vector2(self.x, self.y)

    def to_tuple(self) -> Tuple[float, float]:
        return (self.x, self.y)


@dataclass
class AABB:
    """Axis-Aligned Bounding Box for collision detection."""
    x: float  # left edge
    y: float  # top edge
    width: float
    height: float

    @property
    def left(self) -> float:
        return self.x

    @property
    def right(self) -> float:
        return self.x + self.width

    @property
    def top(self) -> float:
        return self.y

    @property
    def bottom(self) -> float:
        return self.y + self.height

    @property
    def center(self) -> Vector2:
        return Vector2(self.x + self.width / 2, self.y + self.height / 2)

    def intersects(self, other: 'AABB') -> bool:
        """Check if this AABB intersects with another."""
        return (self.left < other.right and
                self.right > other.left and
                self.top < other.bottom and
                self.bottom > other.top)

    def contains_point(self, x: float, y: float) -> bool:
        """Check if a point is inside this AABB."""
        return (self.left <= x <= self.right and
                self.top <= y <= self.bottom)
