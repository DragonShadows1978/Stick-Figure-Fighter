"""Stage geometry. Concrete stages are game content and live outside the engine."""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .geometry import AABB, Vector2


@dataclass
class Platform:
    """A platform in the stage."""
    x: float
    y: float
    width: float
    height: float = 10  # Default thin platform
    passthrough: bool = True  # Can drop through
    contact_offset: float = 5.0  # Offset contact zone for early landing detection (pixels above surface)

    def get_aabb(self) -> AABB:
        return AABB(self.x, self.y, self.width, self.height)

    @property
    def surface_y(self) -> float:
        """Y position of the top surface."""
        return self.y

    @property
    def contact_zone_y(self) -> float:
        """Y position of the contact detection zone (offset volume)."""
        return self.y - self.contact_offset


@dataclass
class ParallaxLayer:
    """A parallax background layer (render-only data carried with the stage)."""
    depth: float  # 0.0 = foreground (moves with camera), 1.0 = background (doesn't move)
    elements: List[Dict]  # List of {'type': 'line'|'rect'|'circle'|'triangle', 'x': int, 'y': int, ...}
    color: Tuple[int, int, int] = (50, 50, 60)


@dataclass
class Stage:
    """A stage definition."""
    name: str
    platforms: List[Platform]
    main_platform: Platform  # The main ground platform
    blast_zones: Dict[str, float]  # left, right, top, bottom
    spawn_points: List[Vector2]  # Where fighters spawn
    parallax_layers: List[ParallaxLayer] = field(default_factory=list)

    @property
    def all_platforms(self) -> List[Platform]:
        """Main platform first, then the rest."""
        return [self.main_platform] + self.platforms
