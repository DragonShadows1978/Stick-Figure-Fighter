"""Per-player input state."""
from dataclasses import dataclass
from enum import Enum, auto


class InputAction(Enum):
    """Abstract input actions."""
    NONE = auto()
    LEFT = auto()
    RIGHT = auto()
    UP = auto()
    DOWN = auto()
    JUMP = auto()
    ATTACK = auto()
    SPECIAL = auto()
    SHIELD = auto()
    GRAB = auto()


@dataclass
class InputState:
    """Current state of inputs for a player."""
    left: bool = False
    right: bool = False
    up: bool = False
    down: bool = False
    jump: bool = False
    attack: bool = False
    special: bool = False
    shield: bool = False
    grab: bool = False

    # For detecting new presses (not held)
    _prev_jump: bool = False
    _prev_attack: bool = False
    _prev_special: bool = False
    _prev_shield: bool = False
    _prev_grab: bool = False

    @property
    def jump_pressed(self) -> bool:
        """True only on the frame jump was first pressed."""
        return self.jump and not self._prev_jump

    @property
    def attack_pressed(self) -> bool:
        return self.attack and not self._prev_attack

    @property
    def special_pressed(self) -> bool:
        return self.special and not self._prev_special

    @property
    def shield_pressed(self) -> bool:
        return self.shield and not self._prev_shield

    @property
    def grab_pressed(self) -> bool:
        return self.grab and not self._prev_grab

    def update_prev(self):
        """Call at end of frame to update previous state."""
        self._prev_jump = self.jump
        self._prev_attack = self.attack
        self._prev_special = self.special
        self._prev_shield = self.shield
        self._prev_grab = self.grab

    def clear(self):
        """Clear all inputs."""
        self.left = False
        self.right = False
        self.up = False
        self.down = False
        self.jump = False
        self.attack = False
        self.special = False
        self.shield = False
        self.grab = False
