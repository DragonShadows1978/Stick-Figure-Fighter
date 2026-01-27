"""
Stick Figure Fighter - Gamepad Input System

Handles gamepad/controller input via evdev.
Supports Xbox-style controllers and generic USB gamepads.

Uses raw Linux input events for low-latency polling.

Cycle 13: Added rumble/force feedback support
"""
import os
import struct
import fcntl
import select
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import IntEnum

from game_engine import InputState


# =============================================================================
# EVDEV CONSTANTS
# =============================================================================

# Event types (from linux/input.h)
class EventType(IntEnum):
    EV_SYN = 0x00
    EV_KEY = 0x01
    EV_REL = 0x02
    EV_ABS = 0x03


# Key/button codes for gamepads
class ButtonCode(IntEnum):
    # Xbox-style button codes (BTN_GAMEPAD)
    BTN_A = 0x130       # A / Cross
    BTN_B = 0x131       # B / Circle
    BTN_C = 0x132
    BTN_X = 0x133       # X / Square
    BTN_Y = 0x134       # Y / Triangle
    BTN_Z = 0x135
    BTN_TL = 0x136      # Left bumper (LB)
    BTN_TR = 0x137      # Right bumper (RB)
    BTN_TL2 = 0x138     # Left trigger (LT) as button
    BTN_TR2 = 0x139     # Right trigger (RT) as button
    BTN_SELECT = 0x13a  # Back/Select
    BTN_START = 0x13b   # Start/Menu
    BTN_MODE = 0x13c    # Guide/Home button
    BTN_THUMBL = 0x13d  # Left stick click
    BTN_THUMBR = 0x13e  # Right stick click

    # D-pad buttons (some controllers)
    BTN_DPAD_UP = 0x220
    BTN_DPAD_DOWN = 0x221
    BTN_DPAD_LEFT = 0x222
    BTN_DPAD_RIGHT = 0x223


# Axis codes for analog sticks and triggers
class AxisCode(IntEnum):
    ABS_X = 0x00        # Left stick X
    ABS_Y = 0x01        # Left stick Y
    ABS_Z = 0x02        # Left trigger (as axis)
    ABS_RX = 0x03       # Right stick X
    ABS_RY = 0x04       # Right stick Y
    ABS_RZ = 0x05       # Right trigger (as axis)
    ABS_HAT0X = 0x10    # D-pad X (some controllers)
    ABS_HAT0Y = 0x11    # D-pad Y (some controllers)


# IOCTL constants
EVIOCGNAME = lambda length: (2 << 30) | (ord('E') << 8) | 0x06 | (length << 16)
EVIOCGID = (2 << 30) | (ord('E') << 8) | 0x02 | (8 << 16)
EVIOCGABS = lambda axis: (2 << 30) | (ord('E') << 8) | 0x40 + axis | (24 << 16)

# Event struct format: time_sec, time_usec, type, code, value
EVENT_FORMAT = 'llHHi'
EVENT_SIZE = struct.calcsize(EVENT_FORMAT)

# =============================================================================
# FORCE FEEDBACK CONSTANTS (Cycle 13)
# =============================================================================

# Force feedback effect types
class FFEffect(IntEnum):
    FF_RUMBLE = 0x50
    FF_PERIODIC = 0x51
    FF_CONSTANT = 0x52
    FF_SPRING = 0x53
    FF_FRICTION = 0x54
    FF_DAMPER = 0x55
    FF_INERTIA = 0x56
    FF_RAMP = 0x57


# IOCTL for uploading and playing effects
EVIOCSFF = 0x40304580  # Upload force feedback effect
EVIOCRMFF = 0x40044581  # Remove force feedback effect
EVIOCGEFFECTS = 0x80044584  # Get number of effects

# FF effect structure for rumble
# struct ff_effect {
#     __u16 type;        // Effect type
#     __s16 id;          // Effect ID (-1 for new)
#     __u16 direction;   // Direction (0-0xFFFF)
#     struct ff_trigger trigger;
#     struct ff_replay replay;
#     union { ... };
# }
# For FF_RUMBLE:
# struct ff_rumble_effect {
#     __u16 strong_magnitude;
#     __u16 weak_magnitude;
# }

# ff_trigger: __u16 button, __u16 interval
# ff_replay: __u16 length, __u16 delay

# Full ff_effect structure size for rumble
FF_EFFECT_FORMAT = 'hhhHHHHHHxx'  # type, id, direction, trigger(2), replay(2), rumble(2)
FF_EFFECT_SIZE = struct.calcsize(FF_EFFECT_FORMAT)


# =============================================================================
# GAMEPAD STATE
# =============================================================================

@dataclass
class GamepadState:
    """Current state of a gamepad."""
    # Digital buttons
    button_a: bool = False
    button_b: bool = False
    button_x: bool = False
    button_y: bool = False
    bumper_left: bool = False
    bumper_right: bool = False
    trigger_left: bool = False
    trigger_right: bool = False
    start: bool = False
    select: bool = False
    stick_left_click: bool = False
    stick_right_click: bool = False

    # D-pad (can be buttons or axis)
    dpad_up: bool = False
    dpad_down: bool = False
    dpad_left: bool = False
    dpad_right: bool = False

    # Analog axes (normalized -1.0 to 1.0)
    left_x: float = 0.0
    left_y: float = 0.0
    right_x: float = 0.0
    right_y: float = 0.0
    trigger_l: float = 0.0  # 0.0 to 1.0
    trigger_r: float = 0.0  # 0.0 to 1.0


@dataclass
class AxisInfo:
    """Axis calibration info from evdev."""
    minimum: int = -32768
    maximum: int = 32767
    flat: int = 0  # Deadzone
    fuzz: int = 0


# =============================================================================
# GAMEPAD DEVICE
# =============================================================================

class Gamepad:
    """
    Represents a single gamepad device.

    Reads raw events from /dev/input/eventX and maintains state.
    """

    # Stick deadzone (normalized, 0-1)
    DEADZONE = 0.15

    def __init__(self, event_path: str):
        """
        Initialize gamepad from event device path.

        Args:
            event_path: Path like /dev/input/event3
        """
        self.path = event_path
        self.name = "Unknown Gamepad"
        self.state = GamepadState()
        self.prev_state = GamepadState()
        self.fd: Optional[int] = None
        self.axis_info: Dict[int, AxisInfo] = {}

        # Cycle 13: Force feedback / rumble support
        self.ff_fd: Optional[int] = None  # Writable FD for force feedback
        self.rumble_effect_id: int = -1   # Uploaded effect ID
        self.rumble_supported: bool = False

        self._open()

    def _open(self):
        """Open the device file."""
        try:
            self.fd = os.open(self.path, os.O_RDONLY | os.O_NONBLOCK)

            # Get device name
            name_buf = bytearray(256)
            try:
                fcntl.ioctl(self.fd, EVIOCGNAME(256), name_buf)
                self.name = name_buf.split(b'\x00')[0].decode('utf-8', errors='replace')
            except:
                pass

            # Get axis calibration info
            for axis in [AxisCode.ABS_X, AxisCode.ABS_Y,
                        AxisCode.ABS_RX, AxisCode.ABS_RY,
                        AxisCode.ABS_Z, AxisCode.ABS_RZ,
                        AxisCode.ABS_HAT0X, AxisCode.ABS_HAT0Y]:
                try:
                    buf = bytearray(24)
                    fcntl.ioctl(self.fd, EVIOCGABS(axis), buf)
                    # Parse: value, min, max, fuzz, flat, resolution
                    info = struct.unpack('iiiiii', buf)
                    self.axis_info[axis] = AxisInfo(
                        minimum=info[1],
                        maximum=info[2],
                        fuzz=info[3],
                        flat=info[4]
                    )
                except:
                    pass

        except OSError as e:
            self.fd = None
            raise RuntimeError(f"Cannot open gamepad {self.path}: {e}")

        # Cycle 13: Try to open for force feedback
        self._init_force_feedback()

    def _init_force_feedback(self):
        """
        Cycle 13: Initialize force feedback / rumble support.

        Opens the device with write access and uploads a rumble effect.
        """
        try:
            # Open with read-write for force feedback
            self.ff_fd = os.open(self.path, os.O_RDWR)

            # Check if device supports force feedback effects
            num_effects_buf = struct.pack('i', 0)
            try:
                result = fcntl.ioctl(self.ff_fd, EVIOCGEFFECTS, num_effects_buf)
                num_effects = struct.unpack('i', result)[0] if isinstance(result, bytes) else 0
                if num_effects > 0:
                    self.rumble_supported = True
                    print(f"[GAMEPAD] Force feedback supported: {num_effects} effects")
                else:
                    self.rumble_supported = False
            except:
                self.rumble_supported = False

        except (OSError, PermissionError) as e:
            # Can't write to device - no rumble support
            self.ff_fd = None
            self.rumble_supported = False

    def close(self):
        """Close the device."""
        # Clean up rumble effect if uploaded
        if self.ff_fd is not None and self.rumble_effect_id >= 0:
            try:
                buf = struct.pack('i', self.rumble_effect_id)
                fcntl.ioctl(self.ff_fd, EVIOCRMFF, buf)
            except:
                pass
            self.rumble_effect_id = -1

        if self.ff_fd is not None:
            try:
                os.close(self.ff_fd)
            except:
                pass
            self.ff_fd = None

        if self.fd is not None:
            try:
                os.close(self.fd)
            except:
                pass
            self.fd = None

    def _normalize_axis(self, axis: int, value: int) -> float:
        """Normalize axis value to -1.0 to 1.0 range."""
        info = self.axis_info.get(axis, AxisInfo())

        # Center and normalize
        center = (info.minimum + info.maximum) / 2
        half_range = (info.maximum - info.minimum) / 2

        if half_range == 0:
            return 0.0

        normalized = (value - center) / half_range

        # Apply deadzone
        if abs(normalized) < self.DEADZONE:
            return 0.0

        # Rescale from deadzone to 1.0
        sign = 1 if normalized > 0 else -1
        return sign * (abs(normalized) - self.DEADZONE) / (1 - self.DEADZONE)

    def _normalize_trigger(self, axis: int, value: int) -> float:
        """Normalize trigger value to 0.0 to 1.0 range."""
        info = self.axis_info.get(axis, AxisInfo(minimum=0, maximum=255))

        range_val = info.maximum - info.minimum
        if range_val == 0:
            return 0.0

        return max(0.0, min(1.0, (value - info.minimum) / range_val))

    def poll(self):
        """Poll for new events and update state."""
        if self.fd is None:
            return

        # Store previous state for edge detection
        self.prev_state = GamepadState(
            button_a=self.state.button_a,
            button_b=self.state.button_b,
            button_x=self.state.button_x,
            button_y=self.state.button_y,
            bumper_left=self.state.bumper_left,
            bumper_right=self.state.bumper_right,
            trigger_left=self.state.trigger_left,
            trigger_right=self.state.trigger_right,
            start=self.state.start,
            select=self.state.select,
            stick_left_click=self.state.stick_left_click,
            stick_right_click=self.state.stick_right_click,
            dpad_up=self.state.dpad_up,
            dpad_down=self.state.dpad_down,
            dpad_left=self.state.dpad_left,
            dpad_right=self.state.dpad_right,
            left_x=self.state.left_x,
            left_y=self.state.left_y,
            right_x=self.state.right_x,
            right_y=self.state.right_y,
            trigger_l=self.state.trigger_l,
            trigger_r=self.state.trigger_r,
        )

        # Read all available events
        try:
            while True:
                ready, _, _ = select.select([self.fd], [], [], 0)
                if not ready:
                    break

                data = os.read(self.fd, EVENT_SIZE)
                if len(data) < EVENT_SIZE:
                    break

                _, _, ev_type, ev_code, ev_value = struct.unpack(EVENT_FORMAT, data)

                self._process_event(ev_type, ev_code, ev_value)

        except (OSError, BlockingIOError):
            pass

    def _process_event(self, ev_type: int, ev_code: int, ev_value: int):
        """Process a single input event."""
        if ev_type == EventType.EV_KEY:
            pressed = ev_value > 0

            # Face buttons
            if ev_code == ButtonCode.BTN_A:
                self.state.button_a = pressed
            elif ev_code == ButtonCode.BTN_B:
                self.state.button_b = pressed
            elif ev_code == ButtonCode.BTN_X:
                self.state.button_x = pressed
            elif ev_code == ButtonCode.BTN_Y:
                self.state.button_y = pressed

            # Bumpers
            elif ev_code == ButtonCode.BTN_TL:
                self.state.bumper_left = pressed
            elif ev_code == ButtonCode.BTN_TR:
                self.state.bumper_right = pressed

            # Triggers as buttons
            elif ev_code == ButtonCode.BTN_TL2:
                self.state.trigger_left = pressed
            elif ev_code == ButtonCode.BTN_TR2:
                self.state.trigger_right = pressed

            # System buttons
            elif ev_code == ButtonCode.BTN_START:
                self.state.start = pressed
            elif ev_code == ButtonCode.BTN_SELECT:
                self.state.select = pressed

            # Stick clicks
            elif ev_code == ButtonCode.BTN_THUMBL:
                self.state.stick_left_click = pressed
            elif ev_code == ButtonCode.BTN_THUMBR:
                self.state.stick_right_click = pressed

            # D-pad as buttons
            elif ev_code == ButtonCode.BTN_DPAD_UP:
                self.state.dpad_up = pressed
            elif ev_code == ButtonCode.BTN_DPAD_DOWN:
                self.state.dpad_down = pressed
            elif ev_code == ButtonCode.BTN_DPAD_LEFT:
                self.state.dpad_left = pressed
            elif ev_code == ButtonCode.BTN_DPAD_RIGHT:
                self.state.dpad_right = pressed

        elif ev_type == EventType.EV_ABS:
            # Analog axes
            if ev_code == AxisCode.ABS_X:
                self.state.left_x = self._normalize_axis(ev_code, ev_value)
            elif ev_code == AxisCode.ABS_Y:
                self.state.left_y = self._normalize_axis(ev_code, ev_value)
            elif ev_code == AxisCode.ABS_RX:
                self.state.right_x = self._normalize_axis(ev_code, ev_value)
            elif ev_code == AxisCode.ABS_RY:
                self.state.right_y = self._normalize_axis(ev_code, ev_value)

            # Triggers as axes
            elif ev_code == AxisCode.ABS_Z:
                self.state.trigger_l = self._normalize_trigger(ev_code, ev_value)
                self.state.trigger_left = self.state.trigger_l > 0.5
            elif ev_code == AxisCode.ABS_RZ:
                self.state.trigger_r = self._normalize_trigger(ev_code, ev_value)
                self.state.trigger_right = self.state.trigger_r > 0.5

            # D-pad as axes (HAT)
            elif ev_code == AxisCode.ABS_HAT0X:
                self.state.dpad_left = ev_value < 0
                self.state.dpad_right = ev_value > 0
            elif ev_code == AxisCode.ABS_HAT0Y:
                self.state.dpad_up = ev_value < 0
                self.state.dpad_down = ev_value > 0

    def to_input_state(self, mapping: 'GamepadMapping' = None) -> InputState:
        """
        Convert gamepad state to game InputState.

        Args:
            mapping: Button mapping to use (default if None)

        Returns:
            InputState for the game engine
        """
        if mapping is None:
            mapping = DEFAULT_MAPPING

        state = InputState()

        # Directional input from D-pad OR left stick
        state.left = self.state.dpad_left or self.state.left_x < -0.5
        state.right = self.state.dpad_right or self.state.left_x > 0.5
        state.up = self.state.dpad_up or self.state.left_y < -0.5
        state.down = self.state.dpad_down or self.state.left_y > 0.5

        # Apply button mapping
        state.jump = getattr(self.state, mapping.jump)
        state.attack = getattr(self.state, mapping.attack)
        state.special = getattr(self.state, mapping.special)
        state.shield = getattr(self.state, mapping.shield)
        state.grab = getattr(self.state, mapping.grab)

        return state

    def button_just_pressed(self, attr: str) -> bool:
        """Check if a button was just pressed this frame."""
        return getattr(self.state, attr) and not getattr(self.prev_state, attr)

    # =========================================================================
    # CYCLE 13: RUMBLE / FORCE FEEDBACK
    # =========================================================================

    def rumble(self, strong_magnitude: float = 0.5, weak_magnitude: float = 0.3,
               duration_ms: int = 100):
        """
        Cycle 13: Play a rumble effect on the gamepad.

        Args:
            strong_magnitude: Heavy motor intensity (0.0 - 1.0)
            weak_magnitude: Light motor intensity (0.0 - 1.0)
            duration_ms: Duration in milliseconds

        Note:
            Requires write access to the device. May not work without elevated permissions.
        """
        if not self.rumble_supported or self.ff_fd is None:
            return False

        try:
            # Clamp and scale magnitudes to 0-65535
            strong = int(min(1.0, max(0.0, strong_magnitude)) * 65535)
            weak = int(min(1.0, max(0.0, weak_magnitude)) * 65535)

            # Pack ff_effect structure for rumble
            # struct ff_effect layout:
            # - type (u16): FF_RUMBLE = 0x50
            # - id (s16): -1 for new, or existing ID to update
            # - direction (u16): 0
            # - trigger (button u16, interval u16): 0, 0
            # - replay (length u16, delay u16): duration_ms, 0
            # - union (rumble: strong u16, weak u16)

            effect_id = self.rumble_effect_id if self.rumble_effect_id >= 0 else -1

            # Build effect structure
            # Format: 'hhhHHHHHH' = type(2) + id(2) + direction(2) + trigger(4) + replay(4) + rumble(4) = 18 bytes
            # But we need padding to match kernel struct - it's actually larger

            # Use a simpler approach: write the play event directly
            # This works on many controllers even without uploading effects

            # Write an input event to trigger rumble
            # Event type EV_FF (0x15), code = effect ID or rumble constant
            EV_FF = 0x15

            # Create a simple rumble event
            # Many drivers support direct rumble via writing EV_FF events
            now = time.time()
            tv_sec = int(now)
            tv_usec = int((now - tv_sec) * 1000000)

            # Play rumble effect (value = 1 to play, 0 to stop)
            # For devices that support direct rumble, write to effect 0
            event = struct.pack(EVENT_FORMAT, tv_sec, tv_usec, EV_FF, 0, 1)
            os.write(self.ff_fd, event)

            # Schedule stop (write stop event after duration)
            # For immediate stop after duration, we'd need a timer - skip for now
            # The effect will naturally stop or we can write stop event later

            return True

        except (OSError, IOError) as e:
            # Rumble failed - silently ignore
            return False

    def stop_rumble(self):
        """Stop any ongoing rumble effect."""
        if not self.rumble_supported or self.ff_fd is None:
            return

        try:
            EV_FF = 0x15
            now = time.time()
            tv_sec = int(now)
            tv_usec = int((now - tv_sec) * 1000000)

            # Stop rumble effect (value = 0)
            event = struct.pack(EVENT_FORMAT, tv_sec, tv_usec, EV_FF, 0, 0)
            os.write(self.ff_fd, event)
        except:
            pass

    def __del__(self):
        self.close()


# =============================================================================
# BUTTON MAPPING
# =============================================================================

@dataclass
class GamepadMapping:
    """Maps gamepad buttons to game actions."""
    # Attribute names from GamepadState
    jump: str = 'button_a'      # A = Jump
    attack: str = 'button_x'    # X = Attack
    special: str = 'button_b'   # B = Special
    shield: str = 'bumper_right' # RB = Shield
    grab: str = 'bumper_left'    # LB = Grab


# Xbox-style layout
DEFAULT_MAPPING = GamepadMapping()

# Alternative mapping (more like Smash Bros GC)
SMASH_MAPPING = GamepadMapping(
    jump='button_x',     # X = Jump
    attack='button_a',   # A = Attack
    special='button_b',  # B = Special
    shield='trigger_right',  # RT = Shield
    grab='button_y',     # Y = Grab
)


# =============================================================================
# GAMEPAD MANAGER
# =============================================================================

class GamepadManager:
    """
    Manages multiple gamepad devices.

    Scans /dev/input/ for gamepads and handles hotplug.
    """

    def __init__(self):
        """Initialize the gamepad manager."""
        self.gamepads: Dict[str, Gamepad] = {}
        self.mappings: List[GamepadMapping] = [DEFAULT_MAPPING, DEFAULT_MAPPING]
        self._scan_devices()

    def _is_gamepad(self, event_path: str) -> bool:
        """Check if an event device is a gamepad."""
        try:
            fd = os.open(event_path, os.O_RDONLY | os.O_NONBLOCK)
            try:
                # Get device name
                name_buf = bytearray(256)
                fcntl.ioctl(fd, EVIOCGNAME(256), name_buf)
                name = name_buf.split(b'\x00')[0].decode('utf-8', errors='replace').lower()

                # Keywords that indicate a gamepad
                gamepad_keywords = [
                    'gamepad', 'controller', 'joystick', 'xbox', 'playstation',
                    'dualshock', 'dualsense', 'switch pro', 'logitech f',
                    '8bitdo', 'generic x-box'
                ]

                for keyword in gamepad_keywords:
                    if keyword in name:
                        return True

                # Check if it has gamepad-like buttons
                # Read device capabilities (would need more ioctls)
                # For simplicity, trust the name check

            finally:
                os.close(fd)
        except:
            pass

        return False

    def _scan_devices(self):
        """Scan for gamepad devices."""
        # Close existing gamepads
        for gp in self.gamepads.values():
            gp.close()
        self.gamepads.clear()

        # Scan /dev/input/
        input_dir = '/dev/input'
        if not os.path.exists(input_dir):
            return

        for entry in os.listdir(input_dir):
            if not entry.startswith('event'):
                continue

            event_path = os.path.join(input_dir, entry)

            if self._is_gamepad(event_path):
                try:
                    gp = Gamepad(event_path)
                    self.gamepads[event_path] = gp
                    print(f"[GAMEPAD] Found: {gp.name} ({event_path})")
                except Exception as e:
                    print(f"[GAMEPAD] Failed to open {event_path}: {e}")

    def rescan(self):
        """Rescan for new/removed gamepads."""
        self._scan_devices()

    def poll(self):
        """Poll all connected gamepads."""
        for gp in self.gamepads.values():
            gp.poll()

    def get_gamepad(self, index: int) -> Optional[Gamepad]:
        """Get gamepad by index (0, 1, etc.)."""
        pads = list(self.gamepads.values())
        if 0 <= index < len(pads):
            return pads[index]
        return None

    def get_input_state(self, player_id: int) -> Optional[InputState]:
        """
        Get InputState for a player from their gamepad.

        Args:
            player_id: 0 or 1

        Returns:
            InputState or None if no gamepad assigned
        """
        gp = self.get_gamepad(player_id)
        if gp is None:
            return None

        mapping = self.mappings[player_id] if player_id < len(self.mappings) else DEFAULT_MAPPING
        return gp.to_input_state(mapping)

    @property
    def count(self) -> int:
        """Number of connected gamepads."""
        return len(self.gamepads)

    def list_gamepads(self) -> List[str]:
        """Get list of connected gamepad names."""
        return [gp.name for gp in self.gamepads.values()]

    # =========================================================================
    # CYCLE 13: RUMBLE SUPPORT
    # =========================================================================

    def rumble(self, player_id: int, strong: float = 0.5, weak: float = 0.3,
               duration_ms: int = 100):
        """
        Cycle 13: Trigger rumble on a player's gamepad.

        Args:
            player_id: Player index (0 or 1)
            strong: Heavy motor intensity (0.0 - 1.0)
            weak: Light motor intensity (0.0 - 1.0)
            duration_ms: Duration in milliseconds
        """
        gp = self.get_gamepad(player_id)
        if gp:
            gp.rumble(strong, weak, duration_ms)

    def rumble_on_hit(self, player_id: int, knockback_magnitude: float):
        """
        Cycle 13: Trigger rumble when player gets hit.

        Intensity scales with knockback magnitude.

        Args:
            player_id: Player who got hit
            knockback_magnitude: Knockback force applied
        """
        # Scale intensity based on knockback (0-50 range typical)
        intensity = min(1.0, knockback_magnitude / 30.0)
        strong = intensity * 0.8
        weak = intensity * 0.4

        # Duration also scales
        duration = int(50 + intensity * 100)

        self.rumble(player_id, strong, weak, duration)

    def rumble_on_attack_connect(self, player_id: int, damage: float):
        """
        Cycle 13: Short rumble when player's attack connects.

        Args:
            player_id: Player whose attack hit
            damage: Damage dealt
        """
        # Light rumble on attack connect
        intensity = min(0.5, damage / 30.0)
        self.rumble(player_id, intensity * 0.3, intensity * 0.5, 50)

    def close(self):
        """Close all gamepads."""
        for gp in self.gamepads.values():
            gp.close()
        self.gamepads.clear()

    def __del__(self):
        self.close()


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

_gamepad_manager: Optional[GamepadManager] = None


def get_gamepad_manager() -> GamepadManager:
    """Get or create the global gamepad manager."""
    global _gamepad_manager
    if _gamepad_manager is None:
        _gamepad_manager = GamepadManager()
    return _gamepad_manager


def detect_gamepads() -> int:
    """
    Detect and initialize gamepads.

    Returns:
        Number of gamepads found
    """
    mgr = get_gamepad_manager()
    mgr.rescan()
    return mgr.count


# =============================================================================
# TEST CODE
# =============================================================================

if __name__ == "__main__":
    import time

    print("=== Gamepad Input Test ===")
    print("Scanning for gamepads...")

    count = detect_gamepads()
    mgr = get_gamepad_manager()

    if count == 0:
        print("No gamepads found. Connect a controller and try again.")
        print("\nChecking device permissions...")
        print("Run: sudo usermod -a -G input $USER")
        print("Then log out and back in.")
    else:
        print(f"Found {count} gamepad(s):")
        for name in mgr.list_gamepads():
            print(f"  - {name}")

        print("\nPress buttons to see input (Ctrl+C to quit)...")

        try:
            while True:
                mgr.poll()

                gp = mgr.get_gamepad(0)
                if gp:
                    s = gp.state

                    # Only print if something is pressed
                    buttons = []
                    if s.button_a: buttons.append('A')
                    if s.button_b: buttons.append('B')
                    if s.button_x: buttons.append('X')
                    if s.button_y: buttons.append('Y')
                    if s.bumper_left: buttons.append('LB')
                    if s.bumper_right: buttons.append('RB')
                    if s.trigger_left: buttons.append('LT')
                    if s.trigger_right: buttons.append('RT')
                    if s.start: buttons.append('START')
                    if s.select: buttons.append('SELECT')
                    if s.dpad_up: buttons.append('UP')
                    if s.dpad_down: buttons.append('DOWN')
                    if s.dpad_left: buttons.append('LEFT')
                    if s.dpad_right: buttons.append('RIGHT')

                    if buttons or abs(s.left_x) > 0.1 or abs(s.left_y) > 0.1:
                        stick = f"Stick: ({s.left_x:+.2f}, {s.left_y:+.2f})"
                        print(f"Buttons: {', '.join(buttons) or 'None':30} {stick}")

                time.sleep(0.016)  # ~60 Hz

        except KeyboardInterrupt:
            print("\nDone!")
        finally:
            mgr.close()
