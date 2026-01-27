"""
Stick Figure Fighter - Input System

Handles keyboard input via X11 and maps to game actions.
Supports multiple players with configurable key bindings.
"""
import ctypes
from ctypes import c_void_p, c_int, c_char_p, c_ulong, POINTER, byref
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum, auto

from game_engine import InputState, Fighter


# =============================================================================
# X11 KEYBOARD QUERY
# =============================================================================

_x11 = ctypes.CDLL("libX11.so.6")

Display = c_void_p
Window = c_ulong
KeySym = c_ulong


class KeyboardStateArray(ctypes.Structure):
    """32-byte array for keyboard state."""
    _fields_ = [("data", ctypes.c_char * 32)]


_x11.XOpenDisplay.argtypes = [c_char_p]
_x11.XOpenDisplay.restype = Display

_x11.XCloseDisplay.argtypes = [Display]
_x11.XCloseDisplay.restype = c_int

_x11.XQueryKeymap.argtypes = [Display, POINTER(KeyboardStateArray)]
_x11.XQueryKeymap.restype = c_int

_x11.XKeysymToKeycode.argtypes = [Display, KeySym]
_x11.XKeysymToKeycode.restype = ctypes.c_ubyte

_x11.XDefaultRootWindow.argtypes = [Display]
_x11.XDefaultRootWindow.restype = Window


# =============================================================================
# KEY DEFINITIONS
# =============================================================================

# X11 KeySyms for common keys
class KeySym:
    """X11 KeySym constants."""
    # Arrow keys
    Left = 0xff51
    Up = 0xff52
    Right = 0xff53
    Down = 0xff54

    # Function keys
    Escape = 0xff1b
    Return = 0xff0d
    Tab = 0xff09
    BackSpace = 0xff08
    Delete = 0xffff
    Space = 0x0020

    # Modifiers
    Shift_L = 0xffe1
    Shift_R = 0xffe2
    Control_L = 0xffe3
    Control_R = 0xffe4
    Alt_L = 0xffe9
    Alt_R = 0xffea

    # Letters (lowercase)
    a = 0x0061
    b = 0x0062
    c = 0x0063
    d = 0x0064
    e = 0x0065
    f = 0x0066
    g = 0x0067
    h = 0x0068
    i = 0x0069
    j = 0x006a
    k = 0x006b
    l = 0x006c
    m = 0x006d
    n = 0x006e
    o = 0x006f
    p = 0x0070
    q = 0x0071
    r = 0x0072
    s = 0x0073
    t = 0x0074
    u = 0x0075
    v = 0x0076
    w = 0x0077
    x = 0x0078
    y = 0x0079
    z = 0x007a

    # Numbers
    n1 = 0x0031
    n2 = 0x0032
    n3 = 0x0033
    n4 = 0x0034
    n5 = 0x0035
    n6 = 0x0036
    n7 = 0x0037
    n8 = 0x0038
    n9 = 0x0039
    n0 = 0x0030

    # Numpad
    KP_0 = 0xffb0
    KP_1 = 0xffb1
    KP_2 = 0xffb2
    KP_3 = 0xffb3
    KP_4 = 0xffb4
    KP_5 = 0xffb5
    KP_6 = 0xffb6
    KP_7 = 0xffb7
    KP_8 = 0xffb8
    KP_9 = 0xffb9
    KP_Enter = 0xff8d

    # Function keys (for training mode)
    F1 = 0xffbe
    F2 = 0xffbf
    F3 = 0xffc0
    F4 = 0xffc1
    F5 = 0xffc2


# =============================================================================
# INPUT ACTION ENUM
# =============================================================================

class InputAction(Enum):
    """Abstract input actions that can be mapped to keys."""
    LEFT = auto()
    RIGHT = auto()
    UP = auto()
    DOWN = auto()
    JUMP = auto()
    ATTACK = auto()
    SPECIAL = auto()
    SHIELD = auto()
    GRAB = auto()


# =============================================================================
# KEY BINDING PRESETS
# =============================================================================

@dataclass
class KeyBindings:
    """Key bindings for a player."""
    left: int
    right: int
    up: int
    down: int
    jump: int
    attack: int
    special: int
    shield: int
    grab: int

    def to_dict(self) -> Dict[int, str]:
        """Convert to keysym -> action name dict."""
        return {
            self.left: 'left',
            self.right: 'right',
            self.up: 'up',
            self.down: 'down',
            self.jump: 'jump',
            self.attack: 'attack',
            self.special: 'special',
            self.shield: 'shield',
            self.grab: 'grab',
        }


# Default bindings
BINDINGS_WASD = KeyBindings(
    left=KeySym.a,
    right=KeySym.d,
    up=KeySym.w,
    down=KeySym.s,
    jump=KeySym.Space,
    attack=KeySym.j,
    special=KeySym.k,
    shield=KeySym.l,
    grab=KeySym.u,
)

BINDINGS_ARROWS = KeyBindings(
    left=KeySym.Left,
    right=KeySym.Right,
    up=KeySym.Up,
    down=KeySym.Down,
    jump=KeySym.Return,
    attack=KeySym.KP_1,
    special=KeySym.KP_2,
    shield=KeySym.KP_3,
    grab=KeySym.KP_0,
)

BINDINGS_IJKL = KeyBindings(
    left=KeySym.j,
    right=KeySym.l,
    up=KeySym.i,
    down=KeySym.k,
    jump=KeySym.n,
    attack=KeySym.m,
    special=KeySym.n8,
    shield=KeySym.n9,
    grab=KeySym.n0,
)


# =============================================================================
# KEYBOARD INPUT HANDLER
# =============================================================================

class KeyboardInput:
    """
    Handles keyboard input polling via X11.

    Uses XQueryKeymap for low-latency key state polling.
    """

    def __init__(self, display_name: str = None):
        """
        Initialize keyboard input.

        Args:
            display_name: X11 display (e.g., ':99'), None for default
        """
        display_bytes = display_name.encode() if display_name else None
        self.display = _x11.XOpenDisplay(display_bytes)
        if not self.display:
            raise RuntimeError(f"Cannot open X11 display: {display_name or 'default'}")

        # Player key bindings
        self.player_bindings: List[KeyBindings] = [
            BINDINGS_WASD,
            BINDINGS_ARROWS,
        ]

        # Cache keycode lookups
        self._keysym_to_keycode: Dict[int, int] = {}
        self._cache_keycodes()

        # Keyboard state buffer
        self._keymap = KeyboardStateArray()

    def _cache_keycodes(self):
        """Pre-cache keysym to keycode mappings."""
        keysyms_to_cache = [
            KeySym.Left, KeySym.Right, KeySym.Up, KeySym.Down,
            KeySym.Space, KeySym.Return, KeySym.Escape, KeySym.Tab,
            KeySym.Shift_L, KeySym.Shift_R, KeySym.Control_L, KeySym.Control_R,
            KeySym.a, KeySym.b, KeySym.c, KeySym.d, KeySym.e, KeySym.f,
            KeySym.g, KeySym.h, KeySym.i, KeySym.j, KeySym.k, KeySym.l,
            KeySym.m, KeySym.n, KeySym.o, KeySym.p, KeySym.q, KeySym.r,
            KeySym.s, KeySym.t, KeySym.u, KeySym.v, KeySym.w, KeySym.x,
            KeySym.y, KeySym.z,
            KeySym.n0, KeySym.n1, KeySym.n2, KeySym.n3, KeySym.n4,
            KeySym.n5, KeySym.n6, KeySym.n7, KeySym.n8, KeySym.n9,
            KeySym.KP_0, KeySym.KP_1, KeySym.KP_2, KeySym.KP_3, KeySym.KP_4,
            KeySym.KP_5, KeySym.KP_6, KeySym.KP_7, KeySym.KP_8, KeySym.KP_9,
            KeySym.KP_Enter,
            # Function keys for training mode (Cycle 5)
            KeySym.F1, KeySym.F2, KeySym.F3, KeySym.F4, KeySym.F5,
        ]

        for keysym in keysyms_to_cache:
            keycode = _x11.XKeysymToKeycode(self.display, keysym)
            if keycode:
                self._keysym_to_keycode[keysym] = keycode

    def poll(self):
        """Poll current keyboard state."""
        _x11.XQueryKeymap(self.display, byref(self._keymap))

    def is_key_pressed(self, keysym: int) -> bool:
        """Check if a key (by keysym) is currently pressed."""
        keycode = self._keysym_to_keycode.get(keysym)
        if keycode is None:
            # Try to get it now
            keycode = _x11.XKeysymToKeycode(self.display, keysym)
            if keycode:
                self._keysym_to_keycode[keysym] = keycode
            else:
                return False

        # Check bit in keymap
        byte_idx = keycode // 8
        bit_idx = keycode % 8

        # Handle both bytes (Python 3) and ctypes array indexing
        try:
            byte_val = self._keymap.data[byte_idx]
            if isinstance(byte_val, bytes):
                byte_val = byte_val[0] if byte_val else 0
            elif isinstance(byte_val, str):
                byte_val = ord(byte_val)
            return bool(byte_val & (1 << bit_idx))
        except (IndexError, TypeError):
            return False

    def get_input_state(self, player_id: int) -> InputState:
        """
        Get the current input state for a player.

        Args:
            player_id: Player index (0 or 1)

        Returns:
            InputState with current button states
        """
        if player_id >= len(self.player_bindings):
            return InputState()

        bindings = self.player_bindings[player_id]
        state = InputState()

        state.left = self.is_key_pressed(bindings.left)
        state.right = self.is_key_pressed(bindings.right)
        state.up = self.is_key_pressed(bindings.up)
        state.down = self.is_key_pressed(bindings.down)
        state.jump = self.is_key_pressed(bindings.jump)
        state.attack = self.is_key_pressed(bindings.attack)
        state.special = self.is_key_pressed(bindings.special)
        state.shield = self.is_key_pressed(bindings.shield)
        state.grab = self.is_key_pressed(bindings.grab)

        return state

    def update_fighter_input(self, fighter: Fighter):
        """
        Update a fighter's input state from keyboard.

        Preserves the previous state for press detection.
        """
        new_state = self.get_input_state(fighter.player_id)

        # Copy previous state
        new_state._prev_jump = fighter.input.jump
        new_state._prev_attack = fighter.input.attack
        new_state._prev_special = fighter.input.special
        new_state._prev_shield = fighter.input.shield
        new_state._prev_grab = fighter.input.grab

        fighter.input = new_state

    def set_player_bindings(self, player_id: int, bindings: KeyBindings):
        """Set custom key bindings for a player."""
        while len(self.player_bindings) <= player_id:
            self.player_bindings.append(KeyBindings(
                left=0, right=0, up=0, down=0,
                jump=0, attack=0, special=0, shield=0, grab=0
            ))
        self.player_bindings[player_id] = bindings
        self._cache_keycodes()

    def close(self):
        """Clean up X11 resources."""
        if self.display:
            _x11.XCloseDisplay(self.display)
            self.display = None

    def __del__(self):
        try:
            self.close()
        except:
            pass


# =============================================================================
# INPUT BUFFER (for fighting game input techniques)
# =============================================================================

@dataclass
class BufferedInput:
    """A buffered input with timing."""
    action: str
    frame: int
    consumed: bool = False


class InputBuffer:
    """
    Input buffer for fighting game techniques.

    Stores recent inputs for combo detection and input leniency.
    """

    def __init__(self, buffer_frames: int = 8):
        """
        Initialize input buffer.

        Args:
            buffer_frames: Number of frames to buffer inputs
        """
        self.buffer_frames = buffer_frames
        self.buffer: List[BufferedInput] = []
        self.current_frame = 0

    def add_input(self, action: str):
        """Add an input to the buffer."""
        self.buffer.append(BufferedInput(action=action, frame=self.current_frame))

    def update(self, current_frame: int):
        """Update buffer, removing old inputs."""
        self.current_frame = current_frame
        cutoff = current_frame - self.buffer_frames

        # Remove old inputs
        self.buffer = [b for b in self.buffer
                       if b.frame >= cutoff and not b.consumed]

    def has_buffered(self, action: str) -> bool:
        """Check if an action is in the buffer (unconsumed)."""
        for inp in self.buffer:
            if inp.action == action and not inp.consumed:
                return True
        return False

    def consume(self, action: str) -> bool:
        """
        Consume a buffered input.

        Returns True if the input was found and consumed.
        """
        for inp in self.buffer:
            if inp.action == action and not inp.consumed:
                inp.consumed = True
                return True
        return False

    def clear(self):
        """Clear the buffer."""
        self.buffer = []


# =============================================================================
# INPUT MANAGER
# =============================================================================

class InputManager:
    """
    High-level input manager for the game.

    Handles keyboard polling, gamepad support, input buffering,
    and distributing inputs to fighters.
    """

    def __init__(self, display_name: str = None, use_buffer: bool = True,
                 enable_gamepad: bool = True):
        """
        Initialize input manager.

        Args:
            display_name: X11 display name
            use_buffer: Whether to use input buffering
            enable_gamepad: Whether to enable gamepad support
        """
        self.keyboard = KeyboardInput(display_name)
        self.use_buffer = use_buffer

        # Gamepad support
        self.gamepad_manager = None
        self.gamepad_enabled = enable_gamepad
        self.player_uses_gamepad: Dict[int, bool] = {0: False, 1: False}

        if enable_gamepad:
            self._init_gamepad()

        # Input buffers per player
        self.buffers: List[InputBuffer] = [
            InputBuffer(),
            InputBuffer(),
        ]

        # System keys
        self.quit_requested = False
        self.pause_requested = False
        self.debug_toggle = False

    def _init_gamepad(self):
        """Initialize gamepad manager if available."""
        try:
            from gamepad_input import GamepadManager
            self.gamepad_manager = GamepadManager()
            count = self.gamepad_manager.count
            if count > 0:
                print(f"[INPUT] {count} gamepad(s) detected")
                # By default, assign gamepad 0 to player 0, etc.
                for i in range(min(count, 2)):
                    self.player_uses_gamepad[i] = True
        except Exception as e:
            print(f"[INPUT] Gamepad init failed: {e}")
            self.gamepad_manager = None

    def poll(self):
        """Poll keyboard and gamepad state."""
        self.keyboard.poll()

        # Poll gamepads
        if self.gamepad_manager:
            self.gamepad_manager.poll()

        # Check system keys (keyboard only)
        self.quit_requested = self.keyboard.is_key_pressed(KeySym.Escape)
        self.pause_requested = self.keyboard.is_key_pressed(KeySym.p)
        self.debug_toggle = self.keyboard.is_key_pressed(KeySym.Tab)

        # Also allow gamepad start button to pause
        if self.gamepad_manager:
            gp = self.gamepad_manager.get_gamepad(0)
            if gp and gp.button_just_pressed('start'):
                self.pause_requested = True

    def update_fighter(self, fighter: Fighter, current_frame: int):
        """
        Update a fighter's input from keyboard or gamepad.

        Args:
            fighter: The fighter to update
            current_frame: Current game frame for buffering
        """
        pid = fighter.player_id

        # Get keyboard input first
        self.keyboard.update_fighter_input(fighter)

        # Merge gamepad input if player uses gamepad
        if self.gamepad_manager and self.player_uses_gamepad.get(pid, False):
            gp_state = self.gamepad_manager.get_input_state(pid)
            if gp_state:
                # Merge: OR the inputs (either keyboard or gamepad)
                fighter.input.left = fighter.input.left or gp_state.left
                fighter.input.right = fighter.input.right or gp_state.right
                fighter.input.up = fighter.input.up or gp_state.up
                fighter.input.down = fighter.input.down or gp_state.down
                fighter.input.jump = fighter.input.jump or gp_state.jump
                fighter.input.attack = fighter.input.attack or gp_state.attack
                fighter.input.special = fighter.input.special or gp_state.special
                fighter.input.shield = fighter.input.shield or gp_state.shield
                fighter.input.grab = fighter.input.grab or gp_state.grab

        if self.use_buffer and pid < len(self.buffers):
            buffer = self.buffers[pid]
            buffer.update(current_frame)

            # Add newly pressed inputs to buffer
            if fighter.input.jump_pressed:
                buffer.add_input('jump')
            if fighter.input.attack_pressed:
                buffer.add_input('attack')
            if fighter.input.special_pressed:
                buffer.add_input('special')
            if fighter.input.shield_pressed:
                buffer.add_input('shield')
            if fighter.input.grab_pressed:
                buffer.add_input('grab')

    def get_buffered_action(self, player_id: int, action: str) -> bool:
        """
        Check if an action is buffered for a player.

        Args:
            player_id: Player index
            action: Action name ('jump', 'attack', etc.)

        Returns:
            True if the action is buffered
        """
        if not self.use_buffer or player_id >= len(self.buffers):
            return False
        return self.buffers[player_id].has_buffered(action)

    def consume_buffered(self, player_id: int, action: str) -> bool:
        """
        Consume a buffered action.

        Args:
            player_id: Player index
            action: Action name

        Returns:
            True if the action was consumed
        """
        if not self.use_buffer or player_id >= len(self.buffers):
            return False
        return self.buffers[player_id].consume(action)

    def rescan_gamepads(self):
        """Rescan for new/removed gamepads."""
        if self.gamepad_manager:
            self.gamepad_manager.rescan()
            count = self.gamepad_manager.count
            if count > 0:
                for i in range(min(count, 2)):
                    self.player_uses_gamepad[i] = True
            print(f"[INPUT] Rescan: {count} gamepad(s)")

    def set_player_gamepad(self, player_id: int, use_gamepad: bool):
        """Set whether a player uses gamepad or keyboard."""
        self.player_uses_gamepad[player_id] = use_gamepad

    def get_gamepad_names(self) -> List[str]:
        """Get list of connected gamepad names."""
        if self.gamepad_manager:
            return self.gamepad_manager.list_gamepads()
        return []

    def close(self):
        """Clean up resources."""
        self.keyboard.close()
        if self.gamepad_manager:
            self.gamepad_manager.close()

    def __del__(self):
        try:
            self.close()
        except:
            pass


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_input_manager(display_name: str = None) -> InputManager:
    """Create an input manager with default settings."""
    return InputManager(display_name, use_buffer=True)


def print_controls():
    """Print the default control scheme."""
    print("\n=== STICK FIGURE FIGHTER CONTROLS ===\n")
    print("PLAYER 1 (WASD):")
    print("  Move:    W/A/S/D")
    print("  Jump:    Space")
    print("  Attack:  J")
    print("  Special: K")
    print("  Shield:  L")
    print("  Grab:    U")
    print()
    print("PLAYER 2 (Arrows):")
    print("  Move:    Arrow Keys")
    print("  Jump:    Enter")
    print("  Attack:  Numpad 1")
    print("  Special: Numpad 2")
    print("  Shield:  Numpad 3")
    print("  Grab:    Numpad 0")
    print()
    print("GAMEPAD (Xbox-style):")
    print("  Move:    D-Pad or Left Stick")
    print("  Jump:    A")
    print("  Attack:  X")
    print("  Special: B")
    print("  Shield:  RB")
    print("  Grab:    LB")
    print("  Pause:   Start")
    print()
    print("SYSTEM:")
    print("  Quit:    Escape")
    print("  Pause:   P / Start")
    print("  Debug:   Tab")
    print()
    print("TRAINING MODE:")
    print("  F1: Toggle CPU opponent")
    print("  F2: Cycle CPU behavior")
    print("  F3: Toggle frame data")
    print("  F4: Cycle CPU difficulty (1-5)")
    print("  F5: Toggle infinite jumps")
    print("  R:  Reset damage to 0%")
    print("  T:  Reset positions")
    print()


if __name__ == "__main__":
    # Test keyboard input
    import time

    print("Testing keyboard input on display :99")
    print("Press keys to see them detected (Escape to quit)")
    print_controls()

    try:
        kb = KeyboardInput(":99")

        while True:
            kb.poll()

            # Check escape
            if kb.is_key_pressed(KeySym.Escape):
                print("Escape pressed - quitting")
                break

            # Check P1 inputs
            p1 = kb.get_input_state(0)
            p2 = kb.get_input_state(1)

            # Print if any input
            p1_active = any([p1.left, p1.right, p1.up, p1.down, p1.jump,
                            p1.attack, p1.special, p1.shield, p1.grab])
            p2_active = any([p2.left, p2.right, p2.up, p2.down, p2.jump,
                            p2.attack, p2.special, p2.shield, p2.grab])

            if p1_active:
                print(f"P1: L={p1.left} R={p1.right} U={p1.up} D={p1.down} "
                      f"J={p1.jump} A={p1.attack} Sp={p1.special} Sh={p1.shield} G={p1.grab}")

            if p2_active:
                print(f"P2: L={p2.left} R={p2.right} U={p2.up} D={p2.down} "
                      f"J={p2.jump} A={p2.attack} Sp={p2.special} Sh={p2.shield} G={p2.grab}")

            time.sleep(0.05)  # 20 Hz polling for test

        kb.close()

    except Exception as e:
        print(f"Error: {e}")
        print("Make sure DISPLAY=:99 and Xvfb is running")
