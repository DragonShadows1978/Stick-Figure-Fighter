"""
Stick Figure Fighter - X11 Rendering System

Renders stick figures and game state using direct X11 calls.
No external game frameworks - pure X11 via ctypes.

Cycle 3: Visual Effects & Game Feel Polish
- Hit sparks, screen shake, damage flash
- Projectile trails
- Shield visualization, dodge afterimages
- Counter/reflector effects, super armor glow
- HUD improvements

Cycle 5: Training Mode Visualization
- Frame data overlay
- CPU behavior indicator
"""
import ctypes
from ctypes import (
    c_void_p, c_int, c_uint, c_ulong, c_char_p, c_long, c_short,
    POINTER, Structure, byref, cast
)
import math
import os
import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass, field
from enum import Enum, auto

from game_engine import (
    GameState, Fighter, Platform, FighterState, Projectile,
    CONFIG, Vector2, AABB, DodgeType, ParallaxLayer
)
from poses import (  # noqa: F401  (re-exported for older imports)
    POSE_ATTACK,
    POSE_ATTACK_WINDUP,
    POSE_COUNTER,
    POSE_DODGING,
    POSE_ENTERING,
    POSE_FALLING,
    POSE_HITSTUN,
    POSE_IDLE,
    POSE_JUMPING,
    POSE_JUMP_CROUCH,
    POSE_JUMP_STRETCH,
    POSE_LANDING_SQUASH,
    POSE_REFLECTING,
    POSE_SHIELDING,
    POSE_SHIELD_STUN,
    POSE_SPECIAL,
    POSE_WALKING,
    StickFigurePose,
    get_pose_for_state,
)


# =============================================================================
# VISUAL EFFECTS SYSTEM (Cycle 3)
# =============================================================================

class EffectType(Enum):
    """Types of visual effects."""
    HIT_SPARK = auto()       # Radiating lines on attack connect
    DAMAGE_FLASH = auto()    # White flash on damaged fighter
    SHIELD_CRACK = auto()    # Crack effect on low shield
    DODGE_AFTERIMAGE = auto()  # Ghost trail during dodge
    COUNTER_AURA = auto()    # Defensive aura during counter stance
    REFLECTOR_HEX = auto()   # Hexagonal flash on reflect
    SUPER_ARMOR_GLOW = auto()  # Golden outline during super armor
    LANDING_DUST = auto()    # Dust puff on landing
    KNOCKBACK_TRAIL = auto() # Speed lines during knockback
    FRAME_ADVANTAGE = auto() # Cycle 13: Frame advantage number display


@dataclass
class VisualEffect:
    """A temporary visual effect in the game world."""
    effect_type: EffectType
    x: float
    y: float
    duration: int          # Total frames
    frame: int = 0         # Current frame
    active: bool = True
    # Effect-specific parameters
    intensity: float = 1.0  # For scaling effects
    angle: float = 0.0      # Direction (for directional effects like sparks)
    color: Tuple[int, int, int] = (255, 255, 255)
    secondary_color: Tuple[int, int, int] = (255, 200, 100)

    def update(self):
        """Advance effect by one frame."""
        self.frame += 1
        if self.frame >= self.duration:
            self.active = False

    @property
    def progress(self) -> float:
        """Effect progress from 0.0 to 1.0."""
        return self.frame / max(1, self.duration)

    @property
    def alpha(self) -> float:
        """Fade-out alpha based on progress."""
        return 1.0 - self.progress


@dataclass
class ScreenShake:
    """
    Screen shake effect with decay.

    Cycle 14: Enhanced with stronger directional bias on powerful hits.
    """
    magnitude: float = 0.0   # Current shake intensity
    duration: int = 0        # Frames remaining
    decay: float = 0.85      # Decay rate per frame (lower = faster decay)
    direction_x: float = 0.0  # Directional bias
    direction_y: float = 0.0

    def apply(self, knockback_magnitude: float, direction_x: float = 0.0, direction_y: float = 0.0):
        """Apply shake based on knockback."""
        # Scale shake to knockback (capped)
        new_magnitude = min(knockback_magnitude * 0.8, 15.0)
        if new_magnitude > self.magnitude:
            self.magnitude = new_magnitude
            self.duration = int(min(knockback_magnitude * 0.5, 12))

            # Cycle 14: Stronger directional bias on powerful hits (knockback > 15)
            if knockback_magnitude > 15:
                # Powerful hit - stronger directional shake
                self.direction_x = direction_x * 0.6
                self.direction_y = direction_y * 0.6
                # Extend duration for dramatic effect
                self.duration = min(self.duration + 3, 15)
            else:
                self.direction_x = direction_x * 0.3
                self.direction_y = direction_y * 0.3

    def update(self) -> Tuple[int, int]:
        """Update shake and return offset (x, y)."""
        if self.magnitude < 0.5 or self.duration <= 0:
            self.magnitude = 0
            return (0, 0)

        # Random shake with directional bias
        offset_x = (np.random.random() * 2 - 1) * self.magnitude + self.direction_x
        offset_y = (np.random.random() * 2 - 1) * self.magnitude + self.direction_y

        # Decay
        self.magnitude *= self.decay
        self.direction_x *= self.decay
        self.direction_y *= self.decay
        self.duration -= 1

        return (int(offset_x), int(offset_y))


@dataclass
class TrailPoint:
    """A point in a trail history."""
    x: float
    y: float
    alpha: float = 1.0


# =============================================================================
# X11 BINDINGS
# =============================================================================

# Load X11 library
_x11 = ctypes.CDLL("libX11.so.6")

# Types
Display = c_void_p
Window = c_ulong
Drawable = c_ulong
GC = c_void_p
Colormap = c_ulong
Pixmap = c_ulong
XID = c_ulong
Atom = c_ulong
Bool = c_int
Status = c_int
KeySym = c_ulong
KeyCode = c_uint


class XGCValues(Structure):
    _fields_ = [
        ("function", c_int),
        ("plane_mask", c_ulong),
        ("foreground", c_ulong),
        ("background", c_ulong),
        ("line_width", c_int),
        ("line_style", c_int),
        ("cap_style", c_int),
        ("join_style", c_int),
        ("fill_style", c_int),
        ("fill_rule", c_int),
        ("arc_mode", c_int),
        ("tile", Pixmap),
        ("stipple", Pixmap),
        ("ts_x_origin", c_int),
        ("ts_y_origin", c_int),
        ("font", XID),
        ("subwindow_mode", c_int),
        ("graphics_exposures", Bool),
        ("clip_x_origin", c_int),
        ("clip_y_origin", c_int),
        ("clip_mask", Pixmap),
        ("dash_offset", c_int),
        ("dashes", c_int),
    ]


class XWindowAttributes(Structure):
    _fields_ = [
        ("x", c_int),
        ("y", c_int),
        ("width", c_int),
        ("height", c_int),
        ("border_width", c_int),
        ("depth", c_int),
        ("visual", c_void_p),
        ("root", Window),
        ("class_", c_int),
        ("bit_gravity", c_int),
        ("win_gravity", c_int),
        ("backing_store", c_int),
        ("backing_planes", c_ulong),
        ("backing_pixel", c_ulong),
        ("save_under", Bool),
        ("colormap", Colormap),
        ("map_installed", Bool),
        ("map_state", c_int),
        ("all_event_masks", c_long),
        ("your_event_mask", c_long),
        ("do_not_propagate_mask", c_long),
        ("override_redirect", Bool),
        ("screen", c_void_p),
    ]


class XEvent(Structure):
    _fields_ = [("type", c_int), ("pad", c_long * 24)]


class XKeyEvent(Structure):
    _fields_ = [
        ("type", c_int),
        ("serial", c_ulong),
        ("send_event", Bool),
        ("display", Display),
        ("window", Window),
        ("root", Window),
        ("subwindow", Window),
        ("time", c_ulong),
        ("x", c_int),
        ("y", c_int),
        ("x_root", c_int),
        ("y_root", c_int),
        ("state", c_uint),
        ("keycode", c_uint),
        ("same_screen", Bool),
    ]


# X11 function declarations
_x11.XOpenDisplay.argtypes = [c_char_p]
_x11.XOpenDisplay.restype = Display

_x11.XCloseDisplay.argtypes = [Display]
_x11.XCloseDisplay.restype = c_int

_x11.XDefaultScreen.argtypes = [Display]
_x11.XDefaultScreen.restype = c_int

_x11.XRootWindow.argtypes = [Display, c_int]
_x11.XRootWindow.restype = Window

_x11.XDefaultColormap.argtypes = [Display, c_int]
_x11.XDefaultColormap.restype = Colormap

_x11.XCreateSimpleWindow.argtypes = [
    Display, Window, c_int, c_int, c_uint, c_uint, c_uint, c_ulong, c_ulong
]
_x11.XCreateSimpleWindow.restype = Window

_x11.XMapWindow.argtypes = [Display, Window]
_x11.XMapWindow.restype = c_int

_x11.XUnmapWindow.argtypes = [Display, Window]
_x11.XUnmapWindow.restype = c_int

_x11.XDestroyWindow.argtypes = [Display, Window]
_x11.XDestroyWindow.restype = c_int

_x11.XSelectInput.argtypes = [Display, Window, c_long]
_x11.XSelectInput.restype = c_int

_x11.XCreateGC.argtypes = [Display, Drawable, c_ulong, POINTER(XGCValues)]
_x11.XCreateGC.restype = GC

_x11.XFreeGC.argtypes = [Display, GC]
_x11.XFreeGC.restype = c_int

_x11.XSetForeground.argtypes = [Display, GC, c_ulong]
_x11.XSetForeground.restype = c_int

_x11.XSetBackground.argtypes = [Display, GC, c_ulong]
_x11.XSetBackground.restype = c_int

_x11.XSetLineAttributes.argtypes = [Display, GC, c_uint, c_int, c_int, c_int]
_x11.XSetLineAttributes.restype = c_int

_x11.XDrawLine.argtypes = [Display, Drawable, GC, c_int, c_int, c_int, c_int]
_x11.XDrawLine.restype = c_int

_x11.XDrawArc.argtypes = [
    Display, Drawable, GC, c_int, c_int, c_uint, c_uint, c_int, c_int
]
_x11.XDrawArc.restype = c_int

_x11.XFillArc.argtypes = [
    Display, Drawable, GC, c_int, c_int, c_uint, c_uint, c_int, c_int
]
_x11.XFillArc.restype = c_int

_x11.XFillRectangle.argtypes = [
    Display, Drawable, GC, c_int, c_int, c_uint, c_uint
]
_x11.XFillRectangle.restype = c_int

_x11.XDrawRectangle.argtypes = [
    Display, Drawable, GC, c_int, c_int, c_uint, c_uint
]
_x11.XDrawRectangle.restype = c_int

_x11.XDrawString.argtypes = [Display, Drawable, GC, c_int, c_int, c_char_p, c_int]
_x11.XDrawString.restype = c_int

_x11.XClearWindow.argtypes = [Display, Window]
_x11.XClearWindow.restype = c_int

_x11.XFlush.argtypes = [Display]
_x11.XFlush.restype = c_int

_x11.XSync.argtypes = [Display, Bool]
_x11.XSync.restype = c_int

_x11.XCreatePixmap.argtypes = [Display, Drawable, c_uint, c_uint, c_uint]
_x11.XCreatePixmap.restype = Pixmap

_x11.XFreePixmap.argtypes = [Display, Pixmap]
_x11.XFreePixmap.restype = c_int

_x11.XCopyArea.argtypes = [
    Display, Drawable, Drawable, GC, c_int, c_int, c_uint, c_uint, c_int, c_int
]
_x11.XCopyArea.restype = c_int

_x11.XPending.argtypes = [Display]
_x11.XPending.restype = c_int

_x11.XNextEvent.argtypes = [Display, POINTER(XEvent)]
_x11.XNextEvent.restype = c_int

_x11.XKeycodeToKeysym.argtypes = [Display, KeyCode, c_int]
_x11.XKeycodeToKeysym.restype = KeySym

_x11.XLookupString.argtypes = [
    POINTER(XKeyEvent), c_char_p, c_int, POINTER(KeySym), c_void_p
]
_x11.XLookupString.restype = c_int

_x11.XGetWindowAttributes.argtypes = [Display, Window, POINTER(XWindowAttributes)]
_x11.XGetWindowAttributes.restype = Status

_x11.XDefaultDepth.argtypes = [Display, c_int]
_x11.XDefaultDepth.restype = c_int

_x11.XBlackPixel.argtypes = [Display, c_int]
_x11.XBlackPixel.restype = c_ulong

_x11.XWhitePixel.argtypes = [Display, c_int]
_x11.XWhitePixel.restype = c_ulong

_x11.XAllocColor.argtypes = [Display, Colormap, c_void_p]
_x11.XAllocColor.restype = Status

_x11.XStoreName.argtypes = [Display, Window, c_char_p]
_x11.XStoreName.restype = c_int

_x11.XSetWMProtocols.argtypes = [Display, Window, POINTER(Atom), c_int]
_x11.XSetWMProtocols.restype = Status

_x11.XInternAtom.argtypes = [Display, c_char_p, Bool]
_x11.XInternAtom.restype = Atom


class XColor(Structure):
    _fields_ = [
        ("pixel", c_ulong),
        ("red", c_short),
        ("green", c_short),
        ("blue", c_short),
        ("flags", c_int),
        ("pad", c_int),
    ]


# Event masks
KeyPressMask = (1 << 0)
KeyReleaseMask = (1 << 1)
ExposureMask = (1 << 15)
StructureNotifyMask = (1 << 17)

# Event types
KeyPress = 2
KeyRelease = 3
Expose = 12
DestroyNotify = 17
ClientMessage = 33


# =============================================================================
# COLOR UTILITIES
# =============================================================================

_pixel_cache = {}

_NAMED_COLORS = {
    'black': (0, 0, 0),
    'white': (255, 255, 255),
    'red': (255, 80, 80),
    'blue': (80, 80, 255),
    'green': (80, 255, 80),
    'yellow': (255, 255, 80),
    'gray': (128, 128, 128),
    'dark_gray': (64, 64, 64),
    'platform': (100, 80, 60),
    'platform_pass': (80, 100, 80),
    'hitbox': (255, 0, 0),
    'hurtbox': (0, 255, 0),
    'p1': (255, 100, 100),
    'p2': (100, 100, 255),
    'damage_bar': (200, 50, 50),
    'stock': (255, 200, 50),
    'spark_white': (255, 255, 255),
    'spark_yellow': (255, 230, 100),
    'spark_orange': (255, 180, 80),
    'shield_green': (100, 200, 100),
    'shield_yellow': (200, 200, 100),
    'shield_red': (200, 100, 100),
    'counter_blue': (100, 150, 255),
    'reflector_cyan': (100, 255, 255),
    'armor_gold': (255, 200, 50),
    'trail_cyan': (80, 180, 255),
    'trail_orange': (255, 120, 50),
}


def _saved_theme() -> str:
    try:
        from settings import get_settings
        return get_settings().gameplay.theme
    except Exception:
        return 'neon'


def rgb_to_pixel(display: Display, screen: int, colormap: Colormap,
                 r: int, g: int, b: int) -> int:
    """Convert RGB values to X11 pixel value (cached: XAllocColor is a server round trip)."""
    key = (r, g, b)
    pixel = _pixel_cache.get(key)
    if pixel is not None:
        return pixel
    _pixel_cache[key] = pixel = _alloc_pixel(display, colormap, r, g, b)
    return pixel


def _alloc_pixel(display, colormap, r, g, b) -> int:
    color = XColor()
    color.red = r * 257  # Scale 0-255 to 0-65535
    color.green = g * 257
    color.blue = b * 257
    color.flags = 7  # DoRed | DoGreen | DoBlue

    status = _x11.XAllocColor(display, colormap, byref(color))
    if status:
        return color.pixel
    else:
        # Fallback: approximate using default visual
        return (r << 16) | (g << 8) | b


# =============================================================================
# STICK FIGURE RENDERER
# =============================================================================

class X11Renderer:
    """X11-based renderer for the game."""

    def __init__(self, width: int = None, height: int = None,
                 display_name: str = None, title: str = "Stick Figure Fighter",
                 theme: str = None):
        self.width = width or CONFIG.screen_width
        self.height = height or CONFIG.screen_height

        # Open display
        display_bytes = display_name.encode() if display_name else None
        self.display = _x11.XOpenDisplay(display_bytes)
        if not self.display:
            raise RuntimeError(f"Cannot open display: {display_name or 'default'}")

        self.screen = _x11.XDefaultScreen(self.display)
        self.root = _x11.XRootWindow(self.display, self.screen)
        self.colormap = _x11.XDefaultColormap(self.display, self.screen)
        self.depth = _x11.XDefaultDepth(self.display, self.screen)

        self.black = _x11.XBlackPixel(self.display, self.screen)
        self.white = _x11.XWhitePixel(self.display, self.screen)

        # Create window
        self.window = _x11.XCreateSimpleWindow(
            self.display, self.root,
            100, 100,  # Position
            self.width, self.height,
            2,  # Border width
            self.black,  # Border color
            self.black   # Background color
        )

        # Set window title
        _x11.XStoreName(self.display, self.window, title.encode())

        # Handle window close
        self.wm_delete = _x11.XInternAtom(self.display, b"WM_DELETE_WINDOW", False)
        wm_protocols = (Atom * 1)(self.wm_delete)
        _x11.XSetWMProtocols(self.display, self.window, wm_protocols, 1)

        # Select input events
        event_mask = KeyPressMask | KeyReleaseMask | ExposureMask | StructureNotifyMask
        _x11.XSelectInput(self.display, self.window, event_mask)

        # Create graphics context
        self.gc = _x11.XCreateGC(self.display, self.window, 0, None)
        _x11.XSetLineAttributes(self.display, self.gc, 3, 0, 1, 1)  # Line width 3

        # Create double buffer
        self.buffer = _x11.XCreatePixmap(
            self.display, self.window,
            self.width, self.height, self.depth
        )

        # Pre-allocate colors
        self.colors = {}
        self._allocate_colors()

        # XRender drawing context + visual theme (None/classic without XRender)
        self.gfx = None
        try:
            from gfx.context import Gfx
            self.gfx = Gfx.create(self.display, self.screen, self.buffer, self.width, self.height)
        except Exception as e:  # missing libXrender etc.
            print(f"[Renderer] XRender unavailable, using classic look ({e})")
        self.frame = 0
        self._rgb = (255, 255, 255)
        self.theme = None
        self.set_theme(theme or os.environ.get("SFF_THEME") or _saved_theme())

        # Map window
        _x11.XMapWindow(self.display, self.window)
        _x11.XFlush(self.display)

        # Window close flag
        self.should_close = False

        # Debug mode
        self.show_hitboxes = False
        self.show_hurtboxes = False

        # Cycle 3: Visual effects system
        self.effects: List[VisualEffect] = []
        self.screen_shake = ScreenShake()
        self.shake_offset = (0, 0)

        # Trail storage for projectiles (keyed by id(projectile))
        self.projectile_trails: dict = {}

        # Afterimage storage for fighters during dodge
        self.fighter_afterimages: dict = {}  # player_id -> list of (x, y, alpha, frame_data)

        # Knockback trail storage for fighters in hitstun
        self.knockback_trails: dict = {}  # player_id -> list of (x, y) positions

    def _allocate_colors(self):
        """Pre-allocate commonly used colors."""
        colors_to_alloc = _NAMED_COLORS

        for name, (r, g, b) in colors_to_alloc.items():
            self.colors[name] = rgb_to_pixel(
                self.display, self.screen, self.colormap, r, g, b
            )

    # =========================================================================
    # THEMES
    # =========================================================================

    @property
    def themed(self) -> bool:
        return self.theme is not None and not self.theme.classic

    def set_theme(self, name: str):
        import themes
        self.theme = themes.create(name, self)
        self.effects = getattr(self, 'effects', [])

    def cycle_theme(self) -> str:
        import themes
        names = themes.available()
        if self.gfx is None:
            return self.theme.name
        i = names.index(self.theme.name) if self.theme.name in names else -1
        self.set_theme(names[(i + 1) % len(names)])
        return self.theme.name

    def set_color(self, color_name: str):
        """Set the current drawing color."""
        if color_name in self.colors:
            _x11.XSetForeground(self.display, self.gc, self.colors[color_name])
        else:
            _x11.XSetForeground(self.display, self.gc, self.white)
        self._rgb = _NAMED_COLORS.get(color_name, (255, 255, 255))

    def set_color_rgb(self, r: int, g: int, b: int):
        """Set color from RGB values."""
        r, g, b = int(r), int(g), int(b)
        pixel = rgb_to_pixel(self.display, self.screen, self.colormap, r, g, b)
        _x11.XSetForeground(self.display, self.gc, pixel)
        self._rgb = (max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)))

    # =========================================================================
    # VISUAL EFFECTS METHODS (Cycle 3)
    # =========================================================================

    def add_hit_effect(self, x: float, y: float, knockback_magnitude: float,
                       angle: float = 0.0, attacker_color: Tuple[int, int, int] = (255, 255, 255),
                       damage: float = 0.0):
        """
        Add a hit spark effect at the specified position.

        Cycle 13: Spark color based on damage dealt:
        - 0-10 damage: White sparks
        - 11-20 damage: Yellow sparks
        - 21-30 damage: Orange sparks
        - 31+ damage: Red sparks
        """
        # Cycle 13: Determine spark color based on damage
        if damage >= 31:
            spark_color = (255, 100, 100)  # Red
        elif damage >= 21:
            spark_color = (255, 180, 80)   # Orange
        elif damage >= 11:
            spark_color = (255, 255, 100)  # Yellow
        else:
            spark_color = (255, 255, 255)  # White

        # Hit spark
        self.effects.append(VisualEffect(
            effect_type=EffectType.HIT_SPARK,
            x=x, y=y,
            duration=8,
            intensity=min(knockback_magnitude / 10.0, 1.5),
            angle=angle,
            color=spark_color,  # Cycle 13: Use damage-based color
            secondary_color=attacker_color
        ))

        # Screen shake scaled to knockback
        direction_x = math.cos(math.radians(angle))
        direction_y = -math.sin(math.radians(angle))
        self.screen_shake.apply(knockback_magnitude, direction_x, direction_y)

    def add_damage_flash(self, fighter: Fighter):
        """Mark a fighter to flash white (stored on fighter, rendered in draw_stick_figure)."""
        # We'll track this via a simple attribute we add
        fighter._damage_flash_frames = 4

    def add_reflect_effect(self, x: float, y: float):
        """Add reflector visual effect."""
        self.effects.append(VisualEffect(
            effect_type=EffectType.REFLECTOR_HEX,
            x=x, y=y,
            duration=10,
            intensity=1.0,
            color=(100, 255, 255)
        ))

    def update_effects(self):
        """Update all visual effects and remove inactive ones."""
        # Update effects
        for effect in self.effects:
            effect.update()

        # Remove inactive effects
        self.effects = [e for e in self.effects if e.active]

        # Update screen shake
        self.shake_offset = self.screen_shake.update()

    def draw_hit_spark(self, effect: VisualEffect):
        """Draw a hit spark effect (radiating lines)."""
        cx = int(effect.x) + self.shake_offset[0]
        cy = int(effect.y) + self.shake_offset[1]

        # Number of lines and length based on intensity
        num_lines = int(6 + effect.intensity * 4)
        base_length = 20 + effect.intensity * 30
        current_length = base_length * (1.0 - effect.progress * 0.5)

        # Cycle 13: Use damage-based color stored in effect.color
        # Fade the color as it progresses
        progress = effect.progress
        fade = 1.0 - progress * 0.6  # Fade from 1.0 to 0.4 over duration
        r = int(effect.color[0] * fade)
        g = int(effect.color[1] * fade)
        b = int(effect.color[2] * fade)
        self.set_color_rgb(r, g, b)

        # Draw radiating lines
        for i in range(num_lines):
            angle = effect.angle + (360.0 / num_lines) * i + (effect.frame * 5)
            rad = math.radians(angle)
            # Lines expand outward from center
            inner_dist = current_length * 0.3 * effect.progress
            outer_dist = current_length * (0.5 + 0.5 * (1.0 - effect.progress))

            x1 = cx + math.cos(rad) * inner_dist
            y1 = cy - math.sin(rad) * inner_dist
            x2 = cx + math.cos(rad) * outer_dist
            y2 = cy - math.sin(rad) * outer_dist

            self.draw_line(x1, y1, x2, y2)

        # Central flash (first few frames) - use spark color for flash too
        if effect.frame < 3:
            # Flash is brighter version of spark color
            flash_r = min(255, int(effect.color[0] + 50))
            flash_g = min(255, int(effect.color[1] + 50))
            flash_b = min(255, int(effect.color[2] + 50))
            self.set_color_rgb(flash_r, flash_g, flash_b)
            radius = int((3 - effect.frame) * 5 * effect.intensity)
            self.draw_circle(cx, cy, max(1, radius), filled=True)

    def draw_reflector_hex(self, effect: VisualEffect):
        """Draw hexagonal reflector flash effect."""
        cx = int(effect.x) + self.shake_offset[0]
        cy = int(effect.y) + self.shake_offset[1]

        # Hexagon expanding then fading
        radius = 30 + effect.progress * 20
        alpha_mod = 1.0 - effect.progress

        self.set_color('reflector_cyan')

        # Draw hexagon
        points = []
        for i in range(6):
            angle = math.radians(60 * i - 30)  # Start pointing up
            px = cx + math.cos(angle) * radius
            py = cy + math.sin(angle) * radius
            points.append((px, py))

        # Draw hexagon edges
        for i in range(6):
            x1, y1 = points[i]
            x2, y2 = points[(i + 1) % 6]
            self.draw_line(x1, y1, x2, y2)

        # Inner hexagon
        inner_radius = radius * 0.6
        for i in range(6):
            angle = math.radians(60 * i - 30)
            x1 = cx + math.cos(angle) * inner_radius
            y1 = cy + math.sin(angle) * inner_radius
            next_angle = math.radians(60 * ((i + 1) % 6) - 30)
            x2 = cx + math.cos(next_angle) * inner_radius
            y2 = cy + math.sin(next_angle) * inner_radius
            self.draw_line(x1, y1, x2, y2)

    def draw_all_effects(self):
        """Draw all active visual effects."""
        for effect in self.effects:
            if effect.effect_type == EffectType.HIT_SPARK:
                self.draw_hit_spark(effect)
            elif effect.effect_type == EffectType.REFLECTOR_HEX:
                self.draw_reflector_hex(effect)
            elif effect.effect_type == EffectType.LANDING_DUST:
                self.draw_landing_dust(effect)
            elif effect.effect_type == EffectType.FRAME_ADVANTAGE:
                self.draw_frame_advantage(effect)

    def draw_frame_advantage(self, effect: VisualEffect):
        """
        Cycle 13: Draw frame advantage indicator.

        Shows +/- frame advantage at the hit location.
        Color coded: green (advantage), red (disadvantage), yellow (neutral)
        """
        cx = int(effect.x) + self.shake_offset[0]
        cy = int(effect.y) + self.shake_offset[1]

        # Fade out over time
        progress = effect.progress
        fade = max(0.3, 1.0 - progress)

        # Get advantage value from intensity (repurposed for this)
        advantage = int(effect.intensity)

        # Build display text
        if advantage > 0:
            text = f"+{advantage}"
        else:
            text = str(advantage)

        # Color with fade
        r = int(effect.color[0] * fade)
        g = int(effect.color[1] * fade)
        b = int(effect.color[2] * fade)
        self.set_color_rgb(r, g, b)

        # Float upward over time
        float_y = cy - int(progress * 30)

        # Draw text (centered)
        text_width = len(text) * 8  # Approximate character width
        self.draw_text(cx - text_width // 2, float_y, text)

    def add_frame_advantage_effect(self, x: float, y: float, advantage: int):
        """
        Cycle 13: Add a frame advantage display effect.

        Args:
            x, y: Position to display
            advantage: Frame advantage value (positive = attacker advantage)
        """
        # Determine color based on advantage
        if advantage > 0:
            color = (100, 255, 100)  # Green - attacker advantage
        elif advantage < 0:
            color = (255, 100, 100)  # Red - defender advantage
        else:
            color = (255, 255, 100)  # Yellow - neutral

        self.effects.append(VisualEffect(
            effect_type=EffectType.FRAME_ADVANTAGE,
            x=x, y=y,
            duration=60,  # 1 second display
            intensity=advantage,  # Store advantage in intensity field
            color=color
        ))

    def draw_landing_dust(self, effect: VisualEffect):
        """Draw dust puff particles on landing."""
        cx = int(effect.x) + self.shake_offset[0]
        cy = int(effect.y) + self.shake_offset[1]

        progress = effect.frame / effect.duration
        if progress >= 1.0:
            return

        # Dust particles expand outward and fade
        num_particles = 6
        base_spread = 15 + progress * 30  # Expands over time
        fade = int(180 * (1.0 - progress))  # Fade out

        # Tan/brown dust color
        r = min(255, 180 + fade // 3)
        g = min(255, 150 + fade // 4)
        b = min(255, 100 + fade // 6)
        self.set_color_rgb(r, g, b)

        for i in range(num_particles):
            angle = (i / num_particles) * math.pi + math.pi * 0.1  # Semicircle upward spread
            px = cx + math.cos(angle) * base_spread * (0.7 + 0.3 * (i % 2))
            py = cy - abs(math.sin(angle)) * base_spread * 0.5 - progress * 10  # Rise up

            # Particle size decreases over time
            size = int(4 * (1.0 - progress * 0.5))
            if size > 0:
                self.draw_circle(int(px), int(py), size, filled=True)

    def add_landing_dust(self, x: float, y: float):
        """Add landing dust effect at fighter's feet."""
        self.effects.append(VisualEffect(
            effect_type=EffectType.LANDING_DUST,
            x=x, y=y,
            duration=12  # Quick dust puff
        ))

    def draw_dodge_afterimages(self, fighter: Fighter):
        """Draw ghost afterimages during dodge (Cycle 3, enhanced Cycle 10)."""
        if fighter.state != FighterState.DODGING:
            # Clear afterimages when not dodging
            self.fighter_afterimages[fighter.player_id] = []
            return

        # Determine afterimage color based on dodge type (Cycle 10)
        # SPOT_DODGE: Blue-ish, ROLL: Green-ish, AIR_DODGE: Orange-ish
        dodge_colors = {
            DodgeType.SPOT_DODGE: (150, 150, 255),      # Blue-ish
            DodgeType.ROLL_FORWARD: (150, 255, 150),    # Green-ish
            DodgeType.ROLL_BACKWARD: (150, 255, 150),   # Green-ish
            DodgeType.AIR_DODGE: (255, 200, 150),       # Orange-ish
        }
        afterimage_color = dodge_colors.get(fighter.dodge_type, (200, 200, 200))

        # Store current position for afterimage
        if fighter.player_id not in self.fighter_afterimages:
            self.fighter_afterimages[fighter.player_id] = []

        afterimages = self.fighter_afterimages[fighter.player_id]

        # Add new afterimage every 3 frames
        if fighter.state_frame % 3 == 0:
            afterimages.append({
                'x': fighter.position.x,
                'y': fighter.position.y,
                'frame': 0,
                'facing_right': fighter.facing_right,
                'state_frame': fighter.state_frame,
                'dodge_type': fighter.dodge_type
            })

        # Keep only last 3 afterimages
        if len(afterimages) > 3:
            afterimages.pop(0)

        # Draw afterimages (oldest to newest)
        for i, img in enumerate(afterimages):
            alpha = 0.2 + (i / len(afterimages)) * 0.3
            # Create a temporary "ghost" position
            ghost_x = img['x'] - fighter.position.x
            ghost_y = img['y'] - fighter.position.y

            # Get color for this afterimage's dodge type
            img_color = dodge_colors.get(img.get('dodge_type'), afterimage_color)

            # Draw a faded version at the old position
            self.draw_stick_figure(fighter,
                                   offset_x=int(ghost_x),
                                   offset_y=int(ghost_y),
                                   alpha=alpha,
                                   is_afterimage=True,
                                   afterimage_color=img_color)

            img['frame'] += 1

    def draw_invincibility_glow(self, fighter: Fighter):
        """Draw pulsing glow effect during dodge invincibility (Cycle 10)."""
        if fighter.state != FighterState.DODGING:
            return

        # Only show glow during i-frames
        if fighter.invincible_frames <= 0:
            return

        shake_x, shake_y = self.shake_offset
        cx = int(fighter.position.x) + shake_x
        cy = int(fighter.position.y - 40) + shake_y  # Center on body

        # Pulsing effect based on state frame
        pulse = 0.5 + 0.5 * math.sin(fighter.state_frame * 0.5)

        # Determine glow color based on dodge type
        if fighter.dodge_type == DodgeType.SPOT_DODGE:
            base_color = (100, 150, 255)  # Blue
        elif fighter.dodge_type in (DodgeType.ROLL_FORWARD, DodgeType.ROLL_BACKWARD):
            base_color = (100, 255, 150)  # Green
        elif fighter.dodge_type == DodgeType.AIR_DODGE:
            base_color = (255, 180, 100)  # Orange
        else:
            base_color = (200, 200, 255)  # Default white-blue

        # Draw multiple concentric circles for glow effect
        for i in range(3):
            radius = int(25 + i * 8 + pulse * 5)
            # Fade outer rings
            intensity = 1.0 - (i / 3) * 0.6
            r = int(base_color[0] * intensity)
            g = int(base_color[1] * intensity)
            b = int(base_color[2] * intensity)
            self.set_color_rgb(r, g, b)
            self.draw_circle(cx, cy, radius, filled=False)

    def draw_knockback_trails(self, fighter: Fighter):
        """
        Draw speed line trails for fighters in knockback/hitstun.

        Cycle 14: Enhanced with particle effects for fast-moving characters.
        Cycle 15: Enhanced with perpendicular speed lines for high knockback (>18).
        """
        pid = fighter.player_id

        if fighter.state == FighterState.HITSTUN:
            # Store positions for trail
            if pid not in self.knockback_trails:
                self.knockback_trails[pid] = []

            trail = self.knockback_trails[pid]

            # Add current position
            trail.append((fighter.position.x, fighter.position.y))

            # Keep only last 8 positions
            if len(trail) > 8:
                trail.pop(0)

            # Calculate velocity for particle intensity
            velocity_mag = fighter.velocity.magnitude() if hasattr(fighter.velocity, 'magnitude') else 0

            # Draw trail as speed lines
            if len(trail) >= 2:
                # Get fighter color for trail
                r, g, b = fighter.color
                shake_x, shake_y = self.shake_offset

                for i in range(len(trail) - 1):
                    # Fade older segments
                    alpha = (i + 1) / len(trail)
                    tr = int(r * alpha * 0.6)
                    tg = int(g * alpha * 0.6)
                    tb = int(b * alpha * 0.6)
                    self.set_color_rgb(tr, tg, tb)

                    x1 = int(trail[i][0]) + shake_x
                    y1 = int(trail[i][1]) + shake_y - 40  # Offset to body center
                    x2 = int(trail[i + 1][0]) + shake_x
                    y2 = int(trail[i + 1][1]) + shake_y - 40

                    # Draw multiple parallel lines for thickness
                    for offset in [-2, 0, 2]:
                        self.draw_line(x1, y1 + offset, x2, y2 + offset)

                # Cycle 15: Enhanced speed lines for high knockback (>18)
                if velocity_mag > 18:
                    # Calculate velocity direction
                    vx = fighter.velocity.x
                    vy = fighter.velocity.y
                    v_mag = max(0.1, velocity_mag)  # Avoid division by zero

                    # Normalized velocity direction
                    dir_x = vx / v_mag
                    dir_y = vy / v_mag

                    # Perpendicular direction for speed lines
                    perp_x = -dir_y
                    perp_y = dir_x

                    # Draw speed line particles behind the fighter
                    num_speed_lines = min(int((velocity_mag - 18) / 3) + 3, 7)

                    for s in range(num_speed_lines):
                        # Speed lines spawn behind the fighter
                        base_offset = 10 + s * 8 + int(np.random.random() * 5)
                        spawn_x = int(fighter.position.x - dir_x * base_offset) + shake_x
                        spawn_y = int(fighter.position.y - 40 - dir_y * base_offset) + shake_y

                        # Perpendicular offset for line spread
                        perp_offset = (np.random.random() - 0.5) * 20
                        spawn_x += int(perp_x * perp_offset)
                        spawn_y += int(perp_y * perp_offset)

                        # Speed line length (10-15 pixels)
                        line_len = 10 + int(np.random.random() * 5)

                        # Line endpoints perpendicular to velocity
                        lx1 = spawn_x - int(perp_x * line_len / 2)
                        ly1 = spawn_y - int(perp_y * line_len / 2)
                        lx2 = spawn_x + int(perp_x * line_len / 2)
                        ly2 = spawn_y + int(perp_y * line_len / 2)

                        # Alpha fade based on distance from fighter
                        fade = 0.8 - (s / num_speed_lines) * 0.5
                        sr = int(r * fade)
                        sg = int(g * fade)
                        sb = int(b * fade)
                        self.set_color_rgb(sr, sg, sb)

                        self.draw_line(lx1, ly1, lx2, ly2)

                # Cycle 14: Add particle effects for fast-moving knockback
                if velocity_mag > 12:
                    # Spawn particles at character position
                    num_particles = min(int(velocity_mag / 4), 5)
                    for p in range(num_particles):
                        # Random offset for particle spread
                        px = int(fighter.position.x) + shake_x + int((np.random.random() - 0.5) * 30)
                        py = int(fighter.position.y) + shake_y - 40 + int((np.random.random() - 0.5) * 30)

                        # Particle size based on velocity
                        particle_size = max(2, min(int(velocity_mag / 6), 5))

                        # White/yellow particles for impact
                        self.set_color_rgb(255, 255 - p * 30, 200 - p * 40)
                        self.draw_circle(px, py, particle_size, filled=True)
        else:
            # Clear trail when not in hitstun
            self.knockback_trails[pid] = []

    def draw_counter_aura(self, fighter: Fighter):
        """Draw counter stance visual aura (Cycle 3)."""
        if not fighter.counter_active:
            return

        shake_x, shake_y = self.shake_offset
        cx = int(fighter.position.x) + shake_x
        cy = int(fighter.position.y - 40) + shake_y

        # Pulsing blue aura
        pulse = 0.5 + 0.5 * math.sin(fighter.state_frame * 0.3)
        radius = int(35 + pulse * 10)

        self.set_color('counter_blue')
        self.draw_circle(cx, cy, radius, filled=False)
        self.draw_circle(cx, cy, radius - 3, filled=False)

        # Inner glow
        self.set_color_rgb(100, 150, 255)
        self.draw_circle(cx, cy, radius - 6, filled=False)

    def draw_reflector_active(self, fighter: Fighter):
        """Draw active reflector shield (Cycle 3)."""
        if not fighter.reflector_active:
            return

        shake_x, shake_y = self.shake_offset
        cx = int(fighter.position.x) + shake_x
        cy = int(fighter.position.y - 40) + shake_y

        # Hexagonal reflector shield
        radius = 40
        self.set_color('reflector_cyan')

        # Draw hexagon
        for i in range(6):
            angle1 = math.radians(60 * i - 30)
            angle2 = math.radians(60 * (i + 1) - 30)
            x1 = cx + math.cos(angle1) * radius
            y1 = cy + math.sin(angle1) * radius
            x2 = cx + math.cos(angle2) * radius
            y2 = cy + math.sin(angle2) * radius
            self.draw_line(x1, y1, x2, y2)

        # Spinning inner element
        inner_angle = fighter.state_frame * 5
        for i in range(3):
            angle = math.radians(inner_angle + 120 * i)
            x1 = cx + math.cos(angle) * 15
            y1 = cy + math.sin(angle) * 15
            x2 = cx + math.cos(angle) * 30
            y2 = cy + math.sin(angle) * 30
            self.draw_line(x1, y1, x2, y2)

    def cleanup_projectile_trails(self, active_projectiles):
        """Remove trail data for projectiles that no longer exist."""
        active_ids = {id(p) for p in active_projectiles}
        self.projectile_trails = {
            pid: trail for pid, trail in self.projectile_trails.items()
            if pid in active_ids
        }

    def clear(self):
        """Clear the buffer."""
        self.set_color('black')
        _x11.XFillRectangle(self.display, self.buffer, self.gc,
                           0, 0, self.width, self.height)

    def flip(self):
        """Copy buffer to window and flush."""
        _x11.XCopyArea(self.display, self.buffer, self.window, self.gc,
                      0, 0, self.width, self.height, 0, 0)
        _x11.XFlush(self.display)

    def draw_line(self, x1: int, y1: int, x2: int, y2: int):
        """Draw a line."""
        if self.themed:
            self.gfx.line(x1, y1, x2, y2, 2.4, self._rgb)
            return
        _x11.XDrawLine(self.display, self.buffer, self.gc,
                      int(x1), int(y1), int(x2), int(y2))

    def draw_rect(self, x: int, y: int, w: int, h: int, filled: bool = False):
        """Draw a rectangle."""
        if self.themed:
            if filled:
                self.gfx.fill_rect(x, y, w, h, self._rgb)
            else:
                self.gfx.round_rect_outline(x, y, w, h, 0, 1.6, self._rgb)
            return
        if filled:
            _x11.XFillRectangle(self.display, self.buffer, self.gc,
                               int(x), int(y), int(w), int(h))
        else:
            _x11.XDrawRectangle(self.display, self.buffer, self.gc,
                               int(x), int(y), int(w), int(h))

    def draw_circle(self, cx: int, cy: int, radius: int, filled: bool = False):
        """Draw a circle."""
        if self.themed:
            if filled:
                self.gfx.circle(cx, cy, radius, self._rgb)
            else:
                self.gfx.ring(cx, cy, radius, 2.0, self._rgb)
            return
        x = int(cx - radius)
        y = int(cy - radius)
        d = int(radius * 2)
        if filled:
            _x11.XFillArc(self.display, self.buffer, self.gc,
                         x, y, d, d, 0, 360 * 64)
        else:
            _x11.XDrawArc(self.display, self.buffer, self.gc,
                         x, y, d, d, 0, 360 * 64)

    def draw_text(self, x: int, y: int, text: str):
        """Draw text."""
        if self.themed:
            self.theme.ensure_built()
            self.theme.text(x, y, text, 11, self._rgb, role="small")
            return
        _x11.XDrawString(self.display, self.buffer, self.gc,
                        int(x), int(y), text.encode(), len(text))

    # =========================================================================
    # FPS COUNTER & FRAME GRAPH (Cycle 17)
    # =========================================================================

    def draw_fps_counter(self, frame_times: List[float]):
        """
        Draw FPS counter in top-left corner.

        Cycle 17: Performance profiling feature.
        Shows rolling 30-frame average FPS and frame time.
        Color-coded: Green (>55fps), Yellow (45-55fps), Red (<45fps).
        """
        if not frame_times:
            return

        # Calculate rolling average from last 30 frames
        recent_frames = frame_times[-30:] if len(frame_times) >= 30 else frame_times
        avg_frame_time = sum(recent_frames) / len(recent_frames)
        fps = 1000 / avg_frame_time if avg_frame_time > 0 else 0

        # Color based on FPS
        if fps >= 55:
            r, g, b = 100, 255, 100  # Green
        elif fps >= 45:
            r, g, b = 255, 255, 100  # Yellow
        else:
            r, g, b = 255, 100, 100  # Red

        # Draw background box
        self.set_color_rgb(20, 20, 30)
        self.draw_rect(5, 5, 95, 35, filled=True)

        # Draw border
        self.set_color_rgb(60, 60, 80)
        self.draw_rect(5, 5, 95, 35, filled=False)

        # Draw FPS text
        self.set_color_rgb(r, g, b)
        self.draw_text(10, 20, f"FPS: {fps:.0f}")

        # Draw frame time
        self.set_color_rgb(180, 180, 200)
        self.draw_text(10, 35, f"{avg_frame_time:.1f}ms")

    def draw_frame_graph(self, frame_times: List[float]):
        """
        Draw frame time graph at bottom-left of screen.

        Cycle 17: Performance profiling feature.
        Shows last 60 frames as vertical bars.
        Reference line at 16.67ms (60fps target).
        Bars above target are shown in red.
        """
        if not frame_times:
            return

        graph_x = 10
        graph_y = self.height - 60
        graph_width = 64
        graph_height = 50
        target_ms = 16.67  # 60 FPS target
        max_ms = 33.33     # 30 FPS (max displayed)

        # Background
        self.set_color_rgb(20, 20, 30)
        self.draw_rect(graph_x - 2, graph_y - 2, graph_width + 4, graph_height + 4, filled=True)

        # Border
        self.set_color_rgb(60, 60, 80)
        self.draw_rect(graph_x - 2, graph_y - 2, graph_width + 4, graph_height + 4, filled=False)

        # Reference line at 16.67ms (60fps)
        ref_y = graph_y + graph_height - int((target_ms / max_ms) * graph_height)
        self.set_color_rgb(100, 100, 100)
        self.draw_line(graph_x, ref_y, graph_x + 60, ref_y)

        # Frame bars (last 60 frames)
        frames_to_show = frame_times[-60:] if len(frame_times) >= 60 else frame_times
        bar_offset = 60 - len(frames_to_show)  # Right-align bars

        for i, ft in enumerate(frames_to_show):
            bar_height = min(int((ft / max_ms) * graph_height), graph_height)
            bar_y = graph_y + graph_height - bar_height

            if ft > target_ms:
                self.set_color_rgb(255, 100, 100)  # Red for dropped frames
            else:
                self.set_color_rgb(100, 255, 100)  # Green for normal

            self.draw_rect(graph_x + bar_offset + i, bar_y, 1, bar_height, filled=True)

        # Label
        self.set_color_rgb(150, 150, 150)
        self.draw_text(graph_x, graph_y - 5, "FRAME")

    def draw_stick_figure(self, fighter: Fighter, offset_x: int = 0, offset_y: int = 0,
                          alpha: float = 1.0, is_afterimage: bool = False,
                          afterimage_color: Optional[Tuple[int, int, int]] = None,
                          hit_freeze_active: bool = False, hit_freeze_frame: int = 0):
        """
        Draw a stick figure fighter.

        Cycle 15: Added hit_freeze_active and hit_freeze_frame for vibration effect.
        """
        pos = fighter.position
        facing = 1 if fighter.facing_right else -1

        # Apply screen shake offset
        shake_x, shake_y = self.shake_offset if not is_afterimage else (0, 0)
        offset_x += shake_x
        offset_y += shake_y

        # Cycle 15: Hit freeze vibration (2-3 pixel oscillation)
        if hit_freeze_active and not is_afterimage:
            # Oscillate rapidly between -2 and +2
            vibration = 3 if (hit_freeze_frame % 2) == 0 else -3
            if fighter.grounded:
                # Grounded fighters vibrate horizontally
                offset_x += vibration
            else:
                # Airborne fighters vibrate vertically
                offset_y += vibration

        # Get pose for current state (Cycle 4: pass fighter for game feel improvements)
        pose = get_pose_for_state(fighter.state, fighter.state_frame, fighter)

        # Determine color
        if is_afterimage and afterimage_color is not None:
            # Use specified afterimage color for dodge type visualization
            r, g, b = afterimage_color
        else:
            r, g, b = fighter.color

            # Damage flash (Cycle 3) - white flash when hit
            if hasattr(fighter, '_damage_flash_frames') and fighter._damage_flash_frames > 0:
                fighter._damage_flash_frames -= 1
                r, g, b = 255, 255, 255

            # Super armor glow (Cycle 3) - golden outline
            elif fighter.super_armor_active:
                r, g, b = 255, 200, 50  # Golden

            # Counter stance (Cycle 3) - blue tint
            elif fighter.counter_active:
                r, g, b = 100, 150, 255  # Blue

            # Flash white if invincible (original behavior)
            elif fighter.invincible_frames > 0 and (fighter.invincible_frames // 4) % 2 == 0:
                r, g, b = 255, 255, 255

        # Apply alpha for afterimages (just darken since X11 doesn't have alpha)
        if is_afterimage:
            r = int(r * alpha)
            g = int(g * alpha)
            b = int(b * alpha)

        self.set_color_rgb(r, g, b)

        # Center point at feet (with offset)
        cx = int(pos.x) + offset_x
        cy = int(pos.y) + offset_y

        # Head
        head_y = cy + int(pose.head_y)
        self.draw_circle(cx, head_y, int(pose.head_radius), filled=True)

        # Spine (body)
        shoulder_y = cy + int(pose.shoulder_y)
        hip_y = cy + int(pose.hip_y)
        self.draw_line(cx, shoulder_y, cx, hip_y)

        # Neck
        self.draw_line(cx, head_y + int(pose.head_radius), cx, shoulder_y)

        # Left arm
        left_elbow_x = cx + int(pose.left_elbow[0] * facing)
        left_elbow_y = shoulder_y + int(pose.left_elbow[1])
        left_hand_x = cx + int(pose.left_hand[0] * facing)
        left_hand_y = shoulder_y + int(pose.left_hand[1])
        self.draw_line(cx, shoulder_y, left_elbow_x, left_elbow_y)
        self.draw_line(left_elbow_x, left_elbow_y, left_hand_x, left_hand_y)

        # Right arm
        right_elbow_x = cx + int(pose.right_elbow[0] * facing)
        right_elbow_y = shoulder_y + int(pose.right_elbow[1])
        right_hand_x = cx + int(pose.right_hand[0] * facing)
        right_hand_y = shoulder_y + int(pose.right_hand[1])
        self.draw_line(cx, shoulder_y, right_elbow_x, right_elbow_y)
        self.draw_line(right_elbow_x, right_elbow_y, right_hand_x, right_hand_y)

        # Left leg
        left_knee_x = cx + int(pose.left_knee[0] * facing)
        left_knee_y = hip_y + int(pose.left_knee[1])
        left_foot_x = cx + int(pose.left_foot[0] * facing)
        left_foot_y = hip_y + int(pose.left_foot[1])
        self.draw_line(cx, hip_y, left_knee_x, left_knee_y)
        self.draw_line(left_knee_x, left_knee_y, left_foot_x, left_foot_y)

        # Right leg
        right_knee_x = cx + int(pose.right_knee[0] * facing)
        right_knee_y = hip_y + int(pose.right_knee[1])
        right_foot_x = cx + int(pose.right_foot[0] * facing)
        right_foot_y = hip_y + int(pose.right_foot[1])
        self.draw_line(cx, hip_y, right_knee_x, right_knee_y)
        self.draw_line(right_knee_x, right_knee_y, right_foot_x, right_foot_y)

        # Super armor glow (Cycle 3) - draw a second outline
        if fighter.super_armor_active and not is_afterimage:
            self.set_color('armor_gold')
            # Draw slightly offset copies for glow effect
            for glow_offset in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
                gx, gy = glow_offset
                self.draw_circle(cx + gx, head_y + gy, int(pose.head_radius) + 1, filled=False)

        # Draw hitboxes if debug mode
        if self.show_hitboxes:
            self.set_color('hitbox')
            for hitbox_aabb in fighter.get_active_hitbox_aabbs():
                self.draw_rect(
                    hitbox_aabb.x, hitbox_aabb.y,
                    hitbox_aabb.width, hitbox_aabb.height,
                    filled=False
                )

        # Draw hurtbox if debug mode
        if self.show_hurtboxes:
            self.set_color('hurtbox')
            hurtbox = fighter.get_hurtbox_aabb()
            self.draw_rect(
                hurtbox.x, hurtbox.y,
                hurtbox.width, hurtbox.height,
                filled=False
            )

    def draw_platform(self, platform: Platform):
        """Draw a platform (with screen shake - Cycle 3)."""
        # Apply screen shake
        shake_x, shake_y = self.shake_offset

        if platform.passthrough:
            self.set_color('platform_pass')
        else:
            self.set_color('platform')

        self.draw_rect(
            platform.x + shake_x, platform.y + shake_y,
            platform.width, platform.height,
            filled=True
        )

        # Draw surface line
        self.set_color('white')
        self.draw_line(
            platform.x + shake_x, platform.y + shake_y,
            platform.x + platform.width + shake_x, platform.y + shake_y
        )

    def draw_projectile(self, projectile: Projectile):
        """Draw a projectile with trail effect (Cycle 3)."""
        if not projectile.active:
            return

        # Apply screen shake
        shake_x, shake_y = self.shake_offset

        # Use projectile's color
        r, g, b = projectile.color

        cx = int(projectile.position.x) + shake_x
        cy = int(projectile.position.y) + shake_y

        # Cycle 3: Trail effect - store position history
        proj_id = id(projectile)
        if proj_id not in self.projectile_trails:
            self.projectile_trails[proj_id] = []

        trail = self.projectile_trails[proj_id]
        trail.append(TrailPoint(projectile.position.x, projectile.position.y))

        # Keep only last 8 positions
        if len(trail) > 8:
            trail.pop(0)

        # Draw trail (from oldest to newest, fading)
        for i, point in enumerate(trail[:-1]):  # Don't draw the current position as trail
            alpha = (i + 1) / len(trail)
            trail_r = int(r * alpha * 0.6)
            trail_g = int(g * alpha * 0.6)
            trail_b = int(b * alpha * 0.6)
            self.set_color_rgb(trail_r, trail_g, trail_b)

            trail_x = int(point.x) + shake_x
            trail_y = int(point.y) + shake_y
            trail_radius = max(int((projectile.width / 2) * alpha * 0.7), 2)
            self.draw_circle(trail_x, trail_y, trail_radius, filled=True)

        # Draw main projectile
        self.set_color_rgb(r, g, b)

        # Draw based on shape (circles for most, rectangles for beams)
        if projectile.width > projectile.height * 2:
            # Beam-like projectile
            self.draw_rect(
                cx - int(projectile.width / 2),
                cy - int(projectile.height / 2),
                int(projectile.width),
                int(projectile.height),
                filled=True
            )
            # Beam trail - draw trailing lines
            if len(trail) > 1:
                self.set_color_rgb(r // 2, g // 2, b // 2)
                for i in range(1, min(4, len(trail))):
                    prev = trail[-i-1]
                    alpha = 1.0 - (i / 4.0)
                    trail_w = int(projectile.width * alpha)
                    trail_h = int(projectile.height * alpha)
                    self.draw_rect(
                        int(prev.x) + shake_x - trail_w // 2,
                        int(prev.y) + shake_y - trail_h // 2,
                        trail_w, trail_h, filled=True
                    )
        else:
            # Circular projectile
            radius = max(int(projectile.width / 2), 5)
            self.draw_circle(cx, cy, radius, filled=True)

            # Add glow effect (slightly larger circle, less bright)
            self.set_color_rgb(r // 2, g // 2, b // 2)
            self.draw_circle(cx, cy, radius + 3, filled=False)

            # Bright core
            self.set_color_rgb(min(255, r + 50), min(255, g + 50), min(255, b + 50))
            core_radius = max(radius // 2, 2)
            self.draw_circle(cx, cy, core_radius, filled=True)

    def draw_shield(self, fighter: 'Fighter'):
        """Draw a shield bubble around a shielding fighter (Cycle 3 enhanced)."""
        if fighter.state != FighterState.SHIELDING:
            return

        # Apply screen shake
        shake_x, shake_y = self.shake_offset

        # Shield size based on health
        base_radius = 50
        shield_ratio = fighter.shield_health / 100.0
        radius = int(base_radius * (0.5 + 0.5 * shield_ratio))

        # Color based on health (green -> yellow -> red) - smoother gradient
        if shield_ratio > 0.6:
            # Green to yellow
            t = (shield_ratio - 0.6) / 0.4
            r = int(100 + (100 * (1 - t)))
            g = 200
            b = int(100 * t)
        elif shield_ratio > 0.3:
            # Yellow to orange
            t = (shield_ratio - 0.3) / 0.3
            r = 200
            g = int(200 * t + 100 * (1 - t))
            b = int(100 * t)
        else:
            # Orange to red
            t = shield_ratio / 0.3
            r = 200
            g = int(100 * t)
            b = int(100 * t)

        self.set_color_rgb(r, g, b)

        # Draw shield bubble with shake offset
        cx = int(fighter.position.x) + shake_x
        cy = int(fighter.position.y - 40) + shake_y  # Center on body
        self.draw_circle(cx, cy, radius, filled=False)

        # Draw inner shield rings for depth effect
        self.set_color_rgb(r // 2, g // 2, b // 2)
        self.draw_circle(cx, cy, radius - 3, filled=False)

        # Cycle 3: Crack effects when shield is low
        if shield_ratio < 0.3:
            self.set_color('white')
            # Draw crack lines
            num_cracks = int((0.3 - shield_ratio) * 10) + 1
            for i in range(num_cracks):
                # Random-ish crack positions (deterministic based on shield health)
                angle = (i * 73 + int(fighter.shield_health * 3)) % 360
                rad = math.radians(angle)

                # Crack from center outward
                inner_r = radius * 0.3
                outer_r = radius * 0.9

                x1 = cx + math.cos(rad) * inner_r
                y1 = cy + math.sin(rad) * inner_r
                x2 = cx + math.cos(rad) * outer_r
                y2 = cy + math.sin(rad) * outer_r

                self.draw_line(x1, y1, x2, y2)

                # Branch cracks
                if i % 2 == 0:
                    branch_angle = rad + 0.4
                    x3 = x1 + math.cos(branch_angle) * (outer_r - inner_r) * 0.5
                    y3 = y1 + math.sin(branch_angle) * (outer_r - inner_r) * 0.5
                    mid_x = (x1 + x2) / 2
                    mid_y = (y1 + y2) / 2
                    self.draw_line(mid_x, mid_y, x3, y3)

        # Highlight on shield (top-left bright spot)
        self.set_color_rgb(min(255, r + 80), min(255, g + 80), min(255, b + 80))
        highlight_x = cx - int(radius * 0.4)
        highlight_y = cy - int(radius * 0.4)
        self.draw_circle(highlight_x, highlight_y, max(3, radius // 6), filled=True)

    def _draw_stock_icon(self, x: int, y: int, fighter: 'Fighter', filled: bool = True):
        """Draw a stick figure silhouette as stock icon (Cycle 19 update).

        Draws a small stick figure in the character's color.
        Lost stocks are drawn with a crossed-out effect (X through figure).
        """
        r, g, b = fighter.color if filled else (80, 80, 80)

        if not filled:
            # Lost stock - darker color
            r, g, b = 40, 40, 40

        self.set_color_rgb(r, g, b)

        # Scale for mini stick figure (about 18px tall)
        # Head (circle at top)
        head_y = y - 6
        head_radius = 3
        self.draw_circle(x, head_y, head_radius, filled=filled)

        # Body (vertical line)
        body_top = head_y + head_radius
        body_bottom = y + 4
        self.draw_line(x, body_top, x, body_bottom)

        # Arms (horizontal-ish line from middle of body)
        arm_y = body_top + 3
        self.draw_line(x - 5, arm_y - 1, x + 5, arm_y - 1)

        # Legs (two diagonal lines from bottom of body)
        self.draw_line(x, body_bottom, x - 4, y + 10)  # Left leg
        self.draw_line(x, body_bottom, x + 4, y + 10)  # Right leg

        # If not filled (lost stock), draw an X through it
        if not filled:
            self.set_color_rgb(100, 40, 40)  # Dark red X
            # Draw X from top-left to bottom-right and top-right to bottom-left
            self.draw_line(x - 6, y - 10, x + 6, y + 10)
            self.draw_line(x + 6, y - 10, x - 6, y + 10)

    def draw_hud(self, game_state: GameState):
        """Draw the HUD with enhanced visuals (Cycle 3 + Cycle 4 combo/timer + Cycle 5 stock icons)."""
        # Player 1 (left side)
        p1 = game_state.fighters[0]

        # Character name (Cycle 3)
        char_name_1 = p1.stats.name if hasattr(p1, 'stats') and p1.stats else "P1"
        self.set_color('white')
        self.draw_text(50, 30, char_name_1)

        # Damage percentage with animation (Cycle 19)
        self._draw_animated_damage(
            50, 50, p1.damage_percent,
            game_state.damage_anim_frames[0],
            game_state.damage_anim_amount[0]
        )

        # P1 stocks as stick figure silhouettes (Cycle 19 update)
        for i in range(CONFIG.stock_count):
            stock_x = 50 + i * 28
            stock_y = 80  # Adjusted for taller stick figures
            if i < p1.stocks:
                self._draw_stock_icon(stock_x, stock_y, p1, filled=True)
            else:
                self._draw_stock_icon(stock_x, stock_y, p1, filled=False)

        # P1 shield health bar (moved down for stick figure icons)
        self._draw_shield_bar(50, 100, p1.shield_health, p1.shield_broken)

        # Cycle 4: P1 combo counter
        self._draw_combo_counter(150, 50, game_state.combo_counts[0], game_state.combo_timers[0])

        # Player 2 (right side)
        p2 = game_state.fighters[1]

        # Character name (Cycle 3)
        char_name_2 = p2.stats.name if hasattr(p2, 'stats') and p2.stats else "P2"
        self.set_color('white')
        self.draw_text(self.width - 150, 30, char_name_2)

        # Damage percentage with animation (Cycle 19)
        self._draw_animated_damage(
            self.width - 150, 50, p2.damage_percent,
            game_state.damage_anim_frames[1],
            game_state.damage_anim_amount[1]
        )

        # P2 stocks as stick figure silhouettes (Cycle 19 update)
        for i in range(CONFIG.stock_count):
            stock_x = self.width - 150 + i * 28
            stock_y = 80  # Adjusted for taller stick figures
            if i < p2.stocks:
                self._draw_stock_icon(stock_x, stock_y, p2, filled=True)
            else:
                self._draw_stock_icon(stock_x, stock_y, p2, filled=False)

        # P2 shield health bar (moved down for stick figure icons)
        self._draw_shield_bar(self.width - 150, 100, p2.shield_health, p2.shield_broken)

        # Cycle 4: P2 combo counter
        self._draw_combo_counter(self.width - 230, 50, game_state.combo_counts[1], game_state.combo_timers[1])

        # Cycle 4: Match timer (if enabled)
        if game_state.time_limit > 0:
            time_str = game_state.get_time_display()
            # Flash red when time is low (last 10 seconds)
            if game_state.time_remaining <= 600:  # 10 seconds
                if (game_state.frame // 15) % 2 == 0:
                    self.set_color_rgb(255, 80, 80)
                else:
                    self.set_color('white')
            else:
                self.set_color('white')
            self.draw_text(self.width // 2 - 20, 40, time_str)
        else:
            # Frame counter (smaller, less prominent)
            self.set_color('dark_gray')
            self.draw_text(self.width // 2 - 30, 20, f"F:{game_state.frame}")

    def _draw_combo_counter(self, x: int, y: int, combo_count: int, combo_timer: int):
        """Draw combo counter when active (Cycle 4)."""
        if combo_count <= 1:
            return  # Don't show for single hits

        # Fade out as combo timer approaches expiry (45 frames max)
        alpha = max(0.3, 1.0 - (combo_timer / 45.0))

        # Color based on combo count
        if combo_count >= 10:
            r, g, b = 255, 50, 50  # Red for high combos
        elif combo_count >= 5:
            r, g, b = 255, 200, 50  # Gold/yellow
        else:
            r, g, b = 100, 200, 255  # Light blue

        # Apply alpha fade
        r = int(r * alpha)
        g = int(g * alpha)
        b = int(b * alpha)

        self.set_color_rgb(r, g, b)
        self.draw_text(x, y, f"{combo_count} HITS!")

    def _get_damage_color(self, damage: float) -> Tuple[int, int, int]:
        """Get color for damage display based on percentage (Cycle 3)."""
        if damage < 50:
            return (255, 255, 255)  # White
        elif damage < 100:
            # White to yellow
            t = (damage - 50) / 50
            return (255, 255, int(255 * (1 - t)))
        elif damage < 150:
            # Yellow to orange
            t = (damage - 100) / 50
            return (255, int(255 - 100 * t), 0)
        else:
            # Orange to red
            t = min((damage - 150) / 100, 1.0)
            return (255, int(155 - 155 * t), 0)

    def _draw_animated_damage(self, x: int, y: int, damage: float,
                               anim_frames: int, anim_amount: float):
        """Draw damage percentage with animation effects (Cycle 19).

        Animation effects:
        - Scale up on hit (briefly larger font)
        - Color flash (white flash on hit)
        - Shake effect on big hits (>30 damage)
        """
        import math

        # Base color based on damage level
        base_r, base_g, base_b = self._get_damage_color(damage)

        # Animation intensity (1.0 at start, 0.0 when done)
        anim_intensity = anim_frames / 20.0 if anim_frames > 0 else 0.0

        # Calculate animation offsets
        offset_x = 0
        offset_y = 0

        if anim_intensity > 0:
            # Color flash: blend toward white on hit
            flash_amount = anim_intensity * 0.7  # 70% flash at peak
            r = int(base_r + (255 - base_r) * flash_amount)
            g = int(base_g + (255 - base_g) * flash_amount)
            b = int(base_b + (255 - base_b) * flash_amount)

            # Shake effect for big hits (>30 damage)
            if anim_amount >= 30:
                shake_magnitude = min(4, int(anim_amount / 15)) * anim_intensity
                offset_x = int(math.sin(anim_frames * 2.5) * shake_magnitude)
                offset_y = int(math.cos(anim_frames * 3.0) * shake_magnitude * 0.5)
        else:
            r, g, b = base_r, base_g, base_b

        # Draw with offset
        self.set_color_rgb(r, g, b)

        # Scale effect: draw larger when animating, larger at high damage
        # High damage (100%+) always shows with emphasis
        damage_text = f"{damage:.0f}%"

        # For significant animation or high damage, draw with visual emphasis
        if anim_intensity > 0.5 or damage >= 100:
            # Draw a subtle glow/shadow behind for emphasis
            shadow_alpha = int(100 * max(anim_intensity, 0.3 if damage >= 100 else 0))
            if shadow_alpha > 20:
                self.set_color_rgb(r // 3, g // 3, b // 3)
                self.draw_text(x + offset_x + 2, y + offset_y + 2, damage_text)
                self.set_color_rgb(r, g, b)

        self.draw_text(x + offset_x, y + offset_y, damage_text)

    def _draw_shield_bar(self, x: int, y: int, shield_health: float, broken: bool):
        """Draw a shield health bar."""
        bar_width = 80
        bar_height = 6

        # Background
        self.set_color('dark_gray')
        self.draw_rect(x, y, bar_width, bar_height, filled=True)

        # Fill based on health
        if broken:
            self.set_color_rgb(100, 50, 50)  # Dark red when broken
        elif shield_health > 60:
            self.set_color_rgb(100, 200, 100)  # Green
        elif shield_health > 30:
            self.set_color_rgb(200, 200, 100)  # Yellow
        else:
            self.set_color_rgb(200, 100, 100)  # Red

        fill_width = int(bar_width * (shield_health / 100.0))
        if fill_width > 0:
            self.draw_rect(x, y, fill_width, bar_height, filled=True)

    def _draw_victory_screen(self, game_state: GameState):
        """Draw polished victory/KO screen overlay."""
        center_x = self.width // 2
        center_y = self.height // 2

        # Semi-transparent dark overlay (simulated with dark rectangles)
        self.set_color_rgb(20, 20, 30)
        for y_off in range(-100, 101, 4):
            alpha = 1.0 - abs(y_off) / 100.0
            gray = int(20 * alpha)
            self.set_color_rgb(gray, gray, int(gray * 1.2))
            self.draw_line(0, center_y + y_off, self.width, center_y + y_off)

        if game_state.winner is not None:
            winner = game_state.fighters[game_state.winner]
            winner_name = winner.stats.name

            # Winner spotlight effect - radiating lines
            self.set_color_rgb(80, 80, 100)
            for i in range(12):
                angle = math.radians(i * 30)
                x1 = center_x + int(math.cos(angle) * 50)
                y1 = center_y - 60 + int(math.sin(angle) * 30)
                x2 = center_x + int(math.cos(angle) * 200)
                y2 = center_y - 60 + int(math.sin(angle) * 120)
                self.draw_line(x1, y1, x2, y2)

            # "GAME!" text
            self.set_color('white')
            self.draw_text(center_x - 35, center_y - 80, "GAME!")

            # Winner name with their color
            r, g, b = winner.color
            self.set_color_rgb(r, g, b)
            name_x = center_x - len(winner_name) * 4  # Approximate centering
            self.draw_text(name_x, center_y - 30, winner_name)

            # "WINS!" in yellow
            self.set_color('yellow')
            self.draw_text(center_x - 30, center_y, "WINS!")

            # Final stats
            self.set_color_rgb(180, 180, 180)
            match_time = game_state.frame / 60.0
            self.draw_text(center_x - 60, center_y + 40, f"Time: {match_time:.1f}s")

            # Show both fighter's final damage
            for i, f in enumerate(game_state.fighters):
                side_x = 80 if i == 0 else self.width - 150
                y_pos = center_y + 60
                self.set_color_rgb(*f.color)
                self.draw_text(side_x, y_pos, f"{f.stats.name}")
                self.set_color('white')
                self.draw_text(side_x, y_pos + 20, f"{f.damage_percent:.0f}%")
                self.draw_text(side_x, y_pos + 40, f"{f.stocks} stocks")
        else:
            # Draw
            self.set_color('white')
            self.draw_text(center_x - 35, center_y - 60, "GAME!")
            self.set_color('yellow')
            self.draw_text(center_x - 30, center_y - 20, "DRAW!")

            # Show both fighter's final stats
            self.set_color_rgb(180, 180, 180)
            match_time = game_state.frame / 60.0
            self.draw_text(center_x - 60, center_y + 30, f"Time: {match_time:.1f}s")

    def draw_frame_data_overlay(self, fighter: 'Fighter', x: int, y: int, training_manager):
        """Draw frame data overlay for a fighter (Cycle 5).

        Args:
            fighter: The fighter to show frame data for
            x: X position for the overlay
            y: Y position for the overlay
            training_manager: TrainingModeManager instance
        """
        lines = training_manager.get_frame_data_text(fighter)
        if not lines:
            return

        # Background box
        box_width = 120
        box_height = 15 * len(lines) + 10
        self.set_color_rgb(20, 20, 40)
        self.draw_rect(x, y, box_width, box_height, filled=True)
        self.set_color_rgb(100, 100, 150)
        self.draw_rect(x, y, box_width, box_height, filled=False)

        # Text lines
        self.set_color('white')
        for i, line in enumerate(lines):
            self.draw_text(x + 5, y + 12 + i * 15, line)

    def draw_training_mode_info(self, game_state: GameState, training_manager):
        """Draw training mode indicators (Cycle 6 enhanced + Cycle 16)."""
        if not training_manager:
            return

        # CPU status indicator (top center)
        if training_manager.cpu_controller.state.enabled:
            cpu_behavior = training_manager.cpu_controller.state.behavior.name
            cpu_level = training_manager.cpu_controller.state.difficulty.value
            self.set_color('yellow')
            self.draw_text(self.width // 2 - 60, self.height - 50, f"CPU: {cpu_behavior} Lv{cpu_level}")

        # Training mode toggles (right side of center area)
        toggle_x = self.width // 2 + 80
        toggle_y = self.height - 50
        toggles = []
        if training_manager.infinite_jumps:
            toggles.append("InfJumps")
        if training_manager.infinite_shields:
            toggles.append("InfShield")
        if training_manager.show_frame_advantage:
            toggles.append("FrameAdv")
        if toggles:
            self.set_color_rgb(100, 200, 255)
            self.draw_text(toggle_x, toggle_y, " ".join(toggles))

        # Recording indicator
        if training_manager.recording_state.is_recording:
            self.set_color_rgb(255, 80, 80)
            frames = len(training_manager.recording_state.recorded_inputs)
            self.draw_text(self.width // 2 - 40, self.height - 70, f"REC: {frames}f")
        elif training_manager.recording_state.is_playing:
            self.set_color_rgb(100, 255, 100)
            idx = training_manager.recording_state.playback_index
            total = len(training_manager.recording_state.recorded_inputs)
            self.draw_text(self.width // 2 - 40, self.height - 70, f"PLAY: {idx}/{total}")

        # Frame data overlays for each fighter
        if training_manager.frame_data_overlay.enabled:
            # P1 frame data (left side)
            self.draw_frame_data_overlay(
                game_state.fighters[0],
                10, self.height - 150,
                training_manager
            )
            # P2 frame data (right side)
            self.draw_frame_data_overlay(
                game_state.fighters[1],
                self.width - 130, self.height - 150,
                training_manager
            )

        # Cycle 16: Draw input history for each player
        if training_manager.show_input_history:
            self._draw_input_history(game_state, training_manager)

        # Cycle 16: Draw move list if enabled
        if training_manager.move_list.enabled:
            self._draw_move_list(training_manager)

        # Cycle 16: Draw enhanced hitbox visualization
        self._draw_enhanced_hitboxes(game_state, training_manager)

        # Cycle 18: Draw enhanced combo counter
        self.draw_training_combo_counter(game_state, training_manager)

        # Training mode controls reminder at bottom (updated for Cycle 18)
        self.set_color_rgb(100, 100, 100)
        self.draw_text(10, self.height - 30, "F1:CPU F2:Behavior F3:FrameData F4:Difficulty F5:InfJumps F6:FrameAdv")
        self.draw_text(10, self.height - 15, "F7:InputHist F8:MoveList Tab:HitboxMode R:Dmg T:Pos")

    def _draw_input_history(self, game_state: GameState, training_manager):
        """
        Cycle 16: Draw input history timeline for each player.

        Shows the last 20 inputs as icons in a vertical scrolling list.
        P1's history is displayed on the left side, P2's on the right.
        """
        # P1 input history (left side, near fighter or fixed position)
        p1_history = training_manager.input_history.get_p1_history()
        p1_x = 15
        p1_y = 150  # Below HUD

        # Draw P1 history panel
        self._draw_input_history_panel(p1_x, p1_y, p1_history, "P1 Inputs")

        # P2 input history (right side)
        p2_history = training_manager.input_history.get_p2_history()
        p2_x = self.width - 100
        p2_y = 150

        # Draw P2 history panel
        self._draw_input_history_panel(p2_x, p2_y, p2_history, "P2 Inputs")

    def _draw_input_history_panel(self, x: int, y: int, history, title: str):
        """Draw a single input history panel."""
        panel_width = 85
        panel_height = 200
        entry_height = 16
        max_visible = 12  # How many entries to show

        # Draw panel background
        self.set_color_rgb(20, 20, 40)
        self.draw_rect(x, y, panel_width, panel_height, filled=True)
        self.set_color_rgb(80, 80, 120)
        self.draw_rect(x, y, panel_width, panel_height, filled=False)

        # Draw title
        self.set_color_rgb(150, 150, 200)
        self.draw_text(x + 5, y + 12, title)

        # Draw input entries (most recent at top)
        visible_entries = list(reversed(history[-max_visible:]))

        for i, entry in enumerate(visible_entries):
            entry_y = y + 25 + i * entry_height
            active_inputs = entry.get_active_inputs()

            if active_inputs:
                # Draw input icons
                input_str = " ".join(active_inputs)

                # Color based on input type (directions vs buttons)
                has_direction = any(c in input_str for c in '←→↑↓')
                has_button = any(c in input_str for c in 'JASGZ')

                if has_direction and has_button:
                    self.set_color_rgb(255, 200, 100)  # Orange for combined
                elif has_button:
                    self.set_color_rgb(100, 255, 100)  # Green for buttons
                else:
                    self.set_color_rgb(200, 200, 255)  # Light blue for directions

                self.draw_text(x + 5, entry_y, input_str)

    def _draw_move_list(self, training_manager):
        """
        Cycle 16: Draw move list panel showing all available moves.

        Displays input notation, frame data, and special properties.
        """
        move_list = training_manager.move_list
        if not move_list._cached_moves:
            return

        # Position: center-right of screen
        panel_x = self.width - 280
        panel_y = 80
        panel_width = 265
        row_height = 18
        visible_moves = move_list.get_visible_moves()
        panel_height = 50 + len(visible_moves) * row_height

        # Background
        self.set_color_rgb(25, 25, 45)
        self.draw_rect(panel_x, panel_y, panel_width, panel_height, filled=True)
        self.set_color_rgb(100, 100, 160)
        self.draw_rect(panel_x, panel_y, panel_width, panel_height, filled=False)

        # Title
        char_name = move_list.current_character_type.name if move_list.current_character_type else "???"
        self.set_color('yellow')
        self.draw_text(panel_x + 5, panel_y + 15, f"{char_name} MOVES")

        # Column headers
        self.set_color_rgb(150, 150, 180)
        self.draw_text(panel_x + 5, panel_y + 32, "Move")
        self.draw_text(panel_x + 75, panel_y + 32, "Input")
        self.draw_text(panel_x + 140, panel_y + 32, "Strt")
        self.draw_text(panel_x + 170, panel_y + 32, "Actv")
        self.draw_text(panel_x + 205, panel_y + 32, "Dmg")

        # Draw moves
        for i, move in enumerate(visible_moves):
            row_y = panel_y + 48 + i * row_height

            # Move name (abbreviated)
            name_short = move.name[:8] if len(move.name) > 8 else move.name
            self.set_color('white')
            self.draw_text(panel_x + 5, row_y, name_short)

            # Input notation
            self.set_color_rgb(200, 200, 100)
            self.draw_text(panel_x + 75, row_y, move.input_notation)

            # Startup frames
            self.set_color_rgb(100, 255, 100)
            self.draw_text(panel_x + 140, row_y, f"{move.startup_frames}f")

            # Active frames
            self.set_color_rgb(255, 200, 100)
            self.draw_text(panel_x + 170, row_y, move.active_frames)

            # Damage
            self.set_color_rgb(255, 100, 100)
            self.draw_text(panel_x + 205, row_y, f"{move.damage:.0f}")

            # Notes (if any)
            if move.notes:
                self.set_color_rgb(150, 150, 255)
                self.draw_text(panel_x + 235, row_y, move.notes[:5])

        # Scroll indicators
        if move_list.can_scroll_up():
            self.set_color_rgb(200, 200, 200)
            self.draw_text(panel_x + panel_width - 15, panel_y + 15, "^")
        if move_list.can_scroll_down():
            self.set_color_rgb(200, 200, 200)
            self.draw_text(panel_x + panel_width - 15, panel_y + panel_height - 10, "v")

    def _draw_enhanced_hitboxes(self, game_state: GameState, training_manager):
        """
        Cycle 16: Draw enhanced hitbox visualization with collision markers.

        Features:
        - Distinct colors for hitboxes (red) and hurtboxes (green)
        - Collision point markers (yellow) where boxes intersect
        - Mode cycling: none/hitbox/hurtbox/both/collision
        """
        hitbox_viz = training_manager.hitbox_viz

        # Skip if mode is NONE (use the simple toggle from renderer.show_hitboxes instead)
        if hitbox_viz.mode.name == 'NONE':
            return

        shake_x, shake_y = self.shake_offset

        for fighter in game_state.fighters:
            # Draw hurtboxes if enabled
            if hitbox_viz.show_hurtboxes:
                r, g, b = hitbox_viz.hurtbox_color
                self.set_color_rgb(r, g, b)
                hurtbox = fighter.get_hurtbox_aabb()
                self.draw_rect(
                    int(hurtbox.x) + shake_x, int(hurtbox.y) + shake_y,
                    int(hurtbox.width), int(hurtbox.height),
                    filled=False
                )
                # Draw X through center for visibility
                cx = int(hurtbox.x + hurtbox.width / 2) + shake_x
                cy = int(hurtbox.y + hurtbox.height / 2) + shake_y
                self.draw_line(cx - 5, cy - 5, cx + 5, cy + 5)
                self.draw_line(cx - 5, cy + 5, cx + 5, cy - 5)

            # Draw hitboxes if enabled
            if hitbox_viz.show_hitboxes:
                r, g, b = hitbox_viz.hitbox_color
                self.set_color_rgb(r, g, b)
                for hitbox_aabb in fighter.get_active_hitbox_aabbs():
                    self.draw_rect(
                        int(hitbox_aabb.x) + shake_x, int(hitbox_aabb.y) + shake_y,
                        int(hitbox_aabb.width), int(hitbox_aabb.height),
                        filled=False
                    )
                    # Draw filled rectangle with "X" pattern for attack hitbox
                    cx = int(hitbox_aabb.x + hitbox_aabb.width / 2) + shake_x
                    cy = int(hitbox_aabb.y + hitbox_aabb.height / 2) + shake_y
                    # Draw diagonal lines inside
                    self.draw_line(
                        int(hitbox_aabb.x) + shake_x, int(hitbox_aabb.y) + shake_y,
                        int(hitbox_aabb.x + hitbox_aabb.width) + shake_x,
                        int(hitbox_aabb.y + hitbox_aabb.height) + shake_y
                    )
                    self.draw_line(
                        int(hitbox_aabb.x + hitbox_aabb.width) + shake_x, int(hitbox_aabb.y) + shake_y,
                        int(hitbox_aabb.x) + shake_x,
                        int(hitbox_aabb.y + hitbox_aabb.height) + shake_y
                    )

        # Draw collision points if in COLLISION mode
        if hitbox_viz.show_collision_points:
            r, g, b = hitbox_viz.collision_color
            self.set_color_rgb(r, g, b)
            for cx, cy in hitbox_viz.get_active_collision_points(game_state.frame):
                # Draw a prominent marker at collision point
                icx = int(cx) + shake_x
                icy = int(cy) + shake_y
                # Draw concentric circles
                self.draw_circle(icx, icy, 8, filled=False)
                self.draw_circle(icx, icy, 4, filled=True)
                # Draw crosshair
                self.draw_line(icx - 12, icy, icx + 12, icy)
                self.draw_line(icx, icy - 12, icx, icy + 12)

    def draw_training_combo_counter(self, game_state: GameState, training_manager):
        """
        Cycle 18: Draw enhanced combo counter display for training mode.

        Features:
        - Large centered display above fighters during active combo
        - Total combo damage accumulator
        - Combo drop indicator with fade out
        - Best combo in session displayed in corner
        """
        if not training_manager.show_combo_counter:
            return

        combo_counter = training_manager.combo_counter
        shake_x, shake_y = self.shake_offset

        # Draw combo display for each player
        for player_id in range(2):
            combo_data = combo_counter.get_combo_display(player_id)
            if not combo_data:
                continue

            # Position above the defender (the opponent being comboed)
            defender_id = 1 - player_id
            if defender_id < len(game_state.fighters):
                defender = game_state.fighters[defender_id]
                display_x = int(defender.position.x) + shake_x
                display_y = int(defender.position.y - 100) + shake_y
            else:
                # Fallback to stored position
                display_x = int(combo_data['position'][0]) + shake_x
                display_y = int(combo_data['position'][1] - 80) + shake_y

            # Apply alpha
            alpha = combo_data['alpha']

            # Draw combo count (large)
            count = combo_data['count']
            damage = combo_data['damage']
            is_active = combo_data['is_active']
            is_dropping = combo_data.get('is_dropping', False)

            if is_active:
                # Active combo - color based on count
                if count >= 10:
                    r, g, b = 255, 80, 80     # Red for high combos
                elif count >= 5:
                    r, g, b = 255, 200, 50   # Gold/yellow
                else:
                    r, g, b = 100, 200, 255  # Light blue

                # Pulse effect when close to dropping
                if is_dropping:
                    pulse = int(abs(math.sin(game_state.frame * 0.3)) * 50)
                    r = min(255, r + pulse)
                    g = max(0, g - pulse)
            else:
                # Dropped combo - gray/red
                r, g, b = 200, 100, 100

            # Apply alpha to color
            r = int(r * alpha)
            g = int(g * alpha)
            b = int(b * alpha)

            # Draw background box for visibility
            box_width = 120
            box_height = 50
            box_x = display_x - box_width // 2
            box_y = display_y - 20

            # Semi-transparent background
            bg_alpha = int(30 * alpha)
            self.set_color_rgb(bg_alpha, bg_alpha, bg_alpha + 10)
            self.draw_rect(box_x, box_y, box_width, box_height, filled=True)

            # Draw combo count text
            self.set_color_rgb(r, g, b)

            if is_active:
                # Active combo text
                combo_text = f"{count} HITS!"
                text_x = display_x - len(combo_text) * 4
                self.draw_text(text_x, display_y, combo_text)

                # Damage accumulator (smaller, below)
                dmg_text = f"{damage:.0f}%"
                self.set_color_rgb(int(200 * alpha), int(200 * alpha), int(200 * alpha))
                self.draw_text(display_x - len(dmg_text) * 3, display_y + 18, dmg_text)
            else:
                # Dropped combo text
                drop_text = f"{count}x DROPPED"
                text_x = display_x - len(drop_text) * 4
                self.draw_text(text_x, display_y, drop_text)

                # Final damage
                dmg_text = f"({damage:.0f}%)"
                self.set_color_rgb(int(150 * alpha), int(150 * alpha), int(150 * alpha))
                self.draw_text(display_x - len(dmg_text) * 3, display_y + 18, dmg_text)

        # Draw best combo stats in corner
        self._draw_best_combo_stats(combo_counter)

    def _draw_best_combo_stats(self, combo_counter):
        """Draw best combo achieved in training session."""
        # Position: top-center area
        x = self.width // 2 - 60
        y = 65

        # Only show if there's been a meaningful combo
        p1_best_count, p1_best_dmg = combo_counter.get_best_combo(0)
        p2_best_count, p2_best_dmg = combo_counter.get_best_combo(1)

        if p1_best_count <= 1 and p2_best_count <= 1:
            return

        # Semi-transparent background
        self.set_color_rgb(20, 20, 30)
        self.draw_rect(x - 5, y - 12, 130, 35, filled=True)
        self.set_color_rgb(60, 60, 80)
        self.draw_rect(x - 5, y - 12, 130, 35, filled=False)

        # Title
        self.set_color_rgb(150, 150, 180)
        self.draw_text(x, y - 2, "BEST COMBO")

        # P1 best (if any)
        if p1_best_count > 1:
            self.set_color_rgb(255, 150, 150)
            self.draw_text(x, y + 12, f"P1:{p1_best_count}x/{p1_best_dmg:.0f}%")

        # P2 best (if any)
        if p2_best_count > 1:
            self.set_color_rgb(150, 150, 255)
            self.draw_text(x + 70, y + 12, f"P2:{p2_best_count}x/{p2_best_dmg:.0f}%")

    def _draw_pause_menu(self, game_state: GameState):
        """Draw pause menu overlay (Cycle 4)."""
        center_x = self.width // 2
        center_y = self.height // 2

        # Darkened overlay (draw horizontal lines for semi-transparent effect)
        self.set_color_rgb(20, 20, 30)
        for y_off in range(0, self.height, 2):
            self.draw_line(0, y_off, self.width, y_off)

        # Pause box background
        box_width = 200
        box_height = 150
        box_x = center_x - box_width // 2
        box_y = center_y - box_height // 2

        self.set_color_rgb(40, 40, 60)
        self.draw_rect(box_x, box_y, box_width, box_height, filled=True)

        self.set_color('white')
        self.draw_rect(box_x, box_y, box_width, box_height, filled=False)

        # "PAUSED" text
        self.set_color('yellow')
        self.draw_text(center_x - 30, center_y - 40, "PAUSED")

        # Menu options
        self.set_color('white')
        self.draw_text(center_x - 50, center_y, "Press P to Resume")
        self.draw_text(center_x - 55, center_y + 25, "Press ESC to Quit")

        # Current stats
        self.set_color_rgb(180, 180, 180)
        match_time = game_state.frame / 60.0
        self.draw_text(center_x - 40, center_y + 60, f"Time: {match_time:.1f}s")

    def draw_parallax_background(self, game_state: GameState):
        """
        Draw parallax background layers based on camera position.

        Cycle 14: Stage Background Parallax Scrolling
        - Far layers move slowly (high depth value)
        - Near layers move faster (low depth value)
        - Camera position is based on fighters' average position
        """
        stage = game_state.stage

        if not stage.parallax_layers:
            return

        # Calculate camera center based on fighters' average position
        # This creates a simple camera that follows the action
        avg_x = sum(f.position.x for f in game_state.fighters) / len(game_state.fighters)
        avg_y = sum(f.position.y for f in game_state.fighters) / len(game_state.fighters)

        # Camera offset from screen center
        screen_center_x = self.width / 2
        screen_center_y = self.height / 2
        camera_offset_x = (avg_x - screen_center_x) * 0.1  # Scale down for subtler effect
        camera_offset_y = (avg_y - screen_center_y) * 0.05  # Less vertical movement

        # Draw each parallax layer (back to front)
        for layer in stage.parallax_layers:
            # Calculate parallax offset
            # High depth = moves less (far background)
            # Low depth = moves more (near foreground)
            parallax_factor = 1.0 - layer.depth
            offset_x = camera_offset_x * parallax_factor
            offset_y = camera_offset_y * parallax_factor

            # Apply screen shake (reduced for far layers)
            shake_x = int(self.shake_offset[0] * parallax_factor)
            shake_y = int(self.shake_offset[1] * parallax_factor)

            # Set layer color
            self.set_color_rgb(*layer.color)

            # Draw each element in the layer
            for elem in layer.elements:
                elem_x = elem['x'] - offset_x + shake_x
                elem_y = elem['y'] - offset_y + shake_y

                if elem['type'] == 'rect':
                    self.draw_rect(
                        int(elem_x), int(elem_y),
                        elem['width'], elem['height'],
                        filled=True
                    )
                elif elem['type'] == 'circle':
                    self.draw_circle(
                        int(elem_x), int(elem_y),
                        elem['radius'],
                        filled=True
                    )
                elif elem['type'] == 'triangle':
                    # Draw a triangle (mountain shape)
                    base_x = int(elem_x)
                    base_y = int(elem_y)
                    width = elem['width']
                    height = elem['height']
                    # Draw mountain as filled polygon (using lines for X11)
                    # Fill by drawing horizontal lines
                    for h in range(height):
                        progress = h / height
                        line_width = int(width * (1.0 - progress * 0.8))
                        center_offset = (width - line_width) // 2
                        y_pos = base_y - h
                        x_start = base_x + center_offset
                        self.draw_line(x_start, y_pos, x_start + line_width, y_pos)
                elif elem['type'] == 'line':
                    self.draw_line(
                        int(elem_x), int(elem_y),
                        int(elem_x + elem.get('x2', 0)),
                        int(elem_y + elem.get('y2', 0))
                    )

    def render(self, game_state: GameState):
        """Render the complete game state with visual effects."""
        self.frame += 1
        if self.themed:
            if not game_state.paused:
                self.update_effects()
            self.theme.render_match(game_state)
            if game_state.paused:
                self._draw_pause_menu(game_state)
            self.flip()
            return
        self._render_classic(game_state)

    def _render_classic(self, game_state: GameState):
        """Original core-X11 renderer."""
        # Update visual effects first (only when not paused)
        if not game_state.paused:
            self.update_effects()

        self.clear()

        # Cycle 14: Draw parallax background layers FIRST (behind everything)
        self.draw_parallax_background(game_state)

        # Draw stage background elements
        stage = game_state.stage

        # Draw blast zone indicators (subtle) - apply shake
        shake_x, shake_y = self.shake_offset
        self.set_color('dark_gray')
        bz = stage.blast_zones
        # Left blast zone line
        self.draw_line(int(bz['left']) + shake_x, 0, int(bz['left']) + shake_x, self.height)
        # Right blast zone line
        self.draw_line(int(bz['right']) + shake_x, 0, int(bz['right']) + shake_x, self.height)
        # Top blast zone line
        self.draw_line(0, int(bz['top']) + shake_y, self.width, int(bz['top']) + shake_y)
        # Bottom blast zone line
        self.draw_line(0, int(bz['bottom']) + shake_y, self.width, int(bz['bottom']) + shake_y)

        # Draw main platform (with shake)
        self.draw_platform(stage.main_platform)

        # Draw other platforms
        for platform in stage.platforms:
            self.draw_platform(platform)

        # Draw dodge afterimages BEFORE fighters (Cycle 3)
        for fighter in game_state.fighters:
            self.draw_dodge_afterimages(fighter)

        # Draw knockback trails BEFORE fighters (Cycle 3)
        for fighter in game_state.fighters:
            self.draw_knockback_trails(fighter)

        # Draw fighters
        # Cycle 15: Pass hit freeze state for vibration effect
        hit_freeze_active = game_state.hit_freeze_frames > 0
        hit_freeze_frame = game_state.hit_freeze_frames

        for fighter in game_state.fighters:
            self.draw_invincibility_glow(fighter)  # Cycle 10: Dodge i-frame glow
            self.draw_stick_figure(fighter, hit_freeze_active=hit_freeze_active,
                                   hit_freeze_frame=hit_freeze_frame)
            self.draw_shield(fighter)  # Draw shield if shielding
            self.draw_counter_aura(fighter)  # Cycle 3: Counter stance
            self.draw_reflector_active(fighter)  # Cycle 3: Reflector

        # Draw projectiles (with trails - Cycle 3)
        for projectile in game_state.projectiles:
            self.draw_projectile(projectile)

        # Cleanup stale projectile trails
        self.cleanup_projectile_trails(game_state.projectiles)

        # Draw visual effects ON TOP (Cycle 3)
        self.draw_all_effects()

        # Draw HUD (not affected by shake)
        self.draw_hud(game_state)

        # Match over message with improved visuals
        if game_state.match_over:
            self._draw_victory_screen(game_state)

        # Cycle 4: Pause menu overlay
        if game_state.paused:
            self._draw_pause_menu(game_state)

        # Flip buffer
        self.flip()

    def process_events(self) -> List[Tuple[int, int, bool]]:
        """
        Process X11 events and return key events.

        Returns list of (keycode, keysym, is_pressed) tuples.
        """
        events = []

        while _x11.XPending(self.display) > 0:
            event = XEvent()
            _x11.XNextEvent(self.display, byref(event))

            if event.type == KeyPress or event.type == KeyRelease:
                key_event = cast(byref(event), POINTER(XKeyEvent)).contents
                keysym = _x11.XKeycodeToKeysym(self.display, key_event.keycode, 0)
                is_pressed = event.type == KeyPress
                events.append((key_event.keycode, keysym, is_pressed))

            elif event.type == ClientMessage:
                # Window close
                self.should_close = True

            elif event.type == Expose:
                # Redraw needed
                pass

        return events

    def close(self):
        """Clean up X11 resources. Safe to call more than once."""
        if not getattr(self, 'display', None):
            return
        if getattr(self, 'buffer', None):
            _x11.XFreePixmap(self.display, self.buffer)
            self.buffer = None
        if getattr(self, 'gc', None):
            _x11.XFreeGC(self.display, self.gc)
            self.gc = None
        if getattr(self, 'window', None):
            _x11.XDestroyWindow(self.display, self.window)
            self.window = None
        _x11.XCloseDisplay(self.display)
        self.display = None

    def __del__(self):
        try:
            self.close()
        except:
            pass


# =============================================================================
# KEY MAPPING
# =============================================================================

# X11 KeySyms for common keys
XK_Left = 0xff51
XK_Up = 0xff52
XK_Right = 0xff53
XK_Down = 0xff54
XK_space = 0x0020
XK_Return = 0xff0d
XK_Escape = 0xff1b
XK_Shift_L = 0xffe1
XK_Shift_R = 0xffe2
XK_Control_L = 0xffe3
XK_Control_R = 0xffe4

# Letter keys (lowercase)
XK_a = 0x0061
XK_b = 0x0062
XK_c = 0x0063
XK_d = 0x0064
XK_e = 0x0065
XK_f = 0x0066
XK_g = 0x0067
XK_h = 0x0068
XK_i = 0x0069
XK_j = 0x006a
XK_k = 0x006b
XK_l = 0x006c
XK_m = 0x006d
XK_n = 0x006e
XK_o = 0x006f
XK_p = 0x0070
XK_q = 0x0071
XK_r = 0x0072
XK_s = 0x0073
XK_t = 0x0074
XK_u = 0x0075
XK_v = 0x0076
XK_w = 0x0077
XK_x = 0x0078
XK_y = 0x0079
XK_z = 0x007a


# Default key mappings for 2 players
DEFAULT_KEY_MAP_P1 = {
    XK_a: 'left',
    XK_d: 'right',
    XK_w: 'up',
    XK_s: 'down',
    XK_space: 'jump',
    XK_j: 'attack',
    XK_k: 'special',
    XK_l: 'shield',
    XK_u: 'grab',
}

DEFAULT_KEY_MAP_P2 = {
    XK_Left: 'left',
    XK_Right: 'right',
    XK_Up: 'up',
    XK_Down: 'down',
    0xff8d: 'jump',  # KP_Enter
    0xffb1: 'attack',  # KP_1
    0xffb2: 'special',  # KP_2
    0xffb3: 'shield',  # KP_3
    0xffb0: 'grab',  # KP_0
}

# Alternative P2 using numpad or IJKL
DEFAULT_KEY_MAP_P2_ALT = {
    XK_Left: 'left',
    XK_Right: 'right',
    XK_Up: 'up',
    XK_Down: 'down',
    XK_Return: 'jump',
    0xffb1: 'attack',  # Numpad 1
    0xffb2: 'special',  # Numpad 2
    0xffb3: 'shield',  # Numpad 3
    XK_Shift_R: 'grab',
}


if __name__ == "__main__":
    # Quick visual test
    from game_engine import create_game, GameLoop

    print("Creating renderer...")
    renderer = X11Renderer(display_name=":99", title="Stick Figure Fighter Test")
    renderer.show_hitboxes = True
    renderer.show_hurtboxes = True

    print("Creating game...")
    game = create_game()

    print("Running render test...")
    import time

    # Simulate a few frames
    for frame in range(180):  # 3 seconds
        # Move fighters around for testing
        p1 = game.fighters[0]
        p2 = game.fighters[1]

        # P1 walks right
        if frame < 60:
            p1.velocity.x = 3
            p1.state = FighterState.WALKING
        elif frame < 90:
            p1.velocity.y = -12
            p1.grounded = False
            p1.state = FighterState.JUMPING
        elif frame < 120:
            p1.state = FighterState.ATTACKING
        else:
            p1.state = FighterState.IDLE
            p1.velocity.x = 0

        # Apply simple physics
        for f in game.fighters:
            if not f.grounded:
                f.velocity.y += 0.6
            f.position.x += f.velocity.x
            f.position.y += f.velocity.y

            # Simple ground check
            if f.position.y > 550:
                f.position.y = 550
                f.velocity.y = 0
                f.grounded = True

        game.frame = frame

        # Render
        renderer.render(game)

        # Process events
        events = renderer.process_events()
        if renderer.should_close:
            break

        time.sleep(1/60)

    renderer.close()
    print("Test complete!")
