"""
Low-level X11 bindings via ctypes for screen capture and input injection.
Provides direct access to libX11, libXext (SHM), and libXtst (XTest).
"""
import ctypes
import ctypes.util
from typing import Optional, Tuple

# ==================== LOAD LIBRARIES ====================

def _load_library(name: str) -> ctypes.CDLL:
    """Load a shared library by name."""
    path = ctypes.util.find_library(name)
    if path is None:
        raise OSError(f"Library '{name}' not found. Install it with: sudo apt install libx11-dev libxext-dev libxtst-dev")
    return ctypes.CDLL(path)

# Load X11 libraries
try:
    libX11 = _load_library('X11')
    libXext = _load_library('Xext')
    libXtst = _load_library('Xtst')
except OSError as e:
    raise ImportError(f"Failed to load X11 libraries: {e}")


# ==================== TYPE DEFINITIONS ====================

# Basic X11 types
Display = ctypes.c_void_p
Window = ctypes.c_ulong
Drawable = ctypes.c_ulong
Pixmap = ctypes.c_ulong
GC = ctypes.c_void_p
XID = ctypes.c_ulong
Time = ctypes.c_ulong
Bool = ctypes.c_int
Status = ctypes.c_int
KeyCode = ctypes.c_ubyte
KeySym = ctypes.c_ulong

# SHM types
ShmSeg = ctypes.c_ulong


class XWindowAttributes(ctypes.Structure):
    """X11 window attributes structure."""
    _fields_ = [
        ('x', ctypes.c_int),
        ('y', ctypes.c_int),
        ('width', ctypes.c_int),
        ('height', ctypes.c_int),
        ('border_width', ctypes.c_int),
        ('depth', ctypes.c_int),
        ('visual', ctypes.c_void_p),
        ('root', Window),
        ('class_', ctypes.c_int),
        ('bit_gravity', ctypes.c_int),
        ('win_gravity', ctypes.c_int),
        ('backing_store', ctypes.c_int),
        ('backing_planes', ctypes.c_ulong),
        ('backing_pixel', ctypes.c_ulong),
        ('save_under', Bool),
        ('colormap', ctypes.c_ulong),
        ('map_installed', Bool),
        ('map_state', ctypes.c_int),
        ('all_event_masks', ctypes.c_long),
        ('your_event_mask', ctypes.c_long),
        ('do_not_propagate_mask', ctypes.c_long),
        ('override_redirect', Bool),
        ('screen', ctypes.c_void_p),
    ]


class XImage(ctypes.Structure):
    """X11 image structure."""
    _fields_ = [
        ('width', ctypes.c_int),
        ('height', ctypes.c_int),
        ('xoffset', ctypes.c_int),
        ('format', ctypes.c_int),
        ('data', ctypes.c_char_p),
        ('byte_order', ctypes.c_int),
        ('bitmap_unit', ctypes.c_int),
        ('bitmap_bit_order', ctypes.c_int),
        ('bitmap_pad', ctypes.c_int),
        ('depth', ctypes.c_int),
        ('bytes_per_line', ctypes.c_int),
        ('bits_per_pixel', ctypes.c_int),
        ('red_mask', ctypes.c_ulong),
        ('green_mask', ctypes.c_ulong),
        ('blue_mask', ctypes.c_ulong),
        ('obdata', ctypes.c_void_p),
        # Function pointers (we won't use these directly)
        ('f_create_image', ctypes.c_void_p),
        ('f_destroy_image', ctypes.c_void_p),
        ('f_get_pixel', ctypes.c_void_p),
        ('f_put_pixel', ctypes.c_void_p),
        ('f_sub_image', ctypes.c_void_p),
        ('f_add_pixel', ctypes.c_void_p),
    ]


class XShmSegmentInfo(ctypes.Structure):
    """X11 SHM segment info structure."""
    _fields_ = [
        ('shmseg', ShmSeg),
        ('shmid', ctypes.c_int),
        ('shmaddr', ctypes.c_void_p),  # Must be c_void_p, not c_char_p
        ('readOnly', Bool),
    ]


# ==================== FUNCTION SIGNATURES ====================

# XOpenDisplay
libX11.XOpenDisplay.argtypes = [ctypes.c_char_p]
libX11.XOpenDisplay.restype = Display

# XCloseDisplay
libX11.XCloseDisplay.argtypes = [Display]
libX11.XCloseDisplay.restype = ctypes.c_int

# XDefaultRootWindow
libX11.XDefaultRootWindow.argtypes = [Display]
libX11.XDefaultRootWindow.restype = Window

# XGetWindowAttributes
libX11.XGetWindowAttributes.argtypes = [Display, Window, ctypes.POINTER(XWindowAttributes)]
libX11.XGetWindowAttributes.restype = Status

# XGetImage
libX11.XGetImage.argtypes = [Display, Drawable, ctypes.c_int, ctypes.c_int,
                            ctypes.c_uint, ctypes.c_uint, ctypes.c_ulong, ctypes.c_int]
libX11.XGetImage.restype = ctypes.POINTER(XImage)

# XDestroyImage
libX11.XDestroyImage.argtypes = [ctypes.POINTER(XImage)]
libX11.XDestroyImage.restype = ctypes.c_int

# XFlush
libX11.XFlush.argtypes = [Display]
libX11.XFlush.restype = ctypes.c_int

# XSync
libX11.XSync.argtypes = [Display, Bool]
libX11.XSync.restype = ctypes.c_int

# XKeysymToKeycode
libX11.XKeysymToKeycode.argtypes = [Display, KeySym]
libX11.XKeysymToKeycode.restype = KeyCode

# XStringToKeysym
libX11.XStringToKeysym.argtypes = [ctypes.c_char_p]
libX11.XStringToKeysym.restype = KeySym

# XDefaultScreen
libX11.XDefaultScreen.argtypes = [Display]
libX11.XDefaultScreen.restype = ctypes.c_int

# XDisplayWidth / XDisplayHeight
libX11.XDisplayWidth.argtypes = [Display, ctypes.c_int]
libX11.XDisplayWidth.restype = ctypes.c_int
libX11.XDisplayHeight.argtypes = [Display, ctypes.c_int]
libX11.XDisplayHeight.restype = ctypes.c_int

# XQueryPointer
libX11.XQueryPointer.argtypes = [Display, Window,
                                 ctypes.POINTER(Window), ctypes.POINTER(Window),
                                 ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
                                 ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
                                 ctypes.POINTER(ctypes.c_uint)]
libX11.XQueryPointer.restype = Bool

# ==================== SHM EXTENSION ====================

# XShmQueryExtension
libXext.XShmQueryExtension.argtypes = [Display]
libXext.XShmQueryExtension.restype = Bool

# XShmAttach
libXext.XShmAttach.argtypes = [Display, ctypes.POINTER(XShmSegmentInfo)]
libXext.XShmAttach.restype = Bool

# XShmDetach
libXext.XShmDetach.argtypes = [Display, ctypes.POINTER(XShmSegmentInfo)]
libXext.XShmDetach.restype = Bool

# XShmCreateImage
libXext.XShmCreateImage.argtypes = [Display, ctypes.c_void_p, ctypes.c_uint,
                                    ctypes.c_int, ctypes.c_char_p,
                                    ctypes.POINTER(XShmSegmentInfo),
                                    ctypes.c_uint, ctypes.c_uint]
libXext.XShmCreateImage.restype = ctypes.POINTER(XImage)

# XShmGetImage
libXext.XShmGetImage.argtypes = [Display, Drawable, ctypes.POINTER(XImage),
                                 ctypes.c_int, ctypes.c_int, ctypes.c_ulong]
libXext.XShmGetImage.restype = Bool

# XShmPutImage
libXext.XShmPutImage.argtypes = [Display, Drawable, GC, ctypes.POINTER(XImage),
                                 ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                 ctypes.c_uint, ctypes.c_uint, Bool]
libXext.XShmPutImage.restype = Bool

# ==================== XTEST EXTENSION ====================

# XTestQueryExtension
libXtst.XTestQueryExtension.argtypes = [Display, ctypes.POINTER(ctypes.c_int),
                                        ctypes.POINTER(ctypes.c_int),
                                        ctypes.POINTER(ctypes.c_int),
                                        ctypes.POINTER(ctypes.c_int)]
libXtst.XTestQueryExtension.restype = Bool

# XTestFakeKeyEvent
libXtst.XTestFakeKeyEvent.argtypes = [Display, ctypes.c_uint, Bool, ctypes.c_ulong]
libXtst.XTestFakeKeyEvent.restype = ctypes.c_int

# XTestFakeButtonEvent
libXtst.XTestFakeButtonEvent.argtypes = [Display, ctypes.c_uint, Bool, ctypes.c_ulong]
libXtst.XTestFakeButtonEvent.restype = ctypes.c_int

# XTestFakeMotionEvent
libXtst.XTestFakeMotionEvent.argtypes = [Display, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_ulong]
libXtst.XTestFakeMotionEvent.restype = ctypes.c_int

# XTestFakeRelativeMotionEvent
libXtst.XTestFakeRelativeMotionEvent.argtypes = [Display, ctypes.c_int, ctypes.c_int, ctypes.c_ulong]
libXtst.XTestFakeRelativeMotionEvent.restype = ctypes.c_int


# ==================== SHM (POSIX) ====================

# libc for shmget, shmat, shmdt, shmctl
libc = ctypes.CDLL(None)

# shmget
libc.shmget.argtypes = [ctypes.c_int, ctypes.c_size_t, ctypes.c_int]
libc.shmget.restype = ctypes.c_int

# shmat
libc.shmat.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
libc.shmat.restype = ctypes.c_void_p

# shmdt
libc.shmdt.argtypes = [ctypes.c_void_p]
libc.shmdt.restype = ctypes.c_int

# shmctl
libc.shmctl.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
libc.shmctl.restype = ctypes.c_int

# Constants
IPC_PRIVATE = 0
IPC_CREAT = 0o1000
IPC_RMID = 0

# ==================== HIGH-LEVEL WRAPPERS ====================

class X11Display:
    """
    High-level wrapper for X11 display operations.

    Manages display connection and provides utility methods for
    window queries and basic operations.
    """

    def __init__(self, display_name: Optional[str] = None):
        """
        Open a connection to the X server.

        Args:
            display_name: X display name (e.g., ':0'), or None for default
        """
        name = display_name.encode() if display_name else None
        self.display = libX11.XOpenDisplay(name)
        if not self.display:
            raise RuntimeError(f"Failed to open X display: {display_name or 'default'}")

        self.screen = libX11.XDefaultScreen(self.display)
        self.root = libX11.XDefaultRootWindow(self.display)

    def close(self):
        """Close the display connection."""
        if self.display:
            libX11.XCloseDisplay(self.display)
            self.display = None

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def get_screen_size(self) -> Tuple[int, int]:
        """Get the screen dimensions."""
        width = libX11.XDisplayWidth(self.display, self.screen)
        height = libX11.XDisplayHeight(self.display, self.screen)
        return width, height

    def get_window_attributes(self, window: int) -> XWindowAttributes:
        """Get attributes of a window."""
        attrs = XWindowAttributes()
        status = libX11.XGetWindowAttributes(self.display, window, ctypes.byref(attrs))
        if not status:
            raise RuntimeError(f"Failed to get window attributes for window {window}")
        return attrs

    def get_mouse_position(self) -> Tuple[int, int]:
        """Get current mouse position."""
        root_return = Window()
        child_return = Window()
        root_x = ctypes.c_int()
        root_y = ctypes.c_int()
        win_x = ctypes.c_int()
        win_y = ctypes.c_int()
        mask = ctypes.c_uint()

        libX11.XQueryPointer(self.display, self.root,
                            ctypes.byref(root_return), ctypes.byref(child_return),
                            ctypes.byref(root_x), ctypes.byref(root_y),
                            ctypes.byref(win_x), ctypes.byref(win_y),
                            ctypes.byref(mask))
        return root_x.value, root_y.value

    def flush(self):
        """Flush the output buffer."""
        libX11.XFlush(self.display)

    def sync(self, discard: bool = False):
        """Synchronize with the X server."""
        libX11.XSync(self.display, discard)

    def keysym_to_keycode(self, keysym: int) -> int:
        """Convert a keysym to a keycode."""
        return libX11.XKeysymToKeycode(self.display, keysym)

    def string_to_keysym(self, string: str) -> int:
        """Convert a string to a keysym."""
        return libX11.XStringToKeysym(string.encode())


class XShmExtension:
    """
    Wrapper for X11 SHM (Shared Memory) extension.

    Provides fast screen capture using shared memory between
    the X server and the client.
    """

    def __init__(self, x11_display: X11Display):
        """
        Initialize SHM extension.

        Args:
            x11_display: An open X11Display instance
        """
        self.x11 = x11_display
        self.display = x11_display.display

        # Check if SHM is available
        if not libXext.XShmQueryExtension(self.display):
            raise RuntimeError("X11 SHM extension not available")

        self.shm_info = None
        self.image = None
        self.width = 0
        self.height = 0
        self._shmaddr_raw = None

    def create_image(self, width: int, height: int, depth: int = None, visual: ctypes.c_void_p = None):
        """
        Create a shared memory image for capturing.

        Args:
            width: Image width
            height: Image height
            depth: Color depth (None to auto-detect from root window)
            visual: Visual pointer (None to auto-detect)
        """
        self.width = width
        self.height = height

        # Get depth and visual from root window if not provided
        if depth is None or visual is None:
            attrs = self.x11.get_window_attributes(self.x11.root)
            if depth is None:
                depth = attrs.depth
            if visual is None:
                visual = attrs.visual

        # Create SHM segment info
        self.shm_info = XShmSegmentInfo()

        # Calculate required size (4 bytes per pixel for BGRA on 24/32 bit displays)
        size = width * height * 4

        # Create shared memory segment FIRST
        self.shm_info.shmid = libc.shmget(IPC_PRIVATE, size, IPC_CREAT | 0o777)
        if self.shm_info.shmid < 0:
            raise RuntimeError("Failed to create shared memory segment")

        # Attach shared memory
        shmaddr = libc.shmat(self.shm_info.shmid, None, 0)
        if shmaddr is None or shmaddr == 0xffffffffffffffff or shmaddr == 0xffffffff:
            libc.shmctl(self.shm_info.shmid, IPC_RMID, None)
            raise RuntimeError("Failed to attach shared memory")

        self._shmaddr_raw = shmaddr
        self.shm_info.shmaddr = shmaddr  # Store as c_void_p (raw integer address)
        self.shm_info.readOnly = False

        # Create XImage structure
        # ZPixmap format = 2
        self.image = libXext.XShmCreateImage(
            self.display,
            visual,
            depth,
            2,  # ZPixmap
            None,  # data (will be set from shm)
            ctypes.byref(self.shm_info),
            width,
            height
        )

        if not self.image:
            libc.shmdt(shmaddr)
            libc.shmctl(self.shm_info.shmid, IPC_RMID, None)
            raise RuntimeError("Failed to create SHM image")

        # Set the data pointer - cast shmaddr to char pointer for XImage
        self.image.contents.data = ctypes.cast(shmaddr, ctypes.c_char_p)

        # Attach to X server
        if not libXext.XShmAttach(self.display, ctypes.byref(self.shm_info)):
            libc.shmdt(shmaddr)
            libc.shmctl(self.shm_info.shmid, IPC_RMID, None)
            raise RuntimeError("Failed to attach SHM to X server")

        # Sync to ensure attachment is complete
        self.x11.sync()

        # Mark segment for removal after detach
        libc.shmctl(self.shm_info.shmid, IPC_RMID, None)

    def capture(self, drawable: int = None, x: int = 0, y: int = 0) -> bytes:
        """
        Capture the screen using SHM.

        Args:
            drawable: Window/pixmap to capture (default: root window)
            x, y: Capture offset

        Returns:
            Raw pixel data as bytes (BGRA format)
        """
        if drawable is None:
            drawable = self.x11.root

        # Capture to SHM image
        success = libXext.XShmGetImage(
            self.display,
            drawable,
            self.image,
            x, y,
            0xFFFFFFFF  # AllPlanes
        )

        if not success:
            raise RuntimeError("Failed to capture screen with SHM")

        # Copy data from shared memory
        size = self.width * self.height * 4
        return ctypes.string_at(self.shm_info.shmaddr, size)

    def cleanup(self):
        """Release SHM resources."""
        if self.shm_info is not None:
            try:
                libXext.XShmDetach(self.display, ctypes.byref(self.shm_info))
            except Exception:
                pass

            if self._shmaddr_raw:
                libc.shmdt(self._shmaddr_raw)
                self._shmaddr_raw = None

            self.shm_info = None

        if self.image:
            # Don't call XDestroyImage as it will try to free the shm data
            self.image = None

    def __del__(self):
        self.cleanup()


class XTestExtension:
    """
    Wrapper for X11 XTest extension.

    Provides keyboard and mouse input simulation.
    """

    def __init__(self, x11_display: X11Display):
        """
        Initialize XTest extension.

        Args:
            x11_display: An open X11Display instance
        """
        self.x11 = x11_display
        self.display = x11_display.display

        # Check if XTest is available
        event_base = ctypes.c_int()
        error_base = ctypes.c_int()
        major = ctypes.c_int()
        minor = ctypes.c_int()

        if not libXtst.XTestQueryExtension(self.display,
                                           ctypes.byref(event_base),
                                           ctypes.byref(error_base),
                                           ctypes.byref(major),
                                           ctypes.byref(minor)):
            raise RuntimeError("X11 XTest extension not available")

        # Keysym mapping for common keys
        self._keysym_map = {
            'a': 0x61, 'b': 0x62, 'c': 0x63, 'd': 0x64, 'e': 0x65,
            'f': 0x66, 'g': 0x67, 'h': 0x68, 'i': 0x69, 'j': 0x6a,
            'k': 0x6b, 'l': 0x6c, 'm': 0x6d, 'n': 0x6e, 'o': 0x6f,
            'p': 0x70, 'q': 0x71, 'r': 0x72, 's': 0x73, 't': 0x74,
            'u': 0x75, 'v': 0x76, 'w': 0x77, 'x': 0x78, 'y': 0x79,
            'z': 0x7a,
            '0': 0x30, '1': 0x31, '2': 0x32, '3': 0x33, '4': 0x34,
            '5': 0x35, '6': 0x36, '7': 0x37, '8': 0x38, '9': 0x39,
            'space': 0x20, 'Return': 0xff0d, 'Escape': 0xff1b,
            'Tab': 0xff09, 'BackSpace': 0xff08,
            'Up': 0xff52, 'Down': 0xff54, 'Left': 0xff51, 'Right': 0xff53,
            'Shift_L': 0xffe1, 'Shift_R': 0xffe2,
            'Control_L': 0xffe3, 'Control_R': 0xffe4,
            'Alt_L': 0xffe9, 'Alt_R': 0xffea,
        }

    def key_press(self, key: str):
        """
        Press a key down.

        Args:
            key: Key name (e.g., 'a', 'space', 'Return')
        """
        keysym = self._get_keysym(key)
        keycode = self.x11.keysym_to_keycode(keysym)
        libXtst.XTestFakeKeyEvent(self.display, keycode, True, 0)
        self.x11.flush()

    def key_release(self, key: str):
        """
        Release a key.

        Args:
            key: Key name
        """
        keysym = self._get_keysym(key)
        keycode = self.x11.keysym_to_keycode(keysym)
        libXtst.XTestFakeKeyEvent(self.display, keycode, False, 0)
        self.x11.flush()

    def key_tap(self, key: str, delay_ms: int = 10):
        """
        Press and release a key.

        Args:
            key: Key name
            delay_ms: Delay between press and release (not enforced here)
        """
        self.key_press(key)
        self.key_release(key)

    def button_press(self, button: int):
        """
        Press a mouse button.

        Args:
            button: Button number (1=left, 2=middle, 3=right)
        """
        libXtst.XTestFakeButtonEvent(self.display, button, True, 0)
        self.x11.flush()

    def button_release(self, button: int):
        """
        Release a mouse button.

        Args:
            button: Button number
        """
        libXtst.XTestFakeButtonEvent(self.display, button, False, 0)
        self.x11.flush()

    def click(self, button: int = 1):
        """
        Click a mouse button.

        Args:
            button: Button number (default: left)
        """
        self.button_press(button)
        self.button_release(button)

    def move_mouse(self, x: int, y: int, screen: int = -1):
        """
        Move mouse to absolute position.

        Args:
            x, y: Target coordinates
            screen: Screen number (-1 for current)
        """
        libXtst.XTestFakeMotionEvent(self.display, screen, x, y, 0)
        self.x11.flush()

    def move_mouse_relative(self, dx: int, dy: int):
        """
        Move mouse by relative amount.

        Args:
            dx, dy: Movement delta
        """
        libXtst.XTestFakeRelativeMotionEvent(self.display, dx, dy, 0)
        self.x11.flush()

    def _get_keysym(self, key: str) -> int:
        """Convert key name to keysym."""
        if key in self._keysym_map:
            return self._keysym_map[key]
        # Try XStringToKeysym
        keysym = self.x11.string_to_keysym(key)
        if keysym == 0:
            raise ValueError(f"Unknown key: {key}")
        return keysym
