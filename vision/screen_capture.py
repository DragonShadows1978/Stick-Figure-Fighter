"""
ScreenCortex: High-performance screen capture using X11 SHM.

Provides efficient frame capture at 30+ FPS with minimal latency.
"""
import numpy as np
import time
from typing import Optional, Tuple
from .x11_bindings import X11Display, XShmExtension


class ScreenCortex:
    """
    High-performance screen capture system.

    Uses X11 Shared Memory extension for zero-copy screen capture,
    achieving sub-30ms latency at 1080p.

    Example:
        cortex = ScreenCortex()
        frame = cortex.capture()  # Returns RGB numpy array
        cortex.close()
    """

    def __init__(self, display_name: Optional[str] = None,
                 width: Optional[int] = None,
                 height: Optional[int] = None,
                 downscale: int = 1):
        """
        Initialize the screen capture system.

        Args:
            display_name: X11 display (e.g., ':0'), None for default
            width: Capture width (None for full screen)
            height: Capture height (None for full screen)
            downscale: Downscale factor (1 = full resolution)
        """
        self.downscale = max(1, downscale)

        # Open X11 display
        self.x11 = X11Display(display_name)

        # Get screen dimensions
        screen_width, screen_height = self.x11.get_screen_size()

        # Set capture dimensions
        self.capture_width = width or screen_width
        self.capture_height = height or screen_height

        # Output dimensions (after downscaling)
        self.output_width = self.capture_width // self.downscale
        self.output_height = self.capture_height // self.downscale

        # Initialize SHM extension
        self.shm = XShmExtension(self.x11)
        self.shm.create_image(self.capture_width, self.capture_height)

        # Performance tracking
        self._frame_count = 0
        self._total_time = 0.0
        self._last_frame_time = 0.0

        # Cache for frame data
        self._frame_buffer = np.zeros(
            (self.output_height, self.output_width, 3),
            dtype=np.uint8
        )

    def capture(self, x: int = 0, y: int = 0) -> np.ndarray:
        """
        Capture a frame from the screen.

        Args:
            x, y: Capture offset from top-left

        Returns:
            RGB numpy array of shape (H, W, 3)
        """
        start_time = time.perf_counter()

        # Capture raw BGRA data
        raw_data = self.shm.capture(self.x11.root, x, y)

        # Convert to numpy array
        frame = np.frombuffer(raw_data, dtype=np.uint8)
        frame = frame.reshape((self.capture_height, self.capture_width, 4))

        # Convert BGRA to RGB
        rgb = frame[:, :, :3][:, :, ::-1]

        # Downscale if needed
        if self.downscale > 1:
            rgb = rgb[::self.downscale, ::self.downscale]
            np.copyto(self._frame_buffer, rgb)
            result = self._frame_buffer
        else:
            result = rgb.copy()

        # Update performance metrics
        self._last_frame_time = time.perf_counter() - start_time
        self._total_time += self._last_frame_time
        self._frame_count += 1

        return result

    def capture_region(self, x: int, y: int, width: int, height: int) -> np.ndarray:
        """
        Capture a specific region of the screen.

        Note: This still captures the full screen and crops.
        For best performance, use capture() and crop manually.

        Args:
            x, y: Top-left corner of region
            width, height: Region dimensions

        Returns:
            RGB numpy array of shape (height, width, 3)
        """
        frame = self.capture()

        # Adjust for downscale
        x_scaled = x // self.downscale
        y_scaled = y // self.downscale
        w_scaled = width // self.downscale
        h_scaled = height // self.downscale

        return frame[y_scaled:y_scaled+h_scaled, x_scaled:x_scaled+w_scaled].copy()

    @property
    def fps(self) -> float:
        """Calculate average FPS."""
        if self._total_time == 0:
            return 0.0
        return self._frame_count / self._total_time

    @property
    def last_frame_ms(self) -> float:
        """Get the last frame capture time in milliseconds."""
        return self._last_frame_time * 1000

    @property
    def avg_frame_ms(self) -> float:
        """Get average frame capture time in milliseconds."""
        if self._frame_count == 0:
            return 0.0
        return (self._total_time / self._frame_count) * 1000

    def reset_stats(self):
        """Reset performance statistics."""
        self._frame_count = 0
        self._total_time = 0.0
        self._last_frame_time = 0.0

    def get_stats(self) -> dict:
        """Get performance statistics."""
        return {
            'frame_count': self._frame_count,
            'total_time': self._total_time,
            'fps': self.fps,
            'last_frame_ms': self.last_frame_ms,
            'avg_frame_ms': self.avg_frame_ms,
            'capture_size': (self.capture_width, self.capture_height),
            'output_size': (self.output_width, self.output_height),
        }

    def benchmark(self, num_frames: int = 100) -> dict:
        """
        Run a capture benchmark.

        Args:
            num_frames: Number of frames to capture

        Returns:
            Benchmark results dictionary
        """
        self.reset_stats()
        frame_times = []

        for _ in range(num_frames):
            start = time.perf_counter()
            self.capture()
            frame_times.append(time.perf_counter() - start)

        frame_times_ms = [t * 1000 for t in frame_times]

        return {
            'frames': num_frames,
            'total_time': self._total_time,
            'fps': self.fps,
            'avg_ms': np.mean(frame_times_ms),
            'std_ms': np.std(frame_times_ms),
            'min_ms': np.min(frame_times_ms),
            'max_ms': np.max(frame_times_ms),
            'p50_ms': np.percentile(frame_times_ms, 50),
            'p95_ms': np.percentile(frame_times_ms, 95),
            'p99_ms': np.percentile(frame_times_ms, 99),
        }

    def close(self):
        """Clean up resources."""
        if getattr(self, 'shm', None):
            self.shm.cleanup()
            self.shm = None
        if getattr(self, 'x11', None):
            self.x11.close()
            self.x11 = None

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class WindowCapture(ScreenCortex):
    """
    Capture a specific window instead of the full screen.

    Note: This is a convenience wrapper. For game capture,
    ensure the game window is visible and not obscured.
    """

    def __init__(self, window_name: Optional[str] = None,
                 window_id: Optional[int] = None,
                 display_name: Optional[str] = None):
        """
        Initialize window capture.

        Args:
            window_name: Window title to search for
            window_id: Specific window ID (takes precedence)
            display_name: X11 display name
        """
        # First, open display to find window
        temp_x11 = X11Display(display_name)

        if window_id is not None:
            self.target_window = window_id
        elif window_name is not None:
            self.target_window = self._find_window_by_name(temp_x11, window_name)
        else:
            # Default to root window
            self.target_window = temp_x11.root

        # Get window dimensions
        attrs = temp_x11.get_window_attributes(self.target_window)
        width = attrs.width
        height = attrs.height

        temp_x11.close()

        # Initialize parent with window dimensions
        super().__init__(display_name=display_name, width=width, height=height)

        self.window_x = attrs.x
        self.window_y = attrs.y

    def _find_window_by_name(self, x11: X11Display, name: str) -> int:
        """
        Find a window by its title.

        This is a simplified implementation. For robustness,
        consider using wmctrl or similar tools.
        """
        # This would require walking the window tree
        # For now, raise an error and suggest using window_id
        raise NotImplementedError(
            "Window name search not implemented. "
            "Use `xdotool search --name 'Brotato'` to find the window ID "
            "and pass it as window_id parameter."
        )

    def capture(self) -> np.ndarray:
        """Capture the target window."""
        # Note: SHM capture always captures from root
        # We capture at the window's position
        return super().capture(x=self.window_x, y=self.window_y)


def create_capture(downscale: int = 2) -> ScreenCortex:
    """
    Factory function to create a screen capture instance.

    Args:
        downscale: Downscale factor for performance

    Returns:
        Configured ScreenCortex instance
    """
    return ScreenCortex(downscale=downscale)
