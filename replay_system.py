"""
Stick Figure Fighter - Replay System (Cycle 12)

Records and plays back match inputs for replay functionality.
Enables saving/loading replays to/from JSON files.

Features:
- Record all InputStates per frame for both players
- Store match metadata (characters, stage, stocks, seed)
- Save replays to JSON files in replays/ directory
- Load and playback replays by applying inputs frame-by-frame
- Replay browser for listing and selecting saved replays
"""
import os
import json
import time
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Tuple, Any
from enum import Enum

from game_engine import InputState


# =============================================================================
# REPLAY DATA STRUCTURES
# =============================================================================

@dataclass
class InputFrame:
    """Input state for a single frame for one player."""
    left: bool = False
    right: bool = False
    up: bool = False
    down: bool = False
    jump: bool = False
    attack: bool = False
    special: bool = False
    shield: bool = False
    grab: bool = False

    @staticmethod
    def from_input_state(state: InputState) -> 'InputFrame':
        """Convert an InputState to an InputFrame."""
        return InputFrame(
            left=state.left,
            right=state.right,
            up=state.up,
            down=state.down,
            jump=state.jump,
            attack=state.attack,
            special=state.special,
            shield=state.shield,
            grab=state.grab
        )

    def to_input_state(self, prev_state: Optional[InputState] = None) -> InputState:
        """Convert back to InputState with previous state for press detection."""
        state = InputState(
            left=self.left,
            right=self.right,
            up=self.up,
            down=self.down,
            jump=self.jump,
            attack=self.attack,
            special=self.special,
            shield=self.shield,
            grab=self.grab
        )
        if prev_state:
            state._prev_jump = prev_state.jump
            state._prev_attack = prev_state.attack
            state._prev_special = prev_state.special
            state._prev_shield = prev_state.shield
            state._prev_grab = prev_state.grab
        return state

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        # Compact format: only include True values
        d = {}
        if self.left: d['l'] = 1
        if self.right: d['r'] = 1
        if self.up: d['u'] = 1
        if self.down: d['d'] = 1
        if self.jump: d['j'] = 1
        if self.attack: d['a'] = 1
        if self.special: d['s'] = 1
        if self.shield: d['sh'] = 1
        if self.grab: d['g'] = 1
        return d

    @staticmethod
    def from_dict(d: Dict) -> 'InputFrame':
        """Create from dictionary."""
        return InputFrame(
            left=bool(d.get('l', 0)),
            right=bool(d.get('r', 0)),
            up=bool(d.get('u', 0)),
            down=bool(d.get('d', 0)),
            jump=bool(d.get('j', 0)),
            attack=bool(d.get('a', 0)),
            special=bool(d.get('s', 0)),
            shield=bool(d.get('sh', 0)),
            grab=bool(d.get('g', 0))
        )


@dataclass
class ReplayMetadata:
    """Metadata for a replay file."""
    version: int = 1
    timestamp: float = 0.0
    p1_character: str = "FLEX"
    p2_character: str = "FLEX"
    stage_name: str = "battlefield"
    stock_count: int = 3
    time_limit: int = 0  # frames (0 = no limit)
    winner: Optional[int] = None
    total_frames: int = 0
    p1_stocks_remaining: int = 0
    p2_stocks_remaining: int = 0
    p1_max_combo: int = 0
    p2_max_combo: int = 0

    def to_dict(self) -> Dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict) -> 'ReplayMetadata':
        return ReplayMetadata(
            version=d.get('version', 1),
            timestamp=d.get('timestamp', 0.0),
            p1_character=d.get('p1_character', 'FLEX'),
            p2_character=d.get('p2_character', 'FLEX'),
            stage_name=d.get('stage_name', 'battlefield'),
            stock_count=d.get('stock_count', 3),
            time_limit=d.get('time_limit', 0),
            winner=d.get('winner'),
            total_frames=d.get('total_frames', 0),
            p1_stocks_remaining=d.get('p1_stocks_remaining', 0),
            p2_stocks_remaining=d.get('p2_stocks_remaining', 0),
            p1_max_combo=d.get('p1_max_combo', 0),
            p2_max_combo=d.get('p2_max_combo', 0)
        )

    @property
    def duration_str(self) -> str:
        """Get formatted duration string."""
        seconds = self.total_frames / 60
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}:{secs:02d}"

    @property
    def date_str(self) -> str:
        """Get formatted date string."""
        if self.timestamp == 0:
            return "Unknown"
        from datetime import datetime
        dt = datetime.fromtimestamp(self.timestamp)
        return dt.strftime("%Y-%m-%d %H:%M")


@dataclass
class Replay:
    """Complete replay data."""
    metadata: ReplayMetadata
    # inputs[frame_number] = (p1_input, p2_input)
    inputs: List[Tuple[InputFrame, InputFrame]] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'metadata': self.metadata.to_dict(),
            'inputs': [
                [p1.to_dict(), p2.to_dict()]
                for p1, p2 in self.inputs
            ]
        }

    @staticmethod
    def from_dict(d: Dict) -> 'Replay':
        """Create from dictionary."""
        metadata = ReplayMetadata.from_dict(d.get('metadata', {}))
        inputs = [
            (InputFrame.from_dict(frame[0]), InputFrame.from_dict(frame[1]))
            for frame in d.get('inputs', [])
        ]
        return Replay(metadata=metadata, inputs=inputs)


# =============================================================================
# REPLAY RECORDER
# =============================================================================

class ReplayRecorder:
    """
    Records match inputs for replay.

    Usage:
        recorder = ReplayRecorder()
        recorder.start_recording(settings)

        # Each frame during match:
        recorder.record_frame(p1_input_state, p2_input_state)

        # When match ends:
        recorder.finish_recording(game_state)
        recorder.save("my_replay.json")
    """

    def __init__(self, replay_dir: str = "replays"):
        """
        Initialize recorder.

        Args:
            replay_dir: Directory to save replays to
        """
        self.replay_dir = replay_dir
        self.is_recording = False
        self.current_replay: Optional[Replay] = None

        # Ensure replay directory exists
        os.makedirs(replay_dir, exist_ok=True)

    def start_recording(self, p1_character: str, p2_character: str,
                       stage_name: str, stock_count: int = 3,
                       time_limit: int = 0):
        """
        Start recording a new replay.

        Args:
            p1_character: P1 character name
            p2_character: P2 character name
            stage_name: Stage name
            stock_count: Number of stocks
            time_limit: Time limit in frames (0 = no limit)
        """
        metadata = ReplayMetadata(
            version=1,
            timestamp=time.time(),
            p1_character=p1_character,
            p2_character=p2_character,
            stage_name=stage_name,
            stock_count=stock_count,
            time_limit=time_limit
        )
        self.current_replay = Replay(metadata=metadata, inputs=[])
        self.is_recording = True

    def record_frame(self, p1_state: InputState, p2_state: InputState):
        """
        Record inputs for the current frame.

        Args:
            p1_state: Player 1 input state
            p2_state: Player 2 input state
        """
        if not self.is_recording or not self.current_replay:
            return

        p1_frame = InputFrame.from_input_state(p1_state)
        p2_frame = InputFrame.from_input_state(p2_state)
        self.current_replay.inputs.append((p1_frame, p2_frame))

    def finish_recording(self, winner: Optional[int] = None,
                        p1_stocks: int = 0, p2_stocks: int = 0,
                        p1_max_combo: int = 0, p2_max_combo: int = 0):
        """
        Finish recording and finalize metadata.

        Args:
            winner: Winner player ID (0 or 1, or None for tie)
            p1_stocks: P1 remaining stocks
            p2_stocks: P2 remaining stocks
            p1_max_combo: P1 max combo achieved
            p2_max_combo: P2 max combo achieved
        """
        if not self.is_recording or not self.current_replay:
            return

        self.current_replay.metadata.winner = winner
        self.current_replay.metadata.total_frames = len(self.current_replay.inputs)
        self.current_replay.metadata.p1_stocks_remaining = p1_stocks
        self.current_replay.metadata.p2_stocks_remaining = p2_stocks
        self.current_replay.metadata.p1_max_combo = p1_max_combo
        self.current_replay.metadata.p2_max_combo = p2_max_combo
        self.is_recording = False

    def save(self, filename: Optional[str] = None) -> str:
        """
        Save the current replay to a file.

        Args:
            filename: Optional filename (auto-generated if not provided)

        Returns:
            Path to saved file
        """
        if not self.current_replay:
            raise ValueError("No replay to save")

        if not filename:
            # Generate filename based on timestamp and characters
            ts = int(self.current_replay.metadata.timestamp)
            p1 = self.current_replay.metadata.p1_character[:4].upper()
            p2 = self.current_replay.metadata.p2_character[:4].upper()
            filename = f"replay_{ts}_{p1}_vs_{p2}.json"

        filepath = os.path.join(self.replay_dir, filename)

        with open(filepath, 'w') as f:
            json.dump(self.current_replay.to_dict(), f)

        print(f"[REPLAY] Saved: {filepath}")
        return filepath

    def discard(self):
        """Discard the current recording."""
        self.current_replay = None
        self.is_recording = False


# =============================================================================
# REPLAY PLAYER
# =============================================================================

class ReplayPlayer:
    """
    Plays back recorded replays.

    Usage:
        player = ReplayPlayer()
        player.load("my_replay.json")

        # Get match settings from replay
        settings = player.get_match_settings()

        # Each frame during playback:
        p1_input, p2_input = player.get_frame_inputs(frame_number)
    """

    def __init__(self, replay_dir: str = "replays"):
        """
        Initialize player.

        Args:
            replay_dir: Directory to load replays from
        """
        self.replay_dir = replay_dir
        self.current_replay: Optional[Replay] = None
        self.current_frame: int = 0
        self._prev_p1_state: Optional[InputState] = None
        self._prev_p2_state: Optional[InputState] = None

    def load(self, filename: str) -> bool:
        """
        Load a replay file.

        Args:
            filename: Filename or full path

        Returns:
            True if loaded successfully
        """
        if not os.path.isabs(filename):
            filepath = os.path.join(self.replay_dir, filename)
        else:
            filepath = filename

        if not os.path.exists(filepath):
            print(f"[REPLAY] File not found: {filepath}")
            return False

        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            self.current_replay = Replay.from_dict(data)
            self.current_frame = 0
            self._prev_p1_state = None
            self._prev_p2_state = None
            print(f"[REPLAY] Loaded: {filepath}")
            print(f"  {self.metadata.p1_character} vs {self.metadata.p2_character}")
            print(f"  Stage: {self.metadata.stage_name}")
            print(f"  Duration: {self.metadata.duration_str}")
            return True
        except Exception as e:
            print(f"[REPLAY] Error loading: {e}")
            return False

    @property
    def metadata(self) -> Optional[ReplayMetadata]:
        """Get current replay metadata."""
        if self.current_replay:
            return self.current_replay.metadata
        return None

    @property
    def total_frames(self) -> int:
        """Get total frames in replay."""
        if self.current_replay:
            return len(self.current_replay.inputs)
        return 0

    @property
    def is_finished(self) -> bool:
        """Check if playback is finished."""
        return self.current_frame >= self.total_frames

    def get_match_settings(self) -> Dict:
        """Get match settings from the loaded replay."""
        if not self.current_replay:
            return {}

        meta = self.current_replay.metadata
        return {
            'p1_character': meta.p1_character,
            'p2_character': meta.p2_character,
            'stage_name': meta.stage_name,
            'stock_count': meta.stock_count,
            'time_limit': meta.time_limit
        }

    def get_frame_inputs(self, frame: int) -> Tuple[InputState, InputState]:
        """
        Get inputs for a specific frame.

        Args:
            frame: Frame number

        Returns:
            Tuple of (p1_input, p2_input) InputStates
        """
        if not self.current_replay or frame >= self.total_frames:
            return InputState(), InputState()

        p1_frame, p2_frame = self.current_replay.inputs[frame]

        # Convert to InputState with previous state for press detection
        p1_state = p1_frame.to_input_state(self._prev_p1_state)
        p2_state = p2_frame.to_input_state(self._prev_p2_state)

        self._prev_p1_state = p1_state
        self._prev_p2_state = p2_state
        self.current_frame = frame + 1

        return p1_state, p2_state

    def reset(self):
        """Reset playback to the beginning."""
        self.current_frame = 0
        self._prev_p1_state = None
        self._prev_p2_state = None


# =============================================================================
# REPLAY BROWSER
# =============================================================================

@dataclass
class ReplayEntry:
    """Entry in the replay browser list."""
    filename: str
    metadata: ReplayMetadata

    @property
    def display_name(self) -> str:
        """Get display name for the replay."""
        return f"{self.metadata.p1_character} vs {self.metadata.p2_character}"

    @property
    def display_info(self) -> str:
        """Get display info line."""
        winner_str = ""
        if self.metadata.winner is not None:
            winner_char = self.metadata.p1_character if self.metadata.winner == 0 else self.metadata.p2_character
            winner_str = f" - {winner_char} wins"
        return f"{self.metadata.date_str} | {self.metadata.duration_str}{winner_str}"


class ReplayBrowser:
    """
    Browse and select saved replays.

    Usage:
        browser = ReplayBrowser()
        browser.refresh()  # Load list of replays

        # Navigation
        browser.move_cursor(1)  # Move down
        browser.move_cursor(-1)  # Move up

        # Selection
        entry = browser.get_selected()
        if entry:
            print(f"Selected: {entry.display_name}")
    """

    def __init__(self, replay_dir: str = "replays"):
        """
        Initialize browser.

        Args:
            replay_dir: Directory containing replay files
        """
        self.replay_dir = replay_dir
        self.entries: List[ReplayEntry] = []
        self.cursor: int = 0
        self.scroll_offset: int = 0
        self.visible_count: int = 8  # How many to show at once

        # Ensure replay directory exists
        os.makedirs(replay_dir, exist_ok=True)

    def refresh(self):
        """Refresh the list of available replays."""
        self.entries = []

        if not os.path.exists(self.replay_dir):
            return

        # Find all .json files in replay directory
        try:
            files = sorted(
                [f for f in os.listdir(self.replay_dir) if f.endswith('.json')],
                reverse=True  # Newest first
            )
        except OSError:
            return

        for filename in files:
            filepath = os.path.join(self.replay_dir, filename)
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                metadata = ReplayMetadata.from_dict(data.get('metadata', {}))
                self.entries.append(ReplayEntry(filename=filename, metadata=metadata))
            except Exception:
                continue  # Skip invalid files

        # Reset cursor if out of bounds
        if self.cursor >= len(self.entries):
            self.cursor = max(0, len(self.entries) - 1)

        print(f"[REPLAY BROWSER] Found {len(self.entries)} replay(s)")

    def move_cursor(self, delta: int):
        """
        Move cursor up or down.

        Args:
            delta: -1 for up, +1 for down
        """
        if not self.entries:
            return

        self.cursor = max(0, min(len(self.entries) - 1, self.cursor + delta))

        # Adjust scroll offset to keep cursor visible
        if self.cursor < self.scroll_offset:
            self.scroll_offset = self.cursor
        elif self.cursor >= self.scroll_offset + self.visible_count:
            self.scroll_offset = self.cursor - self.visible_count + 1

    def get_selected(self) -> Optional[ReplayEntry]:
        """Get the currently selected replay entry."""
        if 0 <= self.cursor < len(self.entries):
            return self.entries[self.cursor]
        return None

    def get_visible_entries(self) -> List[Tuple[int, ReplayEntry]]:
        """Get entries currently visible in the browser."""
        start = self.scroll_offset
        end = min(start + self.visible_count, len(self.entries))
        return [(i, self.entries[i]) for i in range(start, end)]

    def delete_selected(self) -> bool:
        """
        Delete the currently selected replay.

        Returns:
            True if deleted successfully
        """
        entry = self.get_selected()
        if not entry:
            return False

        filepath = os.path.join(self.replay_dir, entry.filename)
        try:
            os.remove(filepath)
            self.entries.remove(entry)
            if self.cursor >= len(self.entries):
                self.cursor = max(0, len(self.entries) - 1)
            print(f"[REPLAY] Deleted: {entry.filename}")
            return True
        except OSError as e:
            print(f"[REPLAY] Failed to delete: {e}")
            return False

    @property
    def is_empty(self) -> bool:
        """Check if there are no replays."""
        return len(self.entries) == 0


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_replay_recorder(replay_dir: str = "replays") -> ReplayRecorder:
    """Get a replay recorder instance."""
    return ReplayRecorder(replay_dir)


def get_replay_player(replay_dir: str = "replays") -> ReplayPlayer:
    """Get a replay player instance."""
    return ReplayPlayer(replay_dir)


def get_replay_browser(replay_dir: str = "replays") -> ReplayBrowser:
    """Get a replay browser instance."""
    return ReplayBrowser(replay_dir)


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("Testing Replay System...")

    # Test recording
    print("\n1. Testing ReplayRecorder...")
    recorder = ReplayRecorder("test_replays")
    recorder.start_recording("RUSH", "ZONE", "battlefield", 3, 0)

    # Simulate some inputs
    for i in range(60):  # 1 second of frames
        p1 = InputState()
        p2 = InputState()
        p1.right = i % 10 < 5
        p1.attack = i % 20 == 0
        p2.left = i % 8 < 4
        p2.jump = i % 30 == 0
        recorder.record_frame(p1, p2)

    recorder.finish_recording(winner=0, p1_stocks=2, p2_stocks=0)
    filepath = recorder.save()
    print(f"  Saved to: {filepath}")

    # Test loading
    print("\n2. Testing ReplayPlayer...")
    player = ReplayPlayer("test_replays")
    if player.load(os.path.basename(filepath)):
        print(f"  Total frames: {player.total_frames}")
        settings = player.get_match_settings()
        print(f"  Settings: {settings}")

        # Read a few frames
        for i in range(3):
            p1, p2 = player.get_frame_inputs(i)
            print(f"  Frame {i}: P1(r={p1.right}, a={p1.attack}) P2(l={p2.left}, j={p2.jump})")

    # Test browser
    print("\n3. Testing ReplayBrowser...")
    browser = ReplayBrowser("test_replays")
    browser.refresh()
    print(f"  Found {len(browser.entries)} replay(s)")
    if browser.entries:
        entry = browser.get_selected()
        print(f"  Selected: {entry.display_name}")
        print(f"  Info: {entry.display_info}")

    # Cleanup test directory
    import shutil
    shutil.rmtree("test_replays", ignore_errors=True)

    print("\nReplay system test complete!")
