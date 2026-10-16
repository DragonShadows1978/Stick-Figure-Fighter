"""
Stick Figure Fighter - Settings Persistence

Handles saving and loading game settings to/from JSON.
"""
import json
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any


# =============================================================================
# SETTINGS DATA
# =============================================================================

@dataclass
class AudioSettings:
    """Audio-related settings."""
    master_volume: float = 1.0
    sfx_volume: float = 1.0
    music_volume: float = 0.7
    muted: bool = False


@dataclass
class ControlSettings:
    """Control-related settings for a player."""
    # Keyboard bindings (keysym values)
    left: int = 0
    right: int = 0
    up: int = 0
    down: int = 0
    jump: int = 0
    attack: int = 0
    special: int = 0
    shield: int = 0
    grab: int = 0

    # Gamepad settings
    use_gamepad: bool = False
    gamepad_mapping: str = 'default'  # 'default' or 'smash'


@dataclass
class GameplaySettings:
    """Gameplay-related settings."""
    default_stock_count: int = 3
    default_time_limit: int = 0  # 0 = infinite
    default_stage: str = 'battlefield'
    show_damage_numbers: bool = True
    show_frame_data: bool = False
    # Cycle 17: Performance profiling
    show_fps: bool = False
    show_frame_graph: bool = False
    # Visual theme: neon, sketch, silhouette or classic
    theme: str = 'neon'


@dataclass
class GameSettings:
    """All game settings."""
    audio: AudioSettings = field(default_factory=AudioSettings)
    player1_controls: ControlSettings = field(default_factory=ControlSettings)
    player2_controls: ControlSettings = field(default_factory=ControlSettings)
    gameplay: GameplaySettings = field(default_factory=GameplaySettings)

    # Character preferences
    last_p1_character: str = 'RUSH'
    last_p2_character: str = 'ZONE'

    # Display settings
    fullscreen: bool = False
    vsync: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary."""
        return {
            'audio': asdict(self.audio),
            'player1_controls': asdict(self.player1_controls),
            'player2_controls': asdict(self.player2_controls),
            'gameplay': asdict(self.gameplay),
            'last_p1_character': self.last_p1_character,
            'last_p2_character': self.last_p2_character,
            'fullscreen': self.fullscreen,
            'vsync': self.vsync,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GameSettings':
        """Create settings from dictionary."""
        settings = cls()

        if 'audio' in data:
            for key, val in data['audio'].items():
                if hasattr(settings.audio, key):
                    setattr(settings.audio, key, val)

        if 'player1_controls' in data:
            for key, val in data['player1_controls'].items():
                if hasattr(settings.player1_controls, key):
                    setattr(settings.player1_controls, key, val)

        if 'player2_controls' in data:
            for key, val in data['player2_controls'].items():
                if hasattr(settings.player2_controls, key):
                    setattr(settings.player2_controls, key, val)

        if 'gameplay' in data:
            for key, val in data['gameplay'].items():
                if hasattr(settings.gameplay, key):
                    setattr(settings.gameplay, key, val)

        for key in ['last_p1_character', 'last_p2_character', 'fullscreen', 'vsync']:
            if key in data:
                setattr(settings, key, data[key])

        return settings


# =============================================================================
# SETTINGS MANAGER
# =============================================================================

class SettingsManager:
    """
    Manages loading and saving game settings.

    Settings are stored in a JSON file in the user's home directory
    or next to the game executable.
    """

    DEFAULT_FILENAME = 'stick_fighter_settings.json'

    def __init__(self, settings_path: Optional[str] = None):
        """
        Initialize the settings manager.

        Args:
            settings_path: Optional path to settings file.
                          If None, uses default location.
        """
        self.settings_path = settings_path or self._get_default_path()
        self.settings = GameSettings()
        self._load_default_keybindings()

    def _get_default_path(self) -> str:
        """Get the default settings file path."""
        # Try current directory first
        local_path = os.path.join(os.path.dirname(__file__), self.DEFAULT_FILENAME)
        if os.path.exists(local_path):
            return local_path

        # Then try user config directory
        config_dir = os.path.expanduser('~/.config/stick_fighter')
        os.makedirs(config_dir, exist_ok=True)
        return os.path.join(config_dir, self.DEFAULT_FILENAME)

    def _load_default_keybindings(self):
        """Load default keyboard bindings."""
        from input_system import KeySym, BINDINGS_WASD, BINDINGS_ARROWS

        # Player 1 defaults (WASD)
        self.settings.player1_controls.left = BINDINGS_WASD.left
        self.settings.player1_controls.right = BINDINGS_WASD.right
        self.settings.player1_controls.up = BINDINGS_WASD.up
        self.settings.player1_controls.down = BINDINGS_WASD.down
        self.settings.player1_controls.jump = BINDINGS_WASD.jump
        self.settings.player1_controls.attack = BINDINGS_WASD.attack
        self.settings.player1_controls.special = BINDINGS_WASD.special
        self.settings.player1_controls.shield = BINDINGS_WASD.shield
        self.settings.player1_controls.grab = BINDINGS_WASD.grab

        # Player 2 defaults (Arrows)
        self.settings.player2_controls.left = BINDINGS_ARROWS.left
        self.settings.player2_controls.right = BINDINGS_ARROWS.right
        self.settings.player2_controls.up = BINDINGS_ARROWS.up
        self.settings.player2_controls.down = BINDINGS_ARROWS.down
        self.settings.player2_controls.jump = BINDINGS_ARROWS.jump
        self.settings.player2_controls.attack = BINDINGS_ARROWS.attack
        self.settings.player2_controls.special = BINDINGS_ARROWS.special
        self.settings.player2_controls.shield = BINDINGS_ARROWS.shield
        self.settings.player2_controls.grab = BINDINGS_ARROWS.grab

    def load(self) -> bool:
        """
        Load settings from file.

        Returns:
            True if settings were loaded successfully
        """
        if not os.path.exists(self.settings_path):
            return False

        try:
            with open(self.settings_path, 'r') as f:
                data = json.load(f)

            self.settings = GameSettings.from_dict(data)
            print(f"[SETTINGS] Loaded from {self.settings_path}")
            return True

        except (json.JSONDecodeError, IOError) as e:
            print(f"[SETTINGS] Failed to load: {e}")
            return False

    def save(self) -> bool:
        """
        Save settings to file.

        Returns:
            True if settings were saved successfully
        """
        try:
            data = self.settings.to_dict()

            with open(self.settings_path, 'w') as f:
                json.dump(data, f, indent=2)

            print(f"[SETTINGS] Saved to {self.settings_path}")
            return True

        except IOError as e:
            print(f"[SETTINGS] Failed to save: {e}")
            return False

    def apply_to_audio(self, audio_system):
        """Apply audio settings to the audio system."""
        audio_system.set_master_volume(self.settings.audio.master_volume)
        audio_system.set_sfx_volume(self.settings.audio.sfx_volume)
        audio_system.muted = self.settings.audio.muted

    def apply_to_input_manager(self, input_manager):
        """Apply control settings to the input manager."""
        from input_system import KeyBindings

        # Player 1 bindings
        p1 = self.settings.player1_controls
        input_manager.keyboard.set_player_bindings(0, KeyBindings(
            left=p1.left, right=p1.right, up=p1.up, down=p1.down,
            jump=p1.jump, attack=p1.attack, special=p1.special,
            shield=p1.shield, grab=p1.grab
        ))

        # Player 2 bindings
        p2 = self.settings.player2_controls
        input_manager.keyboard.set_player_bindings(1, KeyBindings(
            left=p2.left, right=p2.right, up=p2.up, down=p2.down,
            jump=p2.jump, attack=p2.attack, special=p2.special,
            shield=p2.shield, grab=p2.grab
        ))

        # Gamepad preferences
        input_manager.set_player_gamepad(0, p1.use_gamepad)
        input_manager.set_player_gamepad(1, p2.use_gamepad)

    def update_from_audio(self, audio_system):
        """Update settings from current audio system state."""
        self.settings.audio.master_volume = audio_system.master_volume
        self.settings.audio.sfx_volume = audio_system.sfx_volume
        self.settings.audio.muted = audio_system.muted


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

_settings_manager: Optional[SettingsManager] = None


def get_settings_manager() -> SettingsManager:
    """Get or create the global settings manager."""
    global _settings_manager
    if _settings_manager is None:
        _settings_manager = SettingsManager()
        _settings_manager.load()
    return _settings_manager


def get_settings() -> GameSettings:
    """Get the current game settings."""
    return get_settings_manager().settings


def save_settings():
    """Save current settings."""
    get_settings_manager().save()


# =============================================================================
# TEST CODE
# =============================================================================

if __name__ == "__main__":
    print("=== Settings Test ===")

    # Create settings manager
    mgr = SettingsManager("/tmp/test_settings.json")

    # Print defaults
    print(f"\nDefault audio volume: {mgr.settings.audio.master_volume}")
    print(f"Default stock count: {mgr.settings.gameplay.default_stock_count}")

    # Modify settings
    mgr.settings.audio.master_volume = 0.5
    mgr.settings.gameplay.default_stock_count = 5
    mgr.settings.last_p1_character = 'GRAB'

    # Save
    mgr.save()
    print("\nSettings saved!")

    # Create new manager and load
    mgr2 = SettingsManager("/tmp/test_settings.json")
    mgr2.load()

    print(f"\nLoaded audio volume: {mgr2.settings.audio.master_volume}")
    print(f"Loaded stock count: {mgr2.settings.gameplay.default_stock_count}")
    print(f"Loaded P1 character: {mgr2.settings.last_p1_character}")

    # Cleanup
    os.remove("/tmp/test_settings.json")
    print("\nTest passed!")
