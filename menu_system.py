"""
Stick Figure Fighter - Menu System (Cycle 6)

Provides menu screens:
- Character Select: Choose from 4 characters with stats preview
- Stage Select: Choose between Battlefield and Final Destination
- Match Settings: Configure stocks, time limit, etc.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, List, Tuple, Callable
import math

from game_engine import Stage, Vector2, CONFIG
from characters import CharacterType, get_character_stats, get_character_color, CHARACTER_COLORS


class MenuState(Enum):
    """Current menu screen."""
    TITLE = auto()              # Title screen with mode selection
    CHARACTER_SELECT = auto()   # P1 + P2 character selection
    STAGE_SELECT = auto()       # Stage selection
    MATCH_SETTINGS = auto()     # Stock count, time limit configuration
    READY = auto()              # All selections made, ready to start
    SETTINGS = auto()           # Settings menu (Audio, Controls, Gameplay)


class GameModeSelection(Enum):
    """Game mode options from title screen."""
    VERSUS = auto()      # 2-player versus mode
    TRAINING = auto()    # Training mode (P1 vs CPU)
    REPLAYS = auto()     # View saved replays
    QUIT = auto()        # Exit game


@dataclass
class CharacterSelectState:
    """State for character selection screen."""
    # Cursor positions for each player (0-3 for 4 characters)
    p1_cursor: int = 0
    p2_cursor: int = 3

    # Locked in selections (None if not locked)
    p1_locked: Optional[CharacterType] = None
    p2_locked: Optional[CharacterType] = None

    # Animation timers
    p1_cursor_anim: int = 0
    p2_cursor_anim: int = 0


@dataclass
class StageSelectState:
    """State for stage selection screen."""
    cursor: int = 0  # 0 = Battlefield, 1 = Final Destination
    locked: Optional[str] = None
    cursor_anim: int = 0


@dataclass
class TitleScreenState:
    """State for title screen."""
    cursor: int = 0  # 0=Versus, 1=Training, 2=Replays, 3=Settings, 4=Quit
    cursor_anim: int = 0


class SettingsSubmenu(Enum):
    """Submenus within settings."""
    MAIN = auto()       # Audio, Controls, Gameplay selection
    AUDIO = auto()      # Master volume, SFX volume, mute
    CONTROLS = auto()   # Keybindings display, gamepad info
    GAMEPLAY = auto()   # Default stocks, time limit, damage numbers


@dataclass
class SettingsMenuState:
    """State for settings menu."""
    submenu: SettingsSubmenu = SettingsSubmenu.MAIN
    main_cursor: int = 0        # 0=Audio, 1=Controls, 2=Gameplay, 3=Back
    audio_cursor: int = 0       # 0=Master, 1=SFX, 2=Mute, 3=Back
    controls_cursor: int = 0    # 0=P1 Mapping, 1=P2 Mapping, 2=Back
    gameplay_cursor: int = 0    # 0=Stocks, 1=Time, 2=Damage Numbers, 3=Visual Theme, 4=Back
    cursor_anim: int = 0


@dataclass
class MatchSettingsState:
    """State for match settings screen."""
    cursor: int = 0  # 0=Stock, 1=Time, 2=Start
    stock_count: int = 3  # 1-5 stocks
    time_limit_idx: int = 0  # Index into TIME_OPTIONS
    cursor_anim: int = 0


# Time limit options (index -> minutes, 0 = no limit)
TIME_OPTIONS = [0, 3, 5, 8]
TIME_LABELS = ["None", "3 min", "5 min", "8 min"]


@dataclass
class MatchSettings:
    """Match configuration settings."""
    stock_count: int = 3  # 1-5 stocks
    time_limit: int = 0  # 0 = no limit, or minutes (1-8)
    stage_name: str = "battlefield"
    p1_character: Optional[str] = None
    p2_character: Optional[str] = None
    is_training_mode: bool = False  # True if training mode selected


class MenuSystem:
    """
    Handles all menu screens and transitions.

    Flow: TITLE -> CHARACTER_SELECT -> STAGE_SELECT -> MATCH_SETTINGS -> READY
    Training: TITLE -> CHARACTER_SELECT (P1 only) -> STAGE_SELECT -> MATCH_SETTINGS -> READY
    """

    # Character order for selection
    CHARACTERS = [CharacterType.RUSH, CharacterType.ZONE, CharacterType.GRAB, CharacterType.FLEX]
    STAGES = ["battlefield", "final_destination"]
    STAGE_NAMES = ["Battlefield", "Final Destination"]
    MODE_OPTIONS = ["VERSUS MODE", "TRAINING MODE", "REPLAYS", "SETTINGS", "QUIT"]

    def __init__(self):
        self.state = MenuState.TITLE
        self.title_state = TitleScreenState()
        self.char_select = CharacterSelectState()
        self.stage_select = StageSelectState()
        self.match_settings_state = MatchSettingsState()
        self.settings = MatchSettings()
        self.settings_menu_state = SettingsMenuState()

        # Game mode selection
        self.game_mode = GameModeSelection.VERSUS
        self.quit_requested = False

        # Settings manager reference (loaded on first access)
        self._settings_manager = None

        # Input debounce
        self._prev_inputs = {'p1': {}, 'p2': {}}
        self._input_repeat_timers = {'p1': {}, 'p2': {}}

        # Callbacks
        self.on_match_start: Optional[Callable[[MatchSettings], None]] = None
        self.on_cursor_move: Optional[Callable[[], None]] = None
        self.on_select: Optional[Callable[[], None]] = None
        self.on_back: Optional[Callable[[], None]] = None
        self.on_settings_changed: Optional[Callable[[], None]] = None
        self.on_theme_changed: Optional[Callable[[str], None]] = None  # called with theme name
        self.available_themes: List[str] = ['classic']  # set by the game from the renderer

        # Cycle 18: Character select sound callbacks
        self.on_char_cursor_move: Optional[Callable[[str], None]] = None  # Called with character name
        self.on_char_locked: Optional[Callable[[str], None]] = None  # Called with character name

    @property
    def settings_manager(self):
        """Get the settings manager (lazy load)."""
        if self._settings_manager is None:
            from settings import get_settings_manager
            self._settings_manager = get_settings_manager()
        return self._settings_manager

    def reset(self):
        """Reset all menu state."""
        self.state = MenuState.TITLE
        self.title_state = TitleScreenState()
        self.char_select = CharacterSelectState()
        self.stage_select = StageSelectState()
        self.match_settings_state = MatchSettingsState()
        self.settings = MatchSettings()
        self.settings_menu_state = SettingsMenuState()
        self.game_mode = GameModeSelection.VERSUS
        self.quit_requested = False

    def update(self, p1_inputs: dict, p2_inputs: dict, frame: int) -> bool:
        """
        Update menu state based on inputs.

        Returns True if menu should continue, False if match should start (or quit requested).
        """
        # Update animation timers
        self.title_state.cursor_anim += 1
        self.char_select.p1_cursor_anim += 1
        self.char_select.p2_cursor_anim += 1
        self.stage_select.cursor_anim += 1
        self.match_settings_state.cursor_anim += 1
        self.settings_menu_state.cursor_anim += 1

        if self.state == MenuState.TITLE:
            return self._update_title(p1_inputs, p2_inputs)
        elif self.state == MenuState.CHARACTER_SELECT:
            return self._update_character_select(p1_inputs, p2_inputs)
        elif self.state == MenuState.STAGE_SELECT:
            return self._update_stage_select(p1_inputs, p2_inputs)
        elif self.state == MenuState.MATCH_SETTINGS:
            return self._update_match_settings(p1_inputs, p2_inputs)
        elif self.state == MenuState.SETTINGS:
            return self._update_settings(p1_inputs, p2_inputs)
        elif self.state == MenuState.READY:
            return False  # Ready to start match

        return True

    def _is_pressed(self, player: str, key: str, inputs: dict) -> bool:
        """Check if a key was just pressed (not held)."""
        prev = self._prev_inputs[player].get(key, False)
        current = inputs.get(key, False)
        self._prev_inputs[player][key] = current
        return current and not prev

    def _update_title(self, p1_inputs: dict, p2_inputs: dict) -> bool:
        """Update title screen with mode selection."""
        # Navigation (either player)
        moved = False
        if self._is_pressed('p1', 'up', p1_inputs) or self._is_pressed('p2', 'up', p2_inputs):
            self.title_state.cursor = (self.title_state.cursor - 1) % 5
            self.title_state.cursor_anim = 0
            moved = True
        elif self._is_pressed('p1', 'down', p1_inputs) or self._is_pressed('p2', 'down', p2_inputs):
            self.title_state.cursor = (self.title_state.cursor + 1) % 5
            self.title_state.cursor_anim = 0
            moved = True

        if moved and self.on_cursor_move:
            self.on_cursor_move()

        # Selection
        if (self._is_pressed('p1', 'attack', p1_inputs) or
            self._is_pressed('p1', 'jump', p1_inputs) or
            self._is_pressed('p2', 'attack', p2_inputs) or
            self._is_pressed('p2', 'jump', p2_inputs)):

            if self.title_state.cursor == 0:
                # Versus Mode
                self.game_mode = GameModeSelection.VERSUS
                self.settings.is_training_mode = False
                self.state = MenuState.CHARACTER_SELECT
                if self.on_select:
                    self.on_select()
            elif self.title_state.cursor == 1:
                # Training Mode
                self.game_mode = GameModeSelection.TRAINING
                self.settings.is_training_mode = True
                self.state = MenuState.CHARACTER_SELECT
                if self.on_select:
                    self.on_select()
            elif self.title_state.cursor == 2:
                # Replays - signal to main.py to handle
                self.game_mode = GameModeSelection.REPLAYS
                if self.on_select:
                    self.on_select()
                return False  # Signal for replay browser
            elif self.title_state.cursor == 3:
                # Settings
                self.settings_menu_state = SettingsMenuState()  # Reset settings menu
                self.state = MenuState.SETTINGS
                if self.on_select:
                    self.on_select()
            elif self.title_state.cursor == 4:
                # Quit
                self.quit_requested = True
                if self.on_select:
                    self.on_select()
                return False  # Signal to exit

        return True

    def _update_character_select(self, p1_inputs: dict, p2_inputs: dict) -> bool:
        """Update character selection screen. Cycle 18: Added character preview sounds."""
        is_training = self.game_mode == GameModeSelection.TRAINING

        # P1 navigation (if not locked)
        if self.char_select.p1_locked is None:
            if self._is_pressed('p1', 'left', p1_inputs):
                self.char_select.p1_cursor = (self.char_select.p1_cursor - 1) % 4
                self.char_select.p1_cursor_anim = 0
                # Cycle 18: Play character preview sound
                char_name = self.CHARACTERS[self.char_select.p1_cursor].name
                if self.on_char_cursor_move:
                    self.on_char_cursor_move(char_name)
                elif self.on_cursor_move:
                    self.on_cursor_move()
            elif self._is_pressed('p1', 'right', p1_inputs):
                self.char_select.p1_cursor = (self.char_select.p1_cursor + 1) % 4
                self.char_select.p1_cursor_anim = 0
                # Cycle 18: Play character preview sound
                char_name = self.CHARACTERS[self.char_select.p1_cursor].name
                if self.on_char_cursor_move:
                    self.on_char_cursor_move(char_name)
                elif self.on_cursor_move:
                    self.on_cursor_move()

            # Lock in selection
            if self._is_pressed('p1', 'attack', p1_inputs) or self._is_pressed('p1', 'jump', p1_inputs):
                self.char_select.p1_locked = self.CHARACTERS[self.char_select.p1_cursor]
                self.settings.p1_character = self.char_select.p1_locked.name
                # Cycle 18: Play character lock sound
                if self.on_char_locked:
                    self.on_char_locked(self.char_select.p1_locked.name)
                elif self.on_select:
                    self.on_select()

                # In training mode, auto-lock P2 as FLEX (CPU)
                if is_training:
                    self.char_select.p2_locked = self.CHARACTERS[self.char_select.p2_cursor]
                    self.settings.p2_character = self.char_select.p2_locked.name
        else:
            # P1 is locked - B/special to unlock
            if self._is_pressed('p1', 'special', p1_inputs):
                self.char_select.p1_locked = None
                self.settings.p1_character = None
                # In training mode, also unlock P2
                if is_training:
                    self.char_select.p2_locked = None
                    self.settings.p2_character = None
                if self.on_back:
                    self.on_back()

        # P2 navigation (only in versus mode, not in training mode)
        if not is_training:
            if self.char_select.p2_locked is None:
                if self._is_pressed('p2', 'left', p2_inputs):
                    self.char_select.p2_cursor = (self.char_select.p2_cursor - 1) % 4
                    self.char_select.p2_cursor_anim = 0
                    # Cycle 18: Play character preview sound
                    char_name = self.CHARACTERS[self.char_select.p2_cursor].name
                    if self.on_char_cursor_move:
                        self.on_char_cursor_move(char_name)
                    elif self.on_cursor_move:
                        self.on_cursor_move()
                elif self._is_pressed('p2', 'right', p2_inputs):
                    self.char_select.p2_cursor = (self.char_select.p2_cursor + 1) % 4
                    self.char_select.p2_cursor_anim = 0
                    # Cycle 18: Play character preview sound
                    char_name = self.CHARACTERS[self.char_select.p2_cursor].name
                    if self.on_char_cursor_move:
                        self.on_char_cursor_move(char_name)
                    elif self.on_cursor_move:
                        self.on_cursor_move()

                # Lock in selection
                if self._is_pressed('p2', 'attack', p2_inputs) or self._is_pressed('p2', 'jump', p2_inputs):
                    self.char_select.p2_locked = self.CHARACTERS[self.char_select.p2_cursor]
                    self.settings.p2_character = self.char_select.p2_locked.name
                    # Cycle 18: Play character lock sound
                    if self.on_char_locked:
                        self.on_char_locked(self.char_select.p2_locked.name)
                    elif self.on_select:
                        self.on_select()
            else:
                # P2 is locked - B/special to unlock
                if self._is_pressed('p2', 'special', p2_inputs):
                    self.char_select.p2_locked = None
                    self.settings.p2_character = None
                    if self.on_back:
                        self.on_back()
        else:
            # In training mode, P1 can change P2's character with up/down
            if self.char_select.p1_locked is None:
                if self._is_pressed('p1', 'up', p1_inputs):
                    self.char_select.p2_cursor = (self.char_select.p2_cursor - 1) % 4
                    self.char_select.p2_cursor_anim = 0
                    # Cycle 18: Play character preview sound for CPU selection
                    char_name = self.CHARACTERS[self.char_select.p2_cursor].name
                    if self.on_char_cursor_move:
                        self.on_char_cursor_move(char_name)
                    elif self.on_cursor_move:
                        self.on_cursor_move()
                elif self._is_pressed('p1', 'down', p1_inputs):
                    self.char_select.p2_cursor = (self.char_select.p2_cursor + 1) % 4
                    self.char_select.p2_cursor_anim = 0
                    # Cycle 18: Play character preview sound for CPU selection
                    char_name = self.CHARACTERS[self.char_select.p2_cursor].name
                    if self.on_char_cursor_move:
                        self.on_char_cursor_move(char_name)
                    elif self.on_cursor_move:
                        self.on_cursor_move()

        # Back to title screen (if neither player is locked)
        if not self.char_select.p1_locked and not self.char_select.p2_locked:
            if self._is_pressed('p1', 'special', p1_inputs) or self._is_pressed('p2', 'special', p2_inputs):
                self.state = MenuState.TITLE
                if self.on_back:
                    self.on_back()
                return True

        # Both locked -> advance to stage select
        if self.char_select.p1_locked and self.char_select.p2_locked:
            self.state = MenuState.STAGE_SELECT

        return True

    def _update_stage_select(self, p1_inputs: dict, p2_inputs: dict) -> bool:
        """Update stage selection screen."""
        # Navigation (either player)
        if self.stage_select.locked is None:
            if self._is_pressed('p1', 'left', p1_inputs) or self._is_pressed('p2', 'left', p2_inputs):
                self.stage_select.cursor = (self.stage_select.cursor - 1) % len(self.STAGES)
                self.stage_select.cursor_anim = 0
                if self.on_cursor_move:
                    self.on_cursor_move()
            elif self._is_pressed('p1', 'right', p1_inputs) or self._is_pressed('p2', 'right', p2_inputs):
                self.stage_select.cursor = (self.stage_select.cursor + 1) % len(self.STAGES)
                self.stage_select.cursor_anim = 0
                if self.on_cursor_move:
                    self.on_cursor_move()

            # Lock in selection -> go to match settings
            if (self._is_pressed('p1', 'attack', p1_inputs) or self._is_pressed('p1', 'jump', p1_inputs) or
                self._is_pressed('p2', 'attack', p2_inputs) or self._is_pressed('p2', 'jump', p2_inputs)):
                self.stage_select.locked = self.STAGES[self.stage_select.cursor]
                self.settings.stage_name = self.stage_select.locked
                self.state = MenuState.MATCH_SETTINGS
                if self.on_select:
                    self.on_select()

        # B to go back to character select
        if self._is_pressed('p1', 'special', p1_inputs) or self._is_pressed('p2', 'special', p2_inputs):
            # Unlock characters and go back
            self.char_select.p1_locked = None
            self.char_select.p2_locked = None
            self.settings.p1_character = None
            self.settings.p2_character = None
            self.state = MenuState.CHARACTER_SELECT
            if self.on_back:
                self.on_back()

        return True

    def _update_match_settings(self, p1_inputs: dict, p2_inputs: dict) -> bool:
        """Update match settings screen (stock count, time limit)."""
        mss = self.match_settings_state

        # Navigation up/down between options
        if self._is_pressed('p1', 'up', p1_inputs) or self._is_pressed('p2', 'up', p2_inputs):
            mss.cursor = (mss.cursor - 1) % 3  # 0=Stock, 1=Time, 2=Start
            mss.cursor_anim = 0
            if self.on_cursor_move:
                self.on_cursor_move()
        elif self._is_pressed('p1', 'down', p1_inputs) or self._is_pressed('p2', 'down', p2_inputs):
            mss.cursor = (mss.cursor + 1) % 3
            mss.cursor_anim = 0
            if self.on_cursor_move:
                self.on_cursor_move()

        # Left/right to change values
        if mss.cursor == 0:  # Stock count
            if self._is_pressed('p1', 'left', p1_inputs) or self._is_pressed('p2', 'left', p2_inputs):
                mss.stock_count = max(1, mss.stock_count - 1)
                if self.on_cursor_move:
                    self.on_cursor_move()
            elif self._is_pressed('p1', 'right', p1_inputs) or self._is_pressed('p2', 'right', p2_inputs):
                mss.stock_count = min(5, mss.stock_count + 1)
                if self.on_cursor_move:
                    self.on_cursor_move()
        elif mss.cursor == 1:  # Time limit
            if self._is_pressed('p1', 'left', p1_inputs) or self._is_pressed('p2', 'left', p2_inputs):
                mss.time_limit_idx = max(0, mss.time_limit_idx - 1)
                if self.on_cursor_move:
                    self.on_cursor_move()
            elif self._is_pressed('p1', 'right', p1_inputs) or self._is_pressed('p2', 'right', p2_inputs):
                mss.time_limit_idx = min(len(TIME_OPTIONS) - 1, mss.time_limit_idx + 1)
                if self.on_cursor_move:
                    self.on_cursor_move()

        # Confirm selection
        if (self._is_pressed('p1', 'attack', p1_inputs) or self._is_pressed('p1', 'jump', p1_inputs) or
            self._is_pressed('p2', 'attack', p2_inputs) or self._is_pressed('p2', 'jump', p2_inputs)):

            if mss.cursor == 2:  # Start button
                # Apply settings
                self.settings.stock_count = mss.stock_count
                self.settings.time_limit = TIME_OPTIONS[mss.time_limit_idx]
                self.state = MenuState.READY
                if self.on_select:
                    self.on_select()
                return False  # Ready to start match

        # B to go back to stage select
        if self._is_pressed('p1', 'special', p1_inputs) or self._is_pressed('p2', 'special', p2_inputs):
            self.stage_select.locked = None
            self.state = MenuState.STAGE_SELECT
            if self.on_back:
                self.on_back()

        return True

    def get_character_info(self, char_type: CharacterType) -> dict:
        """Get character info for display."""
        stats = get_character_stats(char_type)
        color = get_character_color(char_type)

        # Calculate relative stats (0-10 scale for display)
        # Speed: based on run_speed (4.5-8 range -> 0-10)
        speed = int((stats.run_speed - 4.5) / 0.35)
        speed = max(0, min(10, speed))

        # Weight: based on weight (85-125 range -> 0-10, inverted so heavy = higher)
        weight = int((stats.weight - 85) / 4)
        weight = max(0, min(10, weight))

        # Power: estimated from character archetype
        power_values = {
            CharacterType.RUSH: 5,
            CharacterType.ZONE: 6,
            CharacterType.GRAB: 10,
            CharacterType.FLEX: 7
        }
        power = power_values.get(char_type, 5)

        # Recovery: based on jump_force and air_speed
        recovery = int((stats.jump_force - 11) / 0.4 + (stats.air_speed - 3) / 0.3)
        recovery = max(0, min(10, recovery))

        return {
            'name': stats.name,
            'color': color,
            'speed': speed,
            'weight': weight,
            'power': power,
            'recovery': recovery,
            'description': self._get_character_description(char_type)
        }

    def _get_character_description(self, char_type: CharacterType) -> str:
        """Get character playstyle description."""
        descriptions = {
            CharacterType.RUSH: "Fast rushdown. Combos & speed.",
            CharacterType.ZONE: "Ranged zoner. Projectiles.",
            CharacterType.GRAB: "Heavy grappler. Power.",
            CharacterType.FLEX: "All-rounder. Balanced."
        }
        return descriptions.get(char_type, "")

    def _update_settings(self, p1_inputs: dict, p2_inputs: dict) -> bool:
        """Update settings menu."""
        sms = self.settings_menu_state
        game_settings = self.settings_manager.settings

        if sms.submenu == SettingsSubmenu.MAIN:
            return self._update_settings_main(p1_inputs, p2_inputs)
        elif sms.submenu == SettingsSubmenu.AUDIO:
            return self._update_settings_audio(p1_inputs, p2_inputs)
        elif sms.submenu == SettingsSubmenu.CONTROLS:
            return self._update_settings_controls(p1_inputs, p2_inputs)
        elif sms.submenu == SettingsSubmenu.GAMEPLAY:
            return self._update_settings_gameplay(p1_inputs, p2_inputs)

        return True

    def _update_settings_main(self, p1_inputs: dict, p2_inputs: dict) -> bool:
        """Update settings main menu (Audio/Controls/Gameplay/Back)."""
        sms = self.settings_menu_state
        moved = False

        # Navigation
        if self._is_pressed('p1', 'up', p1_inputs) or self._is_pressed('p2', 'up', p2_inputs):
            sms.main_cursor = (sms.main_cursor - 1) % 4
            sms.cursor_anim = 0
            moved = True
        elif self._is_pressed('p1', 'down', p1_inputs) or self._is_pressed('p2', 'down', p2_inputs):
            sms.main_cursor = (sms.main_cursor + 1) % 4
            sms.cursor_anim = 0
            moved = True

        if moved and self.on_cursor_move:
            self.on_cursor_move()

        # Selection
        if (self._is_pressed('p1', 'attack', p1_inputs) or
            self._is_pressed('p1', 'jump', p1_inputs) or
            self._is_pressed('p2', 'attack', p2_inputs) or
            self._is_pressed('p2', 'jump', p2_inputs)):

            if sms.main_cursor == 0:
                sms.submenu = SettingsSubmenu.AUDIO
                sms.audio_cursor = 0
            elif sms.main_cursor == 1:
                sms.submenu = SettingsSubmenu.CONTROLS
                sms.controls_cursor = 0
            elif sms.main_cursor == 2:
                sms.submenu = SettingsSubmenu.GAMEPLAY
                sms.gameplay_cursor = 0
            elif sms.main_cursor == 3:
                # Back - save settings and return to title
                self.settings_manager.save()
                self.state = MenuState.TITLE
                if self.on_back:
                    self.on_back()
                return True

            if self.on_select:
                self.on_select()

        # Back button
        if self._is_pressed('p1', 'special', p1_inputs) or self._is_pressed('p2', 'special', p2_inputs):
            self.settings_manager.save()
            self.state = MenuState.TITLE
            if self.on_back:
                self.on_back()

        return True

    def _update_settings_audio(self, p1_inputs: dict, p2_inputs: dict) -> bool:
        """Update audio settings submenu."""
        sms = self.settings_menu_state
        audio = self.settings_manager.settings.audio
        moved = False

        # Navigation up/down
        if self._is_pressed('p1', 'up', p1_inputs) or self._is_pressed('p2', 'up', p2_inputs):
            sms.audio_cursor = (sms.audio_cursor - 1) % 4
            sms.cursor_anim = 0
            moved = True
        elif self._is_pressed('p1', 'down', p1_inputs) or self._is_pressed('p2', 'down', p2_inputs):
            sms.audio_cursor = (sms.audio_cursor + 1) % 4
            sms.cursor_anim = 0
            moved = True

        if moved and self.on_cursor_move:
            self.on_cursor_move()

        # Left/right to change values
        left = self._is_pressed('p1', 'left', p1_inputs) or self._is_pressed('p2', 'left', p2_inputs)
        right = self._is_pressed('p1', 'right', p1_inputs) or self._is_pressed('p2', 'right', p2_inputs)

        if sms.audio_cursor == 0:  # Master volume
            if left:
                audio.master_volume = max(0.0, audio.master_volume - 0.1)
                if self.on_settings_changed:
                    self.on_settings_changed()
            elif right:
                audio.master_volume = min(1.0, audio.master_volume + 0.1)
                if self.on_settings_changed:
                    self.on_settings_changed()
        elif sms.audio_cursor == 1:  # SFX volume
            if left:
                audio.sfx_volume = max(0.0, audio.sfx_volume - 0.1)
                if self.on_settings_changed:
                    self.on_settings_changed()
            elif right:
                audio.sfx_volume = min(1.0, audio.sfx_volume + 0.1)
                if self.on_settings_changed:
                    self.on_settings_changed()
        elif sms.audio_cursor == 2:  # Mute toggle
            if left or right:
                audio.muted = not audio.muted
                if self.on_settings_changed:
                    self.on_settings_changed()
                if self.on_cursor_move:
                    self.on_cursor_move()

        # Selection (toggle mute or go back)
        if (self._is_pressed('p1', 'attack', p1_inputs) or
            self._is_pressed('p1', 'jump', p1_inputs) or
            self._is_pressed('p2', 'attack', p2_inputs) or
            self._is_pressed('p2', 'jump', p2_inputs)):

            if sms.audio_cursor == 2:  # Toggle mute
                audio.muted = not audio.muted
                if self.on_settings_changed:
                    self.on_settings_changed()
            elif sms.audio_cursor == 3:  # Back
                sms.submenu = SettingsSubmenu.MAIN
                if self.on_back:
                    self.on_back()

            if self.on_select:
                self.on_select()

        # Back button
        if self._is_pressed('p1', 'special', p1_inputs) or self._is_pressed('p2', 'special', p2_inputs):
            sms.submenu = SettingsSubmenu.MAIN
            if self.on_back:
                self.on_back()

        return True

    def _update_settings_controls(self, p1_inputs: dict, p2_inputs: dict) -> bool:
        """Update controls settings submenu."""
        sms = self.settings_menu_state
        moved = False

        # Navigation
        if self._is_pressed('p1', 'up', p1_inputs) or self._is_pressed('p2', 'up', p2_inputs):
            sms.controls_cursor = (sms.controls_cursor - 1) % 3
            sms.cursor_anim = 0
            moved = True
        elif self._is_pressed('p1', 'down', p1_inputs) or self._is_pressed('p2', 'down', p2_inputs):
            sms.controls_cursor = (sms.controls_cursor + 1) % 3
            sms.cursor_anim = 0
            moved = True

        if moved and self.on_cursor_move:
            self.on_cursor_move()

        # Left/right to cycle gamepad mapping
        left = self._is_pressed('p1', 'left', p1_inputs) or self._is_pressed('p2', 'left', p2_inputs)
        right = self._is_pressed('p1', 'right', p1_inputs) or self._is_pressed('p2', 'right', p2_inputs)

        settings = self.settings_manager.settings
        if sms.controls_cursor == 0:  # P1 mapping
            if left or right:
                if settings.player1_controls.gamepad_mapping == 'default':
                    settings.player1_controls.gamepad_mapping = 'smash'
                else:
                    settings.player1_controls.gamepad_mapping = 'default'
                if self.on_cursor_move:
                    self.on_cursor_move()
        elif sms.controls_cursor == 1:  # P2 mapping
            if left or right:
                if settings.player2_controls.gamepad_mapping == 'default':
                    settings.player2_controls.gamepad_mapping = 'smash'
                else:
                    settings.player2_controls.gamepad_mapping = 'default'
                if self.on_cursor_move:
                    self.on_cursor_move()

        # Selection
        if (self._is_pressed('p1', 'attack', p1_inputs) or
            self._is_pressed('p1', 'jump', p1_inputs) or
            self._is_pressed('p2', 'attack', p2_inputs) or
            self._is_pressed('p2', 'jump', p2_inputs)):

            if sms.controls_cursor == 2:  # Back
                sms.submenu = SettingsSubmenu.MAIN
                if self.on_back:
                    self.on_back()
                return True

            if self.on_select:
                self.on_select()

        # Back button
        if self._is_pressed('p1', 'special', p1_inputs) or self._is_pressed('p2', 'special', p2_inputs):
            sms.submenu = SettingsSubmenu.MAIN
            if self.on_back:
                self.on_back()

        return True

    def _update_settings_gameplay(self, p1_inputs: dict, p2_inputs: dict) -> bool:
        """Update gameplay settings submenu."""
        sms = self.settings_menu_state
        gameplay = self.settings_manager.settings.gameplay
        moved = False

        # Navigation
        if self._is_pressed('p1', 'up', p1_inputs) or self._is_pressed('p2', 'up', p2_inputs):
            sms.gameplay_cursor = (sms.gameplay_cursor - 1) % 5
            sms.cursor_anim = 0
            moved = True
        elif self._is_pressed('p1', 'down', p1_inputs) or self._is_pressed('p2', 'down', p2_inputs):
            sms.gameplay_cursor = (sms.gameplay_cursor + 1) % 5
            sms.cursor_anim = 0
            moved = True

        if moved and self.on_cursor_move:
            self.on_cursor_move()

        # Left/right to change values
        left = self._is_pressed('p1', 'left', p1_inputs) or self._is_pressed('p2', 'left', p2_inputs)
        right = self._is_pressed('p1', 'right', p1_inputs) or self._is_pressed('p2', 'right', p2_inputs)

        if sms.gameplay_cursor == 0:  # Default stocks
            if left:
                gameplay.default_stock_count = max(1, gameplay.default_stock_count - 1)
            elif right:
                gameplay.default_stock_count = min(5, gameplay.default_stock_count + 1)
        elif sms.gameplay_cursor == 1:  # Default time
            time_options = [0, 3, 5, 8]
            current_idx = time_options.index(gameplay.default_time_limit) if gameplay.default_time_limit in time_options else 0
            if left:
                current_idx = max(0, current_idx - 1)
            elif right:
                current_idx = min(len(time_options) - 1, current_idx + 1)
            gameplay.default_time_limit = time_options[current_idx]
        elif sms.gameplay_cursor == 2:  # Damage numbers
            if left or right:
                gameplay.show_damage_numbers = not gameplay.show_damage_numbers
                if self.on_cursor_move:
                    self.on_cursor_move()
        elif sms.gameplay_cursor == 3:  # Visual theme
            if left or right:
                self.cycle_theme(-1 if left else 1)

        # Selection
        if (self._is_pressed('p1', 'attack', p1_inputs) or
            self._is_pressed('p1', 'jump', p1_inputs) or
            self._is_pressed('p2', 'attack', p2_inputs) or
            self._is_pressed('p2', 'jump', p2_inputs)):

            if sms.gameplay_cursor == 2:  # Toggle damage numbers
                gameplay.show_damage_numbers = not gameplay.show_damage_numbers
            elif sms.gameplay_cursor == 3:  # Next theme
                self.cycle_theme(1)
            elif sms.gameplay_cursor == 4:  # Back
                sms.submenu = SettingsSubmenu.MAIN
                if self.on_back:
                    self.on_back()
                return True

            if self.on_select:
                self.on_select()

        # Back button
        if self._is_pressed('p1', 'special', p1_inputs) or self._is_pressed('p2', 'special', p2_inputs):
            sms.submenu = SettingsSubmenu.MAIN
            if self.on_back:
                self.on_back()

        return True

    def cycle_theme(self, step: int = 1) -> str:
        """Switch the visual theme setting and notify the game."""
        gameplay = self.settings_manager.settings.gameplay
        names = self.available_themes or ['classic']
        i = names.index(gameplay.theme) if gameplay.theme in names else 0
        gameplay.theme = names[(i + step) % len(names)]
        if self.on_theme_changed:
            self.on_theme_changed(gameplay.theme)
        if self.on_cursor_move:
            self.on_cursor_move()
        return gameplay.theme

    def is_ready_to_start(self) -> bool:
        """Check if all selections are made."""
        return self.state == MenuState.READY


class MenuRenderer:
    """
    Renders menu screens using X11.
    Works with the existing X11Renderer.
    """

    def __init__(self, renderer):
        """Initialize with an X11Renderer instance."""
        self.renderer = renderer
        self.menu = None

    def set_menu(self, menu: MenuSystem):
        """Set the menu system to render."""
        self.menu = menu

    def render(self, frame: int):
        """Render the current menu screen."""
        if not self.menu:
            return

        if getattr(self.renderer, 'themed', False):
            if not hasattr(self, '_themed_view'):
                from menu_view import ThemedMenuView
                self._themed_view = ThemedMenuView(self.renderer)
            self._themed_view.menu = self.menu
            self._themed_view.render(frame)
            self.renderer.flip()
            return

        # Clear background
        self.renderer.set_color_rgb(20, 20, 30)
        self.renderer.draw_rect(0, 0, self.renderer.width, self.renderer.height, filled=True)

        if self.menu.state == MenuState.TITLE:
            self._render_title(frame)
        elif self.menu.state == MenuState.CHARACTER_SELECT:
            self._render_character_select(frame)
        elif self.menu.state == MenuState.STAGE_SELECT:
            self._render_stage_select(frame)
        elif self.menu.state == MenuState.MATCH_SETTINGS:
            self._render_match_settings(frame)
        elif self.menu.state == MenuState.SETTINGS:
            self._render_settings(frame)

        self.renderer.flip()

    def _render_title(self, frame: int):
        """Render title screen with mode selection."""
        center_x = self.renderer.width // 2
        center_y = self.renderer.height // 2

        # Title text - large and prominent
        self.renderer.set_color_rgb(255, 200, 50)
        self.renderer.draw_text(center_x - 120, 80, "STICK FIGURE FIGHTER")

        # Animated stick figures on sides
        self._render_title_stick_figures(frame)

        # Menu options
        options = ["VERSUS MODE", "TRAINING MODE", "REPLAYS", "SETTINGS", "QUIT"]
        option_colors = [
            (100, 255, 100),  # Green for versus
            (100, 200, 255),  # Blue for training
            (255, 200, 100),  # Orange for replays
            (200, 180, 255),  # Purple for settings
            (255, 100, 100),  # Red for quit
        ]
        option_descriptions = [
            "2 Player Local Battle",
            "Practice vs CPU",
            "Watch Saved Matches",
            "Audio, Controls & Gameplay",
            "Exit Game"
        ]

        option_y = center_y - 80
        for i, opt in enumerate(options):
            is_selected = self.menu.title_state.cursor == i
            color = option_colors[i]

            if is_selected:
                # Highlight box
                box_w = 220
                box_h = 45
                box_x = center_x - box_w // 2
                box_y = option_y - 8

                # Pulsing highlight
                pulse = int(math.sin(frame * 0.15) * 20) + 40
                self.renderer.set_color_rgb(pulse, pulse, pulse + 20)
                self.renderer.draw_rect(box_x, box_y, box_w, box_h, filled=True)

                # Border
                self.renderer.set_color_rgb(*color)
                self.renderer.draw_rect(box_x, box_y, box_w, box_h, filled=False)

                # Cursor arrows
                arrow_bob = int(math.sin(frame * 0.2) * 3)
                self.renderer.draw_line(box_x - 15 + arrow_bob, option_y + 8, box_x - 5 + arrow_bob, option_y + 3)
                self.renderer.draw_line(box_x - 15 + arrow_bob, option_y + 8, box_x - 5 + arrow_bob, option_y + 13)
                self.renderer.draw_line(box_x + box_w + 15 - arrow_bob, option_y + 8, box_x + box_w + 5 - arrow_bob, option_y + 3)
                self.renderer.draw_line(box_x + box_w + 15 - arrow_bob, option_y + 8, box_x + box_w + 5 - arrow_bob, option_y + 13)

                # Option text (bright)
                self.renderer.set_color('white')
            else:
                # Dimmed text
                self.renderer.set_color_rgb(120, 120, 130)

            self.renderer.draw_text(center_x - len(opt) * 4, option_y + 5, opt)

            # Description for selected
            if is_selected:
                self.renderer.set_color_rgb(150, 150, 160)
                self.renderer.draw_text(center_x - len(option_descriptions[i]) * 3, option_y + 28, option_descriptions[i])

            option_y += 55

        # Controls hint
        self.renderer.set_color_rgb(80, 80, 90)
        self.renderer.draw_text(center_x - 100, self.renderer.height - 30,
                               "Up/Down: Select | Attack/Jump: Confirm")

    def _render_title_stick_figures(self, frame: int):
        """Render animated stick figures on title screen."""
        # Left figure (red/RUSH-like)
        left_x = 150
        left_y = 300

        # Simple idle animation
        bob = int(math.sin(frame * 0.1) * 3)

        self.renderer.set_color_rgb(255, 100, 100)
        # Head
        self.renderer.draw_circle(left_x, left_y - 50 + bob, 20, filled=True)
        # Body
        self.renderer.draw_line(left_x, left_y - 30 + bob, left_x, left_y + 30 + bob)
        # Arms (fighting pose)
        self.renderer.draw_line(left_x, left_y - 10 + bob, left_x + 30, left_y - 20 + bob)
        self.renderer.draw_line(left_x, left_y - 10 + bob, left_x - 20, left_y + bob)
        # Legs
        self.renderer.draw_line(left_x, left_y + 30 + bob, left_x - 15, left_y + 70 + bob)
        self.renderer.draw_line(left_x, left_y + 30 + bob, left_x + 15, left_y + 70 + bob)

        # Right figure (blue/ZONE-like)
        right_x = self.renderer.width - 150
        right_y = 300
        bob2 = int(math.sin(frame * 0.1 + math.pi) * 3)

        self.renderer.set_color_rgb(100, 200, 255)
        # Head
        self.renderer.draw_circle(right_x, right_y - 50 + bob2, 20, filled=True)
        # Body
        self.renderer.draw_line(right_x, right_y - 30 + bob2, right_x, right_y + 30 + bob2)
        # Arms (defensive pose)
        self.renderer.draw_line(right_x, right_y - 10 + bob2, right_x - 30, right_y - 20 + bob2)
        self.renderer.draw_line(right_x, right_y - 10 + bob2, right_x + 20, right_y + bob2)
        # Legs
        self.renderer.draw_line(right_x, right_y + 30 + bob2, right_x - 15, right_y + 70 + bob2)
        self.renderer.draw_line(right_x, right_y + 30 + bob2, right_x + 15, right_y + 70 + bob2)

    def _render_character_select(self, frame: int):
        """Render character selection screen."""
        is_training = self.menu.game_mode == GameModeSelection.TRAINING

        # Title
        self.renderer.set_color('white')
        if is_training:
            self.renderer.draw_text(self.renderer.width // 2 - 120, 30, "TRAINING MODE - SELECT FIGHTERS")
            # Show training mode indicator
            self.renderer.set_color_rgb(100, 200, 255)
            self.renderer.draw_text(self.renderer.width // 2 - 100, 55, "P1 selects, CPU controls P2")
        else:
            self.renderer.draw_text(self.renderer.width // 2 - 80, 40, "SELECT YOUR FIGHTER")

        # Character slots (4 characters in a row)
        slot_width = 200
        slot_height = 280
        total_width = slot_width * 4 + 30 * 3  # 4 slots + 3 gaps
        start_x = (self.renderer.width - total_width) // 2
        slot_y = 100

        for i, char_type in enumerate(self.menu.CHARACTERS):
            slot_x = start_x + i * (slot_width + 30)
            self._render_character_slot(slot_x, slot_y, slot_width, slot_height, char_type, i, frame)

        # Player cursors
        p1_slot_x = start_x + self.menu.char_select.p1_cursor * (slot_width + 30)
        p2_slot_x = start_x + self.menu.char_select.p2_cursor * (slot_width + 30)

        # P1 cursor (above slot)
        if self.menu.char_select.p1_locked is None:
            cursor_bob = int(math.sin(frame * 0.15) * 5)
            self._render_cursor(p1_slot_x + slot_width // 2, slot_y - 20 + cursor_bob,
                              (255, 100, 100), "P1", frame)
        else:
            # Locked indicator
            self.renderer.set_color_rgb(255, 100, 100)
            self.renderer.draw_text(p1_slot_x + slot_width // 2 - 20, slot_y - 25, "P1 READY")

        # P2 cursor (below slot) - shows "CPU" in training mode
        p2_label = "CPU" if is_training else "P2"
        if self.menu.char_select.p2_locked is None:
            cursor_bob = int(math.sin(frame * 0.15 + math.pi) * 5)
            p2_color = (150, 150, 180) if is_training else (100, 100, 255)
            self._render_cursor(p2_slot_x + slot_width // 2, slot_y + slot_height + 30 + cursor_bob,
                              p2_color, p2_label, frame, down=True)
        else:
            # Locked indicator
            if is_training:
                self.renderer.set_color_rgb(150, 150, 180)
                self.renderer.draw_text(p2_slot_x + slot_width // 2 - 25, slot_y + slot_height + 35, "CPU READY")
            else:
                self.renderer.set_color_rgb(100, 100, 255)
                self.renderer.draw_text(p2_slot_x + slot_width // 2 - 20, slot_y + slot_height + 35, "P2 READY")

        # Controls hint
        self.renderer.set_color_rgb(100, 100, 100)
        if is_training:
            self.renderer.draw_text(20, self.renderer.height - 30,
                                   "L/R: P1 Char | U/D: CPU Char | Attack/Jump: Confirm | Special: Back")
        else:
            self.renderer.draw_text(20, self.renderer.height - 30,
                                   "Left/Right: Select | Attack/Jump: Confirm | Special: Cancel")

    def _render_character_slot(self, x: int, y: int, width: int, height: int,
                               char_type: CharacterType, index: int, frame: int):
        """Render a single character slot."""
        info = self.menu.get_character_info(char_type)
        r, g, b = info['color']

        # Background
        self.renderer.set_color_rgb(40, 40, 50)
        self.renderer.draw_rect(x, y, width, height, filled=True)

        # Border (highlight if selected by either player)
        p1_here = self.menu.char_select.p1_cursor == index
        p2_here = self.menu.char_select.p2_cursor == index
        p1_locked = self.menu.char_select.p1_locked == char_type
        p2_locked = self.menu.char_select.p2_locked == char_type

        if p1_locked or p2_locked:
            # Gold border for locked
            self.renderer.set_color_rgb(255, 200, 50)
            self.renderer.draw_rect(x-2, y-2, width+4, height+4, filled=False)
        elif p1_here or p2_here:
            # Colored border for hover
            if p1_here:
                self.renderer.set_color_rgb(255, 100, 100)
            else:
                self.renderer.set_color_rgb(100, 100, 255)
            self.renderer.draw_rect(x-1, y-1, width+2, height+2, filled=False)
        else:
            self.renderer.set_color_rgb(80, 80, 80)
            self.renderer.draw_rect(x, y, width, height, filled=False)

        # Character silhouette (larger stick figure with animation)
        center_x = x + width // 2
        silhouette_y = y + 60

        # Check if this character is locked by either player
        p1_locked = self.menu.char_select.p1_locked == char_type
        p2_locked = self.menu.char_select.p2_locked == char_type
        is_locked = p1_locked or p2_locked

        self._render_character_silhouette_animated(center_x, silhouette_y, r, g, b, frame, is_locked, char_type)

        # Character name
        self.renderer.set_color_rgb(r, g, b)
        self.renderer.draw_text(center_x - len(info['name']) * 4, y + 160, info['name'])

        # Description
        self.renderer.set_color_rgb(180, 180, 180)
        desc = info['description']
        self.renderer.draw_text(x + 10, y + 185, desc)

        # Stats bars
        stats_y = y + 210
        self._render_stat_bar(x + 10, stats_y, "SPD", info['speed'], (100, 200, 255))
        self._render_stat_bar(x + 10, stats_y + 18, "WGT", info['weight'], (200, 150, 100))
        self._render_stat_bar(x + 10, stats_y + 36, "PWR", info['power'], (255, 100, 100))
        self._render_stat_bar(x + 10, stats_y + 54, "REC", info['recovery'], (100, 255, 150))

    def _render_character_silhouette(self, cx: int, cy: int, r: int, g: int, b: int):
        """Render a stick figure silhouette (basic version)."""
        self.renderer.set_color_rgb(r, g, b)

        # Head
        self.renderer.draw_circle(cx, cy - 30, 15, filled=True)

        # Body
        self.renderer.draw_line(cx, cy - 15, cx, cy + 25)

        # Arms
        self.renderer.draw_line(cx, cy - 5, cx - 20, cy + 10)
        self.renderer.draw_line(cx, cy - 5, cx + 20, cy + 10)

        # Legs
        self.renderer.draw_line(cx, cy + 25, cx - 15, cy + 55)
        self.renderer.draw_line(cx, cy + 25, cx + 15, cy + 55)

    def _render_character_silhouette_animated(self, cx: int, cy: int, r: int, g: int, b: int,
                                              frame: int, is_locked: bool,
                                              char_type: Optional[CharacterType] = None):
        """Render a LARGER animated stick figure silhouette (Cycle 9 + Cycle 17).

        When is_locked is True, show attack pose. Otherwise, show breathing idle.
        Scale is 2.0x compared to basic silhouette.

        Cycle 17: Character-specific preview effects:
        - RUSH: Speed afterimage trail
        - ZONE: Orbiting projectile particles
        - GRAB: Power flex with dust
        - FLEX: Shield shimmer glow
        """
        self.renderer.set_color_rgb(r, g, b)

        # Scale factor for larger character preview
        scale = 2.0

        # Breathing animation - subtle vertical bob
        breath = math.sin(frame * 0.08) * 2 * scale

        # Head offset for breathing
        head_bob = int(breath * 0.5)

        # Scaled dimensions
        head_radius = int(15 * scale)
        neck_to_hip = int(40 * scale)
        arm_length = int(25 * scale)
        leg_length = int(35 * scale)

        # Head position
        head_y = int(cy - 35 * scale + head_bob)
        neck_y = head_y + head_radius
        hip_y = neck_y + neck_to_hip
        shoulder_y = neck_y + int(10 * scale)

        # =======================================================================
        # Cycle 17: RUSH - Speed Afterimage Trail (draw BEFORE main figure)
        # =======================================================================
        if char_type == CharacterType.RUSH:
            # Draw 4 ghost copies trailing behind, with decreasing alpha (simulated)
            for i in range(4, 0, -1):
                ghost_offset = 8 + i * 6
                alpha_factor = 1.0 - (i * 0.22)
                ghost_x = cx - ghost_offset + int(math.sin(frame * 0.3 + i) * 3)

                # Ghost color (faded)
                gr = int(r * alpha_factor * 0.6)
                gg = int(g * alpha_factor * 0.6)
                gb = int(b * alpha_factor * 0.6)
                self.renderer.set_color_rgb(gr, gg, gb)

                # Draw ghost stick figure (simplified)
                ghost_head_y = int(cy - 35 * scale + int(math.sin(frame * 0.1 + i * 0.3) * 2))
                self.renderer.draw_circle(ghost_x, ghost_head_y, int(head_radius * 0.9), filled=False)
                ghost_neck_y = ghost_head_y + int(head_radius * 0.9)
                ghost_hip_y = ghost_neck_y + int(neck_to_hip * 0.9)
                ghost_shoulder_y = ghost_neck_y + int(10 * scale * 0.9)
                # Body
                self.renderer.draw_line(ghost_x, ghost_neck_y, ghost_x, ghost_hip_y)
                # Arms (simplified)
                self.renderer.draw_line(ghost_x, ghost_shoulder_y,
                                       ghost_x + int(15 * scale), ghost_shoulder_y + int(20 * scale))
                self.renderer.draw_line(ghost_x, ghost_shoulder_y,
                                       ghost_x - int(15 * scale), ghost_shoulder_y + int(20 * scale))
                # Legs
                self.renderer.draw_line(ghost_x, ghost_hip_y,
                                       ghost_x + int(10 * scale), ghost_hip_y + int(28 * scale))
                self.renderer.draw_line(ghost_x, ghost_hip_y,
                                       ghost_x - int(10 * scale), ghost_hip_y + int(28 * scale))

            # Reset to main color
            self.renderer.set_color_rgb(r, g, b)

        # =======================================================================
        # Cycle 17: ZONE - Orbiting Projectile Particles (draw BEFORE main figure)
        # =======================================================================
        if char_type == CharacterType.ZONE:
            # Draw 3 energy orbs orbiting the character
            orbit_radius = 40 * scale
            for i in range(3):
                angle = (frame * 3 + i * 120) % 360
                orb_x = cx + int(math.cos(math.radians(angle)) * orbit_radius)
                orb_y = int(cy - 20 * scale + math.sin(math.radians(angle)) * orbit_radius * 0.5)

                # Outer glow (cyan)
                self.renderer.set_color_rgb(100, 200, 255)
                self.renderer.draw_circle(orb_x, orb_y, 8, filled=True)

                # Inner core (white)
                self.renderer.set_color_rgb(220, 240, 255)
                self.renderer.draw_circle(orb_x, orb_y, 4, filled=True)

            # Reset to main color
            self.renderer.set_color_rgb(r, g, b)

        # =======================================================================
        # Cycle 17: FLEX - Shield Shimmer Glow (draw BEFORE main figure)
        # =======================================================================
        if char_type == CharacterType.FLEX:
            # Pulsing hexagonal shield
            pulse = math.sin(frame * 0.1) * 0.3 + 1.0  # 0.7 to 1.3
            shield_radius = 35 * scale * pulse
            shield_cy = cy - int(25 * scale)

            # Draw hexagon (6 points, slowly rotating)
            hex_points = []
            for i in range(6):
                angle = i * 60 + frame * 2
                hx = cx + int(math.cos(math.radians(angle)) * shield_radius)
                hy = shield_cy + int(math.sin(math.radians(angle)) * shield_radius * 0.6)
                hex_points.append((hx, hy))

            # Draw hexagon edges with shimmer
            for i in range(6):
                x1, y1 = hex_points[i]
                x2, y2 = hex_points[(i + 1) % 6]

                # Shimmer color based on position
                shimmer = int(math.sin(frame * 0.15 + i) * 50) + 150
                self.renderer.set_color_rgb(shimmer, shimmer + 50, shimmer)
                self.renderer.draw_line(x1, y1, x2, y2)

            # Reset to main color
            self.renderer.set_color_rgb(r, g, b)

        # =======================================================================
        # Cycle 17: GRAB - Power Flex Animation
        # =======================================================================
        grab_flexing = False
        grab_flex_progress = 0.0
        if char_type == CharacterType.GRAB:
            flex_cycle = frame % 90
            if flex_cycle < 20:
                grab_flexing = True
                grab_flex_progress = flex_cycle / 20.0

                # Ground dust particles at feet
                dust_count = int(flex_cycle / 4) + 1
                for d in range(dust_count):
                    dust_x = cx + int((d - dust_count // 2) * 12 + math.sin(frame * 0.5 + d) * 5)
                    dust_y = hip_y + leg_length - int(flex_cycle * 0.8) - d * 3
                    dust_size = max(2, 5 - d)

                    self.renderer.set_color_rgb(180, 150, 120)
                    self.renderer.draw_circle(dust_x, dust_y, dust_size, filled=True)

                # Reset to main color
                self.renderer.set_color_rgb(r, g, b)

        if is_locked:
            # ATTACK POSE: Arms raised, legs wide, dynamic stance
            punch_cycle = (frame % 30) / 30.0  # 0.5 second punch animation cycle

            # Head
            self.renderer.draw_circle(cx, head_y, head_radius, filled=True)

            # Body - slight lean forward
            lean = int(3 * scale)
            self.renderer.draw_line(cx, neck_y, cx + lean, hip_y)

            # Arms - punch pose (one arm forward, one back)
            # Right arm punching forward
            arm_extend = int(math.sin(punch_cycle * math.pi * 2) * 10 * scale)
            right_arm_x = cx + arm_length + abs(arm_extend)
            right_arm_y = shoulder_y - int(10 * scale) - arm_extend // 2
            self.renderer.draw_line(cx + lean // 2, shoulder_y, right_arm_x, right_arm_y)

            # Left arm back/ready position
            left_arm_x = cx - int(15 * scale)
            left_arm_y = shoulder_y + int(5 * scale)
            self.renderer.draw_line(cx + lean // 2, shoulder_y, left_arm_x, left_arm_y)

            # Legs - wide fighting stance
            leg_spread = int(20 * scale)
            self.renderer.draw_line(cx + lean, hip_y, cx - leg_spread + lean, hip_y + leg_length)
            self.renderer.draw_line(cx + lean, hip_y, cx + leg_spread + lean, hip_y + leg_length)

            # Draw glow/power effect behind figure (simple lines)
            self.renderer.set_color_rgb(min(255, r + 50), min(255, g + 50), min(255, b + 50))
            for i in range(3):
                glow_offset = (frame * 3 + i * 120) % 360
                glow_x = cx + int(math.cos(math.radians(glow_offset)) * 25 * scale)
                glow_y = cy + int(math.sin(math.radians(glow_offset)) * 15 * scale)
                self.renderer.draw_line(glow_x - 3, glow_y, glow_x + 3, glow_y)
        else:
            # IDLE POSE: Relaxed with breathing animation
            # Cycle 17: Modify arms for GRAB flexing
            arm_raise = 0
            if grab_flexing:
                arm_raise = int(grab_flex_progress * 30 * scale)

            # Head (shake during GRAB flex)
            actual_cx = cx
            if grab_flexing:
                shake = int(math.sin(frame * 0.8) * 2)
                actual_cx = cx + shake
            self.renderer.draw_circle(actual_cx, head_y, head_radius, filled=True)

            # Body
            self.renderer.draw_line(actual_cx, neck_y, actual_cx, hip_y)

            # Arms - relaxed at sides with slight sway (or raised for GRAB)
            arm_sway = int(math.sin(frame * 0.06) * 3 * scale)

            if grab_flexing:
                # Arms raised in flex pose
                right_arm_end_x = actual_cx + int(25 * scale)
                right_arm_end_y = shoulder_y - arm_raise
                self.renderer.draw_line(actual_cx, shoulder_y, right_arm_end_x, right_arm_end_y)

                left_arm_end_x = actual_cx - int(25 * scale)
                left_arm_end_y = shoulder_y - arm_raise
                self.renderer.draw_line(actual_cx, shoulder_y, left_arm_end_x, left_arm_end_y)
            else:
                # Right arm
                right_arm_end_x = actual_cx + int(20 * scale) + arm_sway
                right_arm_end_y = shoulder_y + int(25 * scale)
                self.renderer.draw_line(actual_cx, shoulder_y, right_arm_end_x, right_arm_end_y)

                # Left arm
                left_arm_end_x = actual_cx - int(20 * scale) - arm_sway
                left_arm_end_y = shoulder_y + int(25 * scale)
                self.renderer.draw_line(actual_cx, shoulder_y, left_arm_end_x, left_arm_end_y)

            # Legs
            leg_spread = int(12 * scale)
            self.renderer.draw_line(actual_cx, hip_y, actual_cx - leg_spread, hip_y + leg_length)
            self.renderer.draw_line(actual_cx, hip_y, actual_cx + leg_spread, hip_y + leg_length)

    def _render_stat_bar(self, x: int, y: int, label: str, value: int, color: Tuple[int, int, int]):
        """Render a stat bar (0-10 scale)."""
        r, g, b = color
        bar_width = 100

        # Label
        self.renderer.set_color_rgb(150, 150, 150)
        self.renderer.draw_text(x, y, label)

        # Background
        self.renderer.set_color_rgb(50, 50, 50)
        self.renderer.draw_rect(x + 35, y - 2, bar_width, 10, filled=True)

        # Fill
        fill_width = int(bar_width * (value / 10))
        if fill_width > 0:
            self.renderer.set_color_rgb(r, g, b)
            self.renderer.draw_rect(x + 35, y - 2, fill_width, 10, filled=True)

    def _render_cursor(self, x: int, y: int, color: Tuple[int, int, int],
                       label: str, frame: int, down: bool = False):
        """Render a player cursor (arrow)."""
        r, g, b = color
        self.renderer.set_color_rgb(r, g, b)

        # Arrow shape pointing down (or up)
        if down:
            # Arrow pointing down
            self.renderer.draw_line(x, y + 15, x - 10, y)
            self.renderer.draw_line(x, y + 15, x + 10, y)
            self.renderer.draw_line(x - 10, y, x + 10, y)
        else:
            # Arrow pointing down at the slot
            self.renderer.draw_line(x, y, x - 10, y - 15)
            self.renderer.draw_line(x, y, x + 10, y - 15)
            self.renderer.draw_line(x - 10, y - 15, x + 10, y - 15)

        # Label
        self.renderer.draw_text(x - 8, y - (25 if not down else -5), label)

    def _render_stage_select(self, frame: int):
        """Render stage selection screen."""
        # Title
        self.renderer.set_color('white')
        self.renderer.draw_text(self.renderer.width // 2 - 60, 40, "SELECT STAGE")

        # Stage preview boxes
        slot_width = 400
        slot_height = 300
        gap = 50
        total_width = slot_width * 2 + gap
        start_x = (self.renderer.width - total_width) // 2
        slot_y = 120

        for i, stage_name in enumerate(self.menu.STAGES):
            slot_x = start_x + i * (slot_width + gap)
            self._render_stage_slot(slot_x, slot_y, slot_width, slot_height,
                                   stage_name, self.menu.STAGE_NAMES[i], i, frame)

        # Cursor
        cursor_x = start_x + self.menu.stage_select.cursor * (slot_width + gap) + slot_width // 2
        cursor_bob = int(math.sin(frame * 0.15) * 5)

        self.renderer.set_color('white')
        # Arrow above slot
        self.renderer.draw_line(cursor_x, slot_y - 20 + cursor_bob,
                               cursor_x - 15, slot_y - 35 + cursor_bob)
        self.renderer.draw_line(cursor_x, slot_y - 20 + cursor_bob,
                               cursor_x + 15, slot_y - 35 + cursor_bob)

        # Characters selected (show at bottom)
        self._render_selected_characters(slot_y + slot_height + 40)

        # Controls hint
        self.renderer.set_color_rgb(100, 100, 100)
        self.renderer.draw_text(20, self.renderer.height - 30,
                               "Left/Right: Select | Attack/Jump: Start Match | Special: Back")

    def _render_stage_slot(self, x: int, y: int, width: int, height: int,
                           stage_id: str, stage_name: str, index: int, frame: int):
        """
        Render a stage selection slot with enhanced thumbnail preview (Cycle 9 + Cycle 18).

        Cycle 18 enhancements:
        - Draw parallax layer color stripes at top of thumbnail
        - Improved platform layout preview with scale matching actual stage
        - Visual indication of stage characteristics
        """
        is_selected = self.menu.stage_select.cursor == index

        # Background with gradient effect (animated for selected)
        if is_selected:
            # Animated gradient background for selected stage
            gradient_phase = (frame * 2) % 60
            for row in range(height // 4):
                # Create scrolling gradient effect
                row_y = y + row * 4
                color_val = 30 + int(math.sin((row + gradient_phase) * 0.15) * 10)
                self.renderer.set_color_rgb(color_val, color_val + 5, color_val + 15)
                self.renderer.draw_rect(x, row_y, width, 4, filled=True)
        else:
            self.renderer.set_color_rgb(30, 35, 45)
            self.renderer.draw_rect(x, y, width, height, filled=True)

        # Cycle 18: Draw parallax layer color preview strips at top
        self._render_parallax_color_strips(x, y, width, stage_id, is_selected, frame)

        # Border
        if is_selected:
            # Animated glowing border
            glow = int(math.sin(frame * 0.1) * 30) + 225
            self.renderer.set_color_rgb(glow, glow - 55, 50)
            self.renderer.draw_rect(x-2, y-2, width+4, height+4, filled=False)
            self.renderer.draw_rect(x-1, y-1, width+2, height+2, filled=False)
        else:
            self.renderer.set_color_rgb(80, 80, 80)
            self.renderer.draw_rect(x, y, width, height, filled=False)

        # Stage preview (simplified platform layout)
        preview_y = y + 50

        # Draw animated background elements for selected stage
        if is_selected:
            self._render_stage_background_animated(x, y, width, height, stage_id, frame)

        # Cycle 18: Enhanced platform preview with better proportions
        self._render_platform_layout_preview(x, y, width, height, stage_id, is_selected, frame)

        # Stage name with banner
        name_y = y + height - 35
        if is_selected:
            # Name banner background
            banner_w = len(stage_name) * 8 + 20
            banner_x = x + width // 2 - banner_w // 2
            self.renderer.set_color_rgb(50, 50, 60)
            self.renderer.draw_rect(banner_x, name_y - 5, banner_w, 25, filled=True)
            self.renderer.set_color_rgb(255, 200, 50)
            self.renderer.draw_rect(banner_x, name_y - 5, banner_w, 25, filled=False)

        self.renderer.set_color('white')
        self.renderer.draw_text(x + width // 2 - len(stage_name) * 4, name_y, stage_name)

        # Cycle 18: Draw stage info tag
        self._render_stage_info_tag(x, y, width, stage_id, is_selected)

    def _render_parallax_color_strips(self, x: int, y: int, width: int, stage_id: str,
                                       is_selected: bool, frame: int):
        """
        Cycle 18: Draw parallax layer color preview strips at top of thumbnail.

        Shows the characteristic colors of each stage's background layers.
        """
        strip_height = 8
        strip_y = y + 5

        if stage_id == "battlefield":
            # Battlefield colors: dark blue-gray mountains, dark green trees, lighter green foliage
            colors = [
                (30, 35, 50),   # Far: Mountains
                (40, 50, 45),   # Mid: Trees
                (50, 60, 55),   # Near: Foliage
            ]
        else:
            # Final Destination colors: white stars, purple nebula, gray debris
            colors = [
                (255, 255, 255),  # Far: Stars
                (60, 40, 80),     # Mid: Nebula
                (70, 70, 80),     # Near: Debris
            ]

        strip_width = (width - 20) // len(colors)
        for i, (r, g, b) in enumerate(colors):
            sx = x + 10 + i * strip_width

            # Animate for selected
            if is_selected:
                pulse = int(math.sin(frame * 0.1 + i) * 10)
                r = min(255, r + pulse)
                g = min(255, g + pulse)
                b = min(255, b + pulse)

            self.renderer.set_color_rgb(r, g, b)
            self.renderer.draw_rect(sx, strip_y, strip_width - 2, strip_height, filled=True)

        # Label
        self.renderer.set_color_rgb(120, 120, 140)
        self.renderer.draw_text(x + 10, strip_y + strip_height + 10, "ATMOSPHERE")

    def _render_platform_layout_preview(self, x: int, y: int, width: int, height: int,
                                          stage_id: str, is_selected: bool, frame: int):
        """
        Cycle 18: Enhanced platform layout preview with proper proportions.

        Draws miniature platform layout that matches actual stage geometry.
        """
        # Preview area bounds
        preview_x = x + 25
        preview_y = y + 80
        preview_width = width - 50
        preview_height = 150

        # Scale factor (actual stage is 1280x720)
        scale_x = preview_width / 1280
        scale_y = preview_height / 550  # Visible play area height

        if stage_id == "battlefield":
            # Main platform: x=240, y=550, width=800, height=50
            main_x = int(preview_x + 240 * scale_x)
            main_y = int(preview_y + 400 * scale_y)
            main_w = int(800 * scale_x)
            main_h = int(30 * scale_y)

            # Draw glow for selected
            if is_selected:
                self.renderer.set_color_rgb(140, 120, 80)
                self.renderer.draw_rect(main_x - 2, main_y - 2, main_w + 4, main_h + 4, filled=True)

            # Main platform
            self.renderer.set_color_rgb(100, 80, 60)
            self.renderer.draw_rect(main_x, main_y, main_w, main_h, filled=True)

            # Left platform: x=300, y=420, width=180
            left_x = int(preview_x + 300 * scale_x)
            left_y = int(preview_y + 270 * scale_y)
            left_w = int(180 * scale_x)
            self.renderer.set_color_rgb(80, 60, 50)
            self.renderer.draw_rect(left_x, left_y, left_w, 6, filled=True)

            # Right platform: x=780, y=420, width=180
            right_x = int(preview_x + 780 * scale_x)
            right_y = int(preview_y + 270 * scale_y)
            right_w = int(180 * scale_x)
            self.renderer.draw_rect(right_x, right_y, right_w, 6, filled=True)

            # Top platform: x=500, y=290, width=280
            top_x = int(preview_x + 500 * scale_x)
            top_y = int(preview_y + 140 * scale_y)
            top_w = int(280 * scale_x)
            self.renderer.set_color_rgb(70, 55, 45)
            self.renderer.draw_rect(top_x, top_y, top_w, 6, filled=True)

            # Platform count label
            self.renderer.set_color_rgb(100, 255, 100)
            self.renderer.draw_text(x + width - 45, preview_y - 5, "3 PLATS")

        else:
            # Final Destination: x=200, y=550, width=880
            main_x = int(preview_x + 200 * scale_x)
            main_y = int(preview_y + 400 * scale_y)
            main_w = int(880 * scale_x)
            main_h = int(30 * scale_y)

            # Draw glow for selected
            if is_selected:
                self.renderer.set_color_rgb(140, 120, 80)
                self.renderer.draw_rect(main_x - 2, main_y - 2, main_w + 4, main_h + 4, filled=True)

            # Main platform
            self.renderer.set_color_rgb(100, 80, 60)
            self.renderer.draw_rect(main_x, main_y, main_w, main_h, filled=True)

            # "FLAT" indicator
            self.renderer.set_color_rgb(255, 200, 100)
            self.renderer.draw_text(x + width - 40, preview_y - 5, "FLAT")

        # Draw stick figure silhouettes to show scale
        if is_selected:
            self._render_scale_figures(preview_x, preview_y, preview_width, preview_height, scale_x, scale_y, frame)

    def _render_scale_figures(self, preview_x: int, preview_y: int, preview_width: int,
                               preview_height: int, scale_x: float, scale_y: float, frame: int):
        """Draw tiny stick figure silhouettes to show platform scale."""
        # Fighter approximate position for Battlefield
        fighter_height = int(60 * scale_y)
        fighter_y = int(preview_y + 400 * scale_y - fighter_height - 5)

        # Left figure
        f1_x = int(preview_x + 400 * scale_x)
        # Right figure
        f2_x = int(preview_x + 880 * scale_x)

        # Animate a subtle bounce
        bounce = int(math.sin(frame * 0.15) * 3)

        # Draw simple stick figure silhouettes
        self.renderer.set_color_rgb(200, 200, 200)
        for fx in [f1_x, f2_x]:
            # Head
            self.renderer.draw_circle(fx, fighter_y + bounce, 4, filled=True)
            # Body
            self.renderer.draw_line(fx, fighter_y + 4 + bounce, fx, fighter_y + fighter_height - 8 + bounce)
            # Arms
            self.renderer.draw_line(fx - 6, fighter_y + 12 + bounce, fx + 6, fighter_y + 12 + bounce)
            # Legs
            self.renderer.draw_line(fx, fighter_y + fighter_height - 8 + bounce,
                                   fx - 5, fighter_y + fighter_height + bounce)
            self.renderer.draw_line(fx, fighter_y + fighter_height - 8 + bounce,
                                   fx + 5, fighter_y + fighter_height + bounce)

    def _render_stage_info_tag(self, x: int, y: int, width: int, stage_id: str, is_selected: bool):
        """
        Cycle 18: Draw info tag with stage characteristics.
        """
        tag_y = y + 30

        if stage_id == "battlefield":
            tag_text = "STANDARD"
            tag_color = (100, 200, 100)  # Green
        else:
            tag_text = "OMEGA"
            tag_color = (200, 100, 255)  # Purple

        tag_x = x + width // 2 - len(tag_text) * 3

        if is_selected:
            r, g, b = tag_color
        else:
            r, g, b = tag_color[0] // 2, tag_color[1] // 2, tag_color[2] // 2

        self.renderer.set_color_rgb(r, g, b)
        self.renderer.draw_text(tag_x, tag_y, tag_text)

    def _render_stage_background_animated(self, x: int, y: int, width: int, height: int,
                                          stage_id: str, frame: int):
        """Render animated background elements for stage preview (Cycle 9)."""
        preview_y = y + 50

        if stage_id == "battlefield":
            # Floating clouds/particles scrolling
            for i in range(5):
                cloud_x = x + ((frame * 2 + i * 80) % (width + 40)) - 20
                cloud_y = preview_y + 20 + i * 25
                self.renderer.set_color_rgb(60, 65, 80)
                self.renderer.draw_circle(cloud_x, cloud_y, 8 + i % 3, filled=True)
        else:
            # Final Destination - cosmic particles/stars
            for i in range(8):
                star_phase = (frame + i * 45) % 120
                star_alpha = abs(60 - star_phase) / 60.0
                brightness = int(50 + star_alpha * 80)

                star_x = x + 20 + ((i * 47 + frame) % (width - 40))
                star_y = preview_y + 10 + (i * 31) % (height - 120)

                self.renderer.set_color_rgb(brightness, brightness, brightness + 30)
                self.renderer.draw_line(star_x - 2, star_y, star_x + 2, star_y)
                self.renderer.draw_line(star_x, star_y - 2, star_x, star_y + 2)

    def _render_selected_characters(self, y: int):
        """Render the selected characters at the bottom."""
        center_x = self.renderer.width // 2

        # P1
        if self.menu.char_select.p1_locked:
            info = self.menu.get_character_info(self.menu.char_select.p1_locked)
            r, g, b = info['color']
            self.renderer.set_color_rgb(r, g, b)
            self.renderer.draw_text(center_x - 150, y, f"P1: {info['name']}")

        # VS
        self.renderer.set_color('white')
        self.renderer.draw_text(center_x - 10, y, "VS")

        # P2
        is_training = self.menu.game_mode == GameModeSelection.TRAINING
        if self.menu.char_select.p2_locked:
            info = self.menu.get_character_info(self.menu.char_select.p2_locked)
            r, g, b = info['color']
            self.renderer.set_color_rgb(r, g, b)
            label = "CPU" if is_training else "P2"
            self.renderer.draw_text(center_x + 80, y, f"{label}: {info['name']}")

    def _render_match_settings(self, frame: int):
        """Render match settings screen (stock count, time limit)."""
        center_x = self.renderer.width // 2
        mss = self.menu.match_settings_state

        # Title
        self.renderer.set_color('white')
        self.renderer.draw_text(center_x - 80, 50, "MATCH SETTINGS")

        # Show selected characters and stage at top
        self._render_match_preview(100)

        # Settings panel
        panel_y = 220
        panel_x = center_x - 200
        panel_w = 400
        panel_h = 250

        # Panel background
        self.renderer.set_color_rgb(35, 35, 45)
        self.renderer.draw_rect(panel_x, panel_y, panel_w, panel_h, filled=True)
        self.renderer.set_color_rgb(80, 80, 100)
        self.renderer.draw_rect(panel_x, panel_y, panel_w, panel_h, filled=False)

        # Settings rows
        row_y = panel_y + 30
        row_height = 60

        # Stock count (row 0)
        self._render_setting_row(
            center_x, row_y, "STOCKS",
            str(mss.stock_count),
            mss.cursor == 0,
            frame,
            show_arrows=True,
            min_val=1, max_val=5, current_val=mss.stock_count
        )
        row_y += row_height

        # Time limit (row 1)
        self._render_setting_row(
            center_x, row_y, "TIME LIMIT",
            TIME_LABELS[mss.time_limit_idx],
            mss.cursor == 1,
            frame,
            show_arrows=True,
            min_val=0, max_val=len(TIME_OPTIONS)-1, current_val=mss.time_limit_idx
        )
        row_y += row_height

        # Start button (row 2)
        self._render_start_button(center_x, row_y + 20, mss.cursor == 2, frame)

        # Controls hint
        self.renderer.set_color_rgb(100, 100, 100)
        self.renderer.draw_text(center_x - 140, self.renderer.height - 30,
                               "Up/Down: Navigate | Left/Right: Change | Attack/Jump: Start")

    def _render_match_preview(self, y: int):
        """Render character matchup preview above settings."""
        center_x = self.renderer.width // 2

        # P1 character
        if self.menu.char_select.p1_locked:
            info = self.menu.get_character_info(self.menu.char_select.p1_locked)
            r, g, b = info['color']
            self.renderer.set_color_rgb(r, g, b)
            self.renderer.draw_text(center_x - 180, y, f"P1: {info['name']}")
            # Small silhouette
            self._render_mini_stick_figure(center_x - 220, y + 40, r, g, b)

        # VS
        self.renderer.set_color_rgb(255, 200, 50)
        self.renderer.draw_text(center_x - 10, y + 20, "VS")

        # P2/CPU character
        is_training = self.menu.game_mode == GameModeSelection.TRAINING
        if self.menu.char_select.p2_locked:
            info = self.menu.get_character_info(self.menu.char_select.p2_locked)
            r, g, b = info['color']
            self.renderer.set_color_rgb(r, g, b)
            label = "CPU" if is_training else "P2"
            self.renderer.draw_text(center_x + 100, y, f"{label}: {info['name']}")
            # Small silhouette
            self._render_mini_stick_figure(center_x + 220, y + 40, r, g, b)

        # Stage name
        stage_name = self.menu.settings.stage_name.replace('_', ' ').title()
        self.renderer.set_color_rgb(150, 150, 160)
        self.renderer.draw_text(center_x - len(stage_name) * 4, y + 80, f"Stage: {stage_name}")

    def _render_mini_stick_figure(self, cx: int, cy: int, r: int, g: int, b: int):
        """Render a small stick figure silhouette."""
        self.renderer.set_color_rgb(r, g, b)
        # Head
        self.renderer.draw_circle(cx, cy - 15, 8, filled=True)
        # Body
        self.renderer.draw_line(cx, cy - 7, cx, cy + 12)
        # Arms
        self.renderer.draw_line(cx, cy - 2, cx - 10, cy + 5)
        self.renderer.draw_line(cx, cy - 2, cx + 10, cy + 5)
        # Legs
        self.renderer.draw_line(cx, cy + 12, cx - 7, cy + 28)
        self.renderer.draw_line(cx, cy + 12, cx + 7, cy + 28)

    def _render_setting_row(self, center_x: int, y: int, label: str, value: str,
                            selected: bool, frame: int, show_arrows: bool = False,
                            min_val: int = 0, max_val: int = 0, current_val: int = 0):
        """Render a settings row with label and value."""
        row_w = 350
        row_h = 45
        row_x = center_x - row_w // 2

        if selected:
            # Highlight background
            pulse = int(math.sin(frame * 0.15) * 15) + 35
            self.renderer.set_color_rgb(pulse, pulse, pulse + 15)
            self.renderer.draw_rect(row_x, y - 5, row_w, row_h, filled=True)

            # Border
            self.renderer.set_color_rgb(100, 200, 255)
            self.renderer.draw_rect(row_x, y - 5, row_w, row_h, filled=False)

        # Label
        if selected:
            self.renderer.set_color('white')
        else:
            self.renderer.set_color_rgb(150, 150, 160)
        self.renderer.draw_text(row_x + 20, y + 10, label)

        # Value
        value_x = row_x + row_w - 100
        if selected:
            self.renderer.set_color_rgb(100, 255, 100)
        else:
            self.renderer.set_color_rgb(180, 180, 190)
        self.renderer.draw_text(value_x, y + 10, value)

        # Arrows (if selected and has range)
        if selected and show_arrows:
            arrow_bob = int(math.sin(frame * 0.2) * 2)

            # Left arrow (if not at min)
            if current_val > min_val:
                self.renderer.set_color_rgb(200, 200, 220)
                ax = value_x - 25 - arrow_bob
                self.renderer.draw_line(ax, y + 15, ax + 10, y + 8)
                self.renderer.draw_line(ax, y + 15, ax + 10, y + 22)

            # Right arrow (if not at max)
            if current_val < max_val:
                self.renderer.set_color_rgb(200, 200, 220)
                ax = value_x + 50 + arrow_bob
                self.renderer.draw_line(ax, y + 15, ax - 10, y + 8)
                self.renderer.draw_line(ax, y + 15, ax - 10, y + 22)

    def _render_start_button(self, center_x: int, y: int, selected: bool, frame: int):
        """Render the START MATCH button."""
        btn_w = 200
        btn_h = 50
        btn_x = center_x - btn_w // 2

        if selected:
            # Pulsing green button
            pulse = int(math.sin(frame * 0.15) * 30) + 80
            self.renderer.set_color_rgb(30, pulse, 30)
            self.renderer.draw_rect(btn_x, y, btn_w, btn_h, filled=True)

            # Bright border
            self.renderer.set_color_rgb(100, 255, 100)
            self.renderer.draw_rect(btn_x, y, btn_w, btn_h, filled=False)

            # Text
            self.renderer.set_color('white')
        else:
            # Dimmed button
            self.renderer.set_color_rgb(50, 60, 50)
            self.renderer.draw_rect(btn_x, y, btn_w, btn_h, filled=True)
            self.renderer.set_color_rgb(80, 100, 80)
            self.renderer.draw_rect(btn_x, y, btn_w, btn_h, filled=False)
            self.renderer.set_color_rgb(120, 140, 120)

        self.renderer.draw_text(center_x - 50, y + 15, "START MATCH")

    def _render_settings(self, frame: int):
        """Render the settings menu."""
        sms = self.menu.settings_menu_state

        if sms.submenu == SettingsSubmenu.MAIN:
            self._render_settings_main(frame)
        elif sms.submenu == SettingsSubmenu.AUDIO:
            self._render_settings_audio(frame)
        elif sms.submenu == SettingsSubmenu.CONTROLS:
            self._render_settings_controls(frame)
        elif sms.submenu == SettingsSubmenu.GAMEPLAY:
            self._render_settings_gameplay(frame)

    def _render_settings_main(self, frame: int):
        """Render the main settings menu."""
        center_x = self.renderer.width // 2
        center_y = self.renderer.height // 2
        sms = self.menu.settings_menu_state

        # Title
        self.renderer.set_color_rgb(200, 180, 255)
        self.renderer.draw_text(center_x - 50, 60, "SETTINGS")

        # Settings gear icon (decorative)
        self._render_gear_icon(center_x, 120, 25, frame)

        # Options
        options = ["AUDIO", "CONTROLS", "GAMEPLAY", "BACK"]
        option_colors = [
            (100, 200, 255),   # Audio - blue
            (255, 200, 100),   # Controls - orange
            (100, 255, 150),   # Gameplay - green
            (180, 180, 180),   # Back - gray
        ]
        option_icons = ["♪", "⊕", "♦", "←"]

        option_y = center_y - 60
        for i, opt in enumerate(options):
            is_selected = sms.main_cursor == i
            color = option_colors[i]

            if is_selected:
                # Highlight box
                box_w = 280
                box_h = 50
                box_x = center_x - box_w // 2
                box_y = option_y - 10

                # Pulsing highlight
                pulse = int(math.sin(frame * 0.15) * 20) + 40
                self.renderer.set_color_rgb(pulse, pulse, pulse + 20)
                self.renderer.draw_rect(box_x, box_y, box_w, box_h, filled=True)

                # Border
                self.renderer.set_color_rgb(*color)
                self.renderer.draw_rect(box_x, box_y, box_w, box_h, filled=False)

                # Cursor arrows
                arrow_bob = int(math.sin(frame * 0.2) * 3)
                self.renderer.draw_line(box_x - 15 + arrow_bob, option_y + 8, box_x - 5 + arrow_bob, option_y + 3)
                self.renderer.draw_line(box_x - 15 + arrow_bob, option_y + 8, box_x - 5 + arrow_bob, option_y + 13)

                # Option text (bright)
                self.renderer.set_color('white')
            else:
                # Dimmed text
                self.renderer.set_color_rgb(120, 120, 130)

            self.renderer.draw_text(center_x - len(opt) * 4, option_y + 5, opt)

            option_y += 70

        # Controls hint
        self.renderer.set_color_rgb(80, 80, 90)
        self.renderer.draw_text(center_x - 120, self.renderer.height - 30,
                               "Up/Down: Select | Attack/Jump: Enter | Special: Back")

    def _render_gear_icon(self, cx: int, cy: int, size: int, frame: int):
        """Render a spinning gear icon."""
        rotation = frame * 0.02
        self.renderer.set_color_rgb(180, 160, 200)

        # Draw gear teeth as lines radiating from center
        num_teeth = 8
        for i in range(num_teeth):
            angle = rotation + (i * math.pi * 2 / num_teeth)
            inner = size * 0.5
            outer = size * 0.9
            x1 = cx + int(math.cos(angle) * inner)
            y1 = cy + int(math.sin(angle) * inner)
            x2 = cx + int(math.cos(angle) * outer)
            y2 = cy + int(math.sin(angle) * outer)
            self.renderer.draw_line(x1, y1, x2, y2)

        # Center circle
        self.renderer.draw_circle(cx, cy, int(size * 0.35), filled=True)

    def _render_settings_audio(self, frame: int):
        """Render the audio settings submenu."""
        center_x = self.renderer.width // 2
        sms = self.menu.settings_menu_state
        audio = self.menu.settings_manager.settings.audio

        # Title
        self.renderer.set_color_rgb(100, 200, 255)
        self.renderer.draw_text(center_x - 80, 60, "AUDIO SETTINGS")

        # Panel
        panel_y = 140
        panel_x = center_x - 250
        panel_w = 500
        panel_h = 320

        self.renderer.set_color_rgb(35, 35, 50)
        self.renderer.draw_rect(panel_x, panel_y, panel_w, panel_h, filled=True)
        self.renderer.set_color_rgb(100, 200, 255)
        self.renderer.draw_rect(panel_x, panel_y, panel_w, panel_h, filled=False)

        # Settings rows
        row_y = panel_y + 40
        row_height = 65

        # Master Volume
        self._render_volume_slider(
            center_x, row_y, "MASTER VOLUME",
            audio.master_volume,
            sms.audio_cursor == 0,
            frame
        )
        row_y += row_height

        # SFX Volume
        self._render_volume_slider(
            center_x, row_y, "SFX VOLUME",
            audio.sfx_volume,
            sms.audio_cursor == 1,
            frame
        )
        row_y += row_height

        # Mute toggle
        self._render_toggle_row(
            center_x, row_y, "MUTE ALL",
            audio.muted,
            sms.audio_cursor == 2,
            frame
        )
        row_y += row_height

        # Back button
        self._render_menu_button(center_x, row_y + 10, "BACK", sms.audio_cursor == 3, frame)

        # Controls hint
        self.renderer.set_color_rgb(80, 80, 90)
        self.renderer.draw_text(center_x - 140, self.renderer.height - 30,
                               "Up/Down: Navigate | Left/Right: Adjust | Special: Back")

    def _render_volume_slider(self, center_x: int, y: int, label: str,
                              value: float, selected: bool, frame: int):
        """Render a volume slider with label."""
        row_w = 420
        row_x = center_x - row_w // 2

        # Highlight if selected
        if selected:
            pulse = int(math.sin(frame * 0.15) * 15) + 35
            self.renderer.set_color_rgb(pulse, pulse, pulse + 20)
            self.renderer.draw_rect(row_x, y - 5, row_w, 50, filled=True)
            self.renderer.set_color_rgb(100, 200, 255)
            self.renderer.draw_rect(row_x, y - 5, row_w, 50, filled=False)

        # Label
        if selected:
            self.renderer.set_color('white')
        else:
            self.renderer.set_color_rgb(150, 150, 160)
        self.renderer.draw_text(row_x + 20, y + 10, label)

        # Slider track
        slider_x = row_x + 200
        slider_w = 180
        slider_y = y + 12

        self.renderer.set_color_rgb(50, 50, 60)
        self.renderer.draw_rect(slider_x, slider_y, slider_w, 12, filled=True)

        # Slider fill
        fill_w = int(slider_w * value)
        if fill_w > 0:
            if selected:
                self.renderer.set_color_rgb(100, 200, 255)
            else:
                self.renderer.set_color_rgb(80, 150, 200)
            self.renderer.draw_rect(slider_x, slider_y, fill_w, 12, filled=True)

        # Value percentage
        pct = f"{int(value * 100)}%"
        self.renderer.set_color_rgb(200, 200, 220)
        self.renderer.draw_text(slider_x + slider_w + 10, y + 10, pct)

        # Arrows if selected
        if selected:
            arrow_bob = int(math.sin(frame * 0.2) * 2)
            if value > 0:
                self.renderer.set_color_rgb(200, 200, 220)
                ax = slider_x - 15 - arrow_bob
                self.renderer.draw_line(ax, slider_y + 6, ax + 8, slider_y)
                self.renderer.draw_line(ax, slider_y + 6, ax + 8, slider_y + 12)
            if value < 1.0:
                ax = slider_x + slider_w + 45 + arrow_bob
                self.renderer.draw_line(ax, slider_y + 6, ax - 8, slider_y)
                self.renderer.draw_line(ax, slider_y + 6, ax - 8, slider_y + 12)

    def _render_toggle_row(self, center_x: int, y: int, label: str,
                           value: bool, selected: bool, frame: int):
        """Render a toggle option row."""
        row_w = 420
        row_x = center_x - row_w // 2

        # Highlight if selected
        if selected:
            pulse = int(math.sin(frame * 0.15) * 15) + 35
            self.renderer.set_color_rgb(pulse, pulse, pulse + 20)
            self.renderer.draw_rect(row_x, y - 5, row_w, 50, filled=True)
            self.renderer.set_color_rgb(100, 200, 255)
            self.renderer.draw_rect(row_x, y - 5, row_w, 50, filled=False)

        # Label
        if selected:
            self.renderer.set_color('white')
        else:
            self.renderer.set_color_rgb(150, 150, 160)
        self.renderer.draw_text(row_x + 20, y + 10, label)

        # Toggle indicator
        toggle_x = row_x + 320
        toggle_w = 60
        toggle_h = 25

        if value:
            # ON state
            self.renderer.set_color_rgb(100, 200, 100)
            self.renderer.draw_rect(toggle_x, y + 5, toggle_w, toggle_h, filled=True)
            self.renderer.set_color('white')
            self.renderer.draw_text(toggle_x + 20, y + 10, "ON")
        else:
            # OFF state
            self.renderer.set_color_rgb(60, 60, 70)
            self.renderer.draw_rect(toggle_x, y + 5, toggle_w, toggle_h, filled=True)
            self.renderer.set_color_rgb(150, 150, 160)
            self.renderer.draw_text(toggle_x + 15, y + 10, "OFF")

        # Border
        self.renderer.set_color_rgb(100, 100, 120)
        self.renderer.draw_rect(toggle_x, y + 5, toggle_w, toggle_h, filled=False)

    def _render_menu_button(self, center_x: int, y: int, text: str, selected: bool, frame: int):
        """Render a menu button."""
        btn_w = 150
        btn_h = 40
        btn_x = center_x - btn_w // 2

        if selected:
            pulse = int(math.sin(frame * 0.15) * 20) + 60
            self.renderer.set_color_rgb(pulse, pulse, pulse + 20)
            self.renderer.draw_rect(btn_x, y, btn_w, btn_h, filled=True)
            self.renderer.set_color_rgb(200, 180, 255)
            self.renderer.draw_rect(btn_x, y, btn_w, btn_h, filled=False)
            self.renderer.set_color('white')
        else:
            self.renderer.set_color_rgb(50, 50, 60)
            self.renderer.draw_rect(btn_x, y, btn_w, btn_h, filled=True)
            self.renderer.set_color_rgb(80, 80, 100)
            self.renderer.draw_rect(btn_x, y, btn_w, btn_h, filled=False)
            self.renderer.set_color_rgb(120, 120, 140)

        self.renderer.draw_text(center_x - len(text) * 4, y + 12, text)

    def _render_settings_controls(self, frame: int):
        """Render the controls settings submenu."""
        center_x = self.renderer.width // 2
        sms = self.menu.settings_menu_state
        settings = self.menu.settings_manager.settings

        # Title
        self.renderer.set_color_rgb(255, 200, 100)
        self.renderer.draw_text(center_x - 100, 60, "CONTROL SETTINGS")

        # Panel
        panel_y = 120
        panel_x = center_x - 300
        panel_w = 600
        panel_h = 380

        self.renderer.set_color_rgb(35, 35, 50)
        self.renderer.draw_rect(panel_x, panel_y, panel_w, panel_h, filled=True)
        self.renderer.set_color_rgb(255, 200, 100)
        self.renderer.draw_rect(panel_x, panel_y, panel_w, panel_h, filled=False)

        # P1 controls section
        section_y = panel_y + 30
        self._render_player_controls_section(
            panel_x + 30, section_y, "PLAYER 1",
            settings.player1_controls,
            (255, 100, 100),
            sms.controls_cursor == 0,
            frame
        )

        # P2 controls section
        self._render_player_controls_section(
            panel_x + 310, section_y, "PLAYER 2",
            settings.player2_controls,
            (100, 100, 255),
            sms.controls_cursor == 1,
            frame
        )

        # Gamepad status
        self._render_gamepad_status(center_x, panel_y + 280, frame)

        # Back button
        self._render_menu_button(center_x, panel_y + 330, "BACK", sms.controls_cursor == 2, frame)

        # Controls hint
        self.renderer.set_color_rgb(80, 80, 90)
        self.renderer.draw_text(center_x - 140, self.renderer.height - 30,
                               "Up/Down: Navigate | Left/Right: Change Mapping | Special: Back")

    def _render_player_controls_section(self, x: int, y: int, title: str,
                                        controls, color: tuple, selected: bool, frame: int):
        """Render a player's control settings section."""
        section_w = 260
        section_h = 200

        # Section background
        if selected:
            pulse = int(math.sin(frame * 0.15) * 10) + 40
            self.renderer.set_color_rgb(pulse, pulse, pulse + 15)
        else:
            self.renderer.set_color_rgb(30, 30, 40)
        self.renderer.draw_rect(x, y, section_w, section_h, filled=True)

        # Border
        if selected:
            self.renderer.set_color_rgb(*color)
        else:
            self.renderer.set_color_rgb(70, 70, 80)
        self.renderer.draw_rect(x, y, section_w, section_h, filled=False)

        # Title
        self.renderer.set_color_rgb(*color)
        self.renderer.draw_text(x + section_w // 2 - len(title) * 4, y + 10, title)

        # Control mapping info
        mapping = controls.gamepad_mapping.upper()
        self.renderer.set_color_rgb(180, 180, 190)
        self.renderer.draw_text(x + 20, y + 45, f"Gamepad: {mapping}")

        # Keyboard info (static display)
        self.renderer.set_color_rgb(120, 120, 130)
        if title == "PLAYER 1":
            self.renderer.draw_text(x + 20, y + 80, "Keyboard: WASD")
            self.renderer.draw_text(x + 20, y + 105, "Space=Jump J=Atk")
            self.renderer.draw_text(x + 20, y + 130, "K=Spcl L=Shield")
        else:
            self.renderer.draw_text(x + 20, y + 80, "Keyboard: Arrows")
            self.renderer.draw_text(x + 20, y + 105, "Enter=Jump Num1=Atk")
            self.renderer.draw_text(x + 20, y + 130, "Num2=Spcl Num3=Shield")

        # Mapping switch arrows if selected
        if selected:
            arrow_bob = int(math.sin(frame * 0.2) * 2)
            self.renderer.set_color_rgb(200, 200, 220)

            # Left arrow
            ax = x + 160
            ay = y + 45
            self.renderer.draw_line(ax - arrow_bob, ay + 5, ax + 6 - arrow_bob, ay)
            self.renderer.draw_line(ax - arrow_bob, ay + 5, ax + 6 - arrow_bob, ay + 10)

            # Right arrow
            ax = x + 240
            self.renderer.draw_line(ax + arrow_bob, ay + 5, ax - 6 + arrow_bob, ay)
            self.renderer.draw_line(ax + arrow_bob, ay + 5, ax - 6 + arrow_bob, ay + 10)

    def _render_gamepad_status(self, center_x: int, y: int, frame: int):
        """Render gamepad detection status."""
        # Try to detect gamepads
        try:
            from gamepad_input import GamepadManager
            gp_mgr = GamepadManager()
            gp_mgr.scan_devices()
            num_gamepads = len(gp_mgr.gamepads)
            gp_mgr.cleanup()
        except Exception:
            num_gamepads = 0

        self.renderer.set_color_rgb(150, 150, 160)
        self.renderer.draw_text(center_x - 80, y, "GAMEPAD STATUS:")

        if num_gamepads > 0:
            self.renderer.set_color_rgb(100, 255, 100)
            self.renderer.draw_text(center_x + 60, y, f"{num_gamepads} connected")
        else:
            self.renderer.set_color_rgb(150, 150, 150)
            self.renderer.draw_text(center_x + 60, y, "None detected")

    def _render_settings_gameplay(self, frame: int):
        """Render the gameplay settings submenu."""
        center_x = self.renderer.width // 2
        sms = self.menu.settings_menu_state
        gameplay = self.menu.settings_manager.settings.gameplay

        # Title
        self.renderer.set_color_rgb(100, 255, 150)
        self.renderer.draw_text(center_x - 110, 60, "GAMEPLAY SETTINGS")

        # Panel
        panel_y = 140
        panel_x = center_x - 250
        panel_w = 500
        panel_h = 385

        self.renderer.set_color_rgb(35, 35, 50)
        self.renderer.draw_rect(panel_x, panel_y, panel_w, panel_h, filled=True)
        self.renderer.set_color_rgb(100, 255, 150)
        self.renderer.draw_rect(panel_x, panel_y, panel_w, panel_h, filled=False)

        # Settings rows
        row_y = panel_y + 40
        row_height = 65

        # Default stocks
        self._render_value_row(
            center_x, row_y, "DEFAULT STOCKS",
            str(gameplay.default_stock_count),
            sms.gameplay_cursor == 0,
            frame,
            min_val=1, max_val=5, current_val=gameplay.default_stock_count
        )
        row_y += row_height

        # Default time
        time_labels = {0: "None", 3: "3 min", 5: "5 min", 8: "8 min"}
        time_label = time_labels.get(gameplay.default_time_limit, "None")
        time_options = [0, 3, 5, 8]
        time_idx = time_options.index(gameplay.default_time_limit) if gameplay.default_time_limit in time_options else 0

        self._render_value_row(
            center_x, row_y, "DEFAULT TIME",
            time_label,
            sms.gameplay_cursor == 1,
            frame,
            min_val=0, max_val=len(time_options) - 1, current_val=time_idx
        )
        row_y += row_height

        # Damage numbers toggle
        self._render_toggle_row(
            center_x, row_y, "DAMAGE NUMBERS",
            gameplay.show_damage_numbers,
            sms.gameplay_cursor == 2,
            frame
        )
        row_y += row_height

        # Visual theme
        import themes
        names = self.menu.available_themes or ['classic']
        theme_idx = names.index(gameplay.theme) if gameplay.theme in names else 0
        self._render_value_row(
            center_x, row_y, "VISUAL THEME",
            themes.title(gameplay.theme),
            sms.gameplay_cursor == 3,
            frame,
            min_val=0, max_val=len(names) - 1, current_val=theme_idx
        )
        row_y += row_height

        # Back button
        self._render_menu_button(center_x, row_y + 10, "BACK", sms.gameplay_cursor == 4, frame)

        # Controls hint
        self.renderer.set_color_rgb(80, 80, 90)
        self.renderer.draw_text(center_x - 140, self.renderer.height - 30,
                               "Up/Down: Navigate | Left/Right: Adjust | Special: Back")

    def _render_value_row(self, center_x: int, y: int, label: str, value: str,
                          selected: bool, frame: int,
                          min_val: int = 0, max_val: int = 0, current_val: int = 0):
        """Render a value adjustment row."""
        row_w = 420
        row_x = center_x - row_w // 2

        # Highlight if selected
        if selected:
            pulse = int(math.sin(frame * 0.15) * 15) + 35
            self.renderer.set_color_rgb(pulse, pulse, pulse + 20)
            self.renderer.draw_rect(row_x, y - 5, row_w, 50, filled=True)
            self.renderer.set_color_rgb(100, 255, 150)
            self.renderer.draw_rect(row_x, y - 5, row_w, 50, filled=False)

        # Label
        if selected:
            self.renderer.set_color('white')
        else:
            self.renderer.set_color_rgb(150, 150, 160)
        self.renderer.draw_text(row_x + 20, y + 10, label)

        # Value
        value_x = row_x + 300
        if selected:
            self.renderer.set_color_rgb(100, 255, 150)
        else:
            self.renderer.set_color_rgb(180, 180, 190)
        self.renderer.draw_text(value_x, y + 10, value)

        # Arrows if selected
        if selected:
            arrow_bob = int(math.sin(frame * 0.2) * 2)

            # Left arrow
            if current_val > min_val:
                self.renderer.set_color_rgb(200, 200, 220)
                ax = value_x - 20 - arrow_bob
                self.renderer.draw_line(ax, y + 15, ax + 8, y + 8)
                self.renderer.draw_line(ax, y + 15, ax + 8, y + 22)

            # Right arrow
            if current_val < max_val:
                self.renderer.set_color_rgb(200, 200, 220)
                ax = value_x + 60 + arrow_bob
                self.renderer.draw_line(ax, y + 15, ax - 8, y + 8)
                self.renderer.draw_line(ax, y + 15, ax - 8, y + 22)
