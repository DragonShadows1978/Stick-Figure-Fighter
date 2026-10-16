#!/usr/bin/env python3
"""
Stick Figure Fighter - Main Entry Point

A platform fighter built entirely from first principles.
No external game engines - pure Python with X11 rendering.

Usage:
    python main.py                     # Full game with menus
    python main.py --quick             # Skip menus, default match
    python main.py --p1 RUSH --p2 GRAB # Quick match with selected characters
    python main.py --headless          # Run headless (for ML training)
    python main.py --test              # Run a quick test match

Characters:
    RUSH - Fast rushdown, combo-focused (red)
    ZONE - Ranged zoner with projectiles (cyan)
    GRAB - Slow grappler, high damage (brown)
    FLEX - Balanced all-rounder (green)

Controls:
    Player 1 (WASD):  W/A/S/D move, Space jump, J attack, K special, L shield
    Player 2 (Arrows): Arrows move, Enter jump, Numpad 1 attack, Numpad 2 special, Numpad 3 shield

    P: Pause/Resume game
    Escape: Quit / Back
    Tab: Toggle hitbox display

Training Mode:
    F1: Toggle CPU opponent
    F2: Cycle CPU behavior (Stand/Jump/Walk/Shield/Attack/Random)
    F3: Toggle frame data overlay
    F4: Cycle CPU difficulty (Level 1-5)
    F5: Toggle infinite jumps
    R: Reset damage to 0%
    T: Reset positions
"""
import sys
import time
import argparse
import os
import math
from typing import Optional, List
from enum import Enum, auto
from dataclasses import dataclass

from game_engine import (
    GameState, GameLoop, Fighter, FighterStats,
    Stage, CONFIG, Vector2, FighterState,
    create_game, run_headless_match, get_observation, get_action_from_index,
    InputState
)
from renderer import X11Renderer
from input_system import InputManager, print_controls, KeySym
from audio_system import (
    AudioManager, GameAudioCallbacks, SoundType,
    get_audio_manager, cleanup_audio
)
from training_mode import (
    TrainingModeManager, CPUBehavior, get_training_manager
)
from menu_system import MenuSystem, MenuRenderer, MenuState, MatchSettings, GameModeSelection
from characters import CharacterType
from replay_system import (
    ReplayRecorder, ReplayPlayer, ReplayBrowser,
    get_replay_recorder, get_replay_player, get_replay_browser
)
from gamepad_input import get_gamepad_manager


# =============================================================================
# GAME MODE ENUM
# =============================================================================

class GameMode(Enum):
    """Current game mode."""
    MENU = auto()           # Character/stage select menus
    COUNTDOWN = auto()      # Pre-match countdown (3-2-1-GO)
    MATCH = auto()          # In a match
    RESULTS = auto()        # Post-match results screen
    PAUSED = auto()         # Match paused
    REPLAY_BROWSER = auto() # Browsing saved replays
    REPLAY_PLAYBACK = auto()  # Playing back a replay


@dataclass
class CountdownState:
    """Countdown state before match starts."""
    countdown_frame: int = 0
    countdown_value: int = 3  # 3, 2, 1, GO
    sound_played: bool = False
    total_frames: int = 180  # 3 seconds at 60fps (1 second per count)
    # Cycle 14: Entrance animation state
    entrance_complete: bool = False
    entrance_frames: int = 60  # 1 second entrance animation
    original_positions: Optional[List[Vector2]] = None  # Store spawn positions


# =============================================================================
# RESULTS SCREEN STATE
# =============================================================================

@dataclass
class MatchResults:
    """Post-match results data."""
    winner: Optional[int]
    p1_character: str
    p2_character: str
    p1_stocks: int
    p2_stocks: int
    p1_damage: float
    p2_damage: float
    p1_max_combo: int
    p2_max_combo: int
    total_frames: int
    stage_name: str

    @property
    def match_time_str(self) -> str:
        """Get formatted match time."""
        seconds = self.total_frames / 60
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}:{secs:02d}"


# =============================================================================
# MAIN GAME CLASS
# =============================================================================

class StickFighterGame:
    """
    Main game class that integrates all systems.
    Now supports full game flow: Menu -> Match -> Results -> Menu
    """

    def __init__(self, display_name: str = ":99", headless: bool = False,
                 skip_menus: bool = False,
                 stage_name: str = "battlefield",
                 p1_character: Optional[str] = None,
                 p2_character: Optional[str] = None,
                 training_mode: bool = False):
        """
        Initialize the game.

        Args:
            display_name: X11 display (e.g., ':99')
            headless: If True, run without rendering
            skip_menus: If True, skip directly to match
            stage_name: 'battlefield' or 'final_destination'
            p1_character: Character for P1 (RUSH, ZONE, GRAB, FLEX)
            p2_character: Character for P2 (RUSH, ZONE, GRAB, FLEX)
            training_mode: If True, start in training mode with CPU
        """
        self.display_name = display_name
        self.headless = headless
        self.skip_menus = skip_menus
        self.training_mode = training_mode

        # Default match settings
        self.default_stage = stage_name
        self.default_p1 = p1_character
        self.default_p2 = p2_character

        # Create renderer (if not headless)
        self.renderer: Optional[X11Renderer] = None
        if not headless:
            self.renderer = X11Renderer(
                display_name=display_name,
                title="Stick Figure Fighter"
            )

        # Create input manager
        self.input_manager = InputManager(display_name)

        # Game mode state
        self.mode = GameMode.MENU if not skip_menus else GameMode.MATCH

        # Menu system
        self.menu_system = MenuSystem()
        self.menu_renderer: Optional[MenuRenderer] = None
        if self.renderer:
            self.menu_renderer = MenuRenderer(self.renderer)
            self.menu_renderer.set_menu(self.menu_system)

        # Current match state (created when starting a match)
        self.game_state: Optional[GameState] = None
        self.game_loop: Optional[GameLoop] = None

        # Match results (populated after match ends)
        self.match_results: Optional[MatchResults] = None
        self.results_timer: int = 0
        self.results_selection: int = 0  # 0 = Rematch, 1 = Character Select

        # Countdown state
        self.countdown_state: Optional[CountdownState] = None
        self.saved_settings: Optional[MatchSettings] = None  # Settings for match being started

        # Cycle 19: Enhanced pause menu state
        self.pause_menu_selection: int = 0  # 0=Resume, 1=Char Select, 2=Title
        self._prev_pause_up: bool = False
        self._prev_pause_down: bool = False
        self._prev_pause_select: bool = False

        # Debug flags
        self.show_hitboxes = False
        self.show_hurtboxes = False
        self._prev_debug_key = False
        self._prev_pause_key = False
        self._prev_escape_key = False

        # Training mode keys
        self._prev_f1_key = False
        self._prev_f2_key = False
        self._prev_f3_key = False
        self._prev_f4_key = False
        self._prev_f5_key = False
        self._prev_f6_key = False
        self._prev_r_key = False
        self._prev_t_key = False

        # Cycle 17: FPS counter key
        self._prev_f10_key = False

        # Game state
        self.quit_requested = False
        self.frame_count = 0

        # Cycle 17: FPS tracking
        self.frame_times: List[float] = []
        self.last_frame_time: float = time.perf_counter()
        self.show_fps: bool = False

        # Audio system
        self.audio: Optional[AudioManager] = None
        self.audio_callbacks: Optional[GameAudioCallbacks] = None
        if not headless:
            self.audio = get_audio_manager()
            self.audio_callbacks = GameAudioCallbacks(self.audio)

            # Connect menu sound callbacks
            self.menu_system.on_cursor_move = lambda: self.audio.play(SoundType.MENU_MOVE, 0.5)
            self.menu_system.on_select = lambda: self.audio.play(SoundType.MENU_SELECT, 0.7)
            self.menu_system.on_back = lambda: self.audio.play(SoundType.MENU_BACK, 0.5)

            # Cycle 18: Character select sound callbacks
            self.menu_system.on_char_cursor_move = lambda char_name: self._play_char_preview(char_name)
            self.menu_system.on_char_locked = lambda char_name: self._play_char_lock(char_name)

        # Training mode
        self.training_manager: Optional[TrainingModeManager] = None
        if not headless:
            self.training_manager = get_training_manager()

        # Replay system
        self.replay_recorder: Optional[ReplayRecorder] = None
        self.replay_player: Optional[ReplayPlayer] = None
        self.replay_browser: Optional[ReplayBrowser] = None
        self.replay_frame: int = 0
        self.replay_paused: bool = False

        if not headless:
            self.replay_recorder = get_replay_recorder()
            self.replay_player = get_replay_player()
            self.replay_browser = get_replay_browser()

        # Wire audio settings callback to actually update audio system
        if self.audio and not headless:
            self.menu_system.on_settings_changed = self._on_settings_changed

        # If skipping menus, create game immediately
        if skip_menus:
            self._start_match_from_args()

    def _on_settings_changed(self):
        """Called when settings are changed in the settings menu."""
        if self.audio and self.menu_system._settings_manager:
            settings = self.menu_system._settings_manager.settings
            # Update audio volumes
            self.audio.set_volume(settings.audio.master_volume)
            # Mute/unmute
            self.audio.set_enabled(not settings.audio.muted)
            # Play a test sound to confirm the change
            if not settings.audio.muted:
                self.audio.play(SoundType.MENU_MOVE, 0.5)

    def _start_match_from_args(self):
        """Start a match using command-line arguments."""
        settings = MatchSettings()
        settings.stage_name = self.default_stage
        settings.p1_character = self.default_p1 or "FLEX"
        settings.p2_character = self.default_p2 or "FLEX"
        settings.is_training_mode = self.training_mode
        self._start_match(settings)

    def _start_match(self, settings: MatchSettings):
        """Start a match with the given settings."""
        # Convert time limit from minutes to frames (0 = no limit)
        time_limit_frames = 0
        if settings.time_limit > 0:
            time_limit_frames = settings.time_limit * 60 * 60  # minutes * seconds * frames

        # Create game state
        self.game_state = create_game(
            settings.stage_name,
            settings.p1_character,
            settings.p2_character,
            stock_count=settings.stock_count,
            time_limit=time_limit_frames
        )

        # Create game loop
        self.game_loop = GameLoop(self.game_state, headless=self.headless)

        # Connect callbacks
        self.game_loop.on_frame_update = self._on_frame_update
        if not self.headless:
            self.game_loop.on_render = self._on_render
        self.game_loop.on_match_end = self._on_match_end

        # Cycle 3: Connect visual effect callbacks
        if not self.headless and self.renderer:
            self.game_loop.on_hit = self._on_hit
            self.game_loop.on_projectile_hit = self._on_projectile_hit
            self.game_loop.on_reflect = self._on_reflect
            self.game_loop.on_land = self._on_land
            # Cycle 9: Dodge sound callback
            self.game_loop.on_dodge = self._on_dodge

        # Enable CPU in training mode
        if settings.is_training_mode and self.training_manager:
            self.training_manager.cpu_controller.enable(CPUBehavior.RANDOM)
            print("[TRAINING] CPU enabled (P2 is CPU controlled)")
            print("[TRAINING] F1: Toggle CPU | F2: Cycle behavior | F3: Frame data | F4: Difficulty")
            print("[TRAINING] F7: Input History | F8: Move List | Tab: Hitbox Mode")

            # Cycle 16: Initialize move list for P1 character
            from characters import get_character_attacks, CharacterType
            try:
                p1_char_type = CharacterType[settings.p1_character or "FLEX"]
                p1_attacks = get_character_attacks(p1_char_type)
                self.training_manager.move_list.set_character(p1_char_type, p1_attacks)
            except (KeyError, ValueError):
                pass  # Invalid character, skip move list init

        # Save settings and start countdown
        self.saved_settings = settings
        self.countdown_state = CountdownState()

        # Start replay recording (if not in replay playback mode)
        if self.replay_recorder and self.mode != GameMode.REPLAY_PLAYBACK:
            self.replay_recorder.start_recording(
                p1_character=settings.p1_character or "FLEX",
                p2_character=settings.p2_character or "FLEX",
                stage_name=settings.stage_name,
                stock_count=settings.stock_count,
                time_limit=time_limit_frames
            )

        # Start countdown mode (plays sounds, shows 3-2-1-GO)
        self.mode = GameMode.COUNTDOWN

        mode_str = "TRAINING" if settings.is_training_mode else "VERSUS"
        print(f"\n=== {mode_str} MATCH START ===")
        print(f"P1: {settings.p1_character} vs P2: {settings.p2_character}")
        print(f"Stage: {settings.stage_name}")
        print(f"Stocks: {settings.stock_count}")
        if settings.time_limit > 0:
            print(f"Time: {settings.time_limit} min")

    def _end_match(self):
        """End the current match and show results."""
        if not self.game_state:
            return

        # Collect match statistics
        settings = self.menu_system.settings

        self.match_results = MatchResults(
            winner=self.game_state.winner,
            p1_character=settings.p1_character or "FLEX",
            p2_character=settings.p2_character or "FLEX",
            p1_stocks=self.game_state.fighters[0].stocks,
            p2_stocks=self.game_state.fighters[1].stocks,
            p1_damage=self.game_state.fighters[0].damage_percent,
            p2_damage=self.game_state.fighters[1].damage_percent,
            p1_max_combo=self.game_state.max_combos[0],
            p2_max_combo=self.game_state.max_combos[1],
            total_frames=self.game_state.frame,
            stage_name=settings.stage_name
        )

        # Save the replay (if recording)
        if self.replay_recorder and self.replay_recorder.is_recording:
            self.replay_recorder.finish_recording(
                winner=self.game_state.winner,
                p1_stocks=self.game_state.fighters[0].stocks,
                p2_stocks=self.game_state.fighters[1].stocks,
                p1_max_combo=self.game_state.max_combos[0],
                p2_max_combo=self.game_state.max_combos[1]
            )
            try:
                self.replay_recorder.save()
            except Exception as e:
                print(f"[REPLAY] Failed to save: {e}")

        self.results_timer = 0
        self.results_selection = 0
        self.mode = GameMode.RESULTS

        # Cycle 15: Stop ambient sound
        if self.audio:
            self.audio.stop_ambient()

        # Play match end sound
        if self.audio:
            self.audio.play(SoundType.MATCH_END, 0.8)

    def _handle_menu_input(self):
        """Handle input while in menu mode."""
        self.input_manager.poll()

        if self.input_manager.quit_requested:
            self.quit_requested = True
            return

        # Get raw inputs for menu navigation
        p1_inputs = {
            'left': self.input_manager.keyboard.is_key_pressed(0x0061),  # 'a'
            'right': self.input_manager.keyboard.is_key_pressed(0x0064),  # 'd'
            'up': self.input_manager.keyboard.is_key_pressed(0x0077),  # 'w'
            'down': self.input_manager.keyboard.is_key_pressed(0x0073),  # 's'
            'attack': self.input_manager.keyboard.is_key_pressed(0x006a),  # 'j'
            'jump': self.input_manager.keyboard.is_key_pressed(0x0020),  # space
            'special': self.input_manager.keyboard.is_key_pressed(0x006b),  # 'k'
        }

        p2_inputs = {
            'left': self.input_manager.keyboard.is_key_pressed(0xff51),  # Left arrow
            'right': self.input_manager.keyboard.is_key_pressed(0xff53),  # Right arrow
            'up': self.input_manager.keyboard.is_key_pressed(0xff52),  # Up arrow
            'down': self.input_manager.keyboard.is_key_pressed(0xff54),  # Down arrow
            'attack': self.input_manager.keyboard.is_key_pressed(0xffb1),  # Numpad 1
            'jump': self.input_manager.keyboard.is_key_pressed(0xff0d),  # Enter
            'special': self.input_manager.keyboard.is_key_pressed(0xffb2),  # Numpad 2
        }

        # Update menu
        continue_menu = self.menu_system.update(p1_inputs, p2_inputs, self.frame_count)

        # Check for quit request from menu
        if self.menu_system.quit_requested:
            self.quit_requested = True
            return

        if not continue_menu:
            if self.menu_system.is_ready_to_start():
                # Play confirm sound
                if self.audio:
                    self.audio.play(SoundType.MENU_SELECT, 0.7)
                # Start the match
                self._start_match(self.menu_system.settings)
            elif self.menu_system.game_mode == GameModeSelection.REPLAYS:
                # Switch to replay browser
                if self.replay_browser:
                    self.replay_browser.refresh()
                    self.mode = GameMode.REPLAY_BROWSER
                    if self.audio:
                        self.audio.play(SoundType.MENU_SELECT, 0.7)
                else:
                    # No replay system, go back to menu
                    self.menu_system.state = MenuState.TITLE
                    return

        # Render menu
        if self.menu_renderer:
            self.menu_renderer.render(self.frame_count)

    def _handle_countdown(self):
        """Handle the pre-match countdown."""
        if not self.countdown_state:
            self.mode = GameMode.MATCH
            return

        self.input_manager.poll()

        if self.input_manager.quit_requested:
            self.quit_requested = True
            return

        cs = self.countdown_state
        cs.countdown_frame += 1

        # Cycle 14: Handle entrance animation (characters drop from top)
        if not cs.entrance_complete and self.game_state:
            # Initialize original positions on first frame
            if cs.original_positions is None:
                cs.original_positions = []
                for fighter in self.game_state.fighters:
                    cs.original_positions.append(Vector2(fighter.position.x, fighter.position.y))
                    # Move fighters to top of screen
                    fighter.position.y = -100
                    fighter.state = FighterState.ENTERING

            # Animate fighters dropping down
            entrance_progress = min(1.0, cs.countdown_frame / cs.entrance_frames)
            # Use easing function for smooth landing (ease-out cubic)
            eased_progress = 1.0 - pow(1.0 - entrance_progress, 3)

            for i, fighter in enumerate(self.game_state.fighters):
                if i < len(cs.original_positions):
                    target_y = cs.original_positions[i].y
                    start_y = -100
                    fighter.position.y = start_y + (target_y - start_y) * eased_progress

                    # Mark entrance complete when reached destination
                    if entrance_progress >= 1.0:
                        fighter.state = FighterState.IDLE

            if entrance_progress >= 1.0:
                cs.entrance_complete = True

        # Determine countdown value based on frame (after entrance animation)
        # Countdown starts after entrance animation
        countdown_start_frame = cs.entrance_frames
        countdown_frame = max(0, cs.countdown_frame - countdown_start_frame)

        # 60 frames per count: 0-59 = "3", 60-119 = "2", 120-179 = "1", 180+ = "GO"
        frames_per_count = 60

        if countdown_frame < frames_per_count:
            new_value = 3
        elif countdown_frame < frames_per_count * 2:
            new_value = 2
        elif countdown_frame < frames_per_count * 3:
            new_value = 1
        else:
            new_value = 0  # GO

        # Play sound when countdown value changes (only after entrance)
        if cs.entrance_complete:
            if new_value != cs.countdown_value or not cs.sound_played:
                cs.countdown_value = new_value
                cs.sound_played = True

                if self.audio:
                    if new_value == 3:
                        self.audio.play(SoundType.COUNTDOWN_3, 0.8)
                    elif new_value == 2:
                        self.audio.play(SoundType.COUNTDOWN_2, 0.8)
                    elif new_value == 1:
                        self.audio.play(SoundType.COUNTDOWN_1, 0.8)
                    elif new_value == 0:
                        self.audio.play(SoundType.COUNTDOWN_GO, 1.0)

        # Countdown complete - start match (after entrance + countdown + GO)
        total_entrance_and_countdown = cs.entrance_frames + cs.total_frames + 30
        if cs.countdown_frame >= total_entrance_and_countdown:
            self.mode = GameMode.MATCH
            self.countdown_state = None

            # Cycle 15: Start stage-specific ambient sound
            if self.audio and self.game_state:
                self.audio.start_ambient(self.game_state.stage.name)

            return

        # Render the countdown
        self._render_countdown()

    def _render_countdown(self):
        """Render the countdown overlay."""
        if not self.renderer or not self.game_state or not self.countdown_state:
            return

        cs = self.countdown_state

        # First render the stage/fighters (with entrance animation positions)
        self._on_render(self.game_state)

        # Don't call flip yet - overlay the countdown

        center_x = self.renderer.width // 2
        center_y = self.renderer.height // 2

        # Cycle 14: Show "READY" text during entrance animation
        if not cs.entrance_complete:
            # Darken overlay slightly
            self.renderer.set_color_rgb(0, 0, 0)
            for y in range(0, self.renderer.height, 6):
                for x in range(0, self.renderer.width, 6):
                    self.renderer.draw_rect(x, y, 2, 2, filled=True)

            # Show "READY" with pulsing effect
            pulse = math.sin(cs.countdown_frame * 0.2) * 0.3 + 0.7
            r = int(255 * pulse)
            g = int(200 * pulse)
            b = int(100 * pulse)
            self.renderer.set_color_rgb(r, g, b)
            self.renderer.draw_text(center_x - 30, center_y - 20, "READY")

            self.renderer.flip()
            return

        # Darken overlay for countdown
        self.renderer.set_color_rgb(0, 0, 0)
        for y in range(0, self.renderer.height, 4):
            for x in range(0, self.renderer.width, 4):
                self.renderer.draw_rect(x, y, 2, 2, filled=True)

        # Display countdown text
        if cs.countdown_value > 0:
            text = str(cs.countdown_value)
            # Pulsing size effect
            countdown_frame = cs.countdown_frame - cs.entrance_frames
            pulse = (countdown_frame % 60) / 60.0
            scale = 1.0 + (1.0 - pulse) * 0.3  # Start big, shrink

            # Draw number with glow effect
            glow_colors = [
                (255, 200, 50),   # Yellow glow
                (255, 255, 100),  # Lighter
                (255, 255, 200),  # Core
            ]

            for i, color in enumerate(glow_colors):
                self.renderer.set_color_rgb(*color)
                offset = (len(glow_colors) - i) * 2
                self.renderer.draw_text(center_x - 15 - offset, center_y - 20 - offset, text)
                self.renderer.draw_text(center_x - 15 + offset, center_y - 20 + offset, text)

            # Main text
            self.renderer.set_color('white')
            self.renderer.draw_text(center_x - 15, center_y - 20, text)
        else:
            # "GO!" text
            self.renderer.set_color_rgb(100, 255, 100)
            self.renderer.draw_text(center_x - 20, center_y - 20, "GO!")

        self.renderer.flip()

    def _handle_results_input(self):
        """Handle input on results screen."""
        self.input_manager.poll()

        if self.input_manager.quit_requested:
            self.quit_requested = True
            return

        self.results_timer += 1

        # Wait a moment before accepting input
        if self.results_timer < 60:
            self._render_results()
            return

        # Navigation
        left = self.input_manager.keyboard.is_key_pressed(0x0061) or self.input_manager.keyboard.is_key_pressed(0xff51)
        right = self.input_manager.keyboard.is_key_pressed(0x0064) or self.input_manager.keyboard.is_key_pressed(0xff53)

        prev_selection = self.results_selection
        if left and not self._prev_left_key:
            self.results_selection = 0
        elif right and not self._prev_right_key:
            self.results_selection = 1
        self._prev_left_key = left
        self._prev_right_key = right

        # Play sound on selection change
        if prev_selection != self.results_selection and self.audio:
            self.audio.play(SoundType.MENU_MOVE, 0.5)

        # Confirm selection
        confirm = (self.input_manager.keyboard.is_key_pressed(0x006a) or  # j
                   self.input_manager.keyboard.is_key_pressed(0x0020) or  # space
                   self.input_manager.keyboard.is_key_pressed(0xffb1) or  # numpad 1
                   self.input_manager.keyboard.is_key_pressed(0xff0d))    # enter

        if confirm and not self._prev_confirm_key:
            if self.audio:
                self.audio.play(SoundType.MENU_SELECT, 0.7)

            if self.results_selection == 0:
                # Rematch - same characters and stage
                self._start_match(self.menu_system.settings)
            else:
                # Back to character select
                self.menu_system.reset()
                self.mode = GameMode.MENU
        self._prev_confirm_key = confirm

        self._render_results()

    def _render_results(self):
        """Render the results screen with victory animations (Cycle 9)."""
        if not self.renderer or not self.match_results:
            return

        r = self.match_results
        center_x = self.renderer.width // 2
        frame = self.results_timer  # Use timer as animation frame

        # Background with spotlight effect
        self.renderer.set_color_rgb(20, 25, 35)
        self.renderer.draw_rect(0, 0, self.renderer.width, self.renderer.height, filled=True)

        # Spotlight effect on winner
        if r.winner is not None:
            spotlight_x = center_x - 200 if r.winner == 0 else center_x + 200
            # Radial spotlight (concentric circles getting darker)
            for radius in range(150, 0, -15):
                brightness = 25 + int((150 - radius) / 150 * 30)
                self.renderer.set_color_rgb(brightness + 5, brightness + 5, brightness + 10)
                self.renderer.draw_circle(spotlight_x, 280, radius, filled=True)

        # Confetti particles (falling rectangles)
        self._render_confetti(frame)

        # Title
        if r.winner is not None:
            winner_name = r.p1_character if r.winner == 0 else r.p2_character
            title = f"PLAYER {r.winner + 1} WINS!"
            color = (255, 100, 100) if r.winner == 0 else (100, 100, 255)
        else:
            title = "DRAW!"
            color = (255, 255, 100)

        # Animated title (pulsing)
        title_scale = 1.0 + math.sin(frame * 0.1) * 0.05
        self.renderer.set_color_rgb(*color)
        self.renderer.draw_text(center_x - len(title) * 5, 60, title)

        # Render winner with victory pose, loser with defeat pose
        self._render_results_characters(r, frame)

        # Stats table
        stats_y = 380
        self.renderer.set_color('white')

        # Stocks
        self.renderer.draw_text(center_x - 200, stats_y, f"Stocks: {r.p1_stocks}")
        self.renderer.draw_text(center_x + 100, stats_y, f"Stocks: {r.p2_stocks}")

        # Damage
        self.renderer.draw_text(center_x - 200, stats_y + 25, f"Damage: {r.p1_damage:.1f}%")
        self.renderer.draw_text(center_x + 100, stats_y + 25, f"Damage: {r.p2_damage:.1f}%")

        # Max combo
        self.renderer.draw_text(center_x - 200, stats_y + 50, f"Max Combo: {r.p1_max_combo}")
        self.renderer.draw_text(center_x + 100, stats_y + 50, f"Max Combo: {r.p2_max_combo}")

        # Match time
        self.renderer.set_color_rgb(150, 150, 150)
        self.renderer.draw_text(center_x - 60, stats_y + 90, f"Match Time: {r.match_time_str}")
        self.renderer.draw_text(center_x - 60, stats_y + 115, f"Stage: {r.stage_name.replace('_', ' ').title()}")

        # Options (only show after delay)
        if self.results_timer >= 60:
            options_y = 450

            # Rematch button
            if self.results_selection == 0:
                self.renderer.set_color_rgb(100, 255, 100)
                self.renderer.draw_rect(center_x - 180, options_y - 5, 120, 35, filled=True)
                self.renderer.set_color_rgb(20, 40, 20)
            else:
                self.renderer.set_color_rgb(80, 80, 80)
                self.renderer.draw_rect(center_x - 180, options_y - 5, 120, 35, filled=False)
                self.renderer.set_color('white')
            self.renderer.draw_text(center_x - 165, options_y + 5, "REMATCH")

            # Character Select button
            if self.results_selection == 1:
                self.renderer.set_color_rgb(100, 255, 100)
                self.renderer.draw_rect(center_x + 60, options_y - 5, 160, 35, filled=True)
                self.renderer.set_color_rgb(20, 40, 20)
            else:
                self.renderer.set_color_rgb(80, 80, 80)
                self.renderer.draw_rect(center_x + 60, options_y - 5, 160, 35, filled=False)
                self.renderer.set_color('white')
            self.renderer.draw_text(center_x + 70, options_y + 5, "CHARACTER SELECT")

            # Controls hint
            self.renderer.set_color_rgb(100, 100, 100)
            self.renderer.draw_text(center_x - 120, 520, "Left/Right: Select | Attack/Jump: Confirm")
        else:
            # "Loading" indicator
            dots = "." * ((self.results_timer // 15) % 4)
            self.renderer.set_color_rgb(100, 100, 100)
            self.renderer.draw_text(center_x - 40, 450, f"Loading{dots}")

        self.renderer.flip()

    def _render_confetti(self, frame: int):
        """Render falling confetti particles (Cycle 9)."""
        if not self.renderer:
            return

        # Confetti colors
        colors = [
            (255, 100, 100),   # Red
            (100, 255, 100),   # Green
            (100, 100, 255),   # Blue
            (255, 255, 100),   # Yellow
            (255, 100, 255),   # Magenta
            (100, 255, 255),   # Cyan
        ]

        # Generate deterministic confetti positions based on frame
        for i in range(30):
            # Pseudo-random seed based on confetti index
            seed = i * 127 + 37
            x_base = (seed * 17) % self.renderer.width
            fall_speed = 2 + (seed % 4)
            sway_amp = 20 + (seed % 30)
            sway_freq = 0.05 + (seed % 10) * 0.01

            # Y position - wraps around
            y = (frame * fall_speed + seed * 7) % (self.renderer.height + 100) - 50

            # X position - sways side to side
            x = x_base + int(math.sin(frame * sway_freq + seed) * sway_amp)

            # Rotation effect (just affects rectangle width/height)
            rotation = (frame * 3 + seed) % 360
            w = 4 + int(abs(math.sin(math.radians(rotation))) * 6)
            h = 4 + int(abs(math.cos(math.radians(rotation))) * 6)

            # Color based on index
            color = colors[i % len(colors)]
            self.renderer.set_color_rgb(*color)
            self.renderer.draw_rect(x, y, w, h, filled=True)

    def _render_results_characters(self, r: 'MatchResults', frame: int):
        """Render winner with victory pose, loser with defeat pose (Cycle 9)."""
        if not self.renderer:
            return

        center_x = self.renderer.width // 2
        from characters import get_character_color

        # Get character colors
        p1_color = get_character_color(CharacterType[r.p1_character])
        p2_color = get_character_color(CharacterType[r.p2_character])

        p1_x = center_x - 200
        p2_x = center_x + 200
        char_y = 280

        # P1 character
        p1_is_winner = r.winner == 0
        self._render_results_stick_figure(p1_x, char_y, p1_color, frame, p1_is_winner)

        # P2 character
        p2_is_winner = r.winner == 1
        self._render_results_stick_figure(p2_x, char_y, p2_color, frame, p2_is_winner)

        # Character names below figures
        self.renderer.set_color_rgb(*p1_color)
        self.renderer.draw_text(p1_x - 25, char_y + 90, f"P1: {r.p1_character}")

        self.renderer.set_color_rgb(*p2_color)
        self.renderer.draw_text(p2_x - 25, char_y + 90, f"P2: {r.p2_character}")

    def _render_results_stick_figure(self, cx: int, cy: int, color: tuple,
                                      frame: int, is_winner: bool):
        """
        Render a stick figure with victory or defeat pose.

        Cycle 9: Initial implementation
        Cycle 15: Enhanced with more dynamic animations, fist pump, dust, sway
        """
        self.renderer.set_color_rgb(*color)

        scale = 1.8  # Larger figure
        head_radius = int(15 * scale)

        if is_winner:
            # VICTORY POSE: Arms raised, jumping celebration with fist pump
            jump_cycle = abs(math.sin(frame * 0.15))
            jump_height = int(jump_cycle * 15)

            # Cycle 15: Alternating arm pump animation
            pump_phase = (frame // 20) % 3  # 0: both up, 1: left pump, 2: right pump

            head_y = cy - 50 - jump_height
            neck_y = head_y + head_radius
            hip_y = neck_y + int(40 * scale)
            shoulder_y = neck_y + int(10 * scale)

            # Head
            self.renderer.draw_circle(cx, head_y, head_radius, filled=True)

            # Body
            self.renderer.draw_line(cx, neck_y, cx, hip_y)

            # Arms with alternating fist pump
            arm_wave = int(math.sin(frame * 0.2) * 5)
            if pump_phase == 0:
                # Both arms up in V
                self.renderer.draw_line(cx, shoulder_y, cx - 30 - arm_wave, shoulder_y - 40)
                self.renderer.draw_line(cx, shoulder_y, cx + 30 + arm_wave, shoulder_y - 40)
            elif pump_phase == 1:
                # Left arm pump (higher), right arm down
                self.renderer.draw_line(cx, shoulder_y, cx - 25, shoulder_y - 50 - int(math.sin(frame * 0.3) * 5))
                self.renderer.draw_line(cx, shoulder_y, cx + 30, shoulder_y + 10)
            else:
                # Right arm pump (higher), left arm down
                self.renderer.draw_line(cx, shoulder_y, cx - 30, shoulder_y + 10)
                self.renderer.draw_line(cx, shoulder_y, cx + 25, shoulder_y - 50 - int(math.sin(frame * 0.3) * 5))

            # Legs - together or slight jump pose
            self.renderer.draw_line(cx, hip_y, cx - 10, hip_y + 50 + jump_height // 2)
            self.renderer.draw_line(cx, hip_y, cx + 10, hip_y + 50 + jump_height // 2)

            # Cycle 15: Ground dust effect when landing from jump
            if jump_cycle < 0.1 and frame % 60 < 15:
                dust_frame = frame % 60
                dust_alpha = 1.0 - (dust_frame / 15)
                dust_spread = dust_frame * 3
                gray = int(150 * dust_alpha)
                self.renderer.set_color_rgb(gray, gray, gray)
                # Left dust cloud
                self.renderer.draw_circle(cx - 15 - dust_spread, hip_y + 55, int(4 + dust_frame * 0.5), filled=True)
                # Right dust cloud
                self.renderer.draw_circle(cx + 15 + dust_spread, hip_y + 55, int(4 + dust_frame * 0.5), filled=True)
                self.renderer.set_color_rgb(*color)

            # Draw celebration sparkles around winner (more sparkles - 6 instead of 4)
            for i in range(6):
                sparkle_angle = (frame * 5 + i * 60) % 360
                sparkle_dist = 55 + int(math.sin(frame * 0.12 + i) * 12)
                sx = cx + int(math.cos(math.radians(sparkle_angle)) * sparkle_dist)
                sy = cy - 30 + int(math.sin(math.radians(sparkle_angle)) * sparkle_dist * 0.6)

                # Cycle 15: Varied sparkle colors
                if i % 3 == 0:
                    self.renderer.set_color_rgb(255, 255, 200)  # Yellow
                elif i % 3 == 1:
                    self.renderer.set_color_rgb(255, 200, 150)  # Orange
                else:
                    self.renderer.set_color_rgb(200, 220, 255)  # Light blue

                # Cross sparkle
                self.renderer.draw_line(sx - 5, sy, sx + 5, sy)
                self.renderer.draw_line(sx, sy - 5, sx, sy + 5)
                # Diagonal sparkle (smaller)
                self.renderer.draw_line(sx - 3, sy - 3, sx + 3, sy + 3)
                self.renderer.draw_line(sx - 3, sy + 3, sx + 3, sy - 3)

            self.renderer.set_color_rgb(*color)

            # Cycle 15: Victory glow ring
            glow_pulse = int(math.sin(frame * 0.1) * 5)
            self.renderer.set_color_rgb(255, 220, 100)
            self.renderer.draw_circle(cx, cy - 20, 65 + glow_pulse, filled=False)
            self.renderer.set_color_rgb(*color)

        else:
            # DEFEAT POSE: Slumped, head down with subtle sway
            # Cycle 15: Enhanced sway animation
            sway_x = int(math.sin(frame * 0.03) * 4)
            slump = int(math.sin(frame * 0.05) * 2)

            # Occasional "sigh" animation (head dips lower)
            sigh_cycle = (frame % 180)  # Every 3 seconds at 60fps
            sigh_extra = 0
            if sigh_cycle < 20:
                sigh_extra = int(math.sin(sigh_cycle * math.pi / 20) * 5)

            head_y = cy - 30 + slump + sigh_extra
            neck_y = head_y + head_radius
            hip_y = neck_y + int(35 * scale)  # Shorter body (slumped)
            shoulder_y = neck_y + int(10 * scale)

            # Head (tilted down, with sway)
            self.renderer.draw_circle(cx + 5 + sway_x, head_y + 10, head_radius, filled=True)

            # Body (slumped forward with sway)
            self.renderer.draw_line(cx + sway_x // 2, neck_y, cx + 10 + sway_x, hip_y)

            # Arms hanging down limply with sway
            self.renderer.draw_line(cx + 5 + sway_x, shoulder_y, cx - 15 + sway_x, hip_y + 10 + slump)
            self.renderer.draw_line(cx + 5 + sway_x, shoulder_y, cx + 25 + sway_x, hip_y + 5 + slump)

            # Legs - standing but defeated with slight sway
            self.renderer.draw_line(cx + 10, hip_y, cx + sway_x // 2, hip_y + 55)
            self.renderer.draw_line(cx + 10, hip_y, cx + 20 + sway_x // 2, hip_y + 55)

            # Cycle 15: Shadow/darkening effect under loser
            self.renderer.set_color_rgb(40, 40, 50)
            shadow_width = 40 + int(abs(sway_x))
            self.renderer.draw_rect(cx - shadow_width // 2, hip_y + 52, shadow_width, 8, filled=True)
            self.renderer.set_color_rgb(*color)

    # =========================================================================
    # Replay Browser
    # =========================================================================

    def _handle_replay_browser(self):
        """Handle input and rendering for the replay browser."""
        if not self.replay_browser or not self.renderer:
            self.mode = GameMode.MENU
            return

        self.input_manager.poll()

        if self.input_manager.quit_requested:
            self.quit_requested = True
            return

        # Navigation
        up = self.input_manager.keyboard.is_key_pressed(0x0077) or self.input_manager.keyboard.is_key_pressed(0xff52)
        down = self.input_manager.keyboard.is_key_pressed(0x0073) or self.input_manager.keyboard.is_key_pressed(0xff54)

        if up and not getattr(self, '_prev_replay_up', False):
            self.replay_browser.move_cursor(-1)
            if self.audio:
                self.audio.play(SoundType.MENU_MOVE, 0.5)
        self._prev_replay_up = up

        if down and not getattr(self, '_prev_replay_down', False):
            self.replay_browser.move_cursor(1)
            if self.audio:
                self.audio.play(SoundType.MENU_MOVE, 0.5)
        self._prev_replay_down = down

        # Back button (special/B)
        back = (self.input_manager.keyboard.is_key_pressed(0x006b) or  # k
                self.input_manager.keyboard.is_key_pressed(0xffb2) or  # numpad 2
                self.input_manager.keyboard.is_key_pressed(0xff1b))    # escape
        if back and not getattr(self, '_prev_replay_back', False):
            self.menu_system.state = MenuState.TITLE
            self.mode = GameMode.MENU
            if self.audio:
                self.audio.play(SoundType.MENU_BACK, 0.5)
            return
        self._prev_replay_back = back

        # Select replay
        confirm = (self.input_manager.keyboard.is_key_pressed(0x006a) or  # j
                   self.input_manager.keyboard.is_key_pressed(0x0020) or  # space
                   self.input_manager.keyboard.is_key_pressed(0xffb1) or  # numpad 1
                   self.input_manager.keyboard.is_key_pressed(0xff0d))    # enter
        if confirm and not getattr(self, '_prev_replay_confirm', False):
            entry = self.replay_browser.get_selected()
            if entry and self.replay_player:
                if self.replay_player.load(entry.filename):
                    self._start_replay_playback()
                    if self.audio:
                        self.audio.play(SoundType.MENU_SELECT, 0.7)
                    return
        self._prev_replay_confirm = confirm

        # Delete replay (X key)
        delete = self.input_manager.keyboard.is_key_pressed(0x0078)  # x
        if delete and not getattr(self, '_prev_replay_delete', False):
            if self.replay_browser.delete_selected():
                if self.audio:
                    self.audio.play(SoundType.MENU_BACK, 0.5)
        self._prev_replay_delete = delete

        # Render browser
        self._render_replay_browser()

    def _render_replay_browser(self):
        """Render the replay browser screen."""
        if not self.renderer or not self.replay_browser:
            return

        # Background
        self.renderer.set_color_rgb(20, 25, 35)
        self.renderer.draw_rect(0, 0, self.renderer.width, self.renderer.height, filled=True)

        center_x = self.renderer.width // 2

        # Title
        self.renderer.set_color_rgb(255, 200, 100)
        self.renderer.draw_text(center_x - 50, 40, "REPLAYS")

        if self.replay_browser.is_empty:
            # No replays message
            self.renderer.set_color_rgb(150, 150, 160)
            self.renderer.draw_text(center_x - 80, 200, "No replays saved yet")
            self.renderer.draw_text(center_x - 100, 240, "Play some matches first!")
        else:
            # Replay list
            y = 100
            visible = self.replay_browser.get_visible_entries()

            for idx, entry in visible:
                is_selected = idx == self.replay_browser.cursor

                if is_selected:
                    # Highlight box
                    self.renderer.set_color_rgb(50, 50, 60)
                    self.renderer.draw_rect(100, y - 5, self.renderer.width - 200, 50, filled=True)

                    self.renderer.set_color_rgb(255, 200, 100)
                    self.renderer.draw_rect(100, y - 5, self.renderer.width - 200, 50, filled=False)

                    # Arrow indicator
                    pulse = int(math.sin(self.frame_count * 0.2) * 3)
                    self.renderer.draw_line(80 + pulse, y + 15, 90 + pulse, y + 10)
                    self.renderer.draw_line(80 + pulse, y + 15, 90 + pulse, y + 20)

                # Match name
                self.renderer.set_color('white' if is_selected else 'gray')
                self.renderer.draw_text(120, y + 5, entry.display_name)

                # Match info
                self.renderer.set_color_rgb(150, 150, 160)
                self.renderer.draw_text(120, y + 25, entry.display_info)

                y += 55

        # Controls hint
        self.renderer.set_color_rgb(80, 80, 90)
        self.renderer.draw_text(center_x - 150, self.renderer.height - 50,
                               "Up/Down: Select | Attack: Play | Special: Back")
        self.renderer.draw_text(center_x - 80, self.renderer.height - 30,
                               "X: Delete Replay")

        self.renderer.flip()

    def _start_replay_playback(self):
        """Start playing back a loaded replay."""
        if not self.replay_player or not self.replay_player.metadata:
            return

        # Create match settings from replay metadata
        settings = MatchSettings()
        replay_settings = self.replay_player.get_match_settings()
        settings.p1_character = replay_settings['p1_character']
        settings.p2_character = replay_settings['p2_character']
        settings.stage_name = replay_settings['stage_name']
        settings.stock_count = replay_settings['stock_count']
        settings.time_limit = replay_settings.get('time_limit', 0) // 3600  # frames to minutes
        settings.is_training_mode = False

        # Store settings for menu system
        self.menu_system.settings = settings

        # Create game state
        time_limit_frames = replay_settings.get('time_limit', 0)

        from game_engine import create_game
        self.game_state = create_game(
            settings.stage_name,
            settings.p1_character,
            settings.p2_character,
            stock_count=settings.stock_count,
            time_limit=time_limit_frames
        )

        # Create game loop (but we'll control input)
        self.game_loop = GameLoop(self.game_state, headless=self.headless)
        self.game_loop.on_frame_update = self._on_replay_frame_update
        if not self.headless:
            self.game_loop.on_render = self._on_render
        self.game_loop.on_match_end = self._on_replay_end

        # Connect effect callbacks
        if not self.headless and self.renderer:
            self.game_loop.on_hit = self._on_hit
            self.game_loop.on_projectile_hit = self._on_projectile_hit
            self.game_loop.on_reflect = self._on_reflect
            self.game_loop.on_land = self._on_land
            self.game_loop.on_dodge = self._on_dodge

        # Reset replay player
        self.replay_player.reset()
        self.replay_frame = 0
        self.replay_paused = False

        self.mode = GameMode.REPLAY_PLAYBACK

        print(f"\n=== REPLAY PLAYBACK ===")
        print(f"{settings.p1_character} vs {settings.p2_character}")
        print(f"Stage: {settings.stage_name}")
        print(f"Total frames: {self.replay_player.total_frames}")

    def _on_replay_frame_update(self, state: GameState):
        """Called every frame during replay playback - applies recorded inputs."""
        self.input_manager.poll()

        if self.input_manager.quit_requested:
            self.quit_requested = True
            if self.game_loop:
                self.game_loop.stop()
            return

        # Check for pause toggle
        pause_key = self.input_manager.pause_requested
        if pause_key and not self._prev_pause_key:
            self.replay_paused = not self.replay_paused
        self._prev_pause_key = pause_key

        # Check escape to exit replay
        escape_key = self.input_manager.keyboard.is_key_pressed(0xff1b)
        if escape_key and not self._prev_escape_key:
            self.mode = GameMode.REPLAY_BROWSER
            if self.game_loop:
                self.game_loop.stop()
            return
        self._prev_escape_key = escape_key

        if not self.replay_paused and self.replay_player:
            if self.replay_frame < self.replay_player.total_frames:
                # Get inputs from replay
                p1_input, p2_input = self.replay_player.get_frame_inputs(self.replay_frame)
                state.fighters[0].input = p1_input
                state.fighters[1].input = p2_input
                self.replay_frame += 1
            else:
                # Replay finished
                if self.game_loop:
                    self.game_loop.stop()

    def _on_replay_end(self, state: GameState):
        """Called when replay playback ends."""
        print("\n=== REPLAY FINISHED ===")
        # Go back to replay browser
        self.mode = GameMode.REPLAY_BROWSER
        if self.replay_browser:
            self.replay_browser.refresh()

    def _handle_replay_playback(self):
        """Handle replay playback mode."""
        if self.game_loop and self.game_state:
            # Run one frame
            if not self.game_loop.step():
                # Playback ended or stopped
                if self.mode == GameMode.REPLAY_PLAYBACK:
                    self._on_replay_end(self.game_state)
            else:
                # Render with replay overlay
                if self.game_loop.on_render and self.game_state:
                    self.game_loop.on_render(self.game_state)

                    # Draw replay indicator overlay
                    self._draw_replay_overlay()
        else:
            self.mode = GameMode.REPLAY_BROWSER

    def _draw_replay_overlay(self):
        """Draw replay indicator and progress."""
        if not self.renderer or not self.replay_player:
            return

        # Already flipped in _on_render, so we need to draw before flip
        # This is called after render, so we need to draw to screen directly

        # Actually, since _on_render already calls flip, we need to modify approach
        # For now, skip the overlay and let the render show the match

    def _on_frame_update(self, state: GameState):
        """Called every simulation frame during a match."""
        # Poll input
        self.input_manager.poll()

        # Check system keys
        if self.input_manager.quit_requested:
            self.quit_requested = True
            if self.game_loop:
                self.game_loop.stop()
            return

        # Toggle pause with P key
        pause_key = self.input_manager.pause_requested
        if pause_key and not self._prev_pause_key:
            state.paused = not state.paused
            if state.paused:
                self.pause_menu_selection = 0  # Reset to Resume on pause
                print("[PAUSED] Navigate with arrows, Enter to select")
        self._prev_pause_key = pause_key

        # Cycle 19: Enhanced pause menu navigation when paused
        if state.paused:
            self._handle_pause_menu_input(state)

        # Check escape key for unpause or back
        escape_key = self.input_manager.keyboard.is_key_pressed(0xff1b)  # Escape
        if escape_key and not self._prev_escape_key:
            if state.paused:
                # ESC unpauses (resume) - same as selecting Resume
                state.paused = False
        self._prev_escape_key = escape_key

        # Toggle debug display / Cycle hitbox mode (Cycle 16: enhanced)
        debug_key = self.input_manager.debug_toggle
        if debug_key and not self._prev_debug_key:
            if self.training_manager:
                # Cycle 16: Use enhanced hitbox visualization modes
                new_mode = self.training_manager.cycle_hitbox_mode()
                print(f"[TRAINING] Hitbox Mode: {new_mode.name}")
                # Also update renderer's simple toggles for compatibility
                if self.renderer:
                    self.renderer.show_hitboxes = self.training_manager.hitbox_viz.show_hitboxes
                    self.renderer.show_hurtboxes = self.training_manager.hitbox_viz.show_hurtboxes
            else:
                # Fallback to simple toggle when not in training mode
                self.show_hitboxes = not self.show_hitboxes
                self.show_hurtboxes = not self.show_hurtboxes
                if self.renderer:
                    self.renderer.show_hitboxes = self.show_hitboxes
                    self.renderer.show_hurtboxes = self.show_hurtboxes
        self._prev_debug_key = debug_key

        # Training mode controls
        if self.training_manager:
            # F1: Toggle CPU
            f1_key = self.input_manager.keyboard.is_key_pressed(0xffbe)  # XK_F1
            if f1_key and not self._prev_f1_key:
                enabled = self.training_manager.toggle_cpu()
                print(f"[TRAINING] CPU: {'ON' if enabled else 'OFF'}")
            self._prev_f1_key = f1_key

            # F2: Cycle CPU behavior
            f2_key = self.input_manager.keyboard.is_key_pressed(0xffbf)  # XK_F2
            if f2_key and not self._prev_f2_key:
                if self.training_manager.cpu_controller.state.enabled:
                    new_behavior = self.training_manager.cycle_cpu_behavior()
                    print(f"[TRAINING] CPU Behavior: {new_behavior.name}")
            self._prev_f2_key = f2_key

            # F3: Toggle frame data overlay
            f3_key = self.input_manager.keyboard.is_key_pressed(0xffc0)  # XK_F3
            if f3_key and not self._prev_f3_key:
                enabled = self.training_manager.toggle_frame_data()
                print(f"[TRAINING] Frame Data: {'ON' if enabled else 'OFF'}")
            self._prev_f3_key = f3_key

            # F4: Cycle CPU difficulty
            f4_key = self.input_manager.keyboard.is_key_pressed(0xffc1)  # XK_F4
            if f4_key and not self._prev_f4_key:
                if self.training_manager.cpu_controller.state.enabled:
                    new_diff = self.training_manager.cycle_cpu_difficulty()
                    print(f"[TRAINING] CPU Difficulty: Lv{new_diff.value}")
            self._prev_f4_key = f4_key

            # F5: Toggle infinite jumps
            f5_key = self.input_manager.keyboard.is_key_pressed(0xffc2)  # XK_F5
            if f5_key and not self._prev_f5_key:
                enabled = self.training_manager.toggle_infinite_jumps()
                print(f"[TRAINING] Infinite Jumps: {'ON' if enabled else 'OFF'}")
            self._prev_f5_key = f5_key

            # F6: Toggle frame advantage display (Cycle 13)
            f6_key = self.input_manager.keyboard.is_key_pressed(0xffc3)  # XK_F6
            if f6_key and not self._prev_f6_key:
                enabled = self.training_manager.toggle_frame_advantage()
                print(f"[TRAINING] Frame Advantage Display: {'ON' if enabled else 'OFF'}")
            self._prev_f6_key = f6_key

            # F7: Toggle input history display (Cycle 16)
            f7_key = self.input_manager.keyboard.is_key_pressed(0xffc4)  # XK_F7
            if f7_key and not getattr(self, '_prev_f7_key', False):
                enabled = self.training_manager.toggle_input_history()
                print(f"[TRAINING] Input History: {'ON' if enabled else 'OFF'}")
            self._prev_f7_key = f7_key

            # F8: Toggle move list display (Cycle 16)
            f8_key = self.input_manager.keyboard.is_key_pressed(0xffc5)  # XK_F8
            if f8_key and not getattr(self, '_prev_f8_key', False):
                enabled = self.training_manager.toggle_move_list()
                print(f"[TRAINING] Move List: {'ON' if enabled else 'OFF'}")
            self._prev_f8_key = f8_key

            # R: Reset damage
            r_key = self.input_manager.keyboard.is_key_pressed(0x0072)  # XK_r
            if r_key and not self._prev_r_key:
                self.training_manager.reset_damage(state)
                print("[TRAINING] Damage reset to 0%")
            self._prev_r_key = r_key

            # T: Reset positions
            t_key = self.input_manager.keyboard.is_key_pressed(0x0074)  # XK_t
            if t_key and not self._prev_t_key:
                self.training_manager.reset_positions(state)
                print("[TRAINING] Positions reset")
            self._prev_t_key = t_key

        # Update fighter inputs (only if not paused)
        if not state.paused:
            for i, fighter in enumerate(state.fighters):
                # Check if this fighter is CPU controlled (P2 = index 1)
                if i == 1 and self.training_manager and self.training_manager.cpu_controller.state.enabled:
                    # Get CPU inputs
                    cpu_inputs = self.training_manager.get_cpu_inputs(
                        fighter, state.fighters[0], state.frame
                    )
                    # Apply CPU inputs directly
                    self._apply_cpu_inputs(fighter, cpu_inputs, state.frame)
                else:
                    # Human player input
                    self.input_manager.update_fighter(fighter, state.frame)

            # Record inputs for replay
            if self.replay_recorder and self.replay_recorder.is_recording:
                self.replay_recorder.record_frame(
                    state.fighters[0].input,
                    state.fighters[1].input
                )

            # Cycle 16: Record inputs for input history display
            if self.training_manager and self.training_manager.show_input_history:
                for i, fighter in enumerate(state.fighters):
                    input_dict = {
                        'left': fighter.input.left if hasattr(fighter, 'input') and fighter.input else False,
                        'right': fighter.input.right if hasattr(fighter, 'input') and fighter.input else False,
                        'up': fighter.input.up if hasattr(fighter, 'input') and fighter.input else False,
                        'down': fighter.input.down if hasattr(fighter, 'input') and fighter.input else False,
                        'jump': fighter.input.jump if hasattr(fighter, 'input') and fighter.input else False,
                        'attack': fighter.input.attack if hasattr(fighter, 'input') and fighter.input else False,
                        'special': fighter.input.special if hasattr(fighter, 'input') and fighter.input else False,
                        'shield': fighter.input.shield if hasattr(fighter, 'input') and fighter.input else False,
                        'grab': fighter.input.grab if hasattr(fighter, 'input') and fighter.input else False,
                    }
                    self.training_manager.record_input_for_history(i, input_dict, state.frame)

            # Update training mode features (infinite shields, auto-reset, etc.)
            if self.training_manager:
                self.training_manager.update(state)

    def _apply_cpu_inputs(self, fighter: Fighter, inputs: dict, frame: int):
        """Apply CPU input dict to fighter."""
        input_state = InputState()
        input_state.left = inputs.get('left', False)
        input_state.right = inputs.get('right', False)
        input_state.up = inputs.get('up', False)
        input_state.down = inputs.get('down', False)
        input_state.jump = inputs.get('jump', False)
        input_state.attack = inputs.get('attack', False)
        input_state.special = inputs.get('special', False)
        input_state.shield = inputs.get('shield', False)
        input_state.grab = inputs.get('grab', False)

        # Update fighter's input state
        fighter.input_state = input_state

    def _on_render(self, state: GameState):
        """Called when rendering a match frame."""
        if self.renderer:
            self.renderer.render(state)

            # Draw training mode overlay (after main render)
            if self.training_manager:
                self.renderer.draw_training_mode_info(state, self.training_manager)

            # Draw pause overlay if paused
            if state.paused:
                self._draw_pause_overlay()

            # Cycle 17: Draw FPS counter and frame graph
            if self.show_fps:
                self.renderer.draw_fps_counter(self.frame_times)
                self.renderer.draw_frame_graph(self.frame_times)

            self.renderer.flip()

            # Process window events
            events = self.renderer.process_events()
            if self.renderer.should_close:
                self.quit_requested = True
                if self.game_loop:
                    self.game_loop.stop()

    def _handle_pause_menu_input(self, state: GameState):
        """Handle pause menu navigation input (Cycle 19)."""
        # Navigate up
        up_key = (self.input_manager.keyboard.is_key_pressed(0xff52) or  # Up arrow
                  self.input_manager.keyboard.is_key_pressed(0x0077))    # W key
        if up_key and not self._prev_pause_up:
            self.pause_menu_selection = (self.pause_menu_selection - 1) % 3
            if self.audio:
                self.audio.play(SoundType.MENU_MOVE, 0.5)
        self._prev_pause_up = up_key

        # Navigate down
        down_key = (self.input_manager.keyboard.is_key_pressed(0xff54) or  # Down arrow
                    self.input_manager.keyboard.is_key_pressed(0x0073))    # S key
        if down_key and not self._prev_pause_down:
            self.pause_menu_selection = (self.pause_menu_selection + 1) % 3
            if self.audio:
                self.audio.play(SoundType.MENU_MOVE, 0.5)
        self._prev_pause_down = down_key

        # Select with Enter/Space/Attack
        select_key = (self.input_manager.keyboard.is_key_pressed(0xff0d) or  # Enter
                      self.input_manager.keyboard.is_key_pressed(0x0020) or  # Space
                      self.input_manager.keyboard.is_key_pressed(0x006a))    # J key (attack)
        if select_key and not self._prev_pause_select:
            if self.audio:
                self.audio.play(SoundType.MENU_SELECT, 0.7)
            self._execute_pause_menu_selection(state)
        self._prev_pause_select = select_key

    def _execute_pause_menu_selection(self, state: GameState):
        """Execute the selected pause menu option (Cycle 19)."""
        if self.pause_menu_selection == 0:
            # Resume
            state.paused = False
        elif self.pause_menu_selection == 1:
            # Return to Character Select
            if not self.skip_menus:
                self.menu_system.reset()
                self.menu_system.state = MenuState.CHARACTER_SELECT
                self.mode = GameMode.MENU
                if self.game_loop:
                    self.game_loop.stop()
            else:
                state.paused = False  # Can't go to char select in quick mode
        elif self.pause_menu_selection == 2:
            # Return to Title / Quit
            if not self.skip_menus:
                self.menu_system.reset()
                self.mode = GameMode.MENU
                if self.game_loop:
                    self.game_loop.stop()
            else:
                self.quit_requested = True
                if self.game_loop:
                    self.game_loop.stop()

    def _draw_pause_overlay(self):
        """Draw enhanced pause menu overlay (Cycle 19)."""
        if not self.renderer or not self.game_state:
            return

        center_x = self.renderer.width // 2
        center_y = self.renderer.height // 2

        # Dark box - taller to fit more options
        box_w, box_h = 320, 280
        self.renderer.set_color_rgb(20, 20, 30)
        self.renderer.draw_rect(center_x - box_w//2, center_y - box_h//2, box_w, box_h, filled=True)

        # Border
        self.renderer.set_color_rgb(100, 100, 120)
        self.renderer.draw_rect(center_x - box_w//2, center_y - box_h//2, box_w, box_h, filled=False)

        # Title
        self.renderer.set_color('white')
        self.renderer.draw_text(center_x - 35, center_y - 110, "PAUSED")

        # Match info section
        match_time = self.game_state.frame / 60
        minutes = int(match_time // 60)
        seconds = int(match_time % 60)
        self.renderer.set_color_rgb(150, 150, 150)
        self.renderer.draw_text(center_x - 50, center_y - 80, f"Time: {minutes}:{seconds:02d}")

        # Show current match settings (stocks/time limit)
        if self.saved_settings:
            stocks_text = f"Stocks: {self.saved_settings.stock_count}"
            time_text = f"Time: {'None' if self.saved_settings.time_limit == 0 else f'{self.saved_settings.time_limit} min'}"
            self.renderer.set_color_rgb(120, 120, 140)
            self.renderer.draw_text(center_x - 60, center_y - 60, stocks_text)
            self.renderer.draw_text(center_x + 30, center_y - 60, time_text)

        # Menu options
        options = [
            ("Resume", 100, 255, 100),           # Green
            ("Character Select", 100, 200, 255), # Cyan
            ("Return to Title", 255, 100, 100),  # Red
        ]

        for i, (text, r, g, b) in enumerate(options):
            y_pos = center_y - 20 + i * 40

            # Highlight selected option
            if i == self.pause_menu_selection:
                # Draw selection highlight
                self.renderer.set_color_rgb(60, 60, 80)
                self.renderer.draw_rect(center_x - 100, y_pos - 5, 200, 30, filled=True)
                # Draw arrow indicator
                self.renderer.set_color_rgb(255, 255, 100)
                self.renderer.draw_text(center_x - 95, y_pos, ">")
                # Brighter text
                self.renderer.set_color_rgb(r, g, b)
            else:
                # Dimmer text for unselected
                self.renderer.set_color_rgb(r // 2, g // 2, b // 2)

            self.renderer.draw_text(center_x - 70, y_pos, text)

        # Controls hint
        self.renderer.set_color_rgb(100, 100, 100)
        self.renderer.draw_text(center_x - 90, center_y + 110, "Arrow/WASD + Enter")

    def _on_match_end(self, state: GameState):
        """Called when match ends."""
        if state.winner is not None:
            print(f"\n=== PLAYER {state.winner + 1} WINS! ===")
        else:
            print("\n=== DRAW! ===")

        print(f"Match lasted {state.frame} frames ({state.frame / 60:.1f} seconds)")
        for i, f in enumerate(state.fighters):
            print(f"Player {i + 1}: {f.stocks} stocks, {f.damage_percent:.1f}% damage")
            print(f"  Max combo: {state.max_combos[i]}")

        # Transition to results screen
        self._end_match()

    # =========================================================================
    # Visual Effect Callbacks
    # =========================================================================

    def _on_hit(self, attacker, defender, hitbox, contact_x, contact_y, knockback_magnitude):
        """Called when an attack hits a fighter."""
        if self.renderer:
            # Add hit spark effect - Cycle 13: pass damage for color variations
            self.renderer.add_hit_effect(
                x=contact_x,
                y=contact_y,
                knockback_magnitude=knockback_magnitude,
                angle=hitbox.angle,
                attacker_color=attacker.color,
                damage=hitbox.damage
            )
            # Flash the defender
            self.renderer.add_damage_flash(defender)

            # Cycle 13: Frame advantage display in training mode
            if self.training_mode and self.training_manager:
                self.training_manager.record_hit(
                    attacker, defender, hitbox, contact_x, contact_y,
                    game_frame=self.game_state.frame if self.game_state else 0
                )
                # Add visual frame advantage effect
                adv_display = self.training_manager.get_frame_advantage_display()
                if adv_display:
                    x, y, text, color = adv_display
                    advantage = self.training_manager.frame_advantage.advantage_frames
                    self.renderer.add_frame_advantage_effect(contact_x, contact_y - 40, advantage)

                # Cycle 16: Record collision point for enhanced hitbox visualization
                self.training_manager.record_collision(
                    contact_x, contact_y,
                    self.game_state.frame if self.game_state else 0
                )

                # Cycle 18: Record hit for combo counter
                attacker_id = attacker.player_id if hasattr(attacker, 'player_id') else 0
                self.training_manager.record_combo_hit(
                    attacker_id, hitbox.damage, (contact_x, contact_y)
                )

        # Play hit sound
        if self.audio_callbacks:
            self.audio_callbacks.on_hit(attacker, defender, hitbox, contact_x, contact_y,
                                        knockback_magnitude)

        # Cycle 13: Gamepad rumble
        gamepad_mgr = get_gamepad_manager()
        # Rumble defender's controller (got hit)
        gamepad_mgr.rumble_on_hit(defender.player_id, knockback_magnitude)
        # Light rumble for attacker (attack connected)
        gamepad_mgr.rumble_on_attack_connect(attacker.player_id, hitbox.damage)

    def _on_projectile_hit(self, projectile, defender, contact_x, contact_y, knockback_magnitude):
        """Called when a projectile hits a fighter."""
        if self.renderer:
            # Add hit spark effect - Cycle 13: pass damage for color variations
            self.renderer.add_hit_effect(
                x=contact_x,
                y=contact_y,
                knockback_magnitude=knockback_magnitude,
                angle=projectile.hitbox.angle,
                attacker_color=projectile.color,
                damage=projectile.hitbox.damage
            )
            # Flash the defender
            self.renderer.add_damage_flash(defender)

            # Cycle 18: Record projectile hit for combo counter in training mode
            if self.training_mode and self.training_manager:
                # Determine attacker ID from projectile owner
                attacker_id = projectile.owner_id if hasattr(projectile, 'owner_id') else 0
                self.training_manager.record_combo_hit(
                    attacker_id, projectile.hitbox.damage, (contact_x, contact_y)
                )

        # Play projectile hit sound
        if self.audio_callbacks:
            self.audio_callbacks.on_projectile_hit(projectile, defender, contact_x, contact_y,
                                                   knockback_magnitude)

        # Cycle 13: Gamepad rumble (defender got hit by projectile)
        gamepad_mgr = get_gamepad_manager()
        gamepad_mgr.rumble_on_hit(defender.player_id, knockback_magnitude)

    def _on_reflect(self, projectile, reflector, x, y):
        """Called when a projectile is reflected."""
        if self.renderer:
            # Add reflector hexagon effect
            self.renderer.add_reflect_effect(x, y)

        # Play reflect sound
        if self.audio_callbacks:
            self.audio_callbacks.on_reflect(projectile, reflector, x, y)

    def _play_char_preview(self, char_name: str):
        """
        Cycle 18: Play character-specific preview sound on cursor move.
        """
        if self.audio and self.audio.sound_library:
            sound_type = self.audio.sound_library.get_character_preview_sound(char_name)
            if sound_type:
                self.audio.play(sound_type, 0.6)
            else:
                # Fallback to menu move sound
                self.audio.play(SoundType.MENU_MOVE, 0.5)

    def _play_char_lock(self, char_name: str):
        """
        Cycle 18: Play character lock confirmation sound.
        """
        if self.audio:
            self.audio.play(SoundType.CHAR_LOCK, 0.7)

    def _on_land(self, fighter, x, y):
        """Called when a fighter lands."""
        if self.renderer:
            # Add landing dust effect
            self.renderer.add_landing_dust(x, y)

        # Play landing sound
        if self.audio_callbacks:
            self.audio_callbacks.on_land(fighter, x, y)

    def _on_dodge(self, fighter, dodge_type):
        """Called when a fighter starts a dodge (Cycle 9)."""
        # Play dodge sound
        if self.audio_callbacks:
            self.audio_callbacks.on_dodge(fighter)

    def run(self, max_frames: int = 0):
        """
        Run the game main loop.

        Args:
            max_frames: Maximum frames to run (0 = until quit)
        """
        print("\n=== STICK FIGURE FIGHTER ===")
        print(f"Display: {self.display_name}")
        print(f"Headless: {self.headless}")

        if not self.headless:
            print("\nControls:")
            print("  P1: WASD + Space/J/K/L")
            print("  P2: Arrows + Enter/Numpad 1,2,3")
            print("  P: Pause | Tab: Hitboxes | ESC: Quit")

        print("\nStarting game...\n")

        # Initialize key state trackers for results screen
        self._prev_left_key = False
        self._prev_right_key = False
        self._prev_confirm_key = False

        try:
            while not self.quit_requested:
                self.frame_count += 1

                # Cycle 17: Track frame time for FPS counter
                current_time = time.perf_counter()
                frame_time_ms = (current_time - self.last_frame_time) * 1000
                self.last_frame_time = current_time
                self.frame_times.append(frame_time_ms)
                if len(self.frame_times) > 60:
                    self.frame_times.pop(0)

                # Cycle 17: F10 toggle for FPS display
                f10_key = self.input_manager.keyboard.is_key_pressed(0xffc7)  # XK_F10
                if f10_key and not self._prev_f10_key:
                    self.show_fps = not self.show_fps
                    print(f"[DEBUG] FPS Display: {'ON' if self.show_fps else 'OFF'}")
                self._prev_f10_key = f10_key

                if self.mode == GameMode.MENU:
                    self._handle_menu_input()
                    time.sleep(1/60)

                elif self.mode == GameMode.COUNTDOWN:
                    self._handle_countdown()
                    time.sleep(1/60)

                elif self.mode == GameMode.MATCH:
                    if self.game_loop:
                        # Run one frame of the match
                        if not self.game_loop.step():
                            # Match ended
                            if self.mode == GameMode.MATCH:  # Still in match mode (not quit)
                                self._end_match()
                        else:
                            # Render
                            if self.game_loop.on_render and self.game_state:
                                self.game_loop.on_render(self.game_state)

                            # Frame timing
                            time.sleep(1/60)
                    else:
                        # No game loop, go back to menu
                        self.mode = GameMode.MENU

                elif self.mode == GameMode.RESULTS:
                    self._handle_results_input()
                    time.sleep(1/60)

                elif self.mode == GameMode.REPLAY_BROWSER:
                    self._handle_replay_browser()
                    time.sleep(1/60)

                elif self.mode == GameMode.REPLAY_PLAYBACK:
                    self._handle_replay_playback()
                    time.sleep(1/60)

                # Max frames limit (for testing)
                if max_frames > 0 and self.frame_count >= max_frames:
                    break

        except KeyboardInterrupt:
            print("\nGame interrupted.")

        return self.game_state

    def cleanup(self):
        """Clean up resources."""
        # Cleanup audio
        if self.audio:
            self.audio.cleanup()

        if self.input_manager:
            self.input_manager.close()
        if self.renderer:
            self.renderer.close()


# =============================================================================
# HEADLESS MODE FOR ML TRAINING
# =============================================================================

def run_training_match(num_frames: int = 3600) -> dict:
    """
    Run a headless match with random inputs for testing ML integration.

    Returns match statistics.
    """
    import numpy as np

    game = create_game()

    def random_input(state: GameState, player_id: int) -> InputState:
        """Generate random inputs for testing."""
        inp = InputState()
        r = np.random.random

        # Movement
        if r() < 0.3:
            inp.left = True
        elif r() < 0.6:
            inp.right = True

        # Vertical
        if r() < 0.1:
            inp.up = True
            inp.jump = True
        elif r() < 0.15:
            inp.down = True

        # Combat
        if r() < 0.15:
            inp.attack = True

        return inp

    final_state = run_headless_match(game, random_input, max_frames=num_frames)

    return {
        'frames': final_state.frame,
        'winner': final_state.winner,
        'p1_stocks': final_state.fighters[0].stocks,
        'p2_stocks': final_state.fighters[1].stocks,
        'p1_damage': final_state.fighters[0].damage_percent,
        'p2_damage': final_state.fighters[1].damage_percent,
    }


def benchmark_headless(num_matches: int = 10, frames_per_match: int = 3600):
    """
    Benchmark headless simulation speed.

    This measures how fast matches can run for ML training.
    """
    import numpy as np

    print(f"\nBenchmarking {num_matches} matches, {frames_per_match} frames each...")

    total_frames = 0
    start_time = time.time()

    for i in range(num_matches):
        result = run_training_match(frames_per_match)
        total_frames += result['frames']

        if (i + 1) % 5 == 0:
            elapsed = time.time() - start_time
            fps = total_frames / elapsed
            print(f"Match {i + 1}/{num_matches} - {fps:.0f} frames/sec")

    elapsed = time.time() - start_time
    avg_fps = total_frames / elapsed

    print(f"\n=== Benchmark Results ===")
    print(f"Total frames: {total_frames}")
    print(f"Total time: {elapsed:.2f}s")
    print(f"Average speed: {avg_fps:.0f} frames/sec")
    print(f"Real-time multiple: {avg_fps / 60:.1f}x")


# =============================================================================
# ML OBSERVATION/ACTION TEST
# =============================================================================

def test_ml_interface():
    """Test the ML observation and action interface."""
    import numpy as np

    print("\n=== Testing ML Interface ===\n")

    game = create_game()

    # Test observation
    obs = get_observation(game, player_id=0)
    print(f"Observation shape: {obs.shape}")
    print(f"Observation (first 10): {obs[:10]}")

    # Test action conversion
    for i in range(10):
        inp = get_action_from_index(i)
        actions = []
        if inp.left: actions.append('L')
        if inp.right: actions.append('R')
        if inp.up: actions.append('U')
        if inp.down: actions.append('D')
        if inp.jump: actions.append('Jump')
        if inp.attack: actions.append('Attack')
        print(f"Action {i}: {' '.join(actions) if actions else 'None'}")

    print("\nML interface OK!")


# =============================================================================
# QUICK TEST
# =============================================================================

def quick_test():
    """Run a quick test to verify the game works."""
    print("\n=== Quick Test ===\n")

    # Test game creation
    print("1. Creating game state...")
    game = create_game()
    assert len(game.fighters) == 2
    assert game.stage is not None
    print("   OK - Game state created")

    # Test physics simulation
    print("2. Testing physics simulation...")
    loop = GameLoop(game, headless=True)
    for _ in range(60):  # 1 second
        loop.step()
    assert game.frame == 60
    print("   OK - Physics simulation works")

    # Test headless match
    print("3. Testing headless match...")
    import numpy as np

    game2 = create_game()
    def test_input(state, pid):
        inp = InputState()
        inp.right = np.random.random() > 0.5
        inp.attack = np.random.random() > 0.9
        return inp

    final = run_headless_match(game2, test_input, max_frames=300)
    assert final.frame > 0
    print(f"   OK - Headless match ran {final.frame} frames")

    # Test observation
    print("4. Testing ML observation...")
    obs = get_observation(game2, 0)
    assert obs.shape[0] == 21
    print(f"   OK - Observation shape: {obs.shape}")

    # Test menu system
    print("5. Testing menu system...")
    from menu_system import MenuSystem
    menu = MenuSystem()
    assert menu.state.name == "TITLE" or menu.state.name == "CHARACTER_SELECT"
    print("   OK - Menu system initialized")

    print("\n=== All Tests Passed! ===\n")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Stick Figure Fighter")
    parser.add_argument('--display', type=str, default=os.environ.get('DISPLAY', ':0'),
                       help='X11 display (default: $DISPLAY, else :0)')
    parser.add_argument('--headless', action='store_true',
                       help='Run in headless mode (no rendering)')
    parser.add_argument('--quick', action='store_true',
                       help='Skip menus and start a quick match')
    parser.add_argument('--stage', type=str, default='battlefield',
                       choices=['battlefield', 'final_destination'],
                       help='Stage to play on')
    parser.add_argument('--p1', type=str, default=None,
                       choices=['RUSH', 'ZONE', 'GRAB', 'FLEX'],
                       help='P1 character (default: FLEX)')
    parser.add_argument('--p2', type=str, default=None,
                       choices=['RUSH', 'ZONE', 'GRAB', 'FLEX'],
                       help='P2 character (default: FLEX)')
    parser.add_argument('--test', action='store_true',
                       help='Run quick test')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run headless benchmark')
    parser.add_argument('--ml-test', action='store_true',
                       help='Test ML interface')
    parser.add_argument('--max-frames', type=int, default=0,
                       help='Maximum frames (0 = no limit)')
    parser.add_argument('--training', action='store_true',
                       help='Start in training mode with CPU opponent')

    args = parser.parse_args()

    if args.test:
        quick_test()
        return

    if args.benchmark:
        benchmark_headless()
        return

    if args.ml_test:
        test_ml_interface()
        return

    # Determine if we should skip menus
    skip_menus = args.quick or args.headless or args.p1 or args.p2 or args.training

    # Run the game
    game = StickFighterGame(
        display_name=args.display,
        headless=args.headless,
        skip_menus=skip_menus,
        stage_name=args.stage,
        p1_character=args.p1,
        p2_character=args.p2,
        training_mode=args.training
    )

    try:
        game.run(max_frames=args.max_frames)
    finally:
        game.cleanup()


if __name__ == "__main__":
    main()
