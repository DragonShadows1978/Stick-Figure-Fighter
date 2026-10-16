"""
Observation and action encoding for ML agents (1v1).

This is the interface the trained policies were built against; changing the
observation layout or action table invalidates existing checkpoints.
"""
import numpy as np

from engine import CONFIG, FighterState, GameState, InputState


def get_observation(game_state: GameState, player_id: int) -> np.ndarray:
    """
    Get observation array for a player (for ML training).

    Returns a flat numpy array with normalized game state.
    """
    player = game_state.fighters[player_id]
    opponent = game_state.fighters[1 - player_id]

    # Normalize positions to [0, 1]
    px = player.position.x / CONFIG.screen_width
    py = player.position.y / CONFIG.screen_height
    ox = opponent.position.x / CONFIG.screen_width
    oy = opponent.position.y / CONFIG.screen_height

    # Normalize velocities (assuming max ~20 units)
    pvx = player.velocity.x / 20.0
    pvy = player.velocity.y / 20.0
    ovx = opponent.velocity.x / 20.0
    ovy = opponent.velocity.y / 20.0

    # Damage (0-999 -> 0-1)
    p_dmg = player.damage_percent / 999.0
    o_dmg = opponent.damage_percent / 999.0

    # Stocks (0-3 -> 0-1)
    p_stocks = player.stocks / CONFIG.stock_count
    o_stocks = opponent.stocks / CONFIG.stock_count

    # State one-hot (simplified - just a few key states)
    state_vec = np.zeros(5)
    if player.state == FighterState.IDLE:
        state_vec[0] = 1
    elif player.state in (FighterState.WALKING, FighterState.RUNNING):
        state_vec[1] = 1
    elif player.state in (FighterState.JUMPING, FighterState.FALLING):
        state_vec[2] = 1
    elif player.state == FighterState.ATTACKING:
        state_vec[3] = 1
    elif player.state == FighterState.HITSTUN:
        state_vec[4] = 1

    # Facing direction
    facing = 1.0 if player.facing_right else -1.0
    o_facing = 1.0 if opponent.facing_right else -1.0

    # Grounded
    grounded = 1.0 if player.grounded else 0.0
    o_grounded = 1.0 if opponent.grounded else 0.0

    # Combine all observations
    obs = np.array([
        px, py, pvx, pvy,
        ox, oy, ovx, ovy,
        p_dmg, o_dmg,
        p_stocks, o_stocks,
        facing, o_facing,
        grounded, o_grounded,
        *state_vec
    ], dtype=np.float32)

    return obs


def get_action_from_index(action_idx: int) -> InputState:
    """
    Convert a discrete action index to InputState.

    Action space:
    0: Nothing
    1: Left
    2: Right
    3: Up/Jump
    4: Down
    5: Attack
    6: Left + Attack
    7: Right + Attack
    8: Up + Attack
    9: Down + Attack
    """
    inp = InputState()

    if action_idx == 0:
        pass  # Nothing
    elif action_idx == 1:
        inp.left = True
    elif action_idx == 2:
        inp.right = True
    elif action_idx == 3:
        inp.up = True
        inp.jump = True
    elif action_idx == 4:
        inp.down = True
    elif action_idx == 5:
        inp.attack = True
    elif action_idx == 6:
        inp.left = True
        inp.attack = True
    elif action_idx == 7:
        inp.right = True
        inp.attack = True
    elif action_idx == 8:
        inp.up = True
        inp.attack = True
    elif action_idx == 9:
        inp.down = True
        inp.attack = True

    return inp


ACTION_SPACE_SIZE = 10
OBSERVATION_SIZE = 21  # Size of get_observation output
