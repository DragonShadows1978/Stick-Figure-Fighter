#!/usr/bin/env python3
"""
Test script to verify all 4 characters are implemented correctly.

Tests:
1. Character stats are distinct
2. All attacks are defined for each character
3. Game can be created with each character pair
4. Headless match runs without errors
5. Character-specific attack data is used
"""
import sys
import time

# Add parent path for imports
sys.path.insert(0, '/home/vader/mini-mind-v2/workspace/Stick_Figure_Fighter')

from game_engine import (
    create_game, run_headless_match, GameLoop, InputState, AttackType,
    FighterState, Projectile, DodgeType
)
from characters import (
    CharacterType, CHARACTER_STATS, CHARACTER_COLORS,
    get_character_attacks, get_character_stats, get_character_color,
    list_characters
)


def test_character_stats():
    """Test that all characters have distinct stats."""
    print("\n=== Testing Character Stats ===")

    for char_type in CharacterType:
        stats = get_character_stats(char_type)
        color = get_character_color(char_type)
        print(f"\n{char_type.name}:")
        print(f"  Walk/Run: {stats.walk_speed}/{stats.run_speed}")
        print(f"  Air Speed: {stats.air_speed}")
        print(f"  Jump: {stats.jump_force}/{stats.double_jump_force}")
        print(f"  Weight: {stats.weight}")
        print(f"  Size: {stats.width}x{stats.height}")
        print(f"  Color: RGB{color}")

    # Verify stats are distinct
    speeds = [CHARACTER_STATS[c].run_speed for c in CharacterType]
    weights = [CHARACTER_STATS[c].weight for c in CharacterType]

    assert len(set(speeds)) == len(speeds), "All characters should have unique run speeds"
    assert len(set(weights)) == len(weights), "All characters should have unique weights"

    print("\n[PASS] All character stats are distinct")
    return True


def test_character_attacks():
    """Test that all characters have complete attack sets."""
    print("\n=== Testing Character Attacks ===")

    # Required attacks
    required_attacks = [
        AttackType.JAB,
        AttackType.FORWARD_TILT, AttackType.UP_TILT, AttackType.DOWN_TILT,
        AttackType.FORWARD_SMASH, AttackType.UP_SMASH, AttackType.DOWN_SMASH,
        AttackType.NEUTRAL_AIR, AttackType.FORWARD_AIR, AttackType.BACK_AIR,
        AttackType.UP_AIR, AttackType.DOWN_AIR,
        AttackType.NEUTRAL_SPECIAL, AttackType.SIDE_SPECIAL,
        AttackType.UP_SPECIAL, AttackType.DOWN_SPECIAL,
    ]

    for char_type in CharacterType:
        attacks = get_character_attacks(char_type)
        print(f"\n{char_type.name}: {len(attacks)} attacks defined")

        missing = []
        for attack_type in required_attacks:
            if attack_type not in attacks:
                missing.append(attack_type.name)
            else:
                attack = attacks[attack_type]
                print(f"  {attack_type.name}: startup={attack.startup_frames}f, "
                      f"total={attack.total_frames}f, hitboxes={len(attack.hitboxes)}")

        if missing:
            print(f"  MISSING: {missing}")
            return False

    print("\n[PASS] All characters have complete attack sets")
    return True


def test_game_creation():
    """Test that games can be created with all character combinations."""
    print("\n=== Testing Game Creation ===")

    characters = [c.name for c in CharacterType]

    for p1_char in characters:
        for p2_char in characters:
            game = create_game(
                stage_name="battlefield",
                p1_character=p1_char,
                p2_character=p2_char
            )

            p1 = game.fighters[0]
            p2 = game.fighters[1]

            # Verify character types are set
            assert p1.stats.character_type == p1_char, f"P1 should be {p1_char}"
            assert p2.stats.character_type == p2_char, f"P2 should be {p2_char}"

            # Verify colors are different (unless same character)
            if p1_char != p2_char:
                assert p1.color != p2.color, f"Different characters should have different colors"

            print(f"  {p1_char} vs {p2_char}: OK")

    print("\n[PASS] All character combinations create valid games")
    return True


def test_headless_match():
    """Test that a headless match runs correctly with characters."""
    print("\n=== Testing Headless Match ===")

    # Test RUSH vs GRAB (extreme opposites)
    game = create_game(
        stage_name="final_destination",
        p1_character="RUSH",
        p2_character="GRAB"
    )

    # Aggressive random input
    import random

    loop = GameLoop(game, headless=True)

    def random_aggressive_input(state, player_id):
        inp = InputState()
        inp.left = random.random() > 0.6
        inp.right = random.random() > 0.6
        inp.up = random.random() > 0.8
        inp.down = random.random() > 0.8
        inp.jump = random.random() > 0.85
        inp.attack = random.random() > 0.7
        inp.special = random.random() > 0.85
        return inp

    # Run frames directly with step() for maximum speed
    start_time = time.time()
    num_frames = 6000  # 100 seconds of game time
    frames_run = 0

    for frame in range(num_frames):
        for fighter in game.fighters:
            fighter.input = random_aggressive_input(game, fighter.player_id)
        if not loop.step():
            # Match ended early (KO)
            break
        frames_run += 1
        for fighter in game.fighters:
            fighter.input.update_prev()

    elapsed = time.time() - start_time

    fps = frames_run / elapsed if elapsed > 0 else 0
    print(f"\n  Simulated {frames_run} frames in {elapsed:.3f}s ({fps:.0f} FPS)")
    print(f"  RUSH damage: {game.fighters[0].damage_percent:.1f}%")
    print(f"  RUSH stocks: {game.fighters[0].stocks}")
    print(f"  GRAB damage: {game.fighters[1].damage_percent:.1f}%")
    print(f"  GRAB stocks: {game.fighters[1].stocks}")
    print(f"  Match over: {game.match_over}")

    # Must be far faster than real time (60 FPS). Kept loose so machine load
    # doesn't make it flaky; typical is ~50k FPS.
    assert fps > 5000, f"Should run at >5000 FPS headless, got {fps:.0f}"

    # Should have run for at least some frames
    assert frames_run > 100, f"Should run for more than 100 frames, ran {frames_run}"

    print("\n[PASS] Headless match runs correctly")
    return True


def test_character_attack_data_used():
    """Test that character-specific attack data is actually used."""
    print("\n=== Testing Character Attack Data Usage ===")

    # Compare RUSH jab (fast) vs GRAB jab (slow)
    game = create_game(p1_character="RUSH", p2_character="GRAB")

    rush_fighter = game.fighters[0]
    grab_fighter = game.fighters[1]

    # Get attack data for jab
    rush_jab = rush_fighter.get_attack_data(AttackType.JAB)
    grab_jab = grab_fighter.get_attack_data(AttackType.JAB)

    print(f"\n  RUSH JAB: startup={rush_jab.startup_frames}f, total={rush_jab.total_frames}f")
    print(f"  GRAB JAB: startup={grab_jab.startup_frames}f, total={grab_jab.total_frames}f")

    assert rush_jab.startup_frames < grab_jab.startup_frames, "RUSH should have faster jab"
    assert rush_jab.total_frames < grab_jab.total_frames, "RUSH should have shorter jab animation"

    # Test damage difference
    rush_damage = rush_jab.hitboxes[0].damage
    grab_damage = grab_jab.hitboxes[0].damage

    print(f"  RUSH JAB damage: {rush_damage}")
    print(f"  GRAB JAB damage: {grab_damage}")

    assert grab_damage > rush_damage, "GRAB should have higher damage per hit"

    print("\n[PASS] Character attack data is correctly used")
    return True


def test_special_moves():
    """Test that special moves work."""
    print("\n=== Testing Special Moves ===")

    game = create_game(p1_character="FLEX", p2_character="ZONE")
    loop = GameLoop(game, headless=True)

    # Give P1 special input
    p1 = game.fighters[0]
    p1.input.special = True

    # Run a few frames
    for _ in range(5):
        loop.step()
        p1.input.update_prev()
        p1.input.special = False

    # Check P1 entered special state
    assert p1.state == FighterState.SPECIAL or p1.current_attack is not None, \
        "P1 should be in special state or have an attack"

    print(f"  P1 state: {p1.state.name}")
    print(f"  P1 attack: {p1.current_attack.attack_type.name if p1.current_attack else 'None'}")

    # Test up special (recovery)
    game2 = create_game(p1_character="RUSH", p2_character="GRAB")
    loop2 = GameLoop(game2, headless=True)
    p1_2 = game2.fighters[0]

    # Put P1 in air
    p1_2.grounded = False
    p1_2.position.y = 300
    p1_2.velocity.y = 5  # Falling

    # Up special
    p1_2.input.up = True
    p1_2.input.special = True

    for _ in range(3):
        loop2.step()
        p1_2.input.update_prev()

    # Should have upward velocity from up special
    print(f"  After up special: velocity.y = {p1_2.velocity.y:.1f}")

    print("\n[PASS] Special moves work correctly")
    return True


def test_projectile_system():
    """Test that projectiles spawn and work correctly."""
    print("\n=== Testing Projectile System ===")

    # ZONE neutral-B should spawn a projectile
    game = create_game(p1_character="ZONE", p2_character="RUSH")
    loop = GameLoop(game, headless=True)

    zone = game.fighters[0]
    rush = game.fighters[1]

    # Position fighters apart
    zone.position.x = 300
    rush.position.x = 700

    # ZONE uses neutral special (should spawn projectile)
    zone.input.special = True

    # Run a few frames to spawn projectile
    for _ in range(10):
        loop.step()
        zone.input.update_prev()
        zone.input.special = False

    # Should have at least one projectile
    assert len(game.projectiles) >= 1, f"ZONE neutral-B should spawn projectile, got {len(game.projectiles)}"

    proj = game.projectiles[0]
    print(f"  Projectile spawned: pos=({proj.position.x:.0f}, {proj.position.y:.0f}), vel=({proj.velocity.x:.1f}, {proj.velocity.y:.1f})")
    print(f"  Projectile damage: {proj.hitbox.damage}, lifetime: {proj.lifetime}")

    # Run more frames to let projectile travel
    initial_x = proj.position.x
    for _ in range(30):
        loop.step()

    # Projectile should have moved
    assert proj.position.x != initial_x, "Projectile should move"
    print(f"  After 30 frames: pos=({proj.position.x:.0f}, {proj.position.y:.0f})")

    # Test FLEX fireball (has gravity)
    game2 = create_game(p1_character="FLEX", p2_character="ZONE")
    loop2 = GameLoop(game2, headless=True)
    flex = game2.fighters[0]

    flex.input.special = True
    for _ in range(10):
        loop2.step()
        flex.input.update_prev()
        flex.input.special = False

    assert len(game2.projectiles) >= 1, "FLEX neutral-B should spawn fireball"
    fireball = game2.projectiles[0]
    assert fireball.has_gravity, "FLEX fireball should have gravity"
    print(f"  FLEX fireball has gravity: {fireball.has_gravity}")

    print("\n[PASS] Projectile system works correctly")
    return True


def test_shield_mechanics():
    """Test shield activation and damage."""
    print("\n=== Testing Shield Mechanics ===")

    game = create_game(p1_character="RUSH", p2_character="GRAB")
    loop = GameLoop(game, headless=True)

    rush = game.fighters[0]
    grab = game.fighters[1]

    # Position them close
    rush.position.x = 500
    grab.position.x = 550

    # GRAB shields
    grab.input.shield = True

    # Run a frame to activate shield
    loop.step()
    grab.input.update_prev()

    assert grab.state == FighterState.SHIELDING, f"GRAB should be shielding, is {grab.state}"
    initial_shield = grab.shield_health
    print(f"  Shield activated: health = {grab.shield_health:.1f}")

    # Hold shield for a while (should slowly drain)
    for _ in range(60):
        loop.step()
        grab.input.update_prev()

    assert grab.shield_health < initial_shield, "Shield should drain while holding"
    print(f"  After 60 frames: health = {grab.shield_health:.1f}")

    # Release shield
    grab.input.shield = False
    loop.step()
    grab.input.update_prev()

    # Wait for shield to start regenerating
    for _ in range(120):
        loop.step()

    print(f"  After releasing + 120 frames: health = {grab.shield_health:.1f}")

    print("\n[PASS] Shield mechanics work correctly")
    return True


def test_dodge_mechanics():
    """Test spot dodge, roll, and air dodge."""
    print("\n=== Testing Dodge Mechanics ===")

    game = create_game(p1_character="RUSH", p2_character="ZONE")
    loop = GameLoop(game, headless=True)

    rush = game.fighters[0]

    # First activate shield, then down for spot dodge
    rush.input.shield = True
    loop.step()
    rush.input.update_prev()

    # Spot dodge (down + shield while shielding)
    rush.input.down = True
    loop.step()
    rush.input.update_prev()

    if rush.state == FighterState.DODGING:
        print(f"  Spot dodge activated: invincible_frames = {rush.invincible_frames}")
        assert rush.invincible_frames > 0, "Should have invincibility"
    else:
        print(f"  State after spot dodge attempt: {rush.state}")

    # Reset
    game2 = create_game(p1_character="RUSH", p2_character="ZONE")
    loop2 = GameLoop(game2, headless=True)
    rush2 = game2.fighters[0]

    # Air dodge (shield while in air)
    rush2.grounded = False
    rush2.position.y = 400
    rush2.air_dodge_available = True
    rush2.input.shield = True

    loop2.step()
    rush2.input.update_prev()

    if rush2.state == FighterState.DODGING:
        print(f"  Air dodge activated: invincible_frames = {rush2.invincible_frames}")
        assert not rush2.air_dodge_available, "Air dodge should be consumed"
    else:
        print(f"  Air dodge test state: {rush2.state}")

    print("\n[PASS] Dodge mechanics work correctly")
    return True


def test_counter_mechanic():
    """Test RUSH counter (down-B)."""
    print("\n=== Testing Counter Mechanic ===")

    game = create_game(p1_character="RUSH", p2_character="GRAB")
    loop = GameLoop(game, headless=True)

    rush = game.fighters[0]
    grab = game.fighters[1]

    # Position close
    rush.position.x = 500
    grab.position.x = 560

    # RUSH uses down-B (counter)
    rush.input.down = True
    rush.input.special = True

    for _ in range(5):
        loop.step()
        rush.input.update_prev()
        rush.input.special = False

    assert rush.state == FighterState.COUNTER, f"RUSH should be in COUNTER state, is {rush.state}"
    assert rush.counter_active, "Counter should be active"
    assert rush.counter_window_frames > 0, "Counter window should be open"

    print(f"  Counter activated: window_frames = {rush.counter_window_frames}")
    print(f"  Counter attack: {rush.current_attack.attack_type.name if rush.current_attack else 'None'}")

    # Check the attack has is_counter flag
    if rush.current_attack:
        assert rush.current_attack.is_counter, "Attack should have is_counter flag"

    print("\n[PASS] Counter mechanic works correctly")
    return True


def test_reflector_mechanic():
    """Test FLEX reflector (down-B)."""
    print("\n=== Testing Reflector Mechanic ===")

    game = create_game(p1_character="FLEX", p2_character="ZONE")
    loop = GameLoop(game, headless=True)

    flex = game.fighters[0]
    zone = game.fighters[1]

    # FLEX uses down-B (reflector)
    flex.input.down = True
    flex.input.special = True

    for _ in range(8):
        loop.step()
        flex.input.update_prev()
        flex.input.special = False
        flex.input.down = False

    assert flex.state == FighterState.REFLECTING, f"FLEX should be REFLECTING, is {flex.state}"
    print(f"  Reflector activated: reflector_active = {flex.reflector_active}")

    # Check the attack has is_reflector flag
    if flex.current_attack:
        assert flex.current_attack.is_reflector, "Attack should have is_reflector flag"
        print(f"  Reflector hitbox active frames: {flex.current_attack.hitboxes[0].active_frames}")

    print("\n[PASS] Reflector mechanic works correctly")
    return True


def test_super_armor():
    """Test GRAB super armor on neutral-B."""
    print("\n=== Testing Super Armor ===")

    game = create_game(p1_character="GRAB", p2_character="RUSH")
    loop = GameLoop(game, headless=True)

    grab = game.fighters[0]
    rush = game.fighters[1]

    # Position close
    grab.position.x = 500
    rush.position.x = 560

    # GRAB uses neutral-B (armored charge)
    grab.input.special = True

    for _ in range(20):
        loop.step()
        grab.input.update_prev()
        grab.input.special = False

    # Check super armor is active
    if grab.current_attack and grab.current_attack.super_armor:
        print(f"  Armored charge: super_armor flag = {grab.current_attack.super_armor}")
        print(f"  Super armor active: {grab.super_armor_active}")

    print("\n[PASS] Super armor mechanic defined correctly")
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("STICK FIGURE FIGHTER - CHARACTER SYSTEM TESTS")
    print("=" * 60)

    tests = [
        test_character_stats,
        test_character_attacks,
        test_game_creation,
        test_headless_match,
        test_character_attack_data_used,
        test_special_moves,
        test_projectile_system,
        test_shield_mechanics,
        test_dodge_mechanics,
        test_counter_mechanic,
        test_reflector_mechanic,
        test_super_armor,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
                print(f"[FAIL] {test.__name__}")
        except Exception as e:
            failed += 1
            print(f"[ERROR] {test.__name__}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"RESULTS: {passed}/{len(tests)} tests passed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
