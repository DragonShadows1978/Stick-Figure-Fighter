#!/usr/bin/env python3
"""
Comprehensive visual effects test - forces combat to trigger all effects.

Tests:
1. Projectile rendering with trails
2. Hit sparks on attack connect
3. Screen shake on strong hits
4. Damage flash when taking damage
5. Shield visualization (bubble, shrink, cracks)
6. Dodge afterimages
7. Counter/reflector effects
8. Super armor glow
9. Knockback trails
10. Landing dust
11. HUD improvements
"""
import sys
import time
sys.path.insert(0, '/home/vader/mini-mind-v2')
sys.path.insert(0, '/home/vader/mini-mind-v2/workspace/Stick_Figure_Fighter')

from game_engine import (
    create_game, GameLoop, InputState, FighterState, Vector2
)
from renderer import X11Renderer


def capture_frame(renderer, game_state, frame_num, desc):
    """Helper to capture a screenshot with description."""
    from vision_tool import capture_screen
    path = f"/tmp/sfighter_test_{frame_num:03d}_{desc}.png"
    renderer.render(game_state)
    time.sleep(0.05)  # Small delay for render to complete
    capture_screen(':99', path)
    print(f"  [{frame_num}] {desc}: {path}")
    return path


def test_projectile_rendering():
    """Test projectile rendering with trails."""
    print("\n=== Test 1: Projectile Rendering ===")

    # Use ZONE (has projectiles) vs FLEX
    game_state = create_game("final_destination", "ZONE", "FLEX")
    renderer = X11Renderer(display_name=":99", title="Projectile Test")
    loop = GameLoop(game_state, headless=False)

    p1 = game_state.fighters[0]  # ZONE
    p2 = game_state.fighters[1]  # FLEX

    # Position fighters far apart
    p1.position = Vector2(400, 500)
    p2.position = Vector2(900, 500)
    p1.facing_right = True

    screenshots = []

    # Fire projectile
    for frame in range(120):
        inp1 = InputState()
        inp2 = InputState()

        if frame == 10:
            inp1.special = True  # Fire neutral-B projectile
            print("  ZONE fires neutral-B")

        if frame == 50:
            inp1.right = True
            inp1.special = True  # Fire side-B beam
            print("  ZONE fires side-B beam")

        p1.input = inp1
        p1.input.update_prev()
        p2.input = inp2
        p2.input.update_prev()

        loop.step()
        renderer.update_effects()
        renderer.render(game_state)

        # Capture at key moments
        if frame in [20, 30, 60, 80]:
            screenshots.append(capture_frame(renderer, game_state, frame, "projectile"))

        time.sleep(1/120)  # Faster for test

    renderer.close()
    print(f"  Projectiles active: {len(game_state.projectiles)}")
    return screenshots


def test_hit_effects():
    """Test hit sparks and damage flash."""
    print("\n=== Test 2: Hit Effects (Sparks, Shake, Flash) ===")

    game_state = create_game("final_destination", "GRAB", "RUSH")
    renderer = X11Renderer(display_name=":99", title="Hit Effects Test")
    loop = GameLoop(game_state, headless=False)

    # Connect hit callback
    def on_hit(attacker, defender, hitbox, cx, cy, kb_mag):
        print(f"  HIT! KB={kb_mag:.1f}")
        renderer.add_hit_effect(cx, cy, kb_mag, hitbox.angle, attacker.color)
        renderer.add_damage_flash(defender)

    loop.on_hit = on_hit

    p1 = game_state.fighters[0]  # GRAB (heavy hitter)
    p2 = game_state.fighters[1]  # RUSH

    # Position fighters close together
    p1.position = Vector2(500, 500)
    p2.position = Vector2(560, 500)  # Close enough to hit
    p1.facing_right = True
    p2.facing_right = False

    screenshots = []

    for frame in range(100):
        inp1 = InputState()
        inp2 = InputState()

        # P1 attacks at frame 10
        if frame == 10:
            inp1.attack = True
            print("  GRAB jabs")

        # P1 strong attack at frame 50
        if frame == 50:
            inp1.right = True
            inp1.attack = True
            print("  GRAB forward tilt (should cause screen shake)")

        p1.input = inp1
        p1.input.update_prev()
        p2.input = inp2
        p2.input.update_prev()

        loop.step()
        renderer.update_effects()
        renderer.render(game_state)

        # Capture during/after hit
        if frame in [18, 25, 58, 65]:
            screenshots.append(capture_frame(renderer, game_state, frame, "hit_effect"))

        time.sleep(1/120)

    renderer.close()
    print(f"  P2 damage: {p2.damage_percent:.1f}%")
    return screenshots


def test_shield_visualization():
    """Test shield bubble with shrink and crack effects."""
    print("\n=== Test 3: Shield Visualization ===")

    game_state = create_game("final_destination", "GRAB", "FLEX")
    renderer = X11Renderer(display_name=":99", title="Shield Test")
    loop = GameLoop(game_state, headless=False)

    p1 = game_state.fighters[0]  # GRAB
    p2 = game_state.fighters[1]  # FLEX

    p1.position = Vector2(500, 500)
    p2.position = Vector2(600, 500)
    p1.facing_right = True

    screenshots = []

    for frame in range(200):
        inp1 = InputState()
        inp2 = InputState()

        # P2 shields
        if 20 < frame < 180:
            inp2.shield = True
            if frame == 21:
                print("  FLEX starts shielding")

        # P1 attacks shield repeatedly
        if frame in [40, 60, 80, 100, 120, 140]:
            inp1.attack = True
            inp1.right = True
            print(f"  GRAB attacks shield (frame {frame})")

        p1.input = inp1
        p1.input.update_prev()
        p2.input = inp2
        p2.input.update_prev()

        loop.step()
        renderer.update_effects()
        renderer.render(game_state)

        if frame in [30, 70, 110, 150]:
            screenshots.append(capture_frame(renderer, game_state, frame, "shield"))
            print(f"    Shield health: {p2.shield_health:.1f}")

        time.sleep(1/120)

    renderer.close()
    print(f"  Final shield health: {p2.shield_health:.1f}")
    return screenshots


def test_dodge_afterimages():
    """Test dodge afterimage effects."""
    print("\n=== Test 4: Dodge Afterimages ===")

    game_state = create_game("final_destination", "RUSH", "FLEX")
    renderer = X11Renderer(display_name=":99", title="Dodge Test")
    loop = GameLoop(game_state, headless=False)

    p1 = game_state.fighters[0]  # RUSH
    p2 = game_state.fighters[1]  # FLEX

    p1.position = Vector2(500, 500)
    p2.position = Vector2(700, 500)

    screenshots = []

    for frame in range(120):
        inp1 = InputState()
        inp2 = InputState()

        # Spot dodge
        if frame == 20:
            inp1.shield = True
            inp1.down = True
            print("  RUSH spot dodges")

        # Roll forward
        if frame == 50:
            inp1.shield = True
            inp1.right = True
            print("  RUSH rolls forward")

        # Air dodge (need to jump first)
        if frame == 80:
            inp1.jump = True
        if frame == 90:
            inp1.shield = True
            print("  RUSH air dodges")

        p1.input = inp1
        p1.input.update_prev()
        p2.input = inp2
        p2.input.update_prev()

        loop.step()

        # Update afterimages
        renderer.draw_dodge_afterimages(p1)
        renderer.update_effects()
        renderer.render(game_state)

        if frame in [25, 30, 55, 60, 95, 100]:
            screenshots.append(capture_frame(renderer, game_state, frame, "dodge"))

        time.sleep(1/120)

    renderer.close()
    return screenshots


def test_counter_reflector():
    """Test counter and reflector visual effects."""
    print("\n=== Test 5: Counter & Reflector Effects ===")

    game_state = create_game("final_destination", "RUSH", "FLEX")
    renderer = X11Renderer(display_name=":99", title="Counter/Reflector Test")
    loop = GameLoop(game_state, headless=False)

    # Connect effect callbacks
    def on_reflect(proj, reflector, x, y):
        print(f"  REFLECT at ({x:.0f}, {y:.0f})")
        renderer.add_reflect_effect(x, y)

    loop.on_reflect = on_reflect

    p1 = game_state.fighters[0]  # RUSH (has counter)
    p2 = game_state.fighters[1]  # FLEX (has reflector)

    p1.position = Vector2(500, 500)
    p2.position = Vector2(700, 500)
    p1.facing_right = True
    p2.facing_right = False

    screenshots = []

    for frame in range(150):
        inp1 = InputState()
        inp2 = InputState()

        # RUSH uses counter
        if frame == 20:
            inp1.down = True
            inp1.special = True
            print("  RUSH uses counter (down-B)")

        # FLEX uses reflector
        if frame == 80:
            inp2.down = True
            inp2.special = True
            print("  FLEX uses reflector (down-B)")

        p1.input = inp1
        p1.input.update_prev()
        p2.input = inp2
        p2.input.update_prev()

        loop.step()

        # Draw counter/reflector auras
        renderer.draw_counter_aura(p1)
        renderer.draw_reflector_active(p2)
        renderer.update_effects()
        renderer.render(game_state)

        if frame in [25, 35, 85, 95]:
            screenshots.append(capture_frame(renderer, game_state, frame, "counter_reflect"))

        time.sleep(1/120)

    renderer.close()
    return screenshots


def test_super_armor():
    """Test super armor glow effect."""
    print("\n=== Test 6: Super Armor Glow ===")

    game_state = create_game("final_destination", "GRAB", "RUSH")
    renderer = X11Renderer(display_name=":99", title="Super Armor Test")
    loop = GameLoop(game_state, headless=False)

    p1 = game_state.fighters[0]  # GRAB (has super armor on neutral-B)
    p2 = game_state.fighters[1]  # RUSH

    p1.position = Vector2(500, 500)
    p2.position = Vector2(700, 500)
    p1.facing_right = True

    screenshots = []

    for frame in range(100):
        inp1 = InputState()
        inp2 = InputState()

        # GRAB uses armored charge (neutral-B)
        if frame == 20:
            inp1.special = True
            print("  GRAB uses armored charge (neutral-B)")

        p1.input = inp1
        p1.input.update_prev()
        p2.input = inp2
        p2.input.update_prev()

        loop.step()
        renderer.update_effects()
        renderer.render(game_state)

        if frame in [30, 40, 50]:
            screenshots.append(capture_frame(renderer, game_state, frame, "super_armor"))
            print(f"    Super armor active: {p1.super_armor_active}")

        time.sleep(1/120)

    renderer.close()
    return screenshots


def test_knockback_trails():
    """Test knockback trail effects during hitstun."""
    print("\n=== Test 7: Knockback Trails ===")

    game_state = create_game("final_destination", "GRAB", "RUSH")
    renderer = X11Renderer(display_name=":99", title="Knockback Test")
    loop = GameLoop(game_state, headless=False)

    def on_hit(attacker, defender, hitbox, cx, cy, kb_mag):
        print(f"  STRONG HIT! KB={kb_mag:.1f}")
        renderer.add_hit_effect(cx, cy, kb_mag, hitbox.angle, attacker.color)
        renderer.add_damage_flash(defender)

    loop.on_hit = on_hit

    p1 = game_state.fighters[0]  # GRAB
    p2 = game_state.fighters[1]  # RUSH

    p1.position = Vector2(400, 500)
    p2.position = Vector2(460, 500)
    p1.facing_right = True

    # Give P2 some damage so knockback is stronger
    p2.damage_percent = 80.0

    screenshots = []

    for frame in range(100):
        inp1 = InputState()
        inp2 = InputState()

        # GRAB smash attack
        if frame == 20:
            inp1.right = True
            inp1.attack = True
            print("  GRAB forward tilt (on 80% damage)")

        p1.input = inp1
        p1.input.update_prev()
        p2.input = inp2
        p2.input.update_prev()

        loop.step()

        # Draw knockback trails
        renderer.draw_knockback_trails(p2)
        renderer.update_effects()
        renderer.render(game_state)

        if frame in [28, 32, 36, 40, 50]:
            screenshots.append(capture_frame(renderer, game_state, frame, "knockback"))

        time.sleep(1/120)

    renderer.close()
    print(f"  P2 final damage: {p2.damage_percent:.1f}%")
    return screenshots


def test_landing_dust():
    """Test landing dust particle effects."""
    print("\n=== Test 8: Landing Dust ===")

    game_state = create_game("final_destination", "FLEX", "RUSH")
    renderer = X11Renderer(display_name=":99", title="Landing Dust Test")
    loop = GameLoop(game_state, headless=False)

    def on_land(fighter, x, y):
        print(f"  LAND! at ({x:.0f}, {y:.0f})")
        renderer.add_landing_dust(x, y)

    loop.on_land = on_land

    p1 = game_state.fighters[0]  # FLEX
    p2 = game_state.fighters[1]  # RUSH

    p1.position = Vector2(500, 500)
    p2.position = Vector2(700, 500)

    screenshots = []

    for frame in range(100):
        inp1 = InputState()
        inp2 = InputState()

        # Jump
        if frame == 10:
            inp1.jump = True
            print("  FLEX jumps")

        # Another jump
        if frame == 50:
            inp1.jump = True
            print("  FLEX jumps again")

        p1.input = inp1
        p1.input.update_prev()
        p2.input = inp2
        p2.input.update_prev()

        loop.step()
        renderer.update_effects()
        renderer.render(game_state)

        if frame in [35, 38, 80, 83]:
            screenshots.append(capture_frame(renderer, game_state, frame, "landing"))

        time.sleep(1/120)

    renderer.close()
    return screenshots


def run_all_tests():
    """Run all visual effects tests."""
    print("=" * 60)
    print("COMPREHENSIVE VISUAL EFFECTS TEST")
    print("=" * 60)

    all_screenshots = []
    test_results = {}

    # Test 1: Projectiles
    try:
        screenshots = test_projectile_rendering()
        all_screenshots.extend(screenshots)
        test_results['projectiles'] = 'PASS'
    except Exception as e:
        test_results['projectiles'] = f'FAIL: {e}'
        print(f"  ERROR: {e}")

    time.sleep(0.5)

    # Test 2: Hit effects
    try:
        screenshots = test_hit_effects()
        all_screenshots.extend(screenshots)
        test_results['hit_effects'] = 'PASS'
    except Exception as e:
        test_results['hit_effects'] = f'FAIL: {e}'
        print(f"  ERROR: {e}")

    time.sleep(0.5)

    # Test 3: Shield
    try:
        screenshots = test_shield_visualization()
        all_screenshots.extend(screenshots)
        test_results['shield'] = 'PASS'
    except Exception as e:
        test_results['shield'] = f'FAIL: {e}'
        print(f"  ERROR: {e}")

    time.sleep(0.5)

    # Test 4: Dodge
    try:
        screenshots = test_dodge_afterimages()
        all_screenshots.extend(screenshots)
        test_results['dodge'] = 'PASS'
    except Exception as e:
        test_results['dodge'] = f'FAIL: {e}'
        print(f"  ERROR: {e}")

    time.sleep(0.5)

    # Test 5: Counter/Reflector
    try:
        screenshots = test_counter_reflector()
        all_screenshots.extend(screenshots)
        test_results['counter_reflector'] = 'PASS'
    except Exception as e:
        test_results['counter_reflector'] = f'FAIL: {e}'
        print(f"  ERROR: {e}")

    time.sleep(0.5)

    # Test 6: Super Armor
    try:
        screenshots = test_super_armor()
        all_screenshots.extend(screenshots)
        test_results['super_armor'] = 'PASS'
    except Exception as e:
        test_results['super_armor'] = f'FAIL: {e}'
        print(f"  ERROR: {e}")

    time.sleep(0.5)

    # Test 7: Knockback Trails
    try:
        screenshots = test_knockback_trails()
        all_screenshots.extend(screenshots)
        test_results['knockback_trails'] = 'PASS'
    except Exception as e:
        test_results['knockback_trails'] = f'FAIL: {e}'
        print(f"  ERROR: {e}")

    time.sleep(0.5)

    # Test 8: Landing Dust
    try:
        screenshots = test_landing_dust()
        all_screenshots.extend(screenshots)
        test_results['landing_dust'] = 'PASS'
    except Exception as e:
        test_results['landing_dust'] = f'FAIL: {e}'
        print(f"  ERROR: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)

    passed = 0
    failed = 0
    for test_name, result in test_results.items():
        status = "✓" if result == 'PASS' else "✗"
        print(f"  {status} {test_name}: {result}")
        if result == 'PASS':
            passed += 1
        else:
            failed += 1

    print(f"\nTotal: {passed} passed, {failed} failed")
    print(f"Screenshots captured: {len(all_screenshots)}")

    return test_results, all_screenshots


if __name__ == "__main__":
    results, screenshots = run_all_tests()
