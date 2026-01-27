#!/usr/bin/env python3
"""Quick targeted tests for visual effects."""
import sys
import time
sys.path.insert(0, '/home/vader/mini-mind-v2')
sys.path.insert(0, '/home/vader/mini-mind-v2/workspace/Stick_Figure_Fighter')

from game_engine import (
    create_game, GameLoop, InputState, FighterState, Vector2, Projectile, Hitbox
)
from renderer import X11Renderer
from vision_tool import capture_screen


def test_projectile_direct():
    """Test projectile rendering by directly adding projectiles."""
    print("=== Direct Projectile Test ===")

    game_state = create_game("final_destination", "ZONE", "FLEX")
    renderer = X11Renderer(display_name=":99", title="Projectile Direct Test")

    p1 = game_state.fighters[0]
    p1.position = Vector2(400, 500)

    # Manually add projectiles to verify rendering
    proj1 = Projectile(
        owner_id=0,
        position=Vector2(500, 450),
        velocity=Vector2(5, 0),
        hitbox=Hitbox(Vector2(0,0), 25, 25, 5.0, 3.0, 0.05, 30, (1, 100)),
        lifetime=180,
        has_gravity=False,
        width=25.0,
        height=25.0,
        color=(80, 180, 255),  # Cyan
    )
    proj2 = Projectile(
        owner_id=0,
        position=Vector2(600, 450),
        velocity=Vector2(8, 0),
        hitbox=Hitbox(Vector2(0,0), 40, 15, 8.0, 4.0, 0.08, 20, (1, 100)),
        lifetime=180,
        has_gravity=False,
        width=40.0,
        height=15.0,
        color=(120, 220, 120),  # Green
    )
    game_state.projectiles = [proj1, proj2]

    renderer.render(game_state)
    time.sleep(0.1)
    capture_screen(':99', '/tmp/test_projectile_direct.png')
    print("  Screenshot: /tmp/test_projectile_direct.png")

    # Move projectiles and render again
    for i in range(30):
        proj1.position.x += proj1.velocity.x
        proj2.position.x += proj2.velocity.x
        renderer.render(game_state)
        time.sleep(0.02)

    capture_screen(':99', '/tmp/test_projectile_trails.png')
    print("  Screenshot with trails: /tmp/test_projectile_trails.png")

    renderer.close()
    return True


def test_hit_sparks_direct():
    """Test hit spark effects directly."""
    print("=== Direct Hit Spark Test ===")

    game_state = create_game("final_destination", "GRAB", "RUSH")
    renderer = X11Renderer(display_name=":99", title="Hit Spark Test")

    p1 = game_state.fighters[0]
    p2 = game_state.fighters[1]
    p1.position = Vector2(400, 500)
    p2.position = Vector2(600, 500)

    # Directly add hit effects
    renderer.add_hit_effect(500, 450, 5.0, 45, (255, 80, 80))  # Small hit
    renderer.add_hit_effect(600, 420, 15.0, 30, (180, 120, 60))  # Strong hit

    renderer.render(game_state)
    time.sleep(0.1)
    capture_screen(':99', '/tmp/test_hit_sparks.png')
    print("  Screenshot: /tmp/test_hit_sparks.png")

    # Update effects and capture decay
    for i in range(20):
        renderer.update_effects()
        renderer.render(game_state)
        time.sleep(0.02)

    capture_screen(':99', '/tmp/test_hit_sparks_decay.png')
    print("  Screenshot after decay: /tmp/test_hit_sparks_decay.png")

    renderer.close()
    return True


def test_shield_direct():
    """Test shield visualization directly."""
    print("=== Direct Shield Test ===")

    game_state = create_game("final_destination", "FLEX", "RUSH")
    renderer = X11Renderer(display_name=":99", title="Shield Test")

    p1 = game_state.fighters[0]
    p2 = game_state.fighters[1]
    p1.position = Vector2(400, 500)
    p2.position = Vector2(700, 500)

    # Set P1 to shielding with full health
    p1.state = FighterState.SHIELDING
    p1.shield_health = 100.0

    renderer.render(game_state)
    time.sleep(0.1)
    capture_screen(':99', '/tmp/test_shield_full.png')
    print("  Full shield: /tmp/test_shield_full.png")

    # Set P1 to shielding with reduced health
    p1.shield_health = 40.0
    renderer.render(game_state)
    time.sleep(0.1)
    capture_screen(':99', '/tmp/test_shield_damaged.png')
    print("  Damaged shield: /tmp/test_shield_damaged.png")

    # Shield break
    p1.shield_health = 0.0
    p1.state = FighterState.SHIELD_STUN
    renderer.render(game_state)
    time.sleep(0.1)
    capture_screen(':99', '/tmp/test_shield_break.png')
    print("  Shield break: /tmp/test_shield_break.png")

    renderer.close()
    return True


def test_dodge_afterimage_direct():
    """Test dodge afterimage effect directly."""
    print("=== Direct Dodge Afterimage Test ===")

    game_state = create_game("final_destination", "RUSH", "FLEX")
    renderer = X11Renderer(display_name=":99", title="Dodge Afterimage Test")

    p1 = game_state.fighters[0]
    p2 = game_state.fighters[1]
    p1.position = Vector2(400, 500)
    p2.position = Vector2(700, 500)

    # Set P1 to dodging with invincibility
    p1.state = FighterState.DODGING
    p1.invincibility_frames = 15

    # Manually store positions for afterimages
    p1.previous_positions = [
        Vector2(350, 500),
        Vector2(360, 500),
        Vector2(370, 500),
        Vector2(380, 500),
        Vector2(390, 500),
    ]

    renderer.draw_dodge_afterimages(p1)
    renderer.render(game_state)
    time.sleep(0.1)
    capture_screen(':99', '/tmp/test_dodge_afterimage.png')
    print("  Dodge afterimage: /tmp/test_dodge_afterimage.png")

    renderer.close()
    return True


def test_super_armor_glow():
    """Test super armor glow effect directly."""
    print("=== Direct Super Armor Glow Test ===")

    game_state = create_game("final_destination", "GRAB", "RUSH")
    renderer = X11Renderer(display_name=":99", title="Super Armor Test")

    p1 = game_state.fighters[0]  # GRAB
    p2 = game_state.fighters[1]
    p1.position = Vector2(400, 500)
    p2.position = Vector2(700, 500)

    # Set P1 to have super armor active
    p1.super_armor_active = True
    p1.state = FighterState.ATTACKING

    renderer.render(game_state)
    time.sleep(0.1)
    capture_screen(':99', '/tmp/test_super_armor.png')
    print("  Super armor glow: /tmp/test_super_armor.png")

    renderer.close()
    return True


def test_counter_reflector():
    """Test counter and reflector visualization."""
    print("=== Direct Counter/Reflector Test ===")

    game_state = create_game("final_destination", "RUSH", "FLEX")
    renderer = X11Renderer(display_name=":99", title="Counter/Reflector Test")

    p1 = game_state.fighters[0]  # RUSH (counter)
    p2 = game_state.fighters[1]  # FLEX (reflector)
    p1.position = Vector2(350, 500)
    p2.position = Vector2(650, 500)

    # Set P1 to counter state
    p1.state = FighterState.COUNTER
    p1.counter_active = True

    # Set P2 to reflector state
    p2.state = FighterState.REFLECTING
    p2.reflector_active = True

    renderer.draw_counter_aura(p1)
    renderer.draw_reflector_active(p2)
    renderer.render(game_state)
    time.sleep(0.1)
    capture_screen(':99', '/tmp/test_counter_reflector.png')
    print("  Counter/Reflector: /tmp/test_counter_reflector.png")

    renderer.close()
    return True


def test_knockback_trail():
    """Test knockback trail effect."""
    print("=== Direct Knockback Trail Test ===")

    game_state = create_game("final_destination", "GRAB", "RUSH")
    renderer = X11Renderer(display_name=":99", title="Knockback Trail Test")

    p1 = game_state.fighters[0]
    p2 = game_state.fighters[1]
    p1.position = Vector2(350, 500)
    p2.position = Vector2(600, 450)

    # Set P2 in hitstun with high velocity
    p2.state = FighterState.HITSTUN
    p2.velocity = Vector2(15, -8)
    p2.hitstun_frames = 30

    # Add knockback trail positions
    p2.knockback_trail = [
        Vector2(500, 480),
        Vector2(520, 475),
        Vector2(540, 465),
        Vector2(560, 460),
        Vector2(580, 455),
    ]

    renderer.draw_knockback_trails(p2)
    renderer.render(game_state)
    time.sleep(0.1)
    capture_screen(':99', '/tmp/test_knockback_trail.png')
    print("  Knockback trail: /tmp/test_knockback_trail.png")

    renderer.close()
    return True


def test_damage_flash():
    """Test damage flash effect."""
    print("=== Direct Damage Flash Test ===")

    game_state = create_game("final_destination", "GRAB", "RUSH")
    renderer = X11Renderer(display_name=":99", title="Damage Flash Test")

    p1 = game_state.fighters[0]
    p2 = game_state.fighters[1]
    p1.position = Vector2(350, 500)
    p2.position = Vector2(600, 500)

    # Add damage flash to P2
    renderer.add_damage_flash(p2)

    renderer.render(game_state)
    time.sleep(0.1)
    capture_screen(':99', '/tmp/test_damage_flash.png')
    print("  Damage flash: /tmp/test_damage_flash.png")

    renderer.close()
    return True


def test_landing_dust():
    """Test landing dust effect."""
    print("=== Direct Landing Dust Test ===")

    game_state = create_game("final_destination", "FLEX", "RUSH")
    renderer = X11Renderer(display_name=":99", title="Landing Dust Test")

    p1 = game_state.fighters[0]
    p2 = game_state.fighters[1]
    p1.position = Vector2(400, 500)
    p2.position = Vector2(700, 500)

    # Add landing dust effects
    renderer.add_landing_dust(400, 500)
    renderer.add_landing_dust(700, 500)

    renderer.render(game_state)
    time.sleep(0.1)
    capture_screen(':99', '/tmp/test_landing_dust.png')
    print("  Landing dust: /tmp/test_landing_dust.png")

    renderer.close()
    return True


def run_all():
    """Run all quick tests."""
    print("=" * 60)
    print("QUICK VISUAL EFFECTS TESTS")
    print("=" * 60)

    results = {}

    tests = [
        ("projectile_direct", test_projectile_direct),
        ("hit_sparks", test_hit_sparks_direct),
        ("shield", test_shield_direct),
        ("dodge_afterimage", test_dodge_afterimage_direct),
        ("super_armor", test_super_armor_glow),
        ("counter_reflector", test_counter_reflector),
        ("knockback_trail", test_knockback_trail),
        ("damage_flash", test_damage_flash),
        ("landing_dust", test_landing_dust),
    ]

    for name, test_func in tests:
        try:
            test_func()
            results[name] = "PASS"
            print(f"  ✓ {name} PASSED\n")
        except Exception as e:
            results[name] = f"FAIL: {e}"
            print(f"  ✗ {name} FAILED: {e}\n")
        time.sleep(0.3)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    passed = sum(1 for v in results.values() if v == "PASS")
    failed = len(results) - passed
    for name, result in results.items():
        status = "✓" if result == "PASS" else "✗"
        print(f"  {status} {name}: {result}")
    print(f"\nTotal: {passed} passed, {failed} failed")

    return results


if __name__ == "__main__":
    run_all()
