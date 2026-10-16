#!/usr/bin/env python3
"""
Test visual effects by running a scripted combat sequence.

This script simulates combat between two fighters to trigger:
- Hit sparks
- Screen shake
- Damage flash
- Shield visualization
- Dodge afterimages
- Projectile trails
"""
import sys
import time
sys.path.insert(0, '/home/vader/mini-mind-v2')

from game_engine import (
    create_game, GameLoop, InputState, FighterState
)
from renderer import X11Renderer


def test_visual_effects():
    """Run visual effects test."""
    print("=== Visual Effects Test ===\n")

    # Create game with ZONE (has projectiles) vs RUSH (has counter)
    game_state = create_game("battlefield", "ZONE", "RUSH")

    # Create renderer
    renderer = X11Renderer(display_name=":99", title="Visual Effects Test")

    # Create game loop
    loop = GameLoop(game_state, headless=False)

    # Connect visual effect callbacks
    def on_hit(attacker, defender, hitbox, cx, cy, kb_mag):
        print(f"  HIT! {attacker.stats.name} hit {defender.stats.name}, KB={kb_mag:.1f}")
        renderer.add_hit_effect(cx, cy, kb_mag, hitbox.angle, attacker.color)
        renderer.add_damage_flash(defender)

    def on_projectile_hit(proj, defender, cx, cy, kb_mag):
        print(f"  PROJECTILE HIT! KB={kb_mag:.1f}")
        renderer.add_hit_effect(cx, cy, kb_mag, proj.hitbox.angle, proj.color)
        renderer.add_damage_flash(defender)

    def on_reflect(proj, reflector, x, y):
        print(f"  REFLECT!")
        renderer.add_reflect_effect(x, y)

    loop.on_hit = on_hit
    loop.on_projectile_hit = on_projectile_hit
    loop.on_reflect = on_reflect
    loop.on_render = lambda state: renderer.render(state)

    # Get fighters
    p1 = game_state.fighters[0]  # ZONE
    p2 = game_state.fighters[1]  # RUSH

    # Scripted sequence
    print("Running scripted combat sequence...\n")

    frames = 0
    screenshot_frames = [30, 80, 140, 200, 280, 400]  # When to screenshot
    screenshot_idx = 0

    while frames < 500 and not game_state.match_over:
        # Create inputs based on frame
        inp1 = InputState()
        inp2 = InputState()

        # Move fighters toward each other
        if frames < 60:
            inp1.right = True
            inp2.left = True

        # ZONE fires projectile (neutral special)
        elif frames == 70:
            inp1.special = True
            print("ZONE fires neutral-B projectile")

        # Fighters approach more
        elif 80 < frames < 120:
            inp1.right = True
            inp2.left = True

        # P1 attacks P2
        elif frames == 130:
            inp1.attack = True
            print("ZONE attacks (jab)")

        # P2 shields
        elif 150 < frames < 200:
            inp2.shield = True
            print("RUSH shields" if frames == 151 else "")

        # P1 attacks shield
        elif frames == 180:
            inp1.attack = True
            inp1.right = True  # Forward tilt
            print("ZONE attacks shield")

        # P2 dodges
        elif frames == 210:
            inp2.down = True
            inp2.shield = True
            print("RUSH spot dodges")

        # More combat
        elif frames == 250:
            inp1.attack = True
            print("ZONE attacks again")

        # P2 counters (RUSH down-B)
        elif 270 < frames < 310:
            inp2.down = True
            inp2.special = True
            print("RUSH uses counter" if frames == 271 else "")

        # P1 attacks into counter
        elif frames == 285:
            inp1.attack = True
            print("ZONE attacks (into counter?)")

        # More projectile
        elif frames == 350:
            inp1.special = True
            print("ZONE fires another projectile")

        # P2 approaches
        elif 360 < frames < 400:
            inp2.left = True

        # P2 attacks
        elif frames == 410:
            inp2.attack = True
            print("RUSH attacks")

        # Strong attack (smash)
        elif frames == 450:
            inp1.right = True
            inp1.attack = True
            inp1.up = True  # up tilt
            print("ZONE up tilt")

        # Apply inputs
        p1.input = inp1
        p1.input.update_prev()
        p2.input = inp2
        p2.input.update_prev()

        # Step simulation
        loop.step()

        # Update effects and render
        renderer.update_effects()
        renderer.render(game_state)

        # Capture screenshots at key moments
        if screenshot_idx < len(screenshot_frames) and frames == screenshot_frames[screenshot_idx]:
            from vision_tool import capture_screen
            path = f"/tmp/sfighter_effects_{screenshot_idx}.png"
            capture_screen(':99', path)
            print(f"  Screenshot saved: {path}")
            screenshot_idx += 1

        frames += 1
        time.sleep(1/60)  # 60 FPS

    print(f"\nTest completed after {frames} frames")
    print(f"P1 damage: {p1.damage_percent:.1f}%")
    print(f"P2 damage: {p2.damage_percent:.1f}%")

    # Final screenshot
    from vision_tool import capture_screen
    capture_screen(':99', '/tmp/sfighter_effects_final.png')
    print("Final screenshot saved: /tmp/sfighter_effects_final.png")

    renderer.close()


if __name__ == "__main__":
    test_visual_effects()
