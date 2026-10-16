"""Stick Figure Fighter stages."""
from engine import CONFIG, ParallaxLayer, Platform, Stage, Vector2


def _default_blast_zones():
    return {
        'left': CONFIG.blast_zone_left,
        'right': CONFIG.blast_zone_right,
        'top': CONFIG.blast_zone_top,
        'bottom': CONFIG.blast_zone_bottom
    }


def battlefield() -> Stage:
    """Battlefield-style stage: main platform plus three floating platforms."""
    main = Platform(
        x=240, y=550, width=800, height=50, passthrough=False, contact_offset=5.0
    )
    left_plat = Platform(x=300, y=420, width=180, passthrough=True, contact_offset=5.0)
    right_plat = Platform(x=780, y=420, width=180, passthrough=True, contact_offset=5.0)
    top_plat = Platform(x=500, y=290, width=280, passthrough=True, contact_offset=5.0)

    # Create parallax background layers for Battlefield
    # Far background (mountains/skyline) - moves very slowly
    far_layer = ParallaxLayer(
        depth=0.9,  # Almost stationary
        color=(30, 35, 50),  # Dark blue-gray
        elements=[
            # Mountain silhouettes
            {'type': 'triangle', 'x': 100, 'y': 600, 'width': 400, 'height': 250},
            {'type': 'triangle', 'x': 400, 'y': 600, 'width': 350, 'height': 200},
            {'type': 'triangle', 'x': 650, 'y': 600, 'width': 450, 'height': 280},
            {'type': 'triangle', 'x': 950, 'y': 600, 'width': 380, 'height': 220},
        ]
    )

    # Mid background (trees/buildings) - moves at medium speed
    mid_layer = ParallaxLayer(
        depth=0.6,
        color=(40, 50, 45),  # Dark forest green
        elements=[
            # Tree-like shapes
            {'type': 'rect', 'x': 50, 'y': 480, 'width': 30, 'height': 120},
            {'type': 'rect', 'x': 150, 'y': 450, 'width': 40, 'height': 150},
            {'type': 'rect', 'x': 280, 'y': 470, 'width': 35, 'height': 130},
            {'type': 'rect', 'x': 900, 'y': 460, 'width': 35, 'height': 140},
            {'type': 'rect', 'x': 1050, 'y': 480, 'width': 40, 'height': 120},
            {'type': 'rect', 'x': 1180, 'y': 440, 'width': 45, 'height': 160},
        ]
    )

    # Near background (foliage/debris) - moves faster
    near_layer = ParallaxLayer(
        depth=0.3,
        color=(50, 60, 55),  # Lighter green
        elements=[
            # Foreground foliage
            {'type': 'circle', 'x': -20, 'y': 580, 'radius': 25},
            {'type': 'circle', 'x': 1300, 'y': 570, 'radius': 30},
            {'type': 'rect', 'x': -50, 'y': 550, 'width': 60, 'height': 100},
            {'type': 'rect', 'x': 1260, 'y': 540, 'width': 70, 'height': 110},
        ]
    )

    return Stage(
        name="Battlefield",
        platforms=[left_plat, right_plat, top_plat],
        main_platform=main,
        blast_zones=_default_blast_zones(),
        spawn_points=[
            Vector2(480, 510),
            Vector2(800, 510)
        ],
        parallax_layers=[far_layer, mid_layer, near_layer]
    )

def final_destination() -> Stage:
    """Final Destination-style stage (flat, no platforms)."""
    main = Platform(
        x=200, y=550, width=880, height=50, passthrough=False, contact_offset=5.0
    )

    # Create parallax background layers for Final Destination (space theme)
    # Far background (stars) - almost stationary
    star_layer = ParallaxLayer(
        depth=0.95,
        color=(255, 255, 255),  # White stars
        elements=[
            # Scattered stars
            {'type': 'circle', 'x': 50, 'y': 100, 'radius': 2},
            {'type': 'circle', 'x': 200, 'y': 80, 'radius': 1},
            {'type': 'circle', 'x': 350, 'y': 150, 'radius': 2},
            {'type': 'circle', 'x': 500, 'y': 50, 'radius': 1},
            {'type': 'circle', 'x': 650, 'y': 120, 'radius': 2},
            {'type': 'circle', 'x': 800, 'y': 60, 'radius': 1},
            {'type': 'circle', 'x': 950, 'y': 140, 'radius': 2},
            {'type': 'circle', 'x': 1100, 'y': 90, 'radius': 1},
            {'type': 'circle', 'x': 1200, 'y': 180, 'radius': 2},
            {'type': 'circle', 'x': 100, 'y': 250, 'radius': 1},
            {'type': 'circle', 'x': 300, 'y': 300, 'radius': 2},
            {'type': 'circle', 'x': 600, 'y': 220, 'radius': 1},
            {'type': 'circle', 'x': 900, 'y': 280, 'radius': 2},
            {'type': 'circle', 'x': 1150, 'y': 350, 'radius': 1},
        ]
    )

    # Mid background (nebula/cosmic clouds) - slow movement
    nebula_layer = ParallaxLayer(
        depth=0.7,
        color=(60, 40, 80),  # Purple nebula
        elements=[
            {'type': 'circle', 'x': 200, 'y': 300, 'radius': 80},
            {'type': 'circle', 'x': 800, 'y': 200, 'radius': 100},
            {'type': 'circle', 'x': 1100, 'y': 350, 'radius': 70},
        ]
    )

    # Near elements (floating debris) - medium movement
    debris_layer = ParallaxLayer(
        depth=0.4,
        color=(70, 70, 80),  # Gray debris
        elements=[
            {'type': 'rect', 'x': -30, 'y': 400, 'width': 40, 'height': 20},
            {'type': 'rect', 'x': 1280, 'y': 380, 'width': 35, 'height': 25},
            {'type': 'circle', 'x': -50, 'y': 300, 'radius': 15},
            {'type': 'circle', 'x': 1320, 'y': 280, 'radius': 12},
        ]
    )

    return Stage(
        name="Final Destination",
        platforms=[],
        main_platform=main,
        blast_zones=_default_blast_zones(),
        spawn_points=[
            Vector2(440, 510),
            Vector2(840, 510)
        ],
        parallax_layers=[star_layer, nebula_layer, debris_layer]
    )


STAGES = {
    "battlefield": battlefield,
    "final_destination": final_destination,
}


def get_stage(name: str) -> Stage:
    """Build a stage by name. Unknown names fall back to Battlefield."""
    return STAGES.get(name.lower(), battlefield)()
