"""
Theme base class.

A Theme owns every pixel of a match frame and the menu UI kit for one art
direction. The base class handles orchestration and effect bookkeeping
(afterimages, trails, camera, pose animation); subclasses paint.
"""
import math
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

from engine import DodgeType, FighterState
from gfx import xrender as X
from gfx.context import Gfx, Image, Shape
from gfx.text import Face, FontStyle
from poses import PoseAnimator, Skeleton

RGB = Tuple[int, int, int]


@dataclass
class FighterLook:
    """Per-frame appearance flags for one fighter."""
    color: RGB
    flash: float = 0.0          # 0..1 white hit flash
    armor: bool = False
    counter: bool = False
    reflector: bool = False
    invincible: bool = False
    blink: bool = False         # respawn invincibility blink phase
    hitstun: bool = False
    shield_stun: bool = False
    alpha: float = 1.0


class FxState:
    """Frame-to-frame visual bookkeeping shared by all themes."""

    def __init__(self):
        self.afterimages: Dict[int, Deque[Tuple[Skeleton, Optional[DodgeType], int]]] = {}
        self.kb_trails: Dict[int, Deque[Tuple[float, float]]] = {}
        self.proj_trails: Dict[int, Deque[Tuple[float, float]]] = {}
        self.camera = (0.0, 0.0)
        self.frame = 0
        self.prev_stocks: Dict[int, int] = {}
        self.ko_flashes: List[dict] = []

    def update_camera(self, game_state, w, h):
        fs = game_state.fighters
        ax = sum(f.position.x for f in fs) / len(fs)
        ay = sum(f.position.y for f in fs) / len(fs)
        tx, ty = (ax - w / 2) * 0.1, (ay - h / 2) * 0.05
        cx, cy = self.camera
        self.camera = (cx + (tx - cx) * 0.12, cy + (ty - cy) * 0.12)

    def update_trails(self, game_state, skeletons: Dict[int, Skeleton]):
        for f in game_state.fighters:
            pid = f.player_id
            imgs = self.afterimages.setdefault(pid, deque(maxlen=4))
            if f.state == FighterState.DODGING:
                if f.state_frame % 3 == 0 and pid in skeletons:
                    imgs.append((skeletons[pid], f.dodge_type, self.frame))
            elif imgs and self.frame - imgs[-1][2] > 10:
                imgs.clear()
            trail = self.kb_trails.setdefault(pid, deque(maxlen=12))
            if f.state == FighterState.HITSTUN and math.hypot(f.velocity.x, f.velocity.y) > 4:
                sk = skeletons.get(pid)
                if sk:
                    trail.append(sk.center)
            elif trail:
                trail.popleft()
            # KO detection (stock lost)
            prev = self.prev_stocks.get(pid)
            if prev is not None and f.stocks < prev:
                self.ko_flashes.append({"pid": pid, "frame": self.frame, "color": None})
            self.prev_stocks[pid] = f.stocks
        live = set()
        for p in game_state.projectiles:
            k = id(p)
            live.add(k)
            self.proj_trails.setdefault(k, deque(maxlen=10)).append((p.position.x, p.position.y))
        for k in list(self.proj_trails):
            if k not in live:
                del self.proj_trails[k]
        self.ko_flashes = [k for k in self.ko_flashes if self.frame - k["frame"] < 40]


DODGE_COLORS = {
    DodgeType.SPOT_DODGE: (150, 170, 255),
    DodgeType.ROLL_FORWARD: (150, 255, 170),
    DodgeType.ROLL_BACKWARD: (150, 255, 170),
    DodgeType.AIR_DODGE: (255, 200, 150),
}


class Theme:
    name = "base"
    title = "Base"
    classic = False

    # palette roles every theme defines
    palette: Dict[str, RGB] = {
        "bg": (16, 18, 28), "panel": (30, 34, 52), "panel_edge": (90, 100, 140),
        "text": (235, 238, 245), "text_dim": (140, 145, 160), "accent": (255, 200, 60),
        "accent2": (90, 200, 255), "good": (110, 230, 120), "warn": (255, 190, 70),
        "bad": (255, 90, 90), "p1": (255, 90, 90), "p2": (90, 140, 255),
    }

    def __init__(self, renderer):
        self.r = renderer
        self.g: Gfx = renderer.gfx
        self.w, self.h = renderer.width, renderer.height
        self.anim = PoseAnimator()
        self.fx = FxState()
        self.built = False
        self._stage_assets: Dict[str, dict] = {}
        self.fonts: Dict[str, Face] = {}

    # ------------------------------------------------------------------ setup

    def ensure_built(self):
        if not self.built:
            self.build()
            self.built = True

    def build(self):
        """Create fonts and theme-wide assets. Subclasses extend."""
        self.fonts["ui"] = Face(self.g, FontStyle())

    def stage_assets(self, stage) -> dict:
        a = self._stage_assets.get(stage.name)
        if a is None:
            a = self._stage_assets[stage.name] = self.build_stage(stage)
        return a

    def build_stage(self, stage) -> dict:
        return {}

    def reset_match(self):
        self.anim.reset()
        self.fx = FxState()

    # ------------------------------------------------------------------ text

    def face(self, role: str = "ui") -> Face:
        return self.fonts.get(role) or self.fonts["ui"]

    def text(self, x, y, s, size=16, color=None, role="ui", align="left", valign="baseline",
             alpha=1.0, glow=None, glow_alpha=0.0, tracking=None) -> float:
        return self.face(role).draw(str(s), x, y, size, color or self.palette["text"], alpha, align,
                                    valign, glow_color=glow, glow_alpha=glow_alpha, tracking=tracking)

    def measure(self, s, size=16, role="ui") -> float:
        return self.face(role).measure(str(s), size)

    # ------------------------------------------------------------------ colors

    def character_color(self, fighter) -> RGB:
        return tuple(int(c) for c in fighter.color)

    def look_for(self, fighter) -> FighterLook:
        look = FighterLook(color=self.character_color(fighter))
        flash = getattr(fighter, "_damage_flash_frames", 0)
        if flash > 0:
            fighter._damage_flash_frames = flash - 1
            look.flash = min(1.0, flash / 4)
        look.armor = fighter.super_armor_active
        look.counter = fighter.counter_active
        look.reflector = fighter.reflector_active
        look.hitstun = fighter.state == FighterState.HITSTUN
        look.shield_stun = fighter.state == FighterState.SHIELD_STUN
        look.invincible = fighter.invincible_frames > 0
        if fighter.invincible_frames > 0 and fighter.state != FighterState.DODGING:
            look.blink = (fighter.invincible_frames // 4) % 2 == 0
        return look

    # ------------------------------------------------------------------ match frame

    def render_match(self, game_state):
        self.ensure_built()
        fx = self.fx
        fx.frame += 1
        fx.update_camera(game_state, self.w, self.h)
        sx, sy = self.r.shake_offset

        skeletons: Dict[int, Skeleton] = {}
        freeze = game_state.hit_freeze_frames
        for f in game_state.fighters:
            ox, oy = sx, sy
            if freeze > 0:
                v = 3 if freeze % 2 == 0 else -3
                if f.grounded:
                    ox += v
                else:
                    oy += v
            skeletons[f.player_id] = self.anim.skeleton(f, f.position.x + ox, f.position.y + oy, fx.frame)
        fx.update_trails(game_state, skeletons)

        stage = game_state.stage
        self.draw_background(game_state, stage, fx.camera, (sx, sy))
        for plat in [stage.main_platform] + stage.platforms:
            self.draw_platform(plat, stage, (sx, sy))
        self.draw_midground(game_state, stage, (sx, sy))

        for f in game_state.fighters:
            for i, (sk, dtype, _) in enumerate(fx.afterimages.get(f.player_id, ())):
                n = len(fx.afterimages[f.player_id])
                self.draw_afterimage(f, sk, dtype, 0.15 + 0.35 * (i + 1) / n)
        for f in game_state.fighters:
            trail = fx.kb_trails.get(f.player_id)
            if trail and len(trail) > 1:
                self.draw_knockback_trail(f, list(trail))

        for f in game_state.fighters:
            sk = skeletons[f.player_id]
            look = self.look_for(f)
            self.draw_fighter_aura(f, sk, look)
            self.draw_fighter(f, sk, look)
            if f.state == FighterState.SHIELDING:
                self.draw_shield(f, sk)
            if look.counter:
                self.draw_counter(f, sk)
            if look.reflector:
                self.draw_reflector(f, sk)
            if self.r.show_hitboxes or self.r.show_hurtboxes:
                self.draw_debug_boxes(f, (sx, sy))

        for p in game_state.projectiles:
            self.draw_projectile(p, list(fx.proj_trails.get(id(p), ())), (sx, sy))

        for e in self.r.effects:
            self.draw_effect(e, (sx, sy))

        self.draw_foreground(game_state, stage, fx.camera, (sx, sy))
        self.draw_hud(game_state)
        if game_state.match_over:
            self.draw_victory(game_state)

    # ------------------------------------------------------------------ hooks (defaults)

    def draw_background(self, game_state, stage, camera, shake):
        self.g.fill_rect(0, 0, self.w, self.h, self.palette["bg"])

    def draw_platform(self, plat, stage, shake):
        self.g.fill_rect(plat.x + shake[0], plat.y + shake[1], plat.width, plat.height,
                         self.palette["panel_edge"])

    def draw_midground(self, game_state, stage, shake):
        pass

    def draw_foreground(self, game_state, stage, camera, shake):
        pass

    def draw_fighter_aura(self, fighter, sk: Skeleton, look: FighterLook):
        pass

    def draw_fighter(self, fighter, sk: Skeleton, look: FighterLook):
        if look.blink:
            return
        col = look.color if look.flash <= 0 else (255, 255, 255)
        s = Shape()
        for _, pts in sk.chains():
            s.polyline(pts, 5)
        s.circle(sk.head[0], sk.head[1], sk.head_r)
        self.g.fill_shape(s, col, look.alpha)

    def draw_afterimage(self, fighter, sk: Skeleton, dodge_type, alpha: float):
        s = Shape()
        for _, pts in sk.chains():
            s.polyline(pts, 4)
        s.circle(sk.head[0], sk.head[1], sk.head_r)
        self.g.fill_shape(s, DODGE_COLORS.get(dodge_type, (200, 200, 220)), alpha * 0.6)

    def draw_knockback_trail(self, fighter, pts):
        n = len(pts)
        for i in range(n - 1):
            t = (i + 1) / n
            self.g.line(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1], 2 + 8 * t,
                        self.character_color(fighter), 0.35 * t)

    def draw_shield(self, fighter, sk: Skeleton):
        hp = max(0.0, fighter.shield_health) / 100.0
        r = 22 + 26 * hp
        col = self.palette["good"] if hp > 0.5 else self.palette["warn"] if hp > 0.25 else self.palette["bad"]
        cx, cy = sk.center
        self.g.circle(cx, cy - 4, r, col, 0.25)
        self.g.ring(cx, cy - 4, r, 2.5, col, 0.8)

    def draw_counter(self, fighter, sk: Skeleton):
        cx, cy = sk.center
        self.g.ring(cx, cy, 44 + 3 * math.sin(self.fx.frame * 0.5), 3, (110, 160, 255), 0.7)

    def draw_reflector(self, fighter, sk: Skeleton):
        cx, cy = sk.center
        pts = [(cx + math.cos(i * math.pi / 3 + self.fx.frame * 0.1) * 48,
                cy + math.sin(i * math.pi / 3 + self.fx.frame * 0.1) * 48) for i in range(6)]
        self.g.polyline(pts, 3, (110, 255, 255), 0.85, closed=True)

    def draw_projectile(self, p, trail, shake):
        x, y = p.position.x + shake[0], p.position.y + shake[1]
        for i, (tx, ty) in enumerate(trail[:-1]):
            t = (i + 1) / len(trail)
            self.g.circle(tx + shake[0], ty + shake[1], p.width * 0.35 * t, p.color, 0.3 * t)
        self.g.circle(x, y, p.width * 0.5, p.color)

    def draw_effect(self, e, shake):
        from renderer import EffectType
        x, y = e.x + shake[0], e.y + shake[1]
        t = e.progress
        if e.effect_type == EffectType.HIT_SPARK:
            n = 8
            for i in range(n):
                a = math.radians(e.angle) + i * 2 * math.pi / n
                r0, r1 = 8 + 30 * t, 20 + 50 * t * (0.6 + 0.4 * e.intensity)
                self.g.line(x + math.cos(a) * r0, y - math.sin(a) * r0, x + math.cos(a) * r1,
                            y - math.sin(a) * r1, 3 * (1 - t) + 1, e.color, 1 - t)
        elif e.effect_type == EffectType.LANDING_DUST:
            for side in (-1, 1):
                self.g.circle(x + side * (8 + 22 * t), y + 38 - 4 * t, 4 + 8 * t, (200, 200, 200), 0.5 * (1 - t))
        elif e.effect_type == EffectType.REFLECTOR_HEX:
            self.g.ring(x, y, 20 + 40 * t, 3, e.color, 1 - t)
        elif e.effect_type == EffectType.FRAME_ADVANTAGE:
            adv = int(e.intensity)
            label = f"+{adv}" if adv > 0 else str(adv)
            self.text(x, y - 30 - 20 * t, label, 18, e.color, align="center", alpha=1 - t * t)

    def draw_debug_boxes(self, fighter, shake):
        sx, sy = shake
        if self.r.show_hurtboxes:
            hb = fighter.get_hurtbox_aabb()
            self.g.round_rect_outline(hb.x + sx, hb.y + sy, hb.width, hb.height, 2, 2, (80, 255, 120), 0.8)
        if self.r.show_hitboxes:
            for b in fighter.get_active_hitbox_aabbs():
                self.g.fill_rect(b.x + sx, b.y + sy, b.width, b.height, (255, 40, 40), 0.3)
                self.g.round_rect_outline(b.x + sx, b.y + sy, b.width, b.height, 2, 2, (255, 60, 60), 0.9)

    def draw_hud(self, game_state):
        pass

    def draw_victory(self, game_state):
        w, h = self.w, self.h
        self.g.fill_rect(0, 0, w, h, (0, 0, 0), 0.45)
        if game_state.winner is not None:
            txt = f"PLAYER {game_state.winner + 1} WINS"
        else:
            txt = "DRAW"
        self.text(w / 2, h / 2, txt, 56, self.palette["accent"], align="center", valign="middle")

    # ------------------------------------------------------------------ UI kit (menus)

    def ui_backdrop(self, frame: int, variant: str = "menu"):
        self.g.fill_rect(0, 0, self.w, self.h, self.palette["bg"])

    def ui_panel(self, x, y, w, h, emphasis: float = 0.0, accent: Optional[RGB] = None):
        self.g.round_rect(x, y, w, h, 10, self.palette["panel"], 0.92)
        edge = accent or self.palette["panel_edge"]
        self.g.round_rect_outline(x, y, w, h, 10, 1.5 + emphasis * 1.5, edge, 0.6 + 0.4 * emphasis)

    def ui_button(self, x, y, w, h, label, selected: bool, frame: int, accent: Optional[RGB] = None,
                  sublabel: Optional[str] = None, size: int = 20):
        acc = accent or self.palette["accent"]
        self.ui_panel(x, y, w, h, 1.0 if selected else 0.0, acc if selected else None)
        col = self.palette["text"] if selected else self.palette["text_dim"]
        cy = y + h / 2 if not sublabel else y + h * 0.42
        self.text(x + w / 2, cy, label, size, col, align="center", valign="middle")
        if sublabel and selected:
            self.text(x + w / 2, y + h * 0.78, sublabel, max(11, size // 2), self.palette["text_dim"],
                      align="center", valign="middle")

    def ui_title(self, x, y, s, size=48, color=None, frame: int = 0, align="center"):
        self.text(x, y, s, size, color or self.palette["accent"], role="title", align=align,
                  valign="middle")

    def ui_bar(self, x, y, w, h, frac: float, color: RGB, bg_alpha: float = 0.35):
        self.g.round_rect(x, y, w, h, h / 2, self.palette["panel"], bg_alpha + 0.4)
        if frac > 0:
            self.g.round_rect(x, y, max(h, w * min(1.0, frac)), h, h / 2, color)

    def ui_toggle(self, x, y, on: bool, frame: int):
        w, h = 46, 22
        self.g.round_rect(x, y, w, h, h / 2, self.palette["good"] if on else self.palette["panel_edge"],
                          0.9 if on else 0.5)
        kx = x + (w - h / 2 - 2) if on else x + h / 2 + 2
        self.g.circle(kx, y + h / 2, h / 2 - 3, self.palette["text"])

    def ui_arrow(self, x, y, direction: int, color: RGB, size: float = 10, alpha: float = 1.0):
        d = direction
        self.g.polygon([(x + d * size, y), (x - d * size * 0.6, y - size), (x - d * size * 0.6, y + size)],
                       color, alpha)

    def ui_figure(self, x, y, color: RGB, frame: int, scale: float = 1.0, pose: str = "idle",
                  facing: int = 1, char_name: Optional[str] = None):
        """A stick figure for menus (character select, title screen, results)."""
        from poses import POSE_IDLE, ATTACK_POSES, POSE_JUMP_STRETCH, skeleton_from_pose
        from engine import AttackType
        p = {"idle": POSE_IDLE, "victory": POSE_JUMP_STRETCH,
             "attack": ATTACK_POSES[AttackType.SIDE_SPECIAL], "guard": ATTACK_POSES[AttackType.JAB]}.get(pose, POSE_IDLE)
        sk = skeleton_from_pose(p, 0, 0, facing).transformed(scale=scale, pivot=(0, 0), dx=x,
                                                             dy=y + math.sin(frame * 0.08) * 2 * scale)
        self.draw_figure_static(sk, color, scale)

    def draw_figure_static(self, sk: Skeleton, color: RGB, scale: float = 1.0):
        s = Shape()
        for _, pts in sk.chains():
            s.polyline(pts, 5 * scale)
        s.circle(sk.head[0], sk.head[1], sk.head_r)
        self.g.fill_shape(s, color)
