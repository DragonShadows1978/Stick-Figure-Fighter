"""
Deterministic headless match scenarios used as golden traces.

Every scenario drives both fighters with seeded, "sticky" random inputs (an
input is held for several frames) so movement, attacks, specials, shields,
dodges, projectiles, reflects and counters all get exercised. Each frame the
full simulation state is hashed; the chained digest is checkpointed so a
refactor that changes behaviour shows exactly when the sim diverged.

Scenarios run in a fixed order in one process on purpose: they share
module-level state (e.g. character moveset caches) exactly like a real
training run does.

Regenerate with:  python3 tests/record_golden.py
"""
import hashlib
import json
import os
import random
from collections import Counter

from game_engine import GameLoop, InputState, create_game

GOLDEN_PATH = os.path.join(os.path.dirname(__file__), "golden", "traces.json")
FRAMES = 3600
CHECKPOINT_EVERY = 30
CHARACTERS = ["RUSH", "ZONE", "GRAB", "FLEX"]

# (left, right, up, down, jump, attack, special, shield) weights per "intent"
INTENTS = [
    ("idle", 3), ("walk", 6), ("jump", 3), ("attack", 6), ("special", 4),
    ("dirspecial", 4), ("shield", 2), ("dodge", 2), ("downspecial", 3),
    ("upspecial", 2), ("drop", 1),
]


def scenarios():
    out = [("default_battlefield", None, None, "battlefield"),
           ("default_fd", None, None, "final_destination")]
    i = 0
    for p1 in CHARACTERS:
        for p2 in CHARACTERS:
            stage = "battlefield" if i % 2 == 0 else "final_destination"
            out.append((f"{p1}_vs_{p2}_{stage}", p1, p2, stage))
            i += 1
    return out


class StickyInput:
    """Seeded input generator that holds an intent for a few frames."""

    def __init__(self, seed):
        self.rng = random.Random(seed)
        self.intent = "idle"
        self.direction = 1
        self.hold = 0
        names = [n for n, _ in INTENTS]
        weights = [w for _, w in INTENTS]
        self._pick = lambda: self.rng.choices(names, weights)[0]

    def next(self, me, opponent):
        if self.hold <= 0:
            self.intent = self._pick()
            toward = 1 if opponent.position.x > me.position.x else -1
            self.direction = toward if self.rng.random() < 0.75 else -toward
            self.hold = self.rng.randint(4, 18)
        self.hold -= 1
        inp = InputState()
        left, right = self.direction < 0, self.direction > 0
        tap = (self.hold % 3) == 0
        it = self.intent
        if it == "walk":
            inp.left, inp.right = left, right
        elif it == "jump":
            inp.left, inp.right = left, right
            inp.jump = tap
            inp.up = True
        elif it == "attack":
            inp.left, inp.right = (left, right) if self.rng.random() < 0.5 else (False, False)
            inp.up = self.rng.random() < 0.2
            inp.down = self.rng.random() < 0.15
            inp.attack = tap
        elif it == "special":
            inp.special = tap
        elif it == "dirspecial":
            inp.left, inp.right = left, right
            inp.special = tap
        elif it == "downspecial":
            inp.down = True
            inp.special = tap
        elif it == "upspecial":
            inp.up = True
            inp.special = tap
        elif it == "shield":
            inp.shield = True
        elif it == "dodge":
            inp.shield = tap
            inp.left, inp.right = left, right
            inp.down = self.rng.random() < 0.3
        elif it == "drop":
            inp.down = True
        return inp


def _f(x):
    return repr(float(x))


def state_digest(state):
    parts = [str(state.frame), str(state.hit_freeze_frames), str(state.match_over),
             str(state.winner), str(state.combo_counts)]
    for f in state.fighters:
        atk = f.current_attack.attack_type.name if f.current_attack else "-"
        parts.append("|".join([
            _f(f.position.x), _f(f.position.y), _f(f.velocity.x), _f(f.velocity.y),
            f.state.name, str(f.state_frame), str(f.grounded), str(f.facing_right),
            str(f.can_double_jump), _f(f.damage_percent), str(f.stocks),
            _f(f.shield_health), str(f.shield_stun), str(f.shield_broken),
            str(f.invincible_frames), str(f.hitstun_frames), atk, str(f.attack_frame),
            str(f.counter_active), str(f.counter_success), str(f.reflector_active),
            str(f.super_armor_active), str(len(f.active_hitboxes)),
            str(f.dodge_frames_remaining),
        ]))
    for p in state.projectiles:
        parts.append("P" + "|".join([
            str(p.owner_id), _f(p.position.x), _f(p.position.y), _f(p.velocity.x),
            _f(p.velocity.y), str(p.lifetime), str(p.active), str(p.reflected),
            _f(p.hitbox.damage),
        ]))
    return "\n".join(parts)


def run_scenario(name, p1, p2, stage, seed):
    game = create_game(stage, p1, p2, stock_count=3)
    loop = GameLoop(game, headless=True)
    events = Counter()
    loop.on_hit = lambda *a: events.update(["hit"])
    loop.on_projectile_hit = lambda *a: events.update(["projectile_hit"])
    loop.on_reflect = lambda *a: events.update(["reflect"])
    loop.on_land = lambda *a: events.update(["land"])
    loop.on_dodge = lambda *a: events.update(["dodge"])
    gens = [StickyInput(seed * 2 + 1), StickyInput(seed * 2 + 2)]

    chain = hashlib.sha256()
    checkpoints = []
    seen_counter = False
    for frame in range(FRAMES):
        p1f, p2f = game.fighters
        p1f.input = gens[0].next(p1f, p2f)
        p2f.input = gens[1].next(p2f, p1f)
        ongoing = loop.step()
        if game.projectiles:
            events["projectile_frames"] += 1
        for f in game.fighters:
            if f.counter_success and not seen_counter:
                events["counter"] += 1
            if f.shield_broken:
                events["shield_broken_frames"] += 1
        seen_counter = any(f.counter_success for f in game.fighters)
        chain.update(state_digest(game).encode())
        if frame % CHECKPOINT_EVERY == 0:
            checkpoints.append(chain.hexdigest()[:16])
        if not ongoing:
            break
    return {
        "frames": game.frame,
        "checkpoints": checkpoints,
        "final": chain.hexdigest(),
        "result": {
            "winner": game.winner,
            "stocks": [f.stocks for f in game.fighters],
            "damage": [round(f.damage_percent, 3) for f in game.fighters],
        },
        "events": dict(events),
    }


def run_all():
    return {name: run_scenario(name, p1, p2, stage, seed)
            for seed, (name, p1, p2, stage) in enumerate(scenarios())}


def load_golden():
    with open(GOLDEN_PATH) as fh:
        return json.load(fh)
