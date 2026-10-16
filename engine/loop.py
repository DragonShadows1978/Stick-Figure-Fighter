"""Fixed-timestep simulation loop: movement, physics, collision and combat."""
import math
import time
from typing import Callable, Optional

from .combat import AttackData, AttackType, default_attack
from .fighter import DodgeType, Fighter, FighterState
from .geometry import AABB, Vector2
from .input import InputState
from .stage import Platform
from .state import GameState


class GameLoop:
    """
    Main game loop with fixed timestep simulation.

    The game runs at a fixed 60 FPS internally regardless of render rate.
    This ensures deterministic physics for replay and netcode.
    """

    def __init__(self, game_state: GameState, headless: bool = False,
                 fallback_attack: Callable[[AttackType], AttackData] = default_attack):
        self.state = game_state
        self.config = game_state.config
        self.headless = headless
        # Move used for any slot a fighter's moveset doesn't define
        self.fallback_attack = fallback_attack
        self.running = False

        # Callbacks for external systems
        self.on_frame_update: Optional[Callable[[GameState], None]] = None
        self.on_render: Optional[Callable[[GameState], None]] = None
        self.on_match_end: Optional[Callable[[GameState], None]] = None

        # Hit callback for visual effects
        # Signature: on_hit(attacker, defender, hitbox, contact_x, contact_y, knockback_magnitude)
        self.on_hit: Optional[Callable] = None

        # Projectile hit callback
        # Signature: on_projectile_hit(projectile, defender, contact_x, contact_y, knockback_magnitude)
        self.on_projectile_hit: Optional[Callable] = None

        # Reflect callback
        # Signature: on_reflect(projectile, reflector, x, y)
        self.on_reflect: Optional[Callable] = None

        # Landing callback for dust effects
        # Signature: on_land(fighter, x, y)
        self.on_land: Optional[Callable] = None

        # Dodge callback for sound/visual effects
        # Signature: on_dodge(fighter, dodge_type)
        self.on_dodge: Optional[Callable] = None

        # Timing
        self.frame_time = self.config.frame_time
        self.accumulator = 0.0
        self.last_time = 0.0

        # Frame skip for rendering (render every N simulation frames)
        self.render_skip = 1
        self.frames_since_render = 0

    def step(self) -> bool:
        """
        Run a single simulation frame.

        Returns True if match is still ongoing, False if over.
        """
        if self.state.match_over:
            return False

        # Check pause state
        if self.state.paused:
            return True

        # Check hit freeze state
        # During hit freeze, skip physics but still render (dramatic pause)
        if self.state.hit_freeze_frames > 0:
            self.state.hit_freeze_frames -= 1
            # Still update input states during freeze
            for fighter in self.state.fighters:
                fighter.input.update_prev()
            return True

        # Update each fighter
        for fighter in self.state.fighters:
            self._update_fighter(fighter)

        # Update projectiles
        self._update_projectiles()

        # Check for knockouts
        self._check_knockouts()

        # Process combat (hitbox vs hurtbox, projectiles)
        self._process_combat()
        self._process_projectile_combat()

        # Clean up inactive projectiles
        self.state.remove_inactive_projectiles()

        # Update combo timers
        self.state.update_combos()

        # Update damage animations
        self.state.update_damage_animations()

        # Update match timer
        if self.state.update_timer():
            # Time's up!
            self.state.determine_time_up_winner()
            return False

        # Check match end
        self.state.check_match_over()

        # Increment frame counter
        self.state.frame += 1

        # Update input previous states
        for fighter in self.state.fighters:
            fighter.input.update_prev()

        return not self.state.match_over

    def _update_projectiles(self):
        """Update all active projectiles."""
        for projectile in self.state.projectiles:
            if projectile.active:
                projectile.update(self.config.gravity)

                # Check if projectile is off-stage
                if self.state.is_in_blast_zone(projectile.position):
                    projectile.active = False

    def _update_fighter(self, fighter: Fighter):
        """Update a single fighter for one frame."""
        # Decrement timers
        if fighter.invincible_frames > 0:
            fighter.invincible_frames -= 1

        if fighter.hitstun_frames > 0:
            fighter.hitstun_frames -= 1
            if fighter.hitstun_frames == 0:
                fighter.state = FighterState.FALLING if not fighter.grounded else FighterState.IDLE

        # Shield regeneration
        if fighter.shield_regen_delay > 0:
            fighter.shield_regen_delay -= 1
        elif fighter.state != FighterState.SHIELDING and fighter.shield_health < 100.0:
            fighter.shield_health = min(100.0, fighter.shield_health + 0.15)

        # Shield stun countdown
        if fighter.shield_stun > 0:
            fighter.shield_stun -= 1
            if fighter.shield_stun == 0 and fighter.state == FighterState.SHIELD_STUN:
                fighter.shield_broken = False
                fighter.state = FighterState.IDLE if fighter.grounded else FighterState.FALLING

        # Counter window countdown
        if fighter.counter_window_frames > 0:
            fighter.counter_window_frames -= 1
            if fighter.counter_window_frames == 0:
                fighter.counter_active = False

        # Dodge frames countdown
        if fighter.dodge_frames_remaining > 0:
            fighter.dodge_frames_remaining -= 1
            if fighter.dodge_frames_remaining == 0:
                fighter.invincible_frames = 0
                fighter.state = FighterState.IDLE if fighter.grounded else FighterState.FALLING

        # Reset air dodge when landing
        if fighter.grounded and not fighter.air_dodge_available:
            fighter.air_dodge_available = True

        # State-based update
        if fighter.state == FighterState.HITSTUN:
            self._update_hitstun(fighter)
        elif fighter.state == FighterState.ATTACKING:
            self._update_attack(fighter)
        elif fighter.state == FighterState.SPECIAL:
            self._update_special(fighter)
        elif fighter.state == FighterState.SHIELDING:
            self._update_shielding(fighter)
        elif fighter.state == FighterState.DODGING:
            self._update_dodge(fighter)
        elif fighter.state == FighterState.COUNTER:
            self._update_counter(fighter)
        elif fighter.state == FighterState.REFLECTING:
            self._update_reflector(fighter)
        elif fighter.state == FighterState.SHIELD_STUN:
            pass  # Just wait out the stun
        else:
            self._update_normal(fighter)

        # Apply physics
        self._apply_physics(fighter)

        # Platform collision
        self._handle_platform_collision(fighter)
        self._update_ground_support(fighter)

        fighter.state_frame += 1

    def _update_normal(self, fighter: Fighter):
        """Update fighter in normal (non-hitstun) state."""
        inp = fighter.input

        # Horizontal movement
        if inp.left and not inp.right:
            if fighter.grounded:
                fighter.velocity.x = -fighter.stats.walk_speed
                fighter.state = FighterState.WALKING
            else:
                fighter.velocity.x = max(
                    fighter.velocity.x - 0.3,
                    -fighter.stats.air_speed
                )
            fighter.facing_right = False
        elif inp.right and not inp.left:
            if fighter.grounded:
                fighter.velocity.x = fighter.stats.walk_speed
                fighter.state = FighterState.WALKING
            else:
                fighter.velocity.x = min(
                    fighter.velocity.x + 0.3,
                    fighter.stats.air_speed
                )
            fighter.facing_right = True
        else:
            if fighter.grounded:
                fighter.velocity.x *= self.config.friction
                if abs(fighter.velocity.x) < 0.1:
                    fighter.velocity.x = 0
                    if fighter.state == FighterState.WALKING:
                        fighter.state = FighterState.IDLE

        # Jumping
        if inp.jump_pressed:
            if fighter.grounded:
                fighter.velocity.y = -fighter.stats.jump_force
                fighter.grounded = False
                fighter.state = FighterState.JUMPING
            elif fighter.can_double_jump:
                fighter.velocity.y = -fighter.stats.double_jump_force
                fighter.can_double_jump = False
                fighter.state = FighterState.JUMPING

        # Fast fall
        if inp.down and not fighter.grounded and fighter.velocity.y > 0:
            fighter.velocity.y = max(fighter.velocity.y,
                                     fighter.stats.fast_fall_speed * 10)

        # Attacks
        if inp.attack_pressed and fighter.state not in (FighterState.ATTACKING,):
            self._start_attack(fighter)

        # Special moves
        if inp.special_pressed and fighter.state not in (FighterState.ATTACKING, FighterState.SPECIAL):
            self._start_special(fighter)

        # Shield
        if inp.shield and fighter.grounded and not fighter.shield_broken:
            if fighter.state not in (FighterState.SHIELDING, FighterState.DODGING,
                                     FighterState.ATTACKING, FighterState.SPECIAL):
                fighter.state = FighterState.SHIELDING
                fighter.state_frame = 0

        # Dodge (shield + direction while grounded, or just shield in air for air dodge)
        if inp.shield_pressed:
            if fighter.grounded:
                if inp.left or inp.right:
                    # Roll
                    self._start_dodge(fighter, DodgeType.ROLL_BACKWARD if inp.left == fighter.facing_right else DodgeType.ROLL_FORWARD)
                elif inp.down and fighter.state == FighterState.SHIELDING:
                    # Spot dodge
                    self._start_dodge(fighter, DodgeType.SPOT_DODGE)
            else:
                # Air dodge
                if fighter.air_dodge_available:
                    self._start_dodge(fighter, DodgeType.AIR_DODGE)

    def _update_hitstun(self, fighter: Fighter):
        """Update fighter in hitstun state."""
        # Allow DI (Directional Influence)
        if fighter.input.left:
            fighter.velocity.x -= 0.1
        elif fighter.input.right:
            fighter.velocity.x += 0.1

    def _update_attack(self, fighter: Fighter):
        """Update fighter during attack animation."""
        fighter.attack_frame += 1
        attack = fighter.current_attack

        if attack is None:
            fighter.state = FighterState.IDLE if fighter.grounded else FighterState.FALLING
            return

        # Update active hitboxes based on frame
        fighter.active_hitboxes = []
        for hitbox in attack.hitboxes:
            start, end = hitbox.active_frames
            if start <= fighter.attack_frame <= end:
                fighter.active_hitboxes.append(hitbox)

        # End attack when animation is done
        if fighter.attack_frame >= attack.total_frames:
            fighter.current_attack = None
            fighter.active_hitboxes = []
            fighter.state = FighterState.IDLE if fighter.grounded else FighterState.FALLING

    def _update_special(self, fighter: Fighter):
        """Update fighter during special move animation."""
        fighter.attack_frame += 1
        attack = fighter.current_attack

        if attack is None:
            fighter.state = FighterState.IDLE if fighter.grounded else FighterState.FALLING
            return

        # Update active hitboxes based on frame
        fighter.active_hitboxes = []
        for hitbox in attack.hitboxes:
            start, end = hitbox.active_frames
            if start <= fighter.attack_frame <= end:
                fighter.active_hitboxes.append(hitbox)

        # End special when animation is done
        if fighter.attack_frame >= attack.total_frames:
            fighter.current_attack = None
            fighter.active_hitboxes = []
            fighter.super_armor_active = False
            fighter.counter_active = False
            fighter.reflector_active = False
            fighter.state = FighterState.IDLE if fighter.grounded else FighterState.FALLING

    def _update_shielding(self, fighter: Fighter):
        """Update fighter while shielding."""
        inp = fighter.input

        # Deplete shield over time (slow drain while holding)
        fighter.shield_health -= 0.1

        # Check for shield break
        if fighter.shield_health <= 0:
            fighter.shield_health = 0
            fighter.shield_broken = True
            fighter.shield_stun = 180  # 3 seconds stun
            fighter.state = FighterState.SHIELD_STUN
            fighter.state_frame = 0
            return

        # Release shield if button released
        if not inp.shield:
            fighter.state = FighterState.IDLE
            fighter.shield_regen_delay = 60  # 1 second before regen starts

        # Can still dodge out of shield
        if inp.down:
            self._start_dodge(fighter, DodgeType.SPOT_DODGE)
        elif inp.left:
            self._start_dodge(fighter, DodgeType.ROLL_BACKWARD if fighter.facing_right else DodgeType.ROLL_FORWARD)
        elif inp.right:
            self._start_dodge(fighter, DodgeType.ROLL_FORWARD if fighter.facing_right else DodgeType.ROLL_BACKWARD)

    def _update_dodge(self, fighter: Fighter):
        """Update fighter during dodge animation."""
        # Apply roll movement
        if fighter.dodge_type in (DodgeType.ROLL_FORWARD, DodgeType.ROLL_BACKWARD):
            direction = 1 if fighter.dodge_type == DodgeType.ROLL_FORWARD else -1
            if not fighter.facing_right:
                direction = -direction
            fighter.velocity.x = direction * fighter.stats.run_speed * 0.8

        # Air dodge can have directional influence
        if fighter.dodge_type == DodgeType.AIR_DODGE:
            inp = fighter.input
            if inp.left:
                fighter.velocity.x = -fighter.stats.air_speed * 0.5
            elif inp.right:
                fighter.velocity.x = fighter.stats.air_speed * 0.5
            if inp.up:
                fighter.velocity.y = -fighter.stats.air_speed * 0.5
            elif inp.down:
                fighter.velocity.y = fighter.stats.air_speed * 0.5

    def _update_counter(self, fighter: Fighter):
        """Update fighter during counter stance."""
        fighter.attack_frame += 1
        attack = fighter.current_attack

        if attack is None:
            fighter.counter_active = False
            fighter.state = FighterState.IDLE if fighter.grounded else FighterState.FALLING
            return

        # Counter success triggers counterattack
        if fighter.counter_success:
            # Counter was triggered - hitbox is now active
            fighter.active_hitboxes = []
            for hitbox in attack.hitboxes:
                start, end = hitbox.active_frames
                if start <= fighter.attack_frame <= end:
                    fighter.active_hitboxes.append(hitbox)

        # End counter when animation is done
        if fighter.attack_frame >= attack.total_frames:
            fighter.current_attack = None
            fighter.active_hitboxes = []
            fighter.counter_active = False
            fighter.counter_success = False
            fighter.state = FighterState.IDLE if fighter.grounded else FighterState.FALLING

    def _update_reflector(self, fighter: Fighter):
        """Update fighter during reflector active frames."""
        fighter.attack_frame += 1
        attack = fighter.current_attack

        if attack is None:
            fighter.reflector_active = False
            fighter.state = FighterState.IDLE if fighter.grounded else FighterState.FALLING
            return

        # Reflector has a hitbox for close-range use
        fighter.active_hitboxes = []
        for hitbox in attack.hitboxes:
            start, end = hitbox.active_frames
            if start <= fighter.attack_frame <= end:
                fighter.active_hitboxes.append(hitbox)
                fighter.reflector_active = True
            else:
                fighter.reflector_active = False

        # End reflector when animation is done
        if fighter.attack_frame >= attack.total_frames:
            fighter.current_attack = None
            fighter.active_hitboxes = []
            fighter.reflector_active = False
            fighter.state = FighterState.IDLE if fighter.grounded else FighterState.FALLING

    def _start_dodge(self, fighter: Fighter, dodge_type: 'DodgeType'):
        """Start a dodge move."""
        fighter.dodge_type = dodge_type
        fighter.state = FighterState.DODGING
        fighter.state_frame = 0

        # Set dodge duration and invincibility based on type
        if dodge_type == DodgeType.SPOT_DODGE:
            fighter.dodge_frames_remaining = 20
            fighter.invincible_frames = 15
        elif dodge_type in (DodgeType.ROLL_FORWARD, DodgeType.ROLL_BACKWARD):
            fighter.dodge_frames_remaining = 30
            fighter.invincible_frames = 20
        elif dodge_type == DodgeType.AIR_DODGE:
            fighter.dodge_frames_remaining = 25
            fighter.invincible_frames = 18
            fighter.air_dodge_available = False

        # Trigger dodge callback for sound/visual effects
        if self.on_dodge:
            self.on_dodge(fighter, dodge_type)

    def _start_attack(self, fighter: Fighter):
        """Start an attack based on current input and state."""
        inp = fighter.input

        # Determine attack type based on input
        if fighter.grounded:
            if inp.up:
                attack_type = AttackType.UP_TILT
            elif inp.down:
                attack_type = AttackType.DOWN_TILT
            elif inp.left or inp.right:
                attack_type = AttackType.FORWARD_TILT
            else:
                attack_type = AttackType.JAB
        else:
            if inp.up:
                attack_type = AttackType.UP_AIR
            elif inp.down:
                attack_type = AttackType.DOWN_AIR
            elif (inp.left and fighter.facing_right) or (inp.right and not fighter.facing_right):
                attack_type = AttackType.BACK_AIR
            elif (inp.right and fighter.facing_right) or (inp.left and not fighter.facing_right):
                attack_type = AttackType.FORWARD_AIR
            else:
                attack_type = AttackType.NEUTRAL_AIR

        # Try character-specific attack first, fall back to default
        attack = fighter.get_attack_data(attack_type)
        if attack is None:
            attack = self.fallback_attack(attack_type)

        fighter.current_attack = attack
        fighter.attack_frame = 0
        fighter.state = FighterState.ATTACKING

    def _start_special(self, fighter: Fighter):
        """Start a special move based on current input."""
        inp = fighter.input

        # Determine special type based on direction
        if inp.up:
            attack_type = AttackType.UP_SPECIAL
            # Up special typically gives vertical boost (recovery)
            if not fighter.grounded:
                fighter.velocity.y = -fighter.stats.jump_force * 0.8
                fighter.can_double_jump = False
        elif inp.down:
            attack_type = AttackType.DOWN_SPECIAL
        elif inp.left or inp.right:
            attack_type = AttackType.SIDE_SPECIAL
            # Side special may give horizontal movement
            direction = -1 if inp.left else 1
            fighter.facing_right = direction > 0
            fighter.velocity.x = direction * fighter.stats.run_speed * 0.5
        else:
            attack_type = AttackType.NEUTRAL_SPECIAL

        # Try character-specific attack first, fall back to default
        attack = fighter.get_attack_data(attack_type)
        if attack is None:
            attack = self.fallback_attack(attack_type)

        fighter.current_attack = attack
        fighter.attack_frame = 0

        # Per-move special mechanics
        if attack.is_counter:
            fighter.state = FighterState.COUNTER
            fighter.counter_active = True
            fighter.counter_window_frames = 15
            fighter.counter_success = False
        elif attack.is_reflector:
            fighter.state = FighterState.REFLECTING
            fighter.reflector_active = True
        else:
            if attack.super_armor:
                fighter.super_armor_active = True
            if attack.projectile is not None:
                self.state.add_projectile(attack.projectile.spawn(
                    fighter.player_id, fighter.position, fighter.facing_right))
            fighter.state = FighterState.SPECIAL

        if attack.on_start is not None:
            attack.on_start(self, fighter)

    def _compute_safe_displacement(self, fighter: Fighter) -> float:
        """
        Displacement bound calculation.

        Computes the maximum safe vertical movement distance to prevent
        penetration through platforms. Returns the bounded y-displacement.
        """
        if fighter.velocity.y <= 0:  # Not falling, no bound needed
            return fighter.velocity.y

        hurtbox = fighter.get_hurtbox_aabb()
        foot_y = fighter.feet_y
        max_displacement = fighter.velocity.y

        for platform in self.state.stage.all_platforms:
            # Skip if no horizontal overlap
            plat_aabb = platform.get_aabb()
            if hurtbox.right < plat_aabb.left or hurtbox.left > plat_aabb.right:
                continue

            # Skip passthrough platforms if dropping through
            if platform.passthrough and fighter.input.down:
                continue

            # Skip passthrough platforms if we're already below them
            if platform.passthrough and foot_y > plat_aabb.top:
                continue

            # Distance to contact zone
            distance_to_contact = platform.contact_zone_y - foot_y

            # If we're above the contact zone and would cross it
            if distance_to_contact >= 0 and distance_to_contact < max_displacement:
                max_displacement = distance_to_contact

        return max_displacement

    def _apply_physics(self, fighter: Fighter):
        """Apply physics (gravity, velocity) to fighter."""
        # Apply gravity
        if not fighter.grounded:
            fighter.velocity.y += self.config.gravity
            fighter.velocity.y = min(fighter.velocity.y, self.config.terminal_velocity)

            # Air friction
            fighter.velocity.x *= self.config.air_friction

        # Compute safe displacement bound for vertical movement
        safe_y_displacement = self._compute_safe_displacement(fighter)

        # Apply velocity with bounded vertical displacement
        fighter.position.x += fighter.velocity.x
        fighter.position.y += safe_y_displacement

    def _handle_platform_collision(self, fighter: Fighter):
        """Handle collision between fighter and platforms."""
        hurtbox = fighter.get_hurtbox_aabb()

        # Check main platform
        main_plat = self.state.stage.main_platform
        self._check_platform_landing(fighter, main_plat, hurtbox)

        # Check other platforms
        for platform in self.state.stage.platforms:
            self._check_platform_landing(fighter, platform, hurtbox)

    def _check_platform_landing(self, fighter: Fighter, platform: Platform,
                                 hurtbox: AABB):
        """
        Check if fighter lands on a platform.

        Uses the contact zone for detection but snaps to exact surface for positioning.
        """
        plat_aabb = platform.get_aabb()

        # Only check if moving downward
        if fighter.velocity.y <= 0:
            return

        # Check if feet are near the platform surface
        foot_y = fighter.feet_y
        prev_foot_y = foot_y - fighter.velocity.y

        # For passthrough platforms, only land if coming from above
        if platform.passthrough:
            if prev_foot_y > plat_aabb.top:
                return
            # Allow dropping through
            if fighter.input.down:
                return

        # Check horizontal overlap
        if hurtbox.right < plat_aabb.left or hurtbox.left > plat_aabb.right:
            return

        # Land when the feet were at/above the surface last frame and have now
        # reached the contact zone (offset volume) just above it. Detecting at
        # the zone rather than the surface prevents penetration; checking the
        # previous position against the surface also catches feet that start
        # inside the zone (e.g. respawning exactly on the floor).
        if prev_foot_y <= plat_aabb.top and foot_y >= platform.contact_zone_y:
            # Land: snap feet to the exact surface, not the contact zone
            fighter.position.y = plat_aabb.top - self.config.foot_offset
            fighter.velocity.y = 0
            fighter.grounded = True
            fighter.can_double_jump = True

            if fighter.state in (FighterState.JUMPING, FighterState.FALLING):
                fighter.state = FighterState.LANDING
                fighter.state_frame = 0
                # Trigger landing dust effect
                if self.on_land:
                    self.on_land(fighter, fighter.position.x, fighter.position.y)

    # States that just mean "standing around"; walking off a ledge turns them into FALLING
    _GROUNDED_IDLE_STATES = (FighterState.IDLE, FighterState.WALKING, FighterState.RUNNING,
                             FighterState.LANDING, FighterState.CROUCH, FighterState.SHIELDING)
    _SUPPORT_TOLERANCE = 0.5

    def _is_supported(self, fighter: Fighter) -> bool:
        """True if the fighter's feet are resting on a platform."""
        hurtbox = fighter.get_hurtbox_aabb()
        foot_y = fighter.feet_y
        inp = fighter.input
        dropping = inp.down and not (inp.attack or inp.special or inp.shield)
        for platform in self.state.stage.all_platforms:
            plat_aabb = platform.get_aabb()
            if hurtbox.right < plat_aabb.left or hurtbox.left > plat_aabb.right:
                continue
            if platform.passthrough and dropping:
                continue
            if abs(foot_y - plat_aabb.top) <= self._SUPPORT_TOLERANCE:
                return True
        return False

    def _update_ground_support(self, fighter: Fighter):
        """Unground fighters that walked off a ledge or dropped through a platform."""
        if not fighter.grounded:
            return
        if not self._is_supported(fighter):
            fighter.grounded = False
            if fighter.state in self._GROUNDED_IDLE_STATES:
                fighter.state = FighterState.FALLING
                fighter.state_frame = 0
        elif fighter.state == FighterState.LANDING and fighter.state_frame > 4:
            fighter.state = FighterState.IDLE

    def _check_knockouts(self):
        """Check if any fighters are in blast zones."""
        for fighter in self.state.fighters:
            if self.state.is_in_blast_zone(fighter.position):
                spawn = self.state.get_spawn_point(fighter.player_id)
                fighter.respawn(spawn)

    def _process_combat(self):
        """Process hitbox vs hurtbox collisions."""
        fighters = self.state.fighters

        # For each fighter with active hitboxes
        for attacker in fighters:
            if not attacker.active_hitboxes:
                continue

            for hitbox in attacker.active_hitboxes:
                hitbox_aabb = hitbox.get_aabb(attacker.position, attacker.facing_right)

                # Check against all other fighters
                for defender in fighters:
                    if defender == attacker:
                        continue
                    if defender.invincible_frames > 0:
                        continue
                    if defender.state == FighterState.HITSTUN:
                        continue  # Prevent infinite combos (simplified)

                    hurtbox_aabb = defender.get_hurtbox_aabb()

                    if hitbox_aabb.intersects(hurtbox_aabb):
                        # Check if defender is shielding
                        if defender.state == FighterState.SHIELDING:
                            # Hit shield instead
                            shield_damage = hitbox.damage * 1.5  # Shield takes more damage
                            defender.shield_health -= shield_damage
                            defender.shield_regen_delay = 90  # Delay regen after being hit

                            # Shield stun
                            defender.shield_stun = int(hitbox.damage * 0.5)

                            # Check for shield break
                            if defender.shield_health <= 0:
                                defender.shield_health = 0
                                defender.shield_broken = True
                                defender.shield_stun = 180  # Long stun
                                defender.state = FighterState.SHIELD_STUN
                                defender.state_frame = 0

                            # Remove hitbox after hit
                            attacker.active_hitboxes.remove(hitbox)
                            break

                        # Check if defender is in counter stance
                        if defender.counter_active and defender.counter_window_frames > 0:
                            # Counter triggered!
                            defender.counter_success = True
                            defender.counter_window_frames = 0

                            # Turn to face attacker
                            defender.facing_right = attacker.position.x > defender.position.x

                            # Counter attack hits harder (1.5x damage boost)
                            if defender.current_attack:
                                for hb in defender.current_attack.hitboxes:
                                    hb.damage *= 1.5
                                    hb.knockback_base *= 1.5

                            # Remove attacker's hitbox
                            attacker.active_hitboxes.remove(hitbox)
                            break

                        # Check if attacker has super armor
                        if attacker.super_armor_active:
                            # Super armor: take damage but don't flinch
                            # (The attacker still hits the defender normally)
                            pass

                        # Check if defender has super armor
                        if defender.super_armor_active:
                            # Take damage but don't get knocked back
                            defender.damage_percent += hitbox.damage
                            attacker.active_hitboxes.remove(hitbox)
                            break

                        # Calculate knockback
                        angle_rad = math.radians(hitbox.angle)
                        if not attacker.facing_right:
                            angle_rad = math.pi - angle_rad

                        kb_magnitude = hitbox.knockback_base + (
                            defender.damage_percent * hitbox.knockback_scaling
                        )
                        knockback = Vector2(
                            math.cos(angle_rad) * kb_magnitude,
                            -math.sin(angle_rad) * kb_magnitude
                        )

                        # Calculate contact point (center of hitbox/hurtbox overlap)
                        contact_x = (hitbox_aabb.center.x + hurtbox_aabb.center.x) / 2
                        contact_y = (hitbox_aabb.center.y + hurtbox_aabb.center.y) / 2

                        defender.apply_knockback(knockback, hitbox.damage)

                        # Apply hit freeze for impactful hits
                        self.state.apply_hit_freeze(knockback.magnitude(), hitbox.damage)

                        # Register hit for combo tracking
                        self.state.register_hit(attacker.player_id, defender.player_id)

                        # Call hit callback for visual effects
                        if self.on_hit:
                            self.on_hit(attacker, defender, hitbox, contact_x, contact_y,
                                       knockback.magnitude())

                        # Remove hitbox after hit (no multi-hit for now)
                        attacker.active_hitboxes.remove(hitbox)
                        break

    def _process_projectile_combat(self):
        """Process projectile collisions with fighters."""
        fighters = self.state.fighters

        for projectile in self.state.projectiles:
            if not projectile.active:
                continue

            proj_aabb = projectile.get_aabb()

            for fighter in fighters:
                # Skip owner (unless reflected)
                if fighter.player_id == projectile.owner_id and not projectile.reflected:
                    continue
                if fighter.invincible_frames > 0:
                    continue

                hurtbox_aabb = fighter.get_hurtbox_aabb()

                if proj_aabb.intersects(hurtbox_aabb):
                    # Check if fighter is shielding
                    if fighter.state == FighterState.SHIELDING:
                        shield_damage = projectile.hitbox.damage * 1.5
                        fighter.shield_health -= shield_damage
                        fighter.shield_regen_delay = 90
                        projectile.active = False
                        break

                    # Check if fighter has reflector active
                    if fighter.reflector_active:
                        reflect_x = projectile.position.x
                        reflect_y = projectile.position.y
                        projectile.reflect()
                        projectile.owner_id = fighter.player_id  # Now belongs to reflector

                        # Call reflect callback for visual effects
                        if self.on_reflect:
                            self.on_reflect(projectile, fighter, reflect_x, reflect_y)
                        break

                    # Normal hit
                    angle_rad = math.radians(projectile.hitbox.angle)
                    if projectile.velocity.x < 0:
                        angle_rad = math.pi - angle_rad

                    kb_magnitude = projectile.hitbox.knockback_base + (
                        fighter.damage_percent * projectile.hitbox.knockback_scaling
                    )
                    knockback = Vector2(
                        math.cos(angle_rad) * kb_magnitude,
                        -math.sin(angle_rad) * kb_magnitude
                    )

                    # Contact point
                    contact_x = projectile.position.x
                    contact_y = projectile.position.y

                    fighter.apply_knockback(knockback, projectile.hitbox.damage)

                    # Apply hit freeze for impactful projectile hits
                    self.state.apply_hit_freeze(knockback.magnitude(), projectile.hitbox.damage)

                    # Register projectile hit for combo tracking
                    self.state.register_hit(projectile.owner_id, fighter.player_id)

                    # Call projectile hit callback for visual effects
                    if self.on_projectile_hit:
                        self.on_projectile_hit(projectile, fighter, contact_x, contact_y,
                                              knockback.magnitude())

                    projectile.active = False
                    break

    def run(self, max_frames: int = 0):
        """
        Run the game loop.

        Args:
            max_frames: Maximum frames to run (0 = until match ends)
        """
        self.running = True
        frames_run = 0

        if self.headless:
            # No wall clock: simulate as fast as possible
            while self.running and (max_frames == 0 or frames_run < max_frames):
                if not self.step():
                    self.running = False
                    break
                frames_run += 1
                if self.on_frame_update:
                    self.on_frame_update(self.state)
            if self.on_match_end:
                self.on_match_end(self.state)
            return self.state

        self.last_time = time.time()

        while self.running and (max_frames == 0 or frames_run < max_frames):
            current_time = time.time()
            delta = current_time - self.last_time
            self.last_time = current_time
            self.accumulator += delta

            # Fixed timestep simulation
            while self.accumulator >= self.frame_time:
                if not self.step():
                    self.running = False
                    break

                self.accumulator -= self.frame_time
                frames_run += 1

                # Callback for external systems
                if self.on_frame_update:
                    self.on_frame_update(self.state)

            # Rendering (if not headless)
            if not self.headless and self.on_render:
                self.frames_since_render += 1
                if self.frames_since_render >= self.render_skip:
                    self.on_render(self.state)
                    self.frames_since_render = 0

        if self.on_match_end:
            self.on_match_end(self.state)

        return self.state

    def stop(self):
        """Stop the game loop."""
        self.running = False



def run_headless_match(game_state: GameState,
                       input_callback: Callable[[GameState, int], InputState],
                       max_frames: int = 10800) -> GameState:
    """
    Run a match in headless mode.

    Args:
        game_state: The game state to run
        input_callback: Function that returns InputState for each player each frame
                       Takes (game_state, player_id) and returns InputState
        max_frames: Maximum frames (default 3 minutes at 60 FPS)

    Returns:
        Final game state
    """
    loop = GameLoop(game_state, headless=True)

    def frame_update(state: GameState):
        for fighter in state.fighters:
            fighter.input = input_callback(state, fighter.player_id)

    loop.on_frame_update = frame_update
    return loop.run(max_frames)

