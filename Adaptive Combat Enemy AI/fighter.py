import math
import random
import pygame

from brain import ACTIONS, clamp, get_distance_zone

WIDTH, HEIGHT = 1280, 760
ARENA_W = 760
ARENA_RECT = pygame.Rect(30, 85, ARENA_W - 60, HEIGHT - 155)

COL_PLAYER = (70, 155, 255)
COL_AI = (235, 78, 88)
COL_LIGHT = (255, 215, 75)
COL_HEAVY = (255, 85, 55)
COL_BLOCK = (95, 195, 255)
COL_DODGE = (110, 240, 175)
COL_BLIND = (205, 130, 255)
COL_STUN = (255, 145, 50)
COL_TEXT = (232, 236, 245)
COL_MUTED = (145, 154, 175)


def angle_diff(a, b):
    # Return the shortest signed angle between two angles.
    return (b - a + math.pi) % (2 * math.pi) - math.pi


def infer_observed_context(fighter, incoming_attack_type=None):
    # Identify the player's or AI's currently visible action.
    if incoming_attack_type in ACTIONS:
        return incoming_attack_type
    if fighter.state in ("LIGHT_WINDUP", "LIGHT_ACTIVE", "LIGHT_RECOVERY"):
        return "light"
    if fighter.state in ("HEAVY_WINDUP", "HEAVY_ACTIVE", "HEAVY_RECOVERY"):
        return "heavy"
    if fighter.state == "BLOCK":
        return "block"
    if fighter.state == "DODGE":
        return "dodge"
    return None


class FloatingText:
    def __init__(self, x, y, text, color, duration=0.85):
        # Floating combat feedback shown above a fighter.
        self.x = x
        self.y = y
        self.text = text
        self.color = color
        self.timer = duration
        self.max_timer = duration

    def update(self, dt):
        # Float upward and fade over time.
        self.y -= 38 * dt
        self.timer -= dt
        return self.timer > 0

    def draw(self, surf, font):
        alpha = int(255 * clamp(self.timer / self.max_timer, 0.0, 1.0))
        img = font.render(self.text, True, self.color)
        img.set_alpha(alpha)
        surf.blit(img, (self.x - img.get_width() // 2, self.y))


class Shockwave:
    def __init__(self, x, y, max_radius, color, duration=0.38):
        # Expanding visual effect for the blind attack.
        self.x = x
        self.y = y
        self.max_radius = max_radius
        self.color = color
        self.timer = duration
        self.max_timer = duration

    def update(self, dt):
        self.timer -= dt
        return self.timer > 0

    def draw(self, surf):
        progress = 1.0 - clamp(self.timer / self.max_timer, 0.0, 1.0)
        radius = max(4, int(self.max_radius * progress))
        alpha = int(220 * (1.0 - progress))
        size = radius * 2 + 8
        tmp = pygame.Surface((size, size), pygame.SRCALPHA)
        pygame.draw.circle(tmp, (*self.color, alpha), (size // 2, size // 2), radius, width=4)
        surf.blit(tmp, (self.x - size // 2, self.y - size // 2))


def draw_cone(surf, cx, cy, angle, radius, spread, color, alpha):
    # Draw an attack area as a translucent fan.
    r = int(radius)
    size = r * 2 + 4
    tmp = pygame.Surface((size, size), pygame.SRCALPHA)
    center = (size // 2, size // 2)
    pts = [center]
    steps = 14
    start = angle - spread / 2.0
    for i in range(steps + 1):
        a = start + spread * (i / steps)
        pts.append((center[0] + math.cos(a) * r, center[1] + math.sin(a) * r))
    pygame.draw.polygon(tmp, (*color, alpha), pts)
    pygame.draw.polygon(tmp, (*color, min(255, alpha + 55)), pts, width=2)
    surf.blit(tmp, (cx - size // 2, cy - size // 2))


class Fighter:
    RADIUS = 22

    def __init__(self, x, y, color, name, is_ai=False):
        # Core position, health, stamina, and combat state.
        self.x = float(x)
        self.y = float(y)
        self.vx = 0.0
        self.vy = 0.0
        self.color = color
        self.name = name
        self.is_ai = is_ai

        self.max_hp = 100.0
        self.hp = 100.0
        self.max_stamina = 100.0
        self.stamina = 100.0

        self.angle = 0.0
        self.state = "IDLE"
        self.state_timer = 0.0
        self.has_hit = False

        self.move_speed = 235.0
        self.speed_mult = 1.0
        self.damage_mult = 1.0

        self.dodge_dx = 0.0
        self.dodge_dy = 0.0

        self.blind_timer = 0.0
        self.blind_cd = 0.0
        self.hit_flash = 0.0
        self.attack_arc_range = 0.0
        self.attack_arc_spread = 0.0

        self.ai_block_timer = 0.0
        self.ai_decision_timer = 0.22
        self.ai_strafe_dir = 1
        self.ai_strafe_timer = 1.0
        self.ai_blind_wander_angle = 0.0
        self.prev_opponent_state = "IDLE"
        self.active_decision = None
        self.block_log_cd = 0.0

    @property
    def is_blinded(self):
        # True while the blindness timer is active.
        return self.blind_timer > 0.0

    @property
    def can_act(self):
        # Light/heavy/dodge/blind actions require idle or block state.
        return self.state in ("IDLE", "BLOCK")

    def begin_decision(self, brain, ctx, action):
        # Attach a unique ID to an AI action for later reward credit.
        did = brain.mint_decision_id()
        self.active_decision = {
            "id": did,
            "ctx": ctx,
            "action": action,
            "got_outcome": False,
        }
        brain.register_trace(did, ctx, action)
        return did

    def close_decision(self, expected_id=None):
        # Clear this fighter's decision when a matching outcome finishes.
        if self.active_decision is None:
            return
        if expected_id is None or self.active_decision["id"] == expected_id:
            self.active_decision = None

    def reward_active_decision(
        self,
        brain,
        reward,
        reason,
        expected_action=None,
        observed_ctx=None,
        close_after=False,
    ):
        # Send the combat outcome back to the AI's learning tables.
        if not self.is_ai or brain is None or self.active_decision is None:
            return
        dec = self.active_decision
        if expected_action is not None and dec["action"] != expected_action:
            return

        orig_ctx = dec["ctx"]
        final_ctx = observed_ctx if observed_ctx in ACTIONS else orig_ctx
        relabeled = observed_ctx in ACTIONS and observed_ctx != orig_ctx

        brain.apply_reward(
            final_ctx,
            dec["action"],
            reward,
            reason,
            decision_id=dec["id"],
            relabeled=relabeled,
        )
        dec["got_outcome"] = True
        if close_after:
            self.close_decision(dec["id"])

    def start_light_attack(self):
        # Spend stamina and enter the light attack sequence.
        if not self.can_act or self.stamina < 10:
            return False
        self.stamina -= 10
        self.state = "LIGHT_WINDUP"
        self.state_timer = 0.14 / self.speed_mult
        self.has_hit = False
        self.attack_arc_range = 78.0
        self.attack_arc_spread = math.radians(80)
        return True

    def start_heavy_attack(self):
        # Spend stamina and enter the slow guard-break attack.
        if not self.can_act or self.stamina < 22:
            return False
        self.stamina -= 22
        self.state = "HEAVY_WINDUP"
        self.state_timer = 0.42 / self.speed_mult
        self.has_hit = False
        self.attack_arc_range = 96.0
        self.attack_arc_spread = math.radians(104)
        return True

    def start_dodge(self, dx, dy):
        # Spend stamina and store the dodge movement direction.
        if not self.can_act or self.stamina < 18:
            return False
        mag = math.hypot(dx, dy)
        if mag < 0.01:
            dx = -math.cos(self.angle)
            dy = -math.sin(self.angle)
        else:
            dx /= mag
            dy /= mag
        self.stamina -= 18
        self.state = "DODGE"
        self.state_timer = 0.25
        self.dodge_dx = dx
        self.dodge_dy = dy
        return True

    def try_blind(self, opponent, texts, shockwaves, brain=None):
        # Blind the opponent or penalize a missed/dodged blind.
        if not self.can_act or self.blind_cd > 0.0 or self.stamina < 15:
            return False
        self.stamina -= 15
        self.blind_cd = 6.0
        blind_range = 240.0
        shockwaves.append(Shockwave(self.x, self.y, blind_range, COL_BLIND))

        opp_obs_ctx = infer_observed_context(opponent)
        dist = math.hypot(opponent.x - self.x, opponent.y - self.y)

        if dist <= blind_range:
            if opponent.state == "DODGE":
                texts.append(FloatingText(opponent.x, opponent.y - 30, "BLIND DODGED!", COL_DODGE))
                if self.is_ai:
                    self.reward_active_decision(
                        brain, -0.45, "BLIND DODGED", "blind", "dodge", close_after=True
                    )
                elif opponent.is_ai:
                    opponent.reward_active_decision(brain, +1.00, "EVADED BLIND", "dodge", "blind")
            else:
                opponent.blind_timer = 2.0
                opponent.ai_blind_wander_angle = random.uniform(0, math.tau)
                texts.append(FloatingText(opponent.x, opponent.y - 34, "BLINDED 2.0s!", COL_BLIND))
                if self.is_ai:
                    self.reward_active_decision(
                        brain, +1.10, "BLIND HIT", "blind", opp_obs_ctx, close_after=True
                    )
                elif opponent.is_ai:
                    opponent.reward_active_decision(brain, -0.90, "GOT BLINDED", observed_ctx="blind")
        else:
            texts.append(FloatingText(self.x, self.y - 30, "BLIND MISSED", COL_MUTED))
            if self.is_ai:
                self.reward_active_decision(
                    brain, -0.35, "BLIND WHIFF", "blind", opp_obs_ctx, close_after=True
                )

        if self.is_ai:
            self.close_decision()
        return True

    def apply_hit(self, attacker, attack_type, texts, brain=None):
        # Resolve dodge, block, damage, stun, and learning outcomes.
        if self.state == "DEAD":
            return

        defender_obs_ctx = infer_observed_context(self)

        if self.state == "DODGE":
            texts.append(FloatingText(self.x, self.y - 28, "DODGED!", COL_DODGE))
            if self.is_ai:
                self.reward_active_decision(brain, +1.00, "EVADED HIT", "dodge", attack_type)
            elif attacker.is_ai:
                attacker.reward_active_decision(brain, -0.45, "ATK DODGED", attack_type, "dodge")
            return

        dx = self.x - attacker.x
        dy = self.y - attacker.y
        dist = math.hypot(dx, dy) or 1.0
        nx, ny = dx / dist, dy / dist

        if attack_type == "light":
            dmg = 10.0 * attacker.damage_mult
            if self.state == "BLOCK":
                chip = dmg * 0.20
                self.hp = max(0.0, self.hp - chip)
                self.stamina = max(0.0, self.stamina - 8.0)
                self.vx += nx * 130.0
                self.vy += ny * 130.0
                texts.append(FloatingText(self.x, self.y - 28, f"BLOCKED (-{chip:.0f})", COL_BLOCK))
                if self.is_ai:
                    self.reward_active_decision(brain, +0.85, "BLOCKED LGT", "block", "light")
                elif attacker.is_ai:
                    attacker.reward_active_decision(brain, -0.30, "LGT BLOCKED", "light", "block")
            else:
                self.hp = max(0.0, self.hp - dmg)
                self.hit_flash = 0.14
                self.vx += nx * 210.0
                self.vy += ny * 210.0
                if self.state == "HEAVY_WINDUP":
                    self.state = "STUNNED"
                    self.state_timer = 0.35
                    texts.append(FloatingText(self.x, self.y - 34, f"INTERRUPT! -{dmg:.0f}", COL_LIGHT))
                    if attacker.is_ai:
                        attacker.reward_active_decision(brain, +1.45, "INTERRUPT!", "light", "heavy")
                    elif self.is_ai:
                        self.reward_active_decision(
                            brain, -1.10, "HVY INTRPT", "heavy", "light", close_after=True
                        )
                else:
                    texts.append(FloatingText(self.x, self.y - 28, f"-{dmg:.0f}", COL_LIGHT))
                    if attacker.is_ai:
                        attacker.reward_active_decision(brain, +0.85, "LIGHT HIT", "light", defender_obs_ctx)
                    elif self.is_ai:
                        self.reward_active_decision(brain, -0.70, "HIT BY LGT", observed_ctx="light")

        elif attack_type == "heavy":
            dmg = 24.0 * attacker.damage_mult
            if self.state == "BLOCK":
                self.hp = max(0.0, self.hp - dmg)
                self.stamina = 0.0
                self.state = "STUNNED"
                self.state_timer = 0.85
                self.hit_flash = 0.25
                self.vx += nx * 360.0
                self.vy += ny * 360.0
                texts.append(FloatingText(self.x, self.y - 38, f"GUARD BREAK! -{dmg:.0f}", COL_STUN))
                if attacker.is_ai:
                    attacker.reward_active_decision(brain, +2.00, "GUARD BREAK", "heavy", "block")
                elif self.is_ai:
                    self.reward_active_decision(
                        brain, -1.80, "GUARD BROKEN", "block", "heavy", close_after=True
                    )
            else:
                self.hp = max(0.0, self.hp - dmg)
                self.hit_flash = 0.20
                self.vx += nx * 320.0
                self.vy += ny * 320.0
                texts.append(FloatingText(self.x, self.y - 32, f"HEAVY -{dmg:.0f}", COL_HEAVY))
                if attacker.is_ai:
                    attacker.reward_active_decision(brain, +1.25, "HEAVY HIT", "heavy", defender_obs_ctx)
                elif self.is_ai:
                    self.reward_active_decision(brain, -1.20, "HIT BY HVY", observed_ctx="heavy")

        if self.hp <= 0.0:
            self.hp = 0.0
            self.state = "DEAD"
            if self.is_ai:
                self.close_decision()

    def update(self, dt, opponent, texts, brain=None):
        # Advance timers, movement, and the fighter's state machine.
        self.blind_timer = max(0.0, self.blind_timer - dt)
        self.blind_cd = max(0.0, self.blind_cd - dt)
        self.hit_flash = max(0.0, self.hit_flash - dt)
        self.block_log_cd = max(0.0, self.block_log_cd - dt)

        if self.state == "DEAD":
            return

        # Turn toward the opponent, or rotate randomly while blinded.
        if not self.is_blinded and self.state in ("IDLE", "BLOCK", "LIGHT_WINDUP", "HEAVY_WINDUP"):
            target_ang = math.atan2(opponent.y - self.y, opponent.x - self.x)
            turn_rate = 12.0 if self.state in ("IDLE", "BLOCK") else 4.0
            self.angle += angle_diff(self.angle, target_ang) * min(1.0, turn_rate * dt)
        elif self.is_blinded:
            self.angle += 2.2 * dt

        if self.state == "DODGE":
            # Move quickly in the stored dodge direction.
            self.x += self.dodge_dx * 540.0 * dt
            self.y += self.dodge_dy * 540.0 * dt
            self.state_timer -= dt
            if self.state_timer <= 0.0:
                if self.is_ai:
                    if self.active_decision and not self.active_decision["got_outcome"]:
                        self.reward_active_decision(
                            brain,
                            -0.15,
                            "EMPTY DODGE",
                            "dodge",
                            infer_observed_context(opponent),
                            close_after=True,
                        )
                    else:
                        self.close_decision()
                self.state = "IDLE"

        elif self.state == "STUNNED":
            # Wait until the stun duration expires.
            self.state_timer -= dt
            if self.state_timer <= 0.0:
                self.state = "IDLE"

        elif self.state == "LIGHT_WINDUP":
            # Finish wind-up, then begin the active hit-check phase.
            self.state_timer -= dt
            if self.state_timer <= 0.0:
                self.state = "LIGHT_ACTIVE"
                self.state_timer = 0.11
                self.has_hit = False

        elif self.state == "LIGHT_ACTIVE":
            # Check for a hit once, then enter recovery.
            if not self.has_hit:
                self._check_melee_hit(opponent, "light", texts, brain)
            self.state_timer -= dt
            if self.state_timer <= 0.0:
                if self.is_ai and not self.has_hit:
                    self.reward_active_decision(
                        brain, -0.35, "LIGHT WHIFF", "light", infer_observed_context(opponent)
                    )
                self.state = "LIGHT_RECOVERY"
                self.state_timer = 0.16 / self.speed_mult

        elif self.state == "LIGHT_RECOVERY":
            # Return to idle after the recovery delay.
            self.state_timer -= dt
            if self.state_timer <= 0.0:
                self.state = "IDLE"
                if self.is_ai:
                    self.close_decision()

        elif self.state == "HEAVY_WINDUP":
            # Finish the heavy attack wind-up.
            self.state_timer -= dt
            if self.state_timer <= 0.0:
                self.state = "HEAVY_ACTIVE"
                self.state_timer = 0.14
                self.has_hit = False

        elif self.state == "HEAVY_ACTIVE":
            # Check for a heavy hit once, then enter recovery.
            if not self.has_hit:
                self._check_melee_hit(opponent, "heavy", texts, brain)
            self.state_timer -= dt
            if self.state_timer <= 0.0:
                if self.is_ai and not self.has_hit:
                    self.reward_active_decision(
                        brain, -0.40, "HEAVY WHIFF", "heavy", infer_observed_context(opponent)
                    )
                self.state = "HEAVY_RECOVERY"
                self.state_timer = 0.34 / self.speed_mult

        elif self.state == "HEAVY_RECOVERY":
            # Return to idle after heavy attack recovery.
            self.state_timer -= dt
            if self.state_timer <= 0.0:
                self.state = "IDLE"
                if self.is_ai:
                    self.close_decision()

        elif self.state == "BLOCK":
            # Consume stamina while blocking.
            self.stamina = max(0.0, self.stamina - 9.0 * dt)
            if self.stamina <= 1.0:
                if self.is_ai:
                    if self.active_decision and not self.active_decision["got_outcome"]:
                        self.reward_active_decision(
                            brain,
                            -0.20,
                            "IDLE BLOCK",
                            "block",
                            infer_observed_context(opponent),
                            close_after=True,
                        )
                    else:
                        self.close_decision()
                self.state = "IDLE"

        elif self.state == "IDLE":
            # Regenerate stamina while idle.
            self.stamina = min(self.max_stamina, self.stamina + 26.0 * dt)

        self.x += self.vx * dt
        self.y += self.vy * dt
        decay = math.exp(-10.0 * dt)
        self.vx *= decay
        self.vy *= decay

        pad = self.RADIUS + 4
        self.x = clamp(self.x, ARENA_RECT.left + pad, ARENA_RECT.right - pad)
        self.y = clamp(self.y, ARENA_RECT.top + pad, ARENA_RECT.bottom - pad)

    def _check_melee_hit(self, opponent, attack_type, texts, brain):
        # Check whether an attack cone reaches and hits the opponent.
        dx = opponent.x - self.x
        dy = opponent.y - self.y
        dist = math.hypot(dx, dy)
        if dist <= self.attack_arc_range + opponent.RADIUS:
            ang_to_opp = math.atan2(dy, dx)
            if abs(angle_diff(self.angle, ang_to_opp)) <= self.attack_arc_spread * 0.55:
                self.has_hit = True
                opponent.apply_hit(self, attack_type, texts, brain)

    def draw(self, surf, small_font):
        # Draw attack cones, fighter state, facing line, and status label.
        cx, cy = int(self.x), int(self.y)

        if self.state in ("LIGHT_WINDUP", "LIGHT_ACTIVE", "HEAVY_WINDUP", "HEAVY_ACTIVE"):
            is_heavy = self.state.startswith("HEAVY")
            is_active = self.state.endswith("ACTIVE")
            base_col = COL_HEAVY if is_heavy else COL_LIGHT
            alpha = 185 if is_active else 65
            draw_cone(
                surf, cx, cy, self.angle, self.attack_arc_range, self.attack_arc_spread, base_col, alpha
            )

        if self.state == "BLOCK":
            pygame.draw.circle(surf, COL_BLOCK, (cx, cy), self.RADIUS + 9, width=4)
        if self.state == "DODGE":
            pygame.draw.circle(surf, COL_DODGE, (cx, cy), self.RADIUS + 6, width=3)

        body_col = self.color
        if self.hit_flash > 0.0:
            body_col = (255, 255, 255)
        elif self.is_blinded:
            body_col = (135, 125, 155)
        elif self.state == "STUNNED":
            body_col = COL_STUN
        elif self.state == "DEAD":
            body_col = (80, 82, 92)

        # Side-view fantasy warrior: boots, greaves, layered armor, helmet and sword.
        facing = 1 if math.cos(self.angle) >= 0 else -1
        if self.is_ai:
            facing = -1
        bob = int(math.sin(pygame.time.get_ticks() * 0.008 + (1 if self.is_ai else 0)) * 2)
        if self.state in ("LIGHT_WINDUP", "HEAVY_WINDUP"):
            lean = -facing * 5
        elif self.state in ("LIGHT_ACTIVE", "HEAVY_ACTIVE"):
            lean = facing * 9
        else:
            lean = 0
        # Ground shadow
        pygame.draw.ellipse(surf, (5, 7, 13), (cx - 39, cy + 35, 78, 17))
        # Legs and metal boots
        hip_y = cy + 8 + bob
        pygame.draw.line(surf, (38, 48, 67), (cx - 9 + lean, hip_y), (cx - 15, cy + 34), 11)
        pygame.draw.line(surf, (38, 48, 67), (cx + 9 + lean, hip_y), (cx + 16, cy + 34), 11)
        pygame.draw.line(surf, (150, 166, 190), (cx - 15, cy + 28), (cx - 23, cy + 36), 8)
        pygame.draw.line(surf, (150, 166, 190), (cx + 16, cy + 28), (cx + 24, cy + 36), 8)
        # Tunic and armored torso
        torso = [(cx - 17 + lean, cy - 25 + bob), (cx + 16 + lean, cy - 25 + bob),
                 (cx + 20 + lean, cy + 10 + bob), (cx + 9 + lean, cy + 19 + bob),
                 (cx - 12 + lean, cy + 19 + bob), (cx - 21 + lean, cy + 8 + bob)]
        pygame.draw.polygon(surf, (25, 34, 52), torso)
        pygame.draw.polygon(surf, body_col, [(cx-14+lean,cy-23+bob),(cx+13+lean,cy-23+bob),
                         (cx+15+lean,cy+7+bob),(cx+7+lean,cy+14+bob),
                         (cx-9+lean,cy+14+bob),(cx-16+lean,cy+6+bob)])
        pygame.draw.line(surf, (215, 222, 235), (cx-11+lean,cy-12+bob),(cx+11+lean,cy-12+bob), 3)
        # Shoulder plates and arm
        pygame.draw.circle(surf, (185, 197, 216), (cx - 17 + lean, cy - 19 + bob), 9)
        pygame.draw.circle(surf, (185, 197, 216), (cx + 17 + lean, cy - 19 + bob), 9)
        arm_end = (cx + facing*28 + lean, cy - 2 + bob)
        pygame.draw.line(surf, (185, 197, 216), (cx + facing*15 + lean, cy - 16 + bob), arm_end, 8)
        # Helmet, face slit, crest
        pygame.draw.circle(surf, (185, 197, 216), (cx + lean, cy - 37 + bob), 15)
        pygame.draw.arc(surf, (45, 56, 76), (cx-15+lean,cy-52+bob,30,29), math.pi, math.tau, 5)
        pygame.draw.line(surf, (38, 48, 66), (cx-8+lean,cy-36+bob),(cx+9+lean,cy-36+bob), 4)
        pygame.draw.polygon(surf, (220, 184, 112), [(cx-5+lean,cy-51+bob),(cx+2+lean,cy-65+bob),(cx+8+lean,cy-49+bob)])
        # Sword angle responds to attack state
        sword_len = 57 if self.state.startswith(("LIGHT", "HEAVY")) else 45
        sword_start = (cx + facing*18 + lean, cy - 8 + bob)
        sword_end = (cx + facing*sword_len + lean, cy - 34 + bob)
        if self.state in ("HEAVY_WINDUP", "HEAVY_ACTIVE"):
            sword_end = (cx + facing*49 + lean, cy - 62 + bob)
        pygame.draw.line(surf, (112, 78, 48), sword_start, (sword_start[0]-facing*5, sword_start[1]+12), 6)
        pygame.draw.line(surf, (224, 231, 245), sword_start, sword_end, 7)
        pygame.draw.line(surf, (105, 190, 255) if not self.is_ai else (255, 126, 115),
                         sword_start, sword_end, 2)
        pygame.draw.line(surf, (213, 178, 112),
                         (sword_start[0]-2, sword_start[1]+3),(sword_start[0]+facing*10,sword_start[1]-5),4)
        # Magical aura while using the special ability / under blind effect
        if self.blind_cd <= 0.0 or self.is_blinded:
            aura_col = COL_BLIND if self.is_blinded else (90, 170, 255) if not self.is_ai else (255, 110, 105)
            pygame.draw.circle(surf, aura_col, (cx, cy-13+bob), 34, 1)
        if self.state == "BLOCK":
            pygame.draw.arc(surf, COL_BLOCK, (cx-35,cy-55,70,90), -1.2, 1.2, 5)
        if self.state == "DODGE":
            pygame.draw.ellipse(surf, COL_DODGE, (cx-42,cy-20,84,58), 3)

        tag, tag_col = None, COL_TEXT
        if self.state == "DEAD":
            tag, tag_col = "KO", (245, 85, 95)
        elif self.state == "STUNNED":
            tag, tag_col = "STUNNED!", COL_STUN
        elif self.is_blinded:
            tag, tag_col = f"BLINDED {self.blind_timer:.1f}s", COL_BLIND
        elif self.state == "HEAVY_WINDUP":
            tag, tag_col = "HEAVY!", COL_HEAVY
        elif self.state == "BLOCK":
            tag, tag_col = "BLOCK", COL_BLOCK

        if tag:
            lbl = small_font.render(tag, True, tag_col)
            surf.blit(lbl, (cx - lbl.get_width() // 2, cy - self.RADIUS - 22))


def update_ai(ai, player, brain, dt, texts, shockwaves):
    # Run the AI's movement, difficulty scaling, and action selection.
    if ai.state == "DEAD" or player.state == "DEAD":
        return

    # Scale AI speed and damage from dynamic difficulty.
    diff = brain.effective_difficulty
    ai.speed_mult = clamp(0.85 + (diff - 0.70) * 0.45, 0.85, 1.25)
    ai.damage_mult = clamp(0.85 + (diff - 0.70) * 0.35, 0.85, 1.15)

    if ai.state == "BLOCK":
        ai.ai_block_timer -= dt
        if ai.ai_block_timer <= 0.0:
            if ai.active_decision and not ai.active_decision["got_outcome"]:
                ai.reward_active_decision(
                    brain, -0.20, "IDLE BLOCK", "block", infer_observed_context(player), close_after=True
                )
            else:
                ai.close_decision()
            ai.state = "IDLE"

    if ai.is_blinded:
        # Wander randomly while blinded.
        if ai.state == "IDLE":
            ai.ai_strafe_timer -= dt
            if ai.ai_strafe_timer <= 0.0:
                ai.ai_blind_wander_angle = random.uniform(0, math.tau)
                ai.ai_strafe_timer = 0.45
            ai.x += math.cos(ai.ai_blind_wander_angle) * ai.move_speed * 0.55 * dt
            ai.y += math.sin(ai.ai_blind_wander_angle) * ai.move_speed * 0.55 * dt
        return

    dx = player.x - ai.x
    dy = player.y - ai.y
    dist = math.hypot(dx, dy) or 1.0
    nx, ny = dx / dist, dy / dist

    ctx, _, zone = brain.get_combat_context(infer_observed_context(player), dist)

    if ai.state == "IDLE":
        # Change strafe direction periodically.
        ai.ai_strafe_timer -= dt
        if ai.ai_strafe_timer <= 0.0:
            ai.ai_strafe_dir *= -1
            ai.ai_strafe_timer = random.uniform(0.65, 1.35)

        desired_dist = 82.0 if zone == "CLOSE" else (112.0 if zone == "MID" else 135.0)
        move_x, move_y = 0.0, 0.0
        # Approach or retreat toward the zone's preferred distance.
        if dist > desired_dist + 16:
            move_x += nx
            move_y += ny
        elif dist < desired_dist - 20:
            move_x -= nx * 0.75
            move_y -= ny * 0.75

        move_x += -ny * ai.ai_strafe_dir * 0.55
        move_y += nx * ai.ai_strafe_dir * 0.55

        # Normalize movement and apply the difficulty-scaled speed.
        mag = math.hypot(move_x, move_y)
        if mag > 0.01:
            spd = ai.move_speed * (0.90 + 0.14 * diff)
            ai.x += (move_x / mag) * spd * dt
            ai.y += (move_y / mag) * spd * dt

    if player.state != ai.prev_opponent_state:
        # React faster when the opponent starts a risky action.
        if player.state in ("LIGHT_WINDUP", "HEAVY_WINDUP", "BLOCK"):
            ai.ai_decision_timer = min(ai.ai_decision_timer, 0.11 / diff)
        ai.prev_opponent_state = player.state

    ai.ai_decision_timer -= dt
    if ai.ai_decision_timer > 0.0 or ai.state != "IDLE":
        return

    # Set the next decision cooldown.
    ai.ai_decision_timer = clamp(0.32 / diff, 0.15, 0.44)

    # Only offer actions that are valid at this distance and stamina.
    allowed = []
    if dist <= 98 and ai.stamina >= 10:
        allowed.append("light")
    if dist <= 114 and ai.stamina >= 22:
        allowed.append("heavy")
    if dist <= 145 and ai.stamina >= 14:
        allowed.append("block")
    if dist <= 165 and ai.stamina >= 18:
        allowed.append("dodge")
    if 50 <= dist <= 232 and ai.blind_cd <= 0.0 and ai.stamina >= 15 and not player.is_blinded:
        allowed.append("blind")

    if not allowed:
        return

    chosen = brain.choose_action(ctx, allowed)
    did = ai.begin_decision(brain, ctx, chosen)

    started = False
    if chosen == "light":
        started = ai.start_light_attack()
    elif chosen == "heavy":
        started = ai.start_heavy_attack()
    elif chosen == "block":
        ai.state = "BLOCK"
        ai.ai_block_timer = 0.40
        started = True
    elif chosen == "dodge":
        started = ai.start_dodge(-ny * ai.ai_strafe_dir, nx * ai.ai_strafe_dir)
    elif chosen == "blind":
        started = ai.try_blind(player, texts, shockwaves, brain)

    if not started:
        ai.close_decision(did)

