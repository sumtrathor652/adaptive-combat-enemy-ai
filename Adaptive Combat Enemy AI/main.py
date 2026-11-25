import asyncio
import math
import pygame

from brain import ACTIONS, PROFILE_PATH, SHORT, AdvancedHybridBrain, clamp
from fighter import (
    ARENA_RECT,
    ARENA_W,
    COL_AI,
    COL_BLIND,
    COL_BLOCK,
    COL_DODGE,
    COL_HEAVY,
    COL_LIGHT,
    COL_MUTED,
    COL_PLAYER,
    COL_TEXT,
    HEIGHT,
    WIDTH,
    Fighter,
    update_ai,
)

HUD_X = ARENA_W
HUD_W = WIDTH - ARENA_W
FPS = 60

BG_DARK = (18, 20, 28)
ARENA_BG = (26, 30, 42)
ARENA_GRID = (36, 42, 58)
HUD_BG = (15, 17, 24)
CARD_BG = (24, 28, 40)
BORDER_COL = (70, 82, 110)

COL_GOOD = (85, 220, 140)
COL_WARN = (255, 190, 75)
COL_BAD = (245, 85, 95)

ACTION_COLORS = {
    "light": COL_LIGHT,
    "heavy": COL_HEAVY,
    "block": COL_BLOCK,
    "dodge": COL_DODGE,
    "blind": COL_BLIND,
}

LOG_TAG_COLORS = {
    "good": COL_GOOD,
    "bad": COL_BAD,
    "warn": COL_WARN,
    "info": COL_MUTED,
}


def draw_bar(surf, x, y, w, h, ratio, fill_col, bg_col=(36, 40, 54), border_col=BORDER_COL):
    # Draw a proportional HUD bar.
    pygame.draw.rect(surf, bg_col, (x, y, w, h), border_radius=4)
    fw = int(w * clamp(ratio, 0.0, 1.0))
    if fw > 0:
        pygame.draw.rect(surf, fill_col, (x, y, fw, h), border_radius=4)
    pygame.draw.rect(surf, border_col, (x, y, w, h), width=1, border_radius=4)


def draw_signed_bar(surf, x, y, w, h, val, max_abs=2.5):
    # Draw positive and negative values around a centered axis.
    pygame.draw.rect(surf, (32, 36, 48), (x, y, w, h), border_radius=3)
    mid = x + w // 2
    norm = clamp(val / max_abs, -1.0, 1.0)
    span = int(abs(norm) * (w // 2))
    if span > 0:
        if norm >= 0:
            pygame.draw.rect(surf, COL_GOOD, (mid, y, span, h), border_radius=2)
        else:
            pygame.draw.rect(surf, COL_BAD, (mid - span, y, span, h), border_radius=2)
    pygame.draw.line(surf, (130, 140, 165), (mid, y), (mid, y + h), 1)
    pygame.draw.rect(surf, BORDER_COL, (x, y, w, h), width=1, border_radius=3)


def q_cell_color(val):
    # Map a Q-value to a green, neutral, or red heatmap color.
    t = clamp(val / 2.0, -1.0, 1.0)
    if t >= 0:
        r = int(30 * (1 - t) + 32 * t)
        g = int(34 * (1 - t) + 150 * t)
        b = int(46 * (1 - t) + 88 * t)
    else:
        u = -t
        r = int(30 * (1 - u) + 165 * u)
        g = int(34 * (1 - u) + 42 * u)
        b = int(46 * (1 - u) + 54 * u)
    return (r, g, b)


def draw_arena(surf, player, ai, texts, shockwaves, fonts, match_winner):
    # Draw health bars, arena, fighters, effects, and match controls.
    title_f, _, small_f, tiny_f = fonts

    surf.blit(small_f.render("PLAYER (BLUE)", True, COL_PLAYER), (30, 14))
    draw_bar(surf, 30, 36, 310, 18, player.hp / player.max_hp, COL_PLAYER)
    surf.blit(tiny_f.render(f"HP {player.hp:.0f}/100", True, COL_TEXT), (38, 38))
    draw_bar(surf, 30, 58, 220, 10, player.stamina / player.max_stamina, COL_GOOD)
    cd_txt = "BLIND READY [E]" if player.blind_cd <= 0 else f"BLIND CD {player.blind_cd:.1f}s"
    surf.blit(tiny_f.render(cd_txt, True, COL_BLIND if player.blind_cd <= 0 else COL_MUTED), (258, 56))

    ai_lbl = small_f.render("ADAPTIVE AI (RED)", True, COL_AI)
    surf.blit(ai_lbl, (ARENA_W - 30 - ai_lbl.get_width(), 14))
    draw_bar(surf, ARENA_W - 340, 36, 310, 18, ai.hp / ai.max_hp, COL_AI)
    surf.blit(tiny_f.render(f"HP {ai.hp:.0f}/100", True, COL_TEXT), (ARENA_W - 332, 38))
    draw_bar(surf, ARENA_W - 250, 58, 220, 10, ai.stamina / ai.max_stamina, COL_WARN)
    ai_cd = "BLIND READY" if ai.blind_cd <= 0 else f"BLIND {ai.blind_cd:.1f}s"
    surf.blit(tiny_f.render(ai_cd, True, COL_MUTED), (ARENA_W - 340, 56))

    # Cinematic side-view fantasy arena: ruined fortress, moonlit sky and stone platform.
    pygame.draw.rect(surf, (9, 15, 31), ARENA_RECT, border_radius=8)
    for yy in range(ARENA_RECT.top, ARENA_RECT.bottom, 1):
        t = (yy - ARENA_RECT.top) / max(1, ARENA_RECT.height)
        col = (int(12+12*t), int(20+10*t), int(42+12*t))
        pygame.draw.line(surf, col, (ARENA_RECT.left, yy), (ARENA_RECT.right, yy))
    # Moon and distant mountain silhouettes
    pygame.draw.circle(surf, (126, 163, 205), (ARENA_RECT.right-105, ARENA_RECT.top+95), 43)
    pygame.draw.circle(surf, (49, 75, 111), (ARENA_RECT.right-105, ARENA_RECT.top+95), 53, 3)
    pygame.draw.polygon(surf, (17, 29, 48), [(30,390),(115,235),(185,370),(260,215),(355,390),(455,250),(560,390),(650,245),(700,360),(700,530),(30,530)])
    # Ancient fortress towers and glowing windows
    for tx in (75, 255, 475, 625):
        pygame.draw.rect(surf, (27, 39, 59), (tx, 195, 38, 280))
        pygame.draw.polygon(surf, (42, 57, 80), [(tx-10,195),(tx+19,157),(tx+48,195)])
        pygame.draw.rect(surf, (239, 177, 91), (tx+14, 250, 9, 24), border_radius=4)
    # Floor platform and engraved stones
    floor_y = ARENA_RECT.bottom - 72
    pygame.draw.rect(surf, (32, 39, 53), (ARENA_RECT.left, floor_y, ARENA_RECT.width, 72))
    pygame.draw.line(surf, (167, 133, 87), (ARENA_RECT.left, floor_y), (ARENA_RECT.right, floor_y), 4)
    for gx in range(ARENA_RECT.left+15, ARENA_RECT.right, 58):
        pygame.draw.line(surf, (54, 65, 82), (gx, floor_y+8), (gx+18, ARENA_RECT.bottom), 2)
    for gx in range(ARENA_RECT.left+25, ARENA_RECT.right, 110):
        pygame.draw.arc(surf, (83, 94, 111), (gx, floor_y+18, 72, 35), 0, math.pi, 2)
    pygame.draw.rect(surf, (93, 111, 137), ARENA_RECT, width=2, border_radius=8)

    for sw in shockwaves:
        sw.draw(surf)
    ai.draw(surf, tiny_f)
    player.draw(surf, tiny_f)
    for ft in texts:
        ft.draw(surf, small_f)

    if player.is_blinded:
        fog = pygame.Surface((ARENA_RECT.width, ARENA_RECT.height), pygame.SRCALPHA)
        fog.fill((12, 6, 22, 195))
        surf.blit(fog, ARENA_RECT.topleft)
        warn = title_f.render(f"BLINDED! ({player.blind_timer:.1f}s)", True, COL_BLIND)
        surf.blit(warn, (ARENA_RECT.centerx - warn.get_width() // 2, ARENA_RECT.centery - 14))

    if match_winner:
        overlay = pygame.Surface((ARENA_RECT.width, 110), pygame.SRCALPHA)
        overlay.fill((10, 12, 20, 220))
        by = ARENA_RECT.centery - 55
        surf.blit(overlay, (ARENA_RECT.left, by))
        col = COL_PLAYER if match_winner == "PLAYER" else COL_AI
        msg = title_f.render(f"{match_winner} WINS — PROFILE SAVED", True, col)
        sub = small_f.render("[R] Next Round (Keep Memory)   |   [F9] Wipe JSON (Cold Start)", True, COL_TEXT)
        surf.blit(msg, (ARENA_RECT.centerx - msg.get_width() // 2, by + 22))
        surf.blit(sub, (ARENA_RECT.centerx - sub.get_width() // 2, by + 66))

    c1 = tiny_f.render(
        "MOVE: W/S FORWARD/BACK  A/D LEFT/RIGHT | LIGHT: J | HEAVY: K | BLOCK: L | DODGE: SPACE | BLIND: E",
        True,
        COL_TEXT,
    )
    c2 = tiny_f.render(
        "DEMO KEYS:   [R] Next Round (Persist Learning)     |     [F9] Wipe JSON & Reset",
        True,
        COL_MUTED,
    )
    surf.blit(c1, (30, HEIGHT - 54))
    surf.blit(c2, (30, HEIGHT - 32))


def draw_telemetry_hud(surf, brain, ai, fonts):
    # Draw Markov, policy, Q-table, difficulty, and reward telemetry.
    _, med_f, small_f, tiny_f = fonts

    pygame.draw.rect(surf, HUD_BG, (HUD_X, 0, HUD_W, HEIGHT))
    pygame.draw.line(surf, BORDER_COL, (HUD_X, 0), (HUD_X, HEIGHT), 2)

    x = HUD_X + 18
    w = HUD_W - 36

    surf.blit(med_f.render("DUAL-BANDIT + MARKOV TELEMETRY", True, COL_TEXT), (x, 14))
    acc_pct = brain.markov_accuracy * 100.0
    sub = (
        f"Acc: {acc_pct:4.1f}% ({brain.markov_correct_preds}/{brain.markov_total_preds})  |  "
        f"H: {brain.entropy_norm:.2f}  |  Fast/Slow: {brain.w_fast:.2f}/{brain.w_slow:.2f}"
    )
    surf.blit(tiny_f.render(sub, True, COL_MUTED), (x, 38))
    pygame.draw.line(surf, BORDER_COL, (x, 58), (x + w, 58), 1)

    # 1. Markov sequence predictor
    surf.blit(small_f.render("1. Markov Predictor (2nd -> 1st Fallback)", True, COL_TEXT), (x, 66))
    pred_act, pred_conf, m_probs = brain.predict_next()
    last_lbl = brain.recent_actions[-1].upper() if len(brain.recent_actions) >= 1 else "NONE"

    pygame.draw.rect(surf, CARD_BG, (x, 88, w, 32), border_radius=5)
    pygame.draw.rect(surf, BORDER_COL, (x, 88, w, 32), width=1, border_radius=5)
    pred_line = (
        f"O{brain.markov_order_used} LAST:{last_lbl:<4} -> "
        f"PRED:{pred_act.upper():<5} ({pred_conf * 100:4.1f}%)"
    )
    surf.blit(small_f.render(pred_line, True, COL_WARN), (x + 12, 95))

    y = 128
    for act in ACTIONS:
        p = m_probs[act]
        lbl = tiny_f.render(f"{act.upper():<6} {p * 100:5.1f}%", True, COL_TEXT if act == pred_act else COL_MUTED)
        surf.blit(lbl, (x, y))
        draw_bar(surf, x + 108, y + 2, w - 108, 12, p, ACTION_COLORS[act])
        y += 20

    # 2. Blended policy for active context & zone
    ctx = brain.active_ctx
    src = brain.active_ctx_source
    zone = brain.active_zone
    pi = brain.softmax_policy(ctx, ACTIONS)
    best_act = max(ACTIONS, key=lambda a: brain.blended_q(ctx, a))

    surf.blit(
        small_f.render(
            f"2. Blended Policy  ctx={SHORT[ctx]}[{src}]  zone={zone}  tau={brain.temperature:.2f}",
            True,
            COL_TEXT,
        ),
        (x, 236),
    )

    y = 260
    for act in ACTIONS:
        qv = brain.blended_q(ctx, act)
        ub = brain.ucb_bonus(ctx, act)
        n_sa = int(brain.visit_counts[ctx][act])
        prob = pi[act] * 100.0
        col = COL_GOOD if act == best_act and qv > 0.05 else COL_TEXT
        surf.blit(tiny_f.render(f"{SHORT[act]} {qv:+5.2f} +{ub:.2f}", True, col), (x, y))
        draw_signed_bar(surf, x + 124, y + 2, w - 216, 13, qv, max_abs=2.5)
        surf.blit(tiny_f.render(f"n={n_sa:<2d} {prob:4.1f}%", True, COL_MUTED), (x + w - 84, y))
        y += 21

    # 3. 5x5 blended Q-table heatmap & status
    surf.blit(small_f.render("3. Blended Q-Table (Rows=Player, Cols=AI)", True, COL_TEXT), (x, 374))

    gx = x
    gy = 398
    cell_w, cell_h = 44, 22
    row_lbl_w = 40

    for ci, a in enumerate(ACTIONS):
        cx = gx + row_lbl_w + ci * (cell_w + 2)
        surf.blit(tiny_f.render(SHORT[a], True, COL_MUTED), (cx + 10, gy))

    for ri, r_act in enumerate(ACTIONS):
        ry = gy + 18 + ri * (cell_h + 3)
        r_col = COL_WARN if r_act == ctx else COL_MUTED
        surf.blit(tiny_f.render(SHORT[r_act], True, r_col), (gx + 2, ry + 4))
        for ci, c_act in enumerate(ACTIONS):
            cx = gx + row_lbl_w + ci * (cell_w + 2)
            qv = brain.blended_q(r_act, c_act)
            pygame.draw.rect(surf, q_cell_color(qv), (cx, ry, cell_w, cell_h), border_radius=3)
            b_col = COL_WARN if r_act == ctx else BORDER_COL
            pygame.draw.rect(surf, b_col, (cx, ry, cell_w, cell_h), width=1, border_radius=3)
            txt = tiny_f.render(f"{qv:+.1f}", True, COL_TEXT)
            surf.blit(txt, (cx + (cell_w - txt.get_width()) // 2, ry + 4))

    rx = gx + row_lbl_w + 5 * (cell_w + 2) + 12
    rw = (x + w) - rx
    surf.blit(tiny_f.render(f"DDA: {brain.effective_difficulty:.2f}x", True, COL_TEXT), (rx, gy + 18))
    diff_norm = (brain.effective_difficulty - 0.70) / (1.50 - 0.70)
    draw_bar(surf, rx, gy + 36, rw, 11, diff_norm, COL_WARN)
    surf.blit(tiny_f.render(f"Spd {ai.speed_mult:.2f}x Dmg {ai.damage_mult:.2f}x", True, COL_MUTED), (rx, gy + 52))

    slow_status = "SLOW: UNLOCKED" if brain.slow_unlocked else f"SLOW: {brain.total_observations}/{brain.slow_unlock_threshold}"
    surf.blit(tiny_f.render(slow_status, True, COL_GOOD if brain.slow_unlocked else COL_MUTED), (rx, gy + 74))

    active_id_str = (
        f"#{ai.active_decision['id']} {SHORT[ai.active_decision['action']]}"
        if ai.active_decision is not None
        else "IDLE (NONE)"
    )
    surf.blit(tiny_f.render(f"DEC: {active_id_str}", True, COL_WARN if ai.active_decision else COL_MUTED), (rx, gy + 94))
    surf.blit(tiny_f.render(f"REC: M{brain.matches_played} P{brain.player_wins}-A{brain.ai_wins}", True, COL_MUTED), (rx, gy + 114))

    rew_col = COL_GOOD if brain.last_reward_val > 0 else (COL_BAD if brain.last_reward_val < 0 else COL_MUTED)
    rew_str = f"{brain.last_reward_val:+.2f} {brain.last_reward_reason}"
    surf.blit(tiny_f.render(rew_str[:22], True, rew_col), (rx, gy + 134))

    pygame.draw.line(surf, BORDER_COL, (x, 558), (x + w, 558), 1)

    # 4. Learning & trace log
    surf.blit(small_f.render("4. Credit Log (* = Relabeled, ~ = Trace)", True, COL_TEXT), (x, 566))
    y = 590
    for msg, tag in list(brain.event_log):
        col = LOG_TAG_COLORS.get(tag, COL_MUTED)
        surf.blit(tiny_f.render(msg[:64], True, col), (x, y))
        y += 19



def draw_main_menu(surf, fonts, pulse):
    """Polished title screen; the original project/game name is intentionally preserved."""
    title_f, med_f, small_f, tiny_f = fonts
    surf.fill((8, 12, 24))
    # Layered atmospheric arena backdrop
    for i in range(12):
        col = (12 + i * 2, 19 + i * 2, 35 + i * 3)
        pygame.draw.rect(surf, col, (0, 250 + i * 34, WIDTH, 34))
    # Distant pillars and glowing arena gate
    for x in (90, 170, WIDTH - 170, WIDTH - 90):
        pygame.draw.rect(surf, (25, 36, 57), (x, 160, 42, 440), border_radius=8)
        pygame.draw.rect(surf, (55, 85, 120), (x + 8, 170, 5, 420), border_radius=3)
    glow = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    pygame.draw.ellipse(glow, (30, 125, 210, 35), (WIDTH//2 - 270, 120, 540, 500), 5)
    pygame.draw.ellipse(glow, (190, 95, 50, 28), (WIDTH//2 - 220, 155, 440, 430), 3)
    surf.blit(glow, (0, 0))
    # Decorative floor rings
    for r in (90, 145, 205):
        pygame.draw.ellipse(surf, (35, 57, 83), (WIDTH//2-r, 465-r//3, r*2, r*2//3), 2)
    # Two illustrated combatants
    for cx, cy, col, direction in ((WIDTH//2-185, 430, COL_PLAYER, 1),
                                   (WIDTH//2+185, 430, COL_AI, -1)):
        pygame.draw.ellipse(surf, (3, 6, 13), (cx-48, cy+42, 96, 20))
        pygame.draw.line(surf, (45, 55, 73), (cx-10,cy+4),(cx-20,cy+42),12)
        pygame.draw.line(surf, (45, 55, 73), (cx+10,cy+4),(cx+19,cy+42),12)
        pygame.draw.polygon(surf, col, [(cx-20,cy-25),(cx+17,cy-25),(cx+22,cy+12),(cx,cy+23),(cx-22,cy+12)])
        pygame.draw.polygon(surf, (205,215,230), [(cx-16,cy-22),(cx+13,cy-22),(cx+16,cy+8),(cx,cy+16),(cx-17,cy+8)], 3)
        pygame.draw.circle(surf, (185,197,216), (cx,cy-39), 17)
        pygame.draw.arc(surf, (35,45,62), (cx-17,cy-56,34,32), math.pi, math.tau, 5)
        pygame.draw.polygon(surf, (220,184,112), [(cx-6,cy-53),(cx+1,cy-70),(cx+8,cy-52)])
        pygame.draw.line(surf, (220,230,245), (cx+direction*14,cy-14),(cx+direction*77,cy-58),7)
        pygame.draw.line(surf, (125,190,245), (cx+direction*14,cy-14),(cx+direction*77,cy-58),2)
        pygame.draw.line(surf, (205,170,110), (cx+direction*12,cy-17),(cx+direction*25,cy-3),5)
    # Original game name is retained
    heading = title_f.render("ADAPTIVE COMBAT ENEMY AI", True, (238, 244, 255))
    surf.blit(heading, (WIDTH//2-heading.get_width()//2, 78))
    sub = small_f.render("LEARN • ADAPT • FIGHT", True, (105, 190, 255))
    surf.blit(sub, (WIDTH//2-sub.get_width()//2, 119))
    tagline = tiny_f.render("A fighter that studies your moves and adapts to your strategy", True, (175, 190, 215))
    surf.blit(tagline, (WIDTH//2-tagline.get_width()//2, 158))
    # Buttons
    buttons = [
        (pygame.Rect(WIDTH//2-125, 535, 250, 52), "START BATTLE", (28, 105, 175)),
        (pygame.Rect(WIDTH//2-125, 600, 250, 42), "HOW TO PLAY", (42, 58, 82)),
    ]
    for rect, label, color in buttons:
        pygame.draw.rect(surf, (5, 9, 18), rect.move(0, 4), border_radius=9)
        pygame.draw.rect(surf, color, rect, border_radius=9)
        pygame.draw.rect(surf, (105, 165, 220), rect, 2, border_radius=9)
        txt = small_f.render(label, True, (245, 248, 255))
        surf.blit(txt, (rect.centerx-txt.get_width()//2, rect.centery-txt.get_height()//2))
    hint = tiny_f.render("ESC  •  EXIT", True, (130, 145, 170))
    surf.blit(hint, (WIDTH//2-hint.get_width()//2, 675))
    return buttons


def draw_how_to_play(surf, fonts):
    title_f, med_f, small_f, tiny_f = fonts
    surf.fill((10, 15, 27))
    heading = title_f.render("HOW TO PLAY", True, COL_TEXT)
    surf.blit(heading, (WIDTH//2-heading.get_width()//2, 105))
    lines = [
        "W / S             Move forward and backward",
        "A / D             Move left and right",
        "J                 Light attack",
        "K                 Heavy attack / guard break",
        "L                 Hold to block",
        "SPACE          Dodge",
        "E                 Blind ability",
        "R                 Start the next round (AI keeps learning)",
        "F9               Reset saved AI learning profile",
    ]
    y = 205
    for line in lines:
        label = small_f.render(line, True, (190, 210, 235))
        surf.blit(label, (WIDTH//2-250, y))
        y += 43
    back = med_f.render("Press ENTER or ESC to return", True, COL_WARN)
    surf.blit(back, (WIDTH//2-back.get_width()//2, 600))


def spawn_fighters():
    # Create the player and AI at opposite sides of the arena.
    player = Fighter(ARENA_RECT.left + 160, ARENA_RECT.bottom - 105, COL_PLAYER, "Player", is_ai=False)
    ai = Fighter(ARENA_RECT.right - 160, ARENA_RECT.bottom - 105, COL_AI, "AI", is_ai=True)
    ai.angle = math.pi
    return player, ai


async def main():
    # Initialize Pygame and run the browser-compatible 60-FPS loop.
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Adaptive Combat Enemy AI")
    clock = pygame.time.Clock()

    fonts = (
        pygame.font.SysFont("consolas", 23, bold=True),
        pygame.font.SysFont("consolas", 17, bold=True),
        pygame.font.SysFont("consolas", 14, bold=True),
        pygame.font.SysFont("consolas", 12),
    )

    brain = AdvancedHybridBrain(PROFILE_PATH)
    player, ai = spawn_fighters()
    texts = []
    shockwaves = []
    match_winner = None
    dda_tick = 0.0
    screen_state = "menu"
    menu_buttons = []

    running = True
    while running:
        dt = min(0.05, clock.tick(FPS) / 1000.0)
        keys = pygame.key.get_pressed()
        # W/S move forward/backward; A/D move left/right.
        move_x = (1 if keys[pygame.K_d] else 0) - (1 if keys[pygame.K_a] else 0)
        move_y = (1 if keys[pygame.K_s] else 0) - (1 if keys[pygame.K_w] else 0)

        for event in pygame.event.get():
            # Menu navigation and in-game controls.
            if event.type == pygame.QUIT:
                running = False
            elif screen_state == "menu":
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key in (pygame.K_RETURN, pygame.K_SPACE):
                        screen_state = "game"
                    elif event.key == pygame.K_h:
                        screen_state = "help"
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if menu_buttons and menu_buttons[0][0].collidepoint(event.pos):
                        screen_state = "game"
                    elif menu_buttons and menu_buttons[1][0].collidepoint(event.pos):
                        screen_state = "help"
            elif screen_state == "help":
                if event.type == pygame.KEYDOWN and event.key in (pygame.K_RETURN, pygame.K_ESCAPE, pygame.K_BACKSPACE):
                    screen_state = "menu"
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    screen_state = "menu"
                elif event.key == pygame.K_r:
                    player, ai = spawn_fighters()
                    texts.clear()
                    shockwaves.clear()
                    match_winner = None
                    brain.reset_round()
                elif event.key == pygame.K_F9:
                    brain.reset_profile()
                    player, ai = spawn_fighters()
                    texts.clear()
                    shockwaves.clear()
                    match_winner = None
                elif match_winner is None and player.state != "DEAD":
                    if event.key == pygame.K_j:
                        if player.start_light_attack():
                            brain.record_player_action("light")
                    elif event.key == pygame.K_k:
                        if player.start_heavy_attack():
                            brain.record_player_action("heavy")
                    elif event.key == pygame.K_SPACE:
                        if player.start_dodge(move_x, move_y):
                            brain.record_player_action("dodge")
                    elif event.key == pygame.K_e:
                        if player.try_blind(ai, texts, shockwaves, brain):
                            brain.record_player_action("blind")
                    elif event.key == pygame.K_l:
                        if player.state == "IDLE" and player.stamina >= 10:
                            player.state = "BLOCK"
                            if player.block_log_cd <= 0.0:
                                brain.record_player_action("block")
                                player.block_log_cd = 1.25

        if screen_state == "game":
            if match_winner is None:
                # Blocking is a held action.
                if keys[pygame.K_l]:
                    if player.state == "IDLE" and player.stamina >= 10:
                        player.state = "BLOCK"
                    if player.state == "BLOCK" and player.block_log_cd <= 0.0:
                        brain.record_player_action("block")
                        player.block_log_cd = 1.25
                elif player.state == "BLOCK":
                    player.state = "IDLE"

                if player.state == "IDLE" and (move_x != 0 or move_y != 0):
                    # Move without affecting the player's state machine.
                    mag = math.hypot(move_x, move_y)
                    player.x += (move_x / mag) * player.move_speed * dt
                    player.y += (move_y / mag) * player.move_speed * dt

                dda_tick += dt
                if dda_tick >= 0.20:
                    # Update difficulty every 0.20 seconds.
                    dda_tick = 0.0
                    brain.update_dda((player.hp - ai.hp) / 100.0)

                update_ai(ai, player, brain, dt, texts, shockwaves)
                player.update(dt, ai, texts, brain)
                ai.update(dt, player, texts, brain)
                # Keep both fighters inside the arena while allowing forward/back movement.
                margin = Fighter.RADIUS + 8
                player.x = clamp(player.x, ARENA_RECT.left + margin, ARENA_RECT.right - margin)
                player.y = clamp(player.y, ARENA_RECT.top + margin, ARENA_RECT.bottom - margin)
                ai.x = clamp(ai.x, ARENA_RECT.left + margin, ARENA_RECT.right - margin)
                ai.y = clamp(ai.y, ARENA_RECT.top + margin, ARENA_RECT.bottom - margin)

                # Separate fighters if they overlap.
                dx = ai.x - player.x
                dy = ai.y - player.y
                dist = math.hypot(dx, dy)
                min_dist = Fighter.RADIUS * 2.0 + 2.0
                if 0.001 < dist < min_dist:
                    push = (min_dist - dist) * 0.5
                    nx, ny = dx / dist, dy / dist
                    player.x -= nx * push
                    player.y -= ny * push
                    ai.x += nx * push
                    ai.y += ny * push

                if player.state == "DEAD":
                    # AI wins and the result updates the profile.
                    match_winner = "AI"
                    brain.save_profile(player_won=False)
                elif ai.state == "DEAD":
                    # Player wins and difficulty is increased slightly.
                    match_winner = "PLAYER"
                    brain.save_profile(player_won=True)

        # Remove expired floating text and shockwaves.
        texts = [t for t in texts if t.update(dt)]
        shockwaves = [s for s in shockwaves if s.update(dt)]

        if screen_state == "menu":
            menu_buttons = draw_main_menu(screen, fonts, pygame.time.get_ticks() / 1000.0)
        elif screen_state == "help":
            draw_how_to_play(screen, fonts)
        else:
            screen.fill(BG_DARK)
            draw_arena(screen, player, ai, texts, shockwaves, fonts, match_winner)
            draw_telemetry_hud(screen, brain, ai, fonts)
        pygame.display.flip()
        await asyncio.sleep(0)  # Yield to the browser event loop.

    pygame.quit()


if __name__ == "__main__":
    asyncio.run(main())