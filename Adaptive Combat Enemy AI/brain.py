import json
import math
import os
import random
import time
from collections import deque

PROFILE_PATH = "player_profile.json"
ACTIONS = ("light", "heavy", "block", "dodge", "blind")
SHORT = {"light": "LGT", "heavy": "HVY", "block": "BLK", "dodge": "DDG", "blind": "BLD"}
DIST_ZONES = ("CLOSE", "MID", "FAR")


def clamp(val, lo, hi):
    # Keep values inside a safe range.
    return max(lo, min(hi, val))


def get_distance_zone(dist):
    # Convert raw distance into a combat zone.
    if dist < 95.0:
        return "CLOSE"
    if dist < 175.0:
        return "MID"
    return "FAR"


class AdvancedHybridBrain:
    """
    Combines:
      1. Decaying 1st- and 2nd-order Markov chain with automatic fallback.
      2. Dual-timescale Q-tables (fast working memory + slow long-term memory).
      3. Shannon-entropy dynamic gating between fast and slow tables.
      4. 3-step eligibility traces with decision-ID & observed-impact credit assignment.
      5. Confidence-scaled UCB + DDA Softmax action selection.
    """

    def __init__(self, profile_path=PROFILE_PATH):
        self.profile_path = profile_path
        # Learning rates: fast adapts quickly; slow retains trends.
        self.alpha_fast = 0.42
        self.alpha_slow = 0.12
        self.markov_decay = 0.86
        self.smoothing = 0.25
        self.trace_decay = 0.55
        self.c_base = 0.65
        self.slow_unlock_threshold = 12

        # Player action-sequence predictions.
        self.markov_1st = self._empty_2d(0.0)
        self.markov_2nd = {
            a: {b: {c: 0.0 for c in ACTIONS} for b in ACTIONS} for a in ACTIONS
        }

        # Fast/slow learned values and visits per context-action pair.
        self.q_fast = self._empty_2d(0.0)
        self.q_slow = self._empty_2d(0.0)
        self.visit_counts = self._empty_2d(0.0)

        self.recent_actions = deque(maxlen=16)
        self.trace_queue = deque(maxlen=3)  # (decision_id, ctx, action)
        self.next_decision_id = 1
        self.total_observations = 0

        # Persisted match & DDA state
        self.matches_played = 0
        self.player_wins = 0
        self.ai_wins = 0
        self.base_difficulty = 1.0
        self.effective_difficulty = 1.0
        self.profile_loaded = False

        # Live telemetry state
        self.pending_prediction = None
        self.markov_total_preds = 0
        self.markov_correct_preds = 0
        self.markov_order_used = 1
        self.last_pred_conf = 0.20

        self.entropy_norm = 1.0
        self.w_fast = 1.0
        self.w_slow = 0.0

        self.active_ctx = "light"
        self.active_ctx_source = "PRED"
        self.active_zone = "MID"
        self.last_reward_val = 0.0
        self.last_reward_reason = "AWAITING EXCHANGE"
        self.q_updates = 0
        self.event_log = deque(maxlen=8)

        self.load_profile()
        self._update_entropy_and_weights()

    @staticmethod
    def _empty_2d(fill=0.0):
        # Create a 5x5 table keyed by context and action.
        return {r: {c: float(fill) for c in ACTIONS} for r in ACTIONS}

    def mint_decision_id(self):
        # Assign an ID so reward credit stays attached to the right decision.
        did = self.next_decision_id
        self.next_decision_id += 1
        return did

    def log(self, msg, tag="info"):
        # Store the newest entries first for HUD display.
        stamp = time.strftime("%H:%M:%S")
        self.event_log.appendleft((f"[{stamp}] {msg}", tag))

    @property
    def slow_unlocked(self):
        # Unlock long-term memory after enough data or a saved profile.
        return self.profile_loaded or self.total_observations >= self.slow_unlock_threshold

    @property
    def markov_accuracy(self):
        if self.markov_total_preds == 0:
            return 0.0
        return self.markov_correct_preds / self.markov_total_preds

    @property
    def temperature(self):
        # Higher difficulty makes action selection less exploratory.
        return clamp(0.62 - 0.24 * (self.effective_difficulty - 0.70), 0.26, 0.65)

    def load_profile(self):
        if not os.path.exists(self.profile_path):
            self.log("Blank state: cold start active.", "info")
            return
        try:
            # Restore all persistent learning and match statistics.
            with open(self.profile_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.matches_played = int(data.get("matches_played", 0))
            self.player_wins = int(data.get("player_wins", 0))
            self.ai_wins = int(data.get("ai_wins", 0))
            self.base_difficulty = clamp(float(data.get("base_difficulty", 1.0)), 0.75, 1.45)
            self.total_observations = int(data.get("total_observations", 0))
            self.q_updates = int(data.get("q_updates", 0))

            m1 = data.get("markov_1st", {})
            qf = data.get("q_fast", {})
            qs = data.get("q_slow", {})
            vc = data.get("visit_counts", {})
            for r in ACTIONS:
                for c in ACTIONS:
                    self.markov_1st[r][c] = max(0.0, float(m1.get(r, {}).get(c, 0.0)))
                    self.q_fast[r][c] = clamp(float(qf.get(r, {}).get(c, 0.0)), -2.5, 2.5)
                    self.q_slow[r][c] = clamp(float(qs.get(r, {}).get(c, 0.0)), -2.5, 2.5)
                    self.visit_counts[r][c] = max(0.0, float(vc.get(r, {}).get(c, 0.0)))

            m2 = data.get("markov_2nd", {})
            for a in ACTIONS:
                for b in ACTIONS:
                    for c in ACTIONS:
                        self.markov_2nd[a][b][c] = max(
                            0.0, float(m2.get(a, {}).get(b, {}).get(c, 0.0))
                        )

            stats = data.get("markov_stats", {})
            self.markov_total_preds = int(stats.get("total_preds", 0))
            self.markov_correct_preds = int(stats.get("correct_preds", 0))
            self.profile_loaded = True
            self.log(f"Loaded JSON ({self.matches_played} matches).", "good")
        except (OSError, ValueError, TypeError):
            self.log("Corrupt JSON ignored; starting blank.", "warn")

    def save_profile(self, player_won):
        # Adjust difficulty once per match based on the player outcome.
        self.matches_played += 1
        if player_won:
            self.player_wins += 1
            self.base_difficulty = clamp(self.base_difficulty + 0.08, 0.75, 1.45)
        else:
            self.ai_wins += 1
            self.base_difficulty = clamp(self.base_difficulty - 0.06, 0.75, 1.45)

        payload = {
            "matches_played": self.matches_played,
            "player_wins": self.player_wins,
            "ai_wins": self.ai_wins,
            "base_difficulty": round(self.base_difficulty, 3),
            "total_observations": self.total_observations,
            "q_updates": self.q_updates,
            "markov_stats": {
                "total_preds": self.markov_total_preds,
                "correct_preds": self.markov_correct_preds,
                "accuracy": round(self.markov_accuracy, 4),
            },
            "markov_1st": {
                r: {c: round(self.markov_1st[r][c], 4) for c in ACTIONS} for r in ACTIONS
            },
            "markov_2nd": {
                a: {
                    b: {c: round(self.markov_2nd[a][b][c], 4) for c in ACTIONS}
                    for b in ACTIONS
                }
                for a in ACTIONS
            },
            "q_fast": {r: {c: round(self.q_fast[r][c], 4) for c in ACTIONS} for r in ACTIONS},
            "q_slow": {r: {c: round(self.q_slow[r][c], 4) for c in ACTIONS} for r in ACTIONS},
            "visit_counts": {
                r: {c: int(self.visit_counts[r][c]) for c in ACTIONS} for r in ACTIONS
            },
        }
        try:
            with open(self.profile_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            self.profile_loaded = True
            self.log("Saved profile to JSON.", "good")
        except OSError:
            self.log("Error saving JSON profile.", "bad")

    def reset_profile(self):
        # Clear learned sequences, Q-values, visits, and match history.
        self.markov_1st = self._empty_2d(0.0)
        self.markov_2nd = {
            a: {b: {c: 0.0 for c in ACTIONS} for b in ACTIONS} for a in ACTIONS
        }
        self.q_fast = self._empty_2d(0.0)
        self.q_slow = self._empty_2d(0.0)
        self.visit_counts = self._empty_2d(0.0)
        self.recent_actions.clear()
        self.trace_queue.clear()
        self.next_decision_id = 1
        self.total_observations = 0
        self.matches_played = 0
        self.player_wins = 0
        self.ai_wins = 0
        self.base_difficulty = 1.0
        self.effective_difficulty = 1.0
        self.pending_prediction = None
        self.markov_total_preds = 0
        self.markov_correct_preds = 0
        self.q_updates = 0
        self.last_reward_val = 0.0
        self.last_reward_reason = "RESET TO ZERO"
        self.profile_loaded = False
        if os.path.exists(self.profile_path):
            try:
                os.remove(self.profile_path)
            except OSError:
                pass
        self.event_log.clear()
        self._update_entropy_and_weights()
        self.log("Wiped JSON; cold start active.", "warn")

    def reset_round(self):
        # Keep learned memory, but reset round-only prediction state.
        self.recent_actions.clear()
        self.trace_queue.clear()
        self.pending_prediction = None
        self._update_entropy_and_weights()
        self.log("New round started.", "info")

    def update_dda(self, hp_diff_norm):
        # Favor the player more strongly as AI gains an HP advantage.
        self.effective_difficulty = clamp(self.base_difficulty + hp_diff_norm * 0.35, 0.70, 1.50)

    def record_player_action(self, action):
        if action not in ACTIONS:
            return

        # Check whether the previous prediction was correct.
        if self.pending_prediction is not None:
            self.markov_total_preds += 1
            if self.pending_prediction == action:
                self.markov_correct_preds += 1

        # Update 2nd-order sequence if we have at least 2 prior actions
        if len(self.recent_actions) >= 2:
            prev2 = self.recent_actions[-2]
            prev1 = self.recent_actions[-1]
            row2 = self.markov_2nd[prev2][prev1]
            for k in ACTIONS:
                row2[k] *= self.markov_decay
            row2[action] += 1.0

        # Update 1st-order transition if we have at least 1 prior action
        if len(self.recent_actions) >= 1:
            prev1 = self.recent_actions[-1]
            row1 = self.markov_1st[prev1]
            for k in ACTIONS:
                row1[k] *= self.markov_decay
            row1[action] += 1.0

        self.recent_actions.append(action)
        self.total_observations += 1
        self._update_entropy_and_weights()

        pred, conf, _ = self.predict_next()
        self.pending_prediction = pred if conf > 0.25 else None

    def _update_entropy_and_weights(self):
        n = len(self.recent_actions)
        if n < 2:
            self.entropy_norm = 1.0
        else:
            # Measure how predictable the recent action pattern is.
            counts = {a: 0 for a in ACTIONS}
            for a in self.recent_actions:
                counts[a] += 1
            entropy = 0.0
            for a in ACTIONS:
                p = counts[a] / n
                if p > 0.0:
                    entropy -= p * math.log(p)
            max_entropy = math.log(len(ACTIONS))
            self.entropy_norm = clamp(entropy / max_entropy, 0.0, 1.0)

        if not self.slow_unlocked:
            self.w_fast = 1.0
            self.w_slow = 0.0
            return

        # Low entropy (predictable pattern) -> higher slow weight; high entropy -> higher fast weight
        slow_share = clamp(0.68 - 0.45 * self.entropy_norm, 0.15, 0.68)
        self.w_slow = slow_share
        self.w_fast = 1.0 - slow_share

    def predict_next(self):
        if len(self.recent_actions) >= 2:
            prev2 = self.recent_actions[-2]
            prev1 = self.recent_actions[-1]
            row2 = self.markov_2nd[prev2][prev1]
            total2 = sum(row2.values())
            if total2 >= 1.5:
                # Prefer 2nd-order prediction when enough data exists.
                denom = total2 + self.smoothing * len(ACTIONS)
                probs = {a: (row2[a] + self.smoothing) / denom for a in ACTIONS}
                best = max(ACTIONS, key=lambda a: probs[a])
                self.markov_order_used = 2
                self.last_pred_conf = probs[best]
                return best, probs[best], probs

        if len(self.recent_actions) >= 1:
            # Fall back to 1st-order prediction.
            prev1 = self.recent_actions[-1]
            row1 = self.markov_1st[prev1]
            denom = sum(row1.values()) + self.smoothing * len(ACTIONS)
            probs = {a: (row1[a] + self.smoothing) / denom for a in ACTIONS}
            best = max(ACTIONS, key=lambda a: probs[a])
            self.markov_order_used = 1
            self.last_pred_conf = probs[best]
            return best, probs[best], probs

        self.markov_order_used = 1
        self.last_pred_conf = 0.20
        return "light", 0.20, {a: 0.20 for a in ACTIONS}

    def get_combat_context(self, player_obs_ctx, dist):
        self.active_zone = get_distance_zone(dist)
        if player_obs_ctx in ACTIONS:
            # Use the player's currently observed action.
            self.active_ctx = player_obs_ctx
            self.active_ctx_source = "LIVE"
        else:
            # Predict the context when no live action is visible.
            pred, _, _ = self.predict_next()
            self.active_ctx = pred
            self.active_ctx_source = "PRED"
        return self.active_ctx, self.active_ctx_source, self.active_zone

    def blended_q(self, ctx, action):
        # Combine short-term and long-term learned values.
        qf = self.q_fast[ctx][action]
        qs = self.q_slow[ctx][action]
        return self.w_fast * qf + self.w_slow * qs

    def ucb_bonus(self, ctx, action):
        row_total = sum(self.visit_counts[ctx].values())
        if row_total <= 0.0:
            return 0.0
        # Higher Markov confidence lowers exploration bonus so strong predictions get exploited
        conf_scale = clamp(1.35 - 0.70 * self.last_pred_conf, 0.65, 1.35)
        n_sa = self.visit_counts[ctx][action]
        return self.c_base * conf_scale * math.sqrt(math.log(row_total + 1.0) / (n_sa + 0.5))

    def softmax_policy(self, ctx, allowed_actions=ACTIONS):
        tau = self.temperature
        # Combine learned value with exploration bonus.
        scores = {a: self.blended_q(ctx, a) + self.ucb_bonus(ctx, a) for a in allowed_actions}
        max_s = max(scores.values())
        exps = {a: math.exp((scores[a] - max_s) / tau) for a in allowed_actions}
        total = sum(exps.values()) or 1.0
        return {a: exps[a] / total for a in allowed_actions}

    def choose_action(self, ctx, allowed_actions):
        probs = self.softmax_policy(ctx, allowed_actions)
        acts = list(allowed_actions)
        weights = [probs[a] for a in acts]
        # Weighted randomness favors, but does not force, the best action.
        return random.choices(acts, weights=weights, k=1)[0]

    def register_trace(self, decision_id, ctx, action):
        # Remember recent decisions for delayed reward propagation.
        self.trace_queue.append((decision_id, ctx, action))

    def apply_reward(self, ctx, action, reward, reason, decision_id=None, relabeled=False):
        if ctx not in ACTIONS or action not in ACTIONS:
            return

        # Update the triggering decision directly.
        self._update_single_cell(ctx, action, reward)
        self.q_updates += 1
        self.last_reward_val = reward
        self.last_reward_reason = reason

        sign = "+" if reward >= 0 else ""
        tag = f"#{decision_id} " if decision_id is not None else ""
        star = "*" if relabeled else ""
        log_tag = "good" if reward > 0 else "bad"
        self.log(
            f"{tag}Q({SHORT[ctx]}{star},{SHORT[action]}) {sign}{reward:.2f} {reason}",
            log_tag,
        )

        # Propagate decayed eligibility trace to earlier steps in the recent sequence
        step = 1
        for prev_id, prev_ctx, prev_act in reversed(list(self.trace_queue)):
            if decision_id is not None and prev_id == decision_id:
                continue
            trace_r = reward * (self.trace_decay ** step)
            if abs(trace_r) >= 0.05:
                self._update_single_cell(prev_ctx, prev_act, trace_r)
                self.q_updates += 1
                self.log(
                    f"~#{prev_id} Q({SHORT[prev_ctx]},{SHORT[prev_act]}) {trace_r:+.2f} (trace)",
                    "info",
                )
            step += 1

    def _update_single_cell(self, ctx, action, reward):
        old_f = self.q_fast[ctx][action]
        # Move fast memory toward the observed reward.
        self.q_fast[ctx][action] = clamp(old_f + self.alpha_fast * (reward - old_f), -2.5, 2.5)
        if self.slow_unlocked:
            old_s = self.q_slow[ctx][action]
            self.q_slow[ctx][action] = clamp(old_s + self.alpha_slow * (reward - old_s), -2.5, 2.5)
        self.visit_counts[ctx][action] += 1.0

