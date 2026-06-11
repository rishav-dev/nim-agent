import math
import random
import time


class CustomAgent:
    """
    Fast adversarial-search Nim agent.

    Uses alpha-beta search with a strong misere-Nim evaluation function.  The
    evaluation is also used as a safe fallback so the agent stays well under the
    1-second-per-action limit.
    """

    def __init__(self):
        self.time_limit = 0.90
        self.deadline = 0.0
        self.cache = {}
        self.search_depth = 2
        self.max_actions_for_search = 30

    def takeAction(self, state):
        self.deadline = time.perf_counter() + self.time_limit
        self.cache = {}

        sticks = getattr(state, "sticks", None)

        # Nim-specific fast path. This avoids expensive enumeration if a hidden
        # test uses larger piles, while still returning a legal Nim action.
        if sticks is not None:
            direct = self._direct_misere_nim_move(state)
            if direct is None:
                return None

            # For normal starter boards, enumerate actions and use alpha-beta.
            # If there are too many actions, use the exact direct fallback.
            total_actions = sum(s for s in sticks if s > 0)
            if total_actions > self.max_actions_for_search:
                return direct

            try:
                actions = state.getActions()
                if not actions:
                    return None
                searched = self._best_action_alpha_beta(state, actions)
                return searched if searched in actions else direct
            except Exception:
                return direct

        # Fallback for any state with the required interface.
        try:
            actions = state.getActions()
            if not actions:
                return None
            searched = self._best_action_alpha_beta(state, actions)
            return searched if searched in actions else random.choice(actions)
        except Exception:
            return None


    # Alpha-beta adversarial search

    def _best_action_alpha_beta(self, state, actions):
        ordered = self._ordered_actions(state, actions)
        best_action = ordered[0]

        if state.player == 0:  # MAX player
            best_value = -math.inf
            alpha, beta = -math.inf, math.inf
            for action in ordered:
                self._check_time()
                value = self._alpha_beta(state.takeAction(action), self.search_depth - 1, alpha, beta)
                if value > best_value:
                    best_value = value
                    best_action = action
                alpha = max(alpha, best_value)
                if alpha >= beta:
                    break
        else:  # MIN player
            best_value = math.inf
            alpha, beta = -math.inf, math.inf
            for action in ordered:
                self._check_time()
                value = self._alpha_beta(state.takeAction(action), self.search_depth - 1, alpha, beta)
                if value < best_value:
                    best_value = value
                    best_action = action
                beta = min(beta, best_value)
                if alpha >= beta:
                    break

        return best_action

    def _alpha_beta(self, state, depth, alpha, beta):
        self._check_time()

        if state.isTerminal():
            return state.getValue()

        if depth <= 0:
            return self._evaluate(state)

        key = (self._state_key(state), depth)
        if key in self.cache:
            return self.cache[key]

        actions = self._ordered_actions(state, state.getActions())

        if state.player == 0:
            value = -math.inf
            for action in actions:
                value = max(value, self._alpha_beta(state.takeAction(action), depth - 1, alpha, beta))
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
        else:
            value = math.inf
            for action in actions:
                value = min(value, self._alpha_beta(state.takeAction(action), depth - 1, alpha, beta))
                beta = min(beta, value)
                if alpha >= beta:
                    break

        self.cache[key] = value
        return value


    # Evaluation function and move ordering

    def _evaluate(self, state):
        """Evaluation from MAX's perspective: positive favors MAX."""
        if state.isTerminal():
            return state.getValue()

        sticks = getattr(state, "sticks", None)
        if sticks is None:
            try:
                return float(state.Evaluate())
            except Exception:
                return 0.0

        player_to_move_wins = self._player_to_move_wins(sticks)

        if state.player == 0:
            return 1.0 if player_to_move_wins else -1.0
        else:
            return -1.0 if player_to_move_wins else 1.0

    def _ordered_actions(self, state, actions):
        scored = []
        for action in actions:
            try:
                child = state.takeAction(action)
                value = self._evaluate(child)
                trap = self._trap_score(child)
                score = (value, trap) if state.player == 0 else (-value, trap)
            except Exception:
                score = (0.0, 0.0)
            scored.append((score, action))

        scored.sort(reverse=True)
        return [action for _, action in scored]

    def _trap_score(self, state):
        """
        Tie-breaker.  Prefer positions where the opponent has fewer winning
        replies, increasing the chance that an imperfect agent makes a mistake.
        """
        if state.isTerminal():
            return 100.0

        sticks = getattr(state, "sticks", None)
        if sticks is not None and sum(s for s in sticks if s > 0) > self.max_actions_for_search:
            return 0.0

        try:
            actions = state.getActions()
        except Exception:
            return 0.0

        if not actions:
            return 100.0

        opponent = state.player
        winning = 0
        for action in actions:
            child_value = self._evaluate(state.takeAction(action))
            if (opponent == 0 and child_value > 0) or (opponent == 1 and child_value < 0):
                winning += 1

        # Smaller winning-reply ratio is better.  Slightly prefer giving more
        # total choices when the ratio is tied.
        return -(winning / float(len(actions))) + 0.001 * len(actions)

    # Misere Nim logic used by the evaluation/fallback

    def _player_to_move_wins(self, sticks):
        piles = [s for s in sticks if s > 0]

        if not piles:
            return True

        if all(s == 1 for s in piles):
            return len(piles) % 2 == 0
        
        nim_sum = 0
        for s in piles:
            nim_sum ^= s
        return nim_sum != 0

    def _direct_misere_nim_move(self, state):
        """Return a legal strong move directly from the pile list."""
        sticks = getattr(state, "sticks", None)
        if sticks is None:
            return None

        nonzero = [(i, s) for i, s in enumerate(sticks) if s > 0]
        if not nonzero:
            return None

        ones = [(i, s) for i, s in nonzero if s == 1]
        big = [(i, s) for i, s in nonzero if s > 1]

        if not big:
            return (1, nonzero[-1][0])

        if len(big) == 1:
            index, size = big[0]
            target = 1 if len(ones) % 2 == 0 else 0
            return (size - target, index)

        nim_sum = 0
        for _, size in nonzero:
            nim_sum ^= size

        if nim_sum != 0:
            for index, size in nonzero:
                target = size ^ nim_sum
                if target < size:
                    return (size - target, index)

        # Losing position against perfect play.
        index, _ = max(nonzero, key=lambda item: item[1])
        return (1, index)

    # Utility helpers

    def _state_key(self, state):
        sticks = getattr(state, "sticks", None)
        if sticks is not None:
            return (tuple(sorted(s for s in sticks if s > 0)), state.player)
        return (repr(state), state.player)

    def _check_time(self):
        if time.perf_counter() >= self.deadline:
            raise TimeoutError