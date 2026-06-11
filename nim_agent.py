import copy
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from state import State


class NimGame(State):
    """
    Nim game state.

    Players alternate removing any positive number of sticks from a single pile.
    The player who takes the last stick loses.

    You may not change the takeAction, isTerminal, getValue,
    getActions functions.
    """

    def __init__(self, Player=0, the_sticks=None):
        """
        Initialize the Nim game state.

        Input  : Player (int)       — 0 for the MAX player, 1 for the MIN player.
                 the_sticks (list)  — list of pile sizes (positive integers);
                 when omitted a random starting position is generated:
                 2–4 piles, each containing 1–6 sticks.
        Output : None
        Example: NimGame(0, [3, 5, 7])  →  the MAX player to move, three piles of 3, 5, 7
                 NimGame()              →  the MAX player to move, random starting position
        """
        State.__init__(self, Player)
        if the_sticks is None:
            num_piles = random.randint(2, 4)
            the_sticks = [random.randint(1, 6) for _ in range(num_piles)]
        self.sticks = the_sticks

    def getActions(self):
        """
        Return all legal moves as (amount, pile_index) pairs.

        A move removes 'amount' sticks (≥ 1) from the pile at 'pile_index'.

        Input  : None
        Output : list of (int, int) — every (k, j) where 1 ≤ k ≤ sticks[j].
        Example: getActions() with sticks=[2, 3]
                 →  [(1,0),(2,0),(1,1),(2,1),(3,1)]
                 getActions() with sticks=[]  →  []
        """
        if len(self.sticks) == 0:
            return []
        the_actions = []
        for j in range(len(self.sticks)):
            for i in range(self.sticks[j]):
                the_actions.append((i + 1, j))
        return the_actions

    def takeAction(self, action):
        """
        Remove sticks according to action and return the resulting state.

        Input  : action (int, int) — (amount, pile_index): remove 'amount' sticks
                 from the pile at 'pile_index'.
        Output : NimGame — new state with the pile reduced and the other player
                 to move.  If the pile reaches zero it is removed entirely.
        Example: takeAction((2, 0)) on NimGame(0, [3, 5])
                 →  NimGame(1, [1, 5])
                 takeAction((3, 0)) on NimGame(0, [3])
                 →  NimGame(1, [0])   (pile 0 is now empty)
        """
        new_sticks = copy.deepcopy(self.sticks)
        new_sticks[action[1]] -= action[0]
        return NimGame(1 if self.player == 0 else 0, new_sticks)

    def isTerminal(self):
        """
        Return True when no legal moves remain (all piles are empty).

        Input  : None
        Output : bool
        Example: isTerminal() on NimGame(0, [])   →  True
                 isTerminal() on NimGame(0, [1])  →  False
        """
        return self.getActions() == []

    def getValue(self):
        """
        Return the utility of this terminal state from the MAX player's perspective.

        Under the misère convention the player who takes the last stick loses.
        Equivalently, the player whose turn it is when no moves remain has won,
        because the opponent just took the last stick.

        Only call this when isTerminal() is True.
        Input  : None
        Output : int — +1 if the MAX player wins, -1 if the MIN player wins.
        Example: getValue() when it is the MAX player's turn and all piles are empty
                 →  1  (the MIN player took the last stick, so the MAX player wins)
        """
        if not self.isTerminal():
            raise ValueError("getValue() called on a non-terminal Nim state.")
        return 1 if self.player == 0 else -1

    def Evaluate(self):
        """
        Return a heuristic estimate of the state's value for the MAX player.

        *** THIS IS A STARTING IMPLEMENTATION. ***
        You are strongly encouraged to replace or extend it.
        The current version always returns 0, giving no guidance to the search.

        Positive values favour the MAX player; negative values favour the MIN player.

        Ideas for a better heuristic:
          - Compute the XOR (nim-sum) of all pile sizes.  In standard Nim a
            nim-sum of 0 is a losing position for the player to move; in the
            misère variant the rule differs only when all piles have size ≤ 1.
          - Penalise (or reward) positions where one pile is very large and the
            rest are small.
          - You can also randomise among equally-scored actions in getBestMove()
            rather than always picking the first one found.

        Input  : None
        Output : float — getValue() if the state is terminal; otherwise a
                 heuristic score where positive values favour the MAX player
                 and negative values favour the MIN player.
        Example: Evaluate() on a terminal state won by the MAX player  →  1
                 Evaluate() on any non-terminal board                  →  0.0  (with this starter code)
        """
        if self.isTerminal():
            return self.getValue()
        return 0

    def getBestMove(self):
        """
        Return a greedy best action using the current Evaluate() function.

        The MAX player picks the action that maximises Evaluate() on the successor state;
        the MIN player picks the action that minimises it.  Ties are broken by the order
        in which getActions() returns moves (deterministic).

        *** YOU ARE ENCOURAGED TO MODIFY THIS FUNCTION. ***
        For example, you could:
          - Sample actions with probability proportional to their Evaluate() scores.
          - Use a deeper look-ahead (call getBestMove recursively or implement
            minimax / alpha-beta search).
          - Combine Evaluate() with getValue() when the successor is terminal.

        Input  : None
        Output : (int, int) | None — the chosen (amount, pile_index) action,
                 or None if the state is terminal.
        Example: getBestMove() on NimGame(0, [3, 5, 7])
                 →  some (amount, pile_index) pair  (all evaluate to 0 with
                    the default heuristic, so the first action is returned)
                 getBestMove() on NimGame(0, [])  →  None
        """
        if self.isTerminal():
            return None
        best_action = None
        best_value = -float('inf') if self.player == 0 else float('inf')
        for action in self.getActions():
            value = self.takeAction(action).Evaluate()
            if self.player == 0 and value > best_value:
                best_value = value
                best_action = action
            elif self.player == 1 and value < best_value:
                best_value = value
                best_action = action
        return best_action

    def __repr__(self):
        """
        Return a human-readable string of the game state.

        Input  : None
        Output : str — player turn and pile sizes.
        Example: repr(NimGame(0, [3, 5, 7]))
                 →  "Player 1's turn. Sticks: 3 5 7"
        """
        return "Player " + str(self.player + 1) + "'s turn. Sticks: " + " ".join(str(s) for s in self.sticks)
