# Misère Nim Agent

A Python agent for misère Nim using iterative-deepening minimax, alpha-beta pruning, transposition caching, and move ordering under a one-second server deadline.

In misère Nim, players alternate removing sticks from a single pile, and the player who takes the last stick loses.

## How the agent chooses a move

`custom_agent.py` contains `CustomAgent.takeAction`, which does this on every turn:

1. **Compute a fallback first.** Before any search, the agent computes a move from the solved misère strategy: play normal Nim (move to nim-sum zero) until only one pile has more than one stick, then leave an odd number of one-stick piles. This takes microseconds and is always a legal move.
2. **Search with a time limit.** Iterative-deepening minimax with alpha-beta pruning, with a transposition cache keyed on the sorted pile sizes, since pile order does not change the game. Moves are ordered with a misère-specific evaluator so pruning cuts more branches. Each completed depth replaces the fallback with a better answer.
3. **Stop on the clock.** A deadline check raises a timeout inside the search. The agent catches it and returns the best move from the last completed depth.

## Why 0.82 seconds

The server allows one second per move. Python startup, state copying, and the call itself are overhead the agent does not control, so it budgets 0.82 seconds for search and leaves the rest as margin. Because a fallback move already exists, a slow board costs search depth, never a forfeited move. Very large boards (more than 60 sticks or 60 actions) skip the search and return the fallback directly.

## Files

```
custom_agent.py   the agent
nim_agent.py      NimGame state class provided with the assignment
state.py          base State class provided with the assignment
```

`nim_agent.py` and `state.py` are the course starter code and define the game rules. The agent is in `custom_agent.py`.

## Run it

No packages beyond the standard library are needed.

```python
from nim_agent import NimGame
from custom_agent import CustomAgent

game = NimGame(0, [3, 5, 7])
print(CustomAgent().takeAction(game))   # (amount, pile_index)
```

Built for a game-playing course at UMass Amherst.
