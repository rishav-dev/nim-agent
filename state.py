class State:
    """
    Abstract base class representing a node in an extensive-form game tree.

    Player 0 is the MAX player; player 1 is the MIN player.
    Subclasses must override takeAction, isTerminal, getValue, and getActions.
    Evaluate and printState are optional but recommended for depth-limited search
    and debugging, respectively.

    In general, you may not change the takeAction, isTerminal, getValue, getActions functions.
    """

    def __init__(self, Player=0):
        """
        Initialize the state with the index of the player to move.

        Input  : Player (int) — 0 for the MAX player, 1 for the MIN player.
        Output : None
        Example: s = State(Player=0)  →  s.player == 0
        """
        self.player = Player

    def takeAction(self, action):
        """
        Return the successor state that results from applying action.

        Input  : action — an element of getActions(); its type is game-specific.
        Output : State  — a new State instance representing the next game position.
        Example: next_state = state.takeAction('call')  →  new PokerState after calling
        """
        pass

    def isTerminal(self):
        """
        Return True if the game has ended at this state and no further moves exist.

        Input  : None
        Output : bool — True when the game is over, False otherwise.
        Example: state.isTerminal()  →  True if a player has won or drawn
        """
        return self.actions == []

    def getValue(self):
        """
        Return the utility of this terminal state from Player 0's perspective.

        Should only be called when isTerminal() is True.
        Input  : None
        Output : numeric — positive values favour Player 0, negative values favour Player 1.
        Example: state.getValue()  →  1 (Player 0 wins), -1 (Player 1 wins), 0 (draw)
        """
        return 0

    def printState(self):
        """
        Print a human-readable representation of the current game state.

        Input  : None
        Output : None (side-effect: prints to stdout)
        Example: state.printState()  →  displays the board or hand to the console
        """
        pass

    def getActions(self):
        """
        Return the list of legal actions available to the player to move.

        Input  : None
        Output : list — each element is a valid argument to takeAction.
        Example: state.getActions()  →  [(0,0),(0,1),(1,0)] for a 2×2 Tic-Tac-Toe board
        """
        pass

    def Evaluate(self):
        """
        Return a heuristic estimate of the state's value for use in depth-limited search.

        Unlike getValue(), this may be called on non-terminal states.
        Positive values favor Player 0; negative values favor Player 1.
        Input  : None
        Output : numeric — heuristic score in the same range as getValue().
        Example: state.Evaluate()  →  0.6 (Player 0 has a material advantage in chess)
        """
        return 0
