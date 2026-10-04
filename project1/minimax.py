from pacman_module.game import Agent, Directions


def key(state):
    """
    Returns a unique identifier for a game state.
    """
    return (
        state.getPacmanPosition(),
        state.getGhostPosition(1),
        state.getGhostDirection(1),
        state.getFood(),
        tuple(state.getCapsules())
    )


class PacmanAgent(Agent):
    """
    Pacman agent based on Full Minimax Search.
    """

    def __init__(self):
        super().__init__()

    def get_action(self, state):
        """
        Given a Pacman game state, returns a legal move using Minimax.
        """
        best_action = Directions.STOP
        best_val = float('-inf')

        for succ, action in state.generatePacmanSuccessors():
            val = self.min_value(succ, 1)
            if val > best_val:
                best_val = val
                best_action = action

        return best_action

    def max_value(self, state):
        """
        Pacman (MAX node) calculates the highest score possible.
        """
        if state.isWin() or state.isLose():
            return state.getScore()

        v = float('-inf')
        for succ, _ in state.generatePacmanSuccessors():
            v = max(v, self.min_value(succ, 1))
        return v

    def min_value(self, state, ghost_index):
        """
        Ghost (MIN node) calculates the lowest score possible.
        """
        if state.isWin() or state.isLose():
            return state.getScore()

        v = float('inf')
        successors = state.generateGhostSuccessors(ghost_index)
        if not successors:
            return state.getScore()

        for succ, _ in successors:
            v = min(v, self.max_value(succ))
        return v
