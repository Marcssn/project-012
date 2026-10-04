from pacman_module.game import Agent, Directions


def evaluation_function(state):
    """
    Evaluates the game state when the cutoff depth is reached.
    """
    if state.isWin():
        return float('inf')
    if state.isLose():
        return float('-inf')

    score = state.getScore()
    pacman_pos = state.getPacmanPosition()
    ghost_pos = state.getGhostPosition(1)

    ghost_dist = abs(pacman_pos[0] - ghost_pos[0]) + abs(
        pacman_pos[1] - ghost_pos[1]
    )

    food_list = state.getFood().asList()
    if food_list:
        min_food_dist = min(
            abs(pacman_pos[0] - f[0]) + abs(pacman_pos[1] - f[1])
            for f in food_list
        )
    else:
        min_food_dist = 0

    eval_score = (
        1.5 * score
        - 1.5 * min_food_dist
        - 3.0 * len(food_list)
    )

    if ghost_dist <= 1:
        eval_score -= 500
    elif ghost_dist <= 2:
        eval_score -= 100

    return eval_score


class PacmanAgent(Agent):
    """
    Pacman agent based on Heuristic Minimax (H-Minimax).
    """

    def __init__(self, depth=4):
        super().__init__()
        self.depth = depth

    def get_action(self, state):
        """
        Selects the best legal move using Depth-Limited H-Minimax.
        """
        best_action = Directions.STOP
        best_val = float('-inf')

        for succ, action in state.generatePacmanSuccessors():
            val = self.min_value(succ, 1, 0)
            if val > best_val:
                best_val = val
                best_action = action

        return best_action

    def max_value(self, state, current_depth):
        """
        Pacman (Maximizer) node.
        """
        if state.isWin() or state.isLose() or current_depth == self.depth:
            return evaluation_function(state)

        v = float('-inf')
        for succ, _ in state.generatePacmanSuccessors():
            v = max(v, self.min_value(succ, 1, current_depth))
        return v

    def min_value(self, state, ghost_index, current_depth):
        """
        Ghost (Minimizer) node.
        """
        if state.isWin() or state.isLose():
            return evaluation_function(state)

        v = float('inf')
        successors = state.generateGhostSuccessors(ghost_index)
        if not successors:
            return evaluation_function(state)

        for succ, _ in successors:
            v = min(v, self.max_value(succ, current_depth + 1))
        return v
