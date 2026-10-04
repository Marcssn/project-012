import heapq
from pacman_module.game import Agent
from pacman_module.pacman import Directions


def key(state):
    """
    Returns a key that uniquely identifies a Pacman game state.
    """
    return (
        state.getPacmanPosition(),
        state.getFood(),
        tuple(state.getCapsules())
    )


def heuristic(state):
    """
    Admissible and consistent heuristic for A* search.
    Computes the maximum Manhattan distance to any remaining food dot.
    """
    pos = state.getPacmanPosition()
    food_grid = state.getFood()
    food_list = food_grid.asList()

    if not food_list:
        return 0

    return max(abs(pos[0] - f[0]) + abs(pos[1] - f[1]) for f in food_list)


class PacmanAgent(Agent):
    """
    A Pacman agent based on A* Search.
    """

    def __init__(self, args):
        self.moves = []

    def get_action(self, state):
        if not self.moves:
            self.moves = self.astar(state)

        try:
            return self.moves.pop(0)
        except IndexError:
            return Directions.STOP

    def astar(self, state):
        """
        Given a pacman game state,
        returns a list of legal moves to solve the search layout using A*.
        """
        counter = 0
        h_start = heuristic(state)
        fringe = [(h_start, counter, state, [], 0)]
        heapq.heapify(fringe)

        cost_so_far = {key(state): 0}

        while fringe:
            f, _, current, path, g = heapq.heappop(fringe)

            if current.isWin():
                return path

            current_key = key(current)
            if cost_so_far.get(current_key, float('inf')) < g:
                continue

            for next_state, action in current.generatePacmanSuccessors():
                next_key = key(next_state)
                new_g = g + 1

                if new_g < cost_so_far.get(next_key, float('inf')):
                    cost_so_far[next_key] = new_g
                    counter += 1
                    f_cost = new_g + heuristic(next_state)
                    entry = (
                        f_cost, counter, next_state, path + [action], new_g
                    )
                    heapq.heappush(fringe, entry)

        return []
