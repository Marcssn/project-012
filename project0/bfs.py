from collections import deque
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


class PacmanAgent(Agent):
    """
    A Pacman agent based on Breadth-First-Search.
    """

    def __init__(self, args):
        self.moves = []

    def get_action(self, state):
        if not self.moves:
            self.moves = self.bfs(state)

        try:
            return self.moves.pop(0)
        except IndexError:
            return Directions.STOP

    def bfs(self, state):
        """
        Given a pacman game state,
        returns a list of legal moves to solve the search layout using BFS.
        """
        path = []
        queue = deque([(state, path)])
        closed = set()

        while queue:
            current, path = queue.popleft()

            if current.isWin():
                return path

            current_key = key(current)
            if current_key not in closed:
                closed.add(current_key)

                for next_state, action in current.generatePacmanSuccessors():
                    next_key = key(next_state)
                    if next_key not in closed:
                        queue.append((next_state, path + [action]))

        return []
