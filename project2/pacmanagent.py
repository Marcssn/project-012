# Complete this class for all parts of the project

import numpy as np
from pacman_module import util
from pacman_module.game import Actions, Agent
from pacman_module.pacman import Directions


class PacmanAgent(Agent):
    """
    Pacman agent that uses the belief state to hunt edible invisible ghosts.
    """

    def __init__(self, args):
        """
        Arguments:
        ----------
        - `args`: Namespace of arguments from command-line prompt.
        """
        self.args = args

    def get_action(self, state, belief_state):
        """
        Given a pacman game state and a belief state,
        returns a legal move.

        Arguments:
        ----------
        - `state`: the current game state.
        - `belief_state`: a list of probability matrices.

        Return:
        -------
        - A legal move as defined in `game.Directions`.
        """
        legal = state.getLegalPacmanActions()
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)

        if not legal:
            return Directions.STOP

        pacman_pos = state.getPacmanPosition()

        # Find the most likely position across all ghosts
        target_pos = None
        max_prob = -1.0

        for b in belief_state:
            prob = np.max(b)
            if prob > max_prob and prob > 0:
                max_prob = prob
                target_coords = np.unravel_index(np.argmax(b), b.shape)
                target_pos = (target_coords[0], target_coords[1])

        if target_pos is None:
            return legal[0]

        # Choose the action that minimizes distance to the target position
        best_action = legal[0]
        min_dist = float('inf')

        for action in legal:
            dx, dy = Actions.directionToVector(action)
            next_pos = (pacman_pos[0] + int(dx), pacman_pos[1] + int(dy))
            dist = util.manhattanDistance(next_pos, target_pos)
            if dist < min_dist:
                min_dist = dist
                best_action = action

        return best_action
