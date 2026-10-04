# Complete this class for all parts of the project

from pacman_module.game import Agent
import numpy as np
from pacman_module import util
from scipy.stats import binom


class BeliefStateAgent(Agent):
    """
    BeliefStateAgent tracks ghost positions using an Exact Bayes Filter.
    """

    def __init__(self, args):
        """
        Arguments:
        ----------
        - `args`: Namespace of arguments from command-line prompt.
        """
        self.args = args

        # Current list of belief states over ghost positions
        self.beliefGhostStates = None

        # Grid of walls (assigned with 'state.getWalls()' method)
        self.walls = None

        # Hyper-parameters
        self.ghost_type = self.args.ghostagent
        self.sensor_variance = self.args.sensorvariance

        self.p = 0.5
        self.n = int(self.sensor_variance / (self.p * (1 - self.p)))

        # Cache transition model to prevent redundant re-computation
        self._transition_cache = {}

        # Lists for storing metric histories across steps
        self.entropy_history = []
        self.error_history = []

    def _get_sensor_model(self, pacman_position, evidence):
        """
        Arguments:
        ----------
        - `pacman_position`: 2D coordinates position of pacman at state x_{t}
        - `evidence`: float noisy distance observed by pacman

        Return:
        -------
        The sensor model represented as a 2D numpy array of size
        [width, height].
        Element at (w, h) is P(E_t=evidence | X_t=(w, h)).
        """
        width = self.walls.width
        height = self.walls.height
        sensor_model = np.zeros((width, height))

        for w in range(width):
            for h in range(height):
                if not self.walls[w][h]:
                    true_dist = util.manhattanDistance(
                        (w, h), pacman_position
                    )
                    # Noise: evidence = true_distance + noise
                    # noise = k - n * p ==> k = evidence - true_dist + n * p
                    k = round(evidence - true_dist + self.n * self.p)
                    if 0 <= k <= self.n:
                        sensor_model[w, h] = binom.pmf(k, self.n, self.p)

        return sensor_model

    def _get_transition_model(self, pacman_position):
        """
        Arguments:
        ----------
        - `pacman_position`: 2D coordinates position of pacman at state x_{t}

        Return:
        -------
        The transition model represented as a 4D numpy array of
        size [width, height, width, height].
        Element at (w1, h1, w2, h2) is P(X_{t+1}=(w1, h1) | X_t=(w2, h2)).
        """
        if pacman_position in self._transition_cache:
            return self._transition_cache[pacman_position]

        width = self.walls.width
        height = self.walls.height
        t_model = np.zeros((width, height, width, height))

        # Ghost parameter mapping based on ghostAgents.py
        param_map = {
            "scared": 8.0,
            "afraid": 2.0,
            "confused": 1.0
        }
        gamma = param_map.get(self.ghost_type, 1.0)

        for w2 in range(width):
            for h2 in range(height):
                if self.walls[w2][h2]:
                    continue

                # Legal actions for ghosts: N, S, E, W
                neighbors = []
                for action in [
                    ((w2, h2 + 1)),
                    ((w2, h2 - 1)),
                    ((w2 + 1, h2)),
                    ((w2 - 1, h2))
                ]:
                    nw, nh = action
                    if 0 <= nw < width and 0 <= nh < height:
                        if not self.walls[nw][nh]:
                            neighbors.append((nw, nh))

                if not neighbors:
                    t_model[w2, h2, w2, h2] = 1.0
                    continue

                if self.ghost_type == "confused":
                    # Uniform distribution over legal moves
                    prob = 1.0 / len(neighbors)
                    for nw, nh in neighbors:
                        t_model[nw, nh, w2, h2] = prob
                else:
                    # Weight by distance to pacman raised to gamma
                    weights = []
                    for nw, nh in neighbors:
                        dist = util.manhattanDistance(
                            (nw, nh), pacman_position
                        )
                        weights.append(dist ** gamma)

                    total_weight = sum(weights)
                    if total_weight > 0:
                        for idx, (nw, nh) in enumerate(neighbors):
                            t_model[nw, nh, w2, h2] = (
                                weights[idx] / total_weight
                            )
                    else:
                        prob = 1.0 / len(neighbors)
                        for nw, nh in neighbors:
                            t_model[nw, nh, w2, h2] = prob

        self._transition_cache[pacman_position] = t_model
        return t_model

    def _get_updated_belief(
        self, belief, evidences, pacman_position, ghosts_eaten
    ):
        """
        Calculates updated belief distribution via Bayes filter.
        """
        width = self.walls.width
        height = self.walls.height
        updated_beliefs = []

        transition_model = self._get_transition_model(pacman_position)

        for idx, b in enumerate(belief):
            if ghosts_eaten[idx]:
                updated_beliefs.append(np.zeros((width, height)))
                continue

            # Step 1: Predict (Time update / Propagation)
            # P(X_{t+1}) = sum_{X_t} P(X_{t+1} | X_t) * P(X_t)
            # transition_model has shape (w1, h1, w2, h2)
            predicted_belief = np.tensordot(
                transition_model, b, axes=((2, 3), (0, 1))
            )

            # Step 2: Update (Sensor observation)
            sensor_model = self._get_sensor_model(
                pacman_position, evidences[idx]
            )
            unnormalized_belief = predicted_belief * sensor_model

            # Normalization
            total = np.sum(unnormalized_belief)
            if total > 0:
                normalized_belief = unnormalized_belief / total
            else:
                # Fallback to uniform distribution over open spaces
                non_wall = 1.0 - np.array(self.walls.data, dtype=float)
                normalized_belief = non_wall / np.sum(non_wall)

            updated_beliefs.append(normalized_belief)

        return updated_beliefs

    def update_belief_state(self, evidences, pacman_position, ghosts_eaten):
        """
        Given a list of (noised) distances from pacman to ghosts,
        returns a list of belief states about ghosts positions.
        """
        belief = self._get_updated_belief(
            self.beliefGhostStates, evidences, pacman_position, ghosts_eaten
        )
        self.beliefGhostStates = belief
        return belief

    def _get_evidence(self, state):
        """
        Computes noisy distances between pacman and ghosts.
        """
        positions = state.getGhostPositions()
        pacman_position = state.getPacmanPosition()
        noisy_distances = []

        for pos in positions:
            true_distance = util.manhattanDistance(pos, pacman_position)
            noise = binom.rvs(self.n, self.p) - self.n * self.p
            noisy_distances.append(true_distance + noise)

        return noisy_distances

    def _record_metrics(self, belief_states, state):
        """
        Records Shannon Entropy (Uncertainty) and Mean Distance Error (Quality)
        at each time step.
        """
        ghost_positions = state.getGhostPositions()
        layout_name = self.args.layout

        for idx, b in enumerate(belief_states):
            # 1. Uncertainty: Shannon Entropy
            flat_b = b[b > 0]
            entropy = float(-np.sum(flat_b * np.log2(flat_b)))
            self.entropy_history.append(entropy)

            # 2. Quality: Mean Distance Error compared to ground truth
            mean_error = 0.0
            if idx < len(ghost_positions):
                true_pos = ghost_positions[idx]
                width, height = b.shape
                for w in range(width):
                    for h in range(height):
                        if b[w, h] > 0:
                            dist = util.manhattanDistance((w, h), true_pos)
                            mean_error += b[w, h] * dist
                self.error_history.append(mean_error)

            # บันทึกลง metrics.csv สำหรับนำไปใช้สร้างกราฟในรายงาน
            with open("metrics.csv", "a") as f:
                f.write(
                    f"{layout_name},{self.ghost_type},{entropy},{mean_error}\n"
                )

    def get_action(self, state):
        """
        Given a pacman game state, returns a belief state.
        """
        if self.beliefGhostStates is None:
            self.beliefGhostStates = state.getGhostBeliefStates()
        if self.walls is None:
            self.walls = state.getWalls()

        evidence = self._get_evidence(state)
        newBeliefStates = self.update_belief_state(
            evidence, state.getPacmanPosition(), state.data._eaten[1:]
        )
        self._record_metrics(self.beliefGhostStates, state)

        return newBeliefStates, evidence
