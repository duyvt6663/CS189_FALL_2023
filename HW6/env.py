import random
from random import shuffle, randrange
import numpy as np
import matplotlib.pyplot as plt

class Env:
    def __init__(self, rows: int, columns: int, openness: float, epsilon: float = 0.0):
        self.rows = rows
        self.columns = columns
        self.epsilon = epsilon
        self.openness = openness
        self.nodes : dict[tuple[int, int], Node] = {}

        self._create_env()

    def init_env(self):
        """
        Call init_env before collecting any trajectories / stepping through the environment.
        """
        self.hidden_state = None
        self.states_so_far = []

    def _create_env(self) -> None:
        """
        This helper function creates the actual environment that the agent navigates in.
        Feel free to read this code if you want; however, DO NOT CHANGE THIS IN ANY WAY
        Feel free to also skip it; you can complete this assignment without understanding
        how exactly the environment was generated.

        Maze generation code taken from:
        https://rosettacode.org/wiki/Maze_generation
        """
        vis = [[0] * self.columns + [1] for _ in range(self.rows)] + [[1] * (self.columns + 1)]
        ver = [["| "] * self.columns + ['|'] for _ in range(self.rows)] + [[]]
        hor = [["+-"] * self.columns + ['+'] for _ in range(self.rows + 1)]

        def walk(x, y):
            vis[y][x] = 1

            d = [(x - 1, y), (x, y + 1), (x + 1, y), (x, y - 1)]
            shuffle(d)
            for (xx, yy) in d:
                if vis[yy][xx]: continue
                if xx == x: hor[max(y, yy)][x] = "+ "
                if yy == y: ver[y][max(x, xx)] = "  "
                walk(xx, yy)

        walk(randrange(self.columns), randrange(self.rows))

        rows = []
        for i, (a, b) in enumerate(zip(hor, ver)):
            if 0 < i < self.rows:
                for j in range(1, len(a) - 1):
                    if a[j] == "+-" and random.random() < self.openness:
                        a[j] = "+ "
            if 0 <= i < self.rows:
                for j in range(1, len(b) - 1):
                    if b[j] == "| " and random.random() < self.openness:
                        b[j] = "  "
            rows.append(list(''.join(a)))
            rows.append(list(''.join(b)))
        rows.pop()
        self.maze = rows
    
        for i, idx1 in enumerate(range(1, len(self.maze), 2)):
            for j, idx2 in enumerate(range(1, len(self.maze[idx1]), 2)):
                node = Node(i, j)
                # if wall is present, then the sensor is blocked by setting it to 1
                # while in the description, it is mentioned that the sensor is blocked by setting it to 0
                if self.maze[idx1][idx2 - 1] == "|":
                    node.blocked["l"] = 1 
                if self.maze[idx1][idx2 + 1] == "|":
                    node.blocked["r"] = 1
                if self.maze[idx1 - 1][idx2] == "-":
                    node.blocked["u"] = 1
                if self.maze[idx1 + 1][idx2] == "-":
                    node.blocked["d"] = 1
                self.nodes[(i, j)] = node

                if i > 0:
                    self.nodes[(i - 1, j)].neighbors.append(node)
                    node.neighbors.append(self.nodes[(i - 1, j)])
                if j > 0:
                    self.nodes[(i, j - 1)].neighbors.append(node)
                    node.neighbors.append(self.nodes[(i, j - 1)])

    def set_epsilon(self, epsilon: float) -> None:
        self.epsilon = epsilon

    def get_neighbors(self, i: int, j: int) -> np.ndarray:
        """
        Returns all the states that are accessible from state (i, j)
        """
        node = self.nodes[(i, j)]
        moves = [move for move, blocked in node.blocked.items() if not blocked]
        next_states = []
        if "r" in moves:
            next_states.append((i, j + 1))
        if "l" in moves:
            next_states.append((i, j - 1))
        if "u" in moves:
            next_states.append((i - 1, j))
        if "d" in moves:
            next_states.append((i + 1, j))
        return np.array(next_states)
    
    def get_true_sensor_reading(self, i: int, j: int) -> np.ndarray:
        """
        Utility method that returns what the actual sensor reading at a given state (i, j) should be.
        This function should be helpful when computing the probability of emitting an observation from a given state.
        """
        return np.array(self.nodes[(i, j)]._true_sensor_reading())

    def step(self) -> np.ndarray:
        """
        This function performs a random walk.
        Every time step() is called, the agent moves to one of its neighbors and emits a 
        sensor observation (that is then corrupted using the probability of error epsilon).
        The true hidden state of the agent, which the environment tracks internally, is 
        also updated.
        """
        if self.hidden_state is None:
            self.hidden_state = (random.randrange(self.rows), random.randrange(self.columns))
        else:
            # Find the node associated with the current robot hidden state
            node = self.nodes[self.hidden_state]
            # Retrieve the set of states that the robot can transition to
            # and choose one uniformly at random
            # This simulates the underlying hidden markov chain
            moves = [move for move, blocked in node.blocked.items() if not blocked]
            move = random.choice(moves)
            if move == "l":
                self.hidden_state = (self.hidden_state[0], self.hidden_state[1] - 1)
            if move == "r":
                self.hidden_state = (self.hidden_state[0], self.hidden_state[1] + 1)
            if move == "u":
                self.hidden_state = (self.hidden_state[0] - 1, self.hidden_state[1])
            if move == "d":
                self.hidden_state = (self.hidden_state[0] + 1, self.hidden_state[1])
        # Once you have arrived at the new state, emit an observation
        # First find the true observation
        true_reading = self.nodes[self.hidden_state]._true_sensor_reading()
        # Then, corrupt it to simulate faulty sensors
        observed_reading = []
        observed_reading.append(1 - true_reading[0] if random.random() < self.epsilon else true_reading[0])
        observed_reading.append(1 - true_reading[1] if random.random() < self.epsilon else true_reading[1])
        observed_reading.append(1 - true_reading[2] if random.random() < self.epsilon else true_reading[2])
        observed_reading.append(1 - true_reading[3] if random.random() < self.epsilon else true_reading[3])
        self.states_so_far.append(self.hidden_state)
        return np.array(observed_reading)
    
    def compute_accuracy(self, predictions: np.ndarray) -> np.ndarray:
        """
        Computes the cumulative accuracy, i.e., the ith index will contain the accuracy
        of the first i predictions when compared to the true hidden states.
        """
        return np.cumsum(np.all(predictions == np.array(self.states_so_far), axis=-1)) / np.arange(1, len(self.states_so_far) + 1) * 100
    
    def plot_env(self) -> None:
        """
        Saves the environment layout to env_layout.png.
        """
        _, ax = plt.subplots()
        markersize = 240 / self.rows + 1
        ax.set_aspect("equal")
        # Plot the maze
        for i in range(len(self.maze)):
            for j in range(len(self.maze[0])):
                if self.maze[i][j] == '|':
                    ax.plot(j, -i, 'k|', markersize=markersize)  # vertical wall
                elif self.maze[i][j] == '-':
                    ax.plot(j, -i, 'k_', markersize=markersize)  # horizontal wall
                elif self.maze[i][j] == ' ':
                    ax.plot(j, -i, 'w_', markersize=markersize)  # empty space

        ax.xaxis.set_tick_params(labelbottom=False)
        ax.yaxis.set_tick_params(labelleft=False)

        ax.set_xticks([])
        ax.set_yticks([])
        plt.savefig("./env_layout.png")

class Node:
    def __init__(self, i: int, j: int) -> None:
        self.i = i
        self.j = j
        self.neighbors: list[Node] = []
        self.blocked : dict[str, bool] = {"r" : 0, "l" : 0, "u" : 0, "d" : 0}

    def _true_sensor_reading(self) -> tuple[int]:
        return [int(self.blocked["l"]), int(self.blocked["r"]), int(self.blocked["u"]), int(self.blocked["d"])]
