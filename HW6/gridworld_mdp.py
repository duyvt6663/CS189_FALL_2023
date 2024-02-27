# Adjusting the grid to be 4x3 and drawing cells
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def create_grid() -> tuple:
    """Draw a grid of cells."""
    fig, ax = plt.subplots(figsize=(8, 6))

    # Set grid limits
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 3)

    # Draw grid cells
    for x in range(5):
        ax.axvline(x, color='black', linestyle='-')
    for y in range(4):
        ax.axhline(y, color='black', linestyle='-')
    
    # Coloring the 1, 1 cell
    ax.add_patch(plt.Rectangle((1, 1), 1, 1, color='grey'))
    # Adding text "Start" to the middle of the cell at (0, 0)
    ax.text(0.5, 0.5, 'START', ha='center', va='center', fontsize=12, color='black')

    ax.text(3.5, 2.5, '+1', ha='center', va='center', fontsize=14, color='black')
    ax.add_patch(plt.Rectangle((3, 2), 1, 1, color='lightgreen'))
    ax.text(3.5, 1.5, '-1', ha='center', va='center', fontsize=14, color='black')
    ax.add_patch(plt.Rectangle((3, 1), 1, 1, color='lightgreen'))

    # Remove axes
    ax.axis('off')

    return fig, ax
    
def add_arrow(
    ax: plt.Axes,
    x: int,
    y: int,
    directions: list[str]
):
    """Add arrows to the cell at (x, y)."""
    for direction in directions:
        if direction == 'up':
            ax.arrow(x + 0.5, y + 0.6, 0, 0.3, head_width=0.05, head_length=0.05, fc='black', ec='black')
        elif direction == 'down':
            ax.arrow(x + 0.5, y + 0.4, 0, -0.3, head_width=0.05, head_length=0.05, fc='black', ec='black')
        elif direction == 'left':
            ax.arrow(x + 0.38, y + 0.5, -0.3, 0, head_width=0.05, head_length=0.05, fc='black', ec='black')
        elif direction == 'right':
            ax.arrow(x + 0.62, y + 0.5, 0.3, 0, head_width=0.05, head_length=0.05, fc='black', ec='black')

def value_iteration(
    gamma: float = 0.999, 
    reward: float = 1
) -> tuple:
    """
    Implement the value iteration algorithm for a 3x4 grid environment.
    """
    # Initialize hyperparameters
    theta = 0.0001  # Small threshold for determining convergence of value function
    grid_shape = (3, 4)
    obstruction = (1, 1)  # Obstruction cell
    positive_reward_state = (0, 3)
    negative_reward_state = (1, 3)

    # Initialize rewards and value function
    rewards = {(x, y): reward for x in range(grid_shape[0]) for y in range(grid_shape[1])}
    rewards[positive_reward_state] = 1
    rewards[negative_reward_state] = -1
    V = {(x, y): 0 for x in range(grid_shape[0]) for y in range(grid_shape[1])}  # Initialize value function to zeros
    V[obstruction] = None  # No value for obstruction
    V[positive_reward_state] = 1  # terminal state
    V[negative_reward_state] = -1  # another terminal state
    
    actions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Right, Down, Left, Up
    
    def get_next_states(state, action):
        """
        Return a list of possible next states from current state and action.
        Order of next states is [intended action, right angle to the right, right angle to the left].
        """
        # first fill the list with the intended action, then the right angle 
        # to the right and then the right angle to the left
        # make sure the intended action is the first in the list
        actions = [action]
        if action in [(0, 1), (0, -1)]:
            actions.extend([(1, 0), (-1, 0)])
        elif action in [(1, 0), (-1, 0)]:
            actions.extend([(0, 1), (0, -1)])
        else:
            raise ValueError("Invalid action")

        # Get next states for each action
        x, y = state
        next_states = []
        for dx, dy in actions:
            next_state = (x + dx, y + dy)
            if 0 <= next_state[0] < grid_shape[0] and \
                0 <= next_state[1] < grid_shape[1] and \
                next_state != obstruction:
                next_states.append(next_state)
            else:
                # If next state is out of bounds or an obstruction, stay in current state
                next_states.append((x, y))
        return next_states
    
    # Value iteration loop
    probs = [0.8, 0.1, 0.1]  # Probabilities for intended action, right angle to the right, right angle to the left
    while True:
        delta = 0
        for state in V:
            # important to skip the obstruction, positive_reward_state and negative_reward_state
            # if not skipping the positive_reward_state and negative_reward_state, 
            # the algorithm is continuing and not episodic
            if state == obstruction or \
                state == positive_reward_state or \
                state == negative_reward_state:
                continue
            v = V[state]
            # V(s) = R(s) + gamma * max_{a}{sum_{s'} P(s' | s, a) * V(s')}
            # where P(s' | s, a) = 0.8 if s' is the intended action and 0.1 for the other right angles directions
            V[state] = rewards[state] + gamma * max(
                            sum([probs[i] * V[next_state] 
                                for i, next_state in enumerate(get_next_states(state, action))]) 
                                    for action in actions
                        )
            
            delta = max(delta, abs(v - V[state]))
        if delta < theta:
            break
    
    # Extract policy from value function
    policy = {}
    for state in V:
        if state == obstruction or \
            state == positive_reward_state or \
            state == negative_reward_state:
            continue
        best_action = max(actions, key=lambda a: 
                        rewards[state] + gamma * sum([probs[i] * V[next_state] 
                            for i, next_state in enumerate(get_next_states(state, a))]))
        policy[state] = [action_to_direction(action) for action in actions if action == best_action]
    
    return V, policy

def action_to_direction(action):
    """
    Return the direction of the action in the array coordinates.
    """

    if action == (0, 1):
        return 'right'
    elif action == (1, 0):
        return 'down'
    elif action == (0, -1):
        return 'left'
    elif action == (-1, 0):
        return 'up'

if __name__ == "__main__":
    # Execute the value iteration function
    V, policy = value_iteration()
    print(f"optimal value: {V}")
    print(f"optimal policy: {policy}")

    fig, ax = create_grid()
    for state, actions in policy.items():
        # array coordinates: state = (x, y) -> grid coordinates (y, 2-x)
        add_arrow(ax, state[1], 2-state[0], actions)
    plt.show()
