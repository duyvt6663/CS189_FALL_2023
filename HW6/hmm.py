import random
from env import Env
import numpy as np
import matplotlib.pyplot as plt
import time

def viterbi(
    observations: list[list[int]], 
    epsilon: float
) -> np.ndarray:
    """
    Params: 
    observations: a list of observations of size (T, 4) where T is the number of observations and
    1. observations[t][0] is the reading of the left sensor at timestep t
    2. observations[t][1] is the reading of the right sensor at timestep t
    3. observations[t][2] is the reading of the up sensor at timestep t
    4. observations[t][3] is the reading of the down sensor at timestep t
    epsilon: the probability of a single sensor failing

    Return: a list of predictions for the agent's true hidden states.
    The expected output is a numpy array of shape (T, 2) where 
    1. (predictions[t][0], predictions[t][1]) is the prediction for the state at timestep t
    """

    # TODO: implement the viterbi algorithm
    num_states = env.rows * env.columns
    T = len(observations)
    viterbi = np.zeros((num_states, T))
    backpointer = np.zeros((num_states, T), dtype=int)
    
    # Initialize base cases (t == 0)
    viterbi[:, 0] = 1/num_states * emission_probs[0, :]
    backpointer[:, 0] = -1
    
    # Run Viterbi for t > 0
    for t in range(1, T):
        for i in range(num_states):
            probs = viterbi[:, t-1] * transition_probs[:, i] 
            max_idx = np.argmax(probs)
            backpointer[i, t] = max_idx
            viterbi[i, t] = probs[max_idx] * emission_probs[t, i]
    
    # Backtrace
    predictions = np.zeros((T, 2), dtype=int)
    best_last_state = np.argmax(viterbi[:, -1])
    for t in range(T - 1, -1, -1):
        predictions[t] = state2idx[best_last_state]
        best_last_state = backpointer[best_last_state, t]
    
    return predictions

def precompute_emission_probs(
    observations: list[list[int]],
    epsilon: float
) -> np.ndarray:
    """ 
    Return: a numpy array of shape (len(observations), num_states) where
    1. emission_probs[t][i] is the probability of emitting the observation at timestep t from state i
    """

    num_states = env.rows * env.columns
    # Assuming state2idx is available and correctly maps state indices to positions
    
    # Step 1: Get true sensor readings for all states
    true_readings = np.array([env.get_true_sensor_reading(*state2idx[i]) for i in range(num_states)])
    
    # Step 2: Vectorize comparisons
    # Convert observations to a NumPy array for efficient broadcasting
    observations_array = np.array(observations)  # Shape: (T, 4), T is the number of observations
    
    # Compare observations with true readings, shape: (T, num_states, 4)
    # Using broadcasting to compare each observation with each state's true reading
    matches = observations_array[:, None, :] == true_readings[None, :, :]
    
    # Calculate emission probabilities
    # If a sensor matches, use 1 - epsilon, otherwise use epsilon
    # This creates an array of shape (T, num_states, 4) with the probability of each sensor reading
    emission_probs_per_sensor = np.where(matches, 1 - epsilon, epsilon)
    
    # Multiply probabilities across sensors for each observation-state pair
    # np.prod along the last axis (axis=-1) to multiply probabilities for all 4 sensors
    emission_probs = np.prod(emission_probs_per_sensor, axis=-1)  # Resulting shape: (T, num_states)
    
    return emission_probs

def precompute_transition_probs():
    """
    Return: a numpy array of shape (num_states, num_states) where
    1. transition_probs[i][j] is the probability of transitioning from state i to state j
    """

    num_states = env.rows * env.columns
    transition_probs = np.zeros((num_states, num_states))
    for i in range(num_states):
        neighbors = env.get_neighbors(*state2idx[i])
        for rj, cj in neighbors:
            j = rj * env.columns + cj
            transition_probs[i, j] = 1 / len(neighbors)
    return transition_probs

def precompute_state2idx():
    """
    Return: a numpy array of shape (num_states, 2) where
    1. state2idx[i][0] is the row of the ith state
    2. state2idx[i][1] is the column of the ith state
    """

    num_states = env.rows * env.columns

    # Generate an array of state indices
    state_indices = np.arange(num_states)

    # Compute rows and columns for each state index
    rows = state_indices // env.columns
    columns = state_indices % env.columns

    # Stack rows and columns to get the state2idx mapping
    state2idx = np.vstack((rows, columns)).T
    return state2idx

if __name__ == '__main__':
    random.seed(12345)
    rows, cols = 16, 16 # dimensions of the environment
    openness = 0.3 # some hyperparameter defining how "open" an environment is
    traj_len = 100 # number of observations to collect, i.e., number of times to call env.step()
    num_traj = 100 # number of trajectories to run per epsilon

    env = Env(rows, cols, openness)
    env.plot_env() # the environment layout should be saved to env_layout.png

    plt.clf()
    """
    The following loop simulates num_traj trajectories for each value of epsilon.
    Since there are 6 values of epsilon being tried here, a total of 6 * num_traj
    trajectories are generated.
    
    For reference, the staff solution takes < 3 minutes to run.
    """
    start = time.time()
    state2idx = precompute_state2idx() # precompute state to index mapping
    transition_probs = precompute_transition_probs() # precompute transition probabilities
    for epsilon in [0.0, 0.05, 0.1, 0.2, 0.25, 0.5]:
        env.set_epsilon(epsilon)
        
        accuracies = []
        for _ in range(num_traj):
            env.init_env()

            observations = []
            for i in range(traj_len):
                obs = env.step()
                observations.append(obs)

            emission_probs = precompute_emission_probs(observations, epsilon) # precompute emission probabilities
            predictions = viterbi(observations, epsilon)

            accuracies.append(env.compute_accuracy(predictions))
        plt.plot(np.mean(accuracies, axis=0), label=f"epsilon={epsilon}")

    plt.xlabel("Number of observations")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right")
    plt.savefig("accuracies.png")

    end = time.time()
    print("Time taken:", end - start)