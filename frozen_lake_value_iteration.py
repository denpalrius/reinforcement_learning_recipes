import gymnasium as gym
import numpy as np

# Set up the environment
env = gym.make("FrozenLake-v1", render_mode="human", is_slippery=False)
env.reset()

# Initialize parameters
gamma = 0.99  # Discount factor
theta = 1e-6  # Convergence threshold
value_table = np.zeros(env.observation_space.n)  # Initialize value function
num_actions = env.action_space.n  # Number of actions in the environment

# Transition probabilities
P = env.unwrapped.P  # Get transition probabilities from the environment

# Value Iteration
def value_iteration():
    while True:
        delta = 0
        # Iterate through each state
        for state in range(env.observation_space.n):
            v = value_table[state]
            max_value = float('-inf')
            # Evaluate the value for each possible action
            for action in range(num_actions):
                action_value = 0
                # Sum over possible next states and their transition probabilities
                for prob, next_state, reward, done in P[state][action]:
                    action_value += prob * (reward + gamma * value_table[next_state])
                max_value = max(max_value, action_value)
            # Update the value table for this state
            value_table[state] = max_value
            delta = max(delta, abs(v - value_table[state]))
        # Convergence check
        if delta < theta:
            break

# Display the updated value table
print("Value Table:")
print(value_table.reshape((4, 4)))  # Reshaped for visualization as a 4x4 grid

# Policy Extraction (after value iteration)
def extract_policy():
    policy = np.zeros(env.observation_space.n, dtype=int)
    for state in range(env.observation_space.n):
        action_values = np.zeros(num_actions)
        for action in range(num_actions):
            action_value = 0
            for prob, next_state, reward, done in P[state][action]:
                action_value += prob * (reward + gamma * value_table[next_state])
            action_values[action] = action_value
        policy[state] = np.argmax(action_values)  # Best action
    return policy

# Run Value Iteration
value_iteration()

# Extract the optimal policy
optimal_policy = extract_policy()

# Display the optimal policy
print("Optimal Policy:")
print(optimal_policy.reshape((4, 4)))  # Reshaped for visualization as a 4x4 grid

# Optional: Testing the policy by simulating it
state = env.reset()[0]
done = False
total_reward = 0

while not done:
    action = optimal_policy[state]  # Choose the optimal action for the current state
    next_state, reward, done, _, _ = env.step(action)
    total_reward += reward
    state = next_state
    env.render()  # Visualize the environment (optional)

print(f"Total reward using optimal policy: {total_reward}")

# Close the environment
env.close()