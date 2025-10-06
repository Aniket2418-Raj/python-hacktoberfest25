import gymnasium as gym
import numpy as np
import time
from gymnasium.envs.registration import register, registry
from gymnasium.envs.toy_text.frozen_lake import generate_random_map


def log(message):
    print(f" {message}")


#  Core Dynamic Programming: Value Iteration
def value_iteration(env, discount=0.99, threshold=1e-8):
    log(" Starting Value Iteration...")
    start_time = time.time()

    transition_probs = env.unwrapped.P  # Access raw environment transitions
    num_states = env.observation_space.n
    num_actions = env.action_space.n

    value_function = np.zeros(num_states)
    policy = np.zeros(num_states, dtype=int)

    iteration = 0
    while True:
        delta = 0
        for state in range(num_states):
            action_values = np.zeros(num_actions)
            for action in range(num_actions):
                if action not in transition_probs[state]:
                    continue
                for prob, next_state, reward, done in transition_probs[state][action]:
                    action_values[action] += prob * (reward + discount * value_function[next_state])
            best_action = np.argmax(action_values)
            delta = max(delta, abs(action_values[best_action] - value_function[state]))
            value_function[state] = action_values[best_action]
            policy[state] = best_action
        iteration += 1
        if delta < threshold:
            break

    elapsed = time.time() - start_time
    log(f" Converged in {iteration} iterations ( {elapsed:.3f} sec)")
    return policy, value_function


#  Test a learned policy by running episodes
def evaluate_policy(env, policy, episodes=100):
    total_rewards = []
    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        reward_sum = 0
        while not done:
            action = policy[state]
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            reward_sum += reward
        total_rewards.append(reward_sum)
    avg_reward = np.mean(total_rewards)
    log(f" Average reward over {episodes} episodes: {avg_reward:.3f}")
    return avg_reward


#  Register custom environments (with checks)
def register_custom_envs():
    log(" Setting up custom Frozen Lake environments...")

    # Remove existing registrations (if re-running)
    for env_id in ["CustomFrozenLake-v1", "ExpandedFrozenLake-v1"]:
        if env_id in registry:
            del registry[env_id]

    # 5x5 custom map
    custom_map = generate_random_map(size=5)

    # Custom Frozen Lake with no slipperiness
    register(
        id="CustomFrozenLake-v1",
        entry_point="gymnasium.envs.toy_text:FrozenLakeEnv",
        kwargs={"desc": custom_map, "is_slippery": False},
        max_episode_steps=100,
    )

    #  Expanded Frozen Lake with a teleport action (action 4)
    class ExpandedFrozenLake(gym.envs.toy_text.FrozenLakeEnv):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.action_space = gym.spaces.Discrete(5)  # Original 4 + 1 teleport

            # Add teleport transitions to the transition table (P)
            for state in self.P:
                teleport_target = self.np_random.integers(0, self.observation_space.n)
                self.P[state][4] = [(1.0, teleport_target, 0.0, False)]

        def step(self, action):
            if action == 4:  # Teleport to random state
                self.s = self.np_random.integers(0, self.observation_space.n)
                return self.s, 0.0, False, False, {}
            return super().step(action)

    # Register the expanded version
    register(
        id="ExpandedFrozenLake-v1",
        entry_point=lambda **kwargs: ExpandedFrozenLake(**kwargs),
        kwargs={"desc": custom_map, "is_slippery": False},
        max_episode_steps=100,
    )


#  Run value iteration across all environments
def run_all_environments():
    register_custom_envs()

    # Setup environments
    envs = {
        "Original FrozenLake": gym.make("FrozenLake-v1", is_slippery=False),
        "Custom FrozenLake": gym.make("CustomFrozenLake-v1"),
        "Expanded FrozenLake": gym.make("ExpandedFrozenLake-v1"),
    }

    results = {}

    for env_name, env in envs.items():
        log(f"\n Running Value Iteration on {env_name}")
        policy, _ = value_iteration(env)
        avg_reward = evaluate_policy(env, policy)
        results[env_name] = avg_reward

    log("\n Summary of All Results:")
    for name, reward in results.items():
        print(f" {name}: Average Reward = {reward:.3f}")


# Run everything
if __name__ == "__main__":
    run_all_environments()
