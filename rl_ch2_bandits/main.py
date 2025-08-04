import os
from typing import Union
import numpy as np

from bandit import Bandit
from plotting import figure_2_1

DIR_PLOTS = "./plots/"
if not os.path.isdir(DIR_PLOTS):
    os.makedirs(DIR_PLOTS, exist_ok=False)


def simulate(
    k: int = 10,
    epsilon: Union[float, list[float]] = 0.0,
    runs: int = 2000,
    timesteps: int = 1000,
):
    """
    Simulate a k-armed bandit for "runs" runs and "timesteps" timesteps per run
    :param k: number of arms
    :param epsilon: probability [0, 1] of choosing a random action at each pull; translates to a greedy strategy if
    epsilon is 0
    :param runs: number of simulations to run per bandit
    :param timesteps: number of timesteps to simulate for each bandit
    :return: --
    """
    epsilon = epsilon if isinstance(epsilon, list) else [epsilon]
    assert all([0 <= eps < 1 for eps in epsilon]), "Epsilon must be between 0 and 1"

    rewards = np.zeros((len(epsilon), runs, timesteps))
    best_action_taken_count = np.zeros(rewards.shape)
    bandits = [Bandit(k=k, epsilon=eps) for eps in epsilon]
    for i, bandit in enumerate(bandits):
        for r in range(runs):
            bandit.reset()  # Run each simulation with a new reward distribution
            for t in range(timesteps):
                action = bandit.act()
                rewards[i, r, t] = bandit.step(action, method="sample_average")
                if action == bandit.best_action:
                    best_action_taken_count[i, r, t] = 1
        figure_2_1(
            bandit.q_star, bandit.reward_noise, f"./plots/fig_2_1_b{i}.png"
        )  # Plot the most recent reward distribution
    return best_action_taken_count.mean(axis=1), rewards.mean(
        axis=1
    )  # Average over runs


if __name__ == "__main__":
    simulate(epsilon=[0.0, 0.10, 0.01])
