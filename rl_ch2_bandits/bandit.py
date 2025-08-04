import numpy as np
from numpy.typing import NDArray
from typing import Optional


class Bandit:
    def __init__(
        self,
        k: int = 10,
        epsilon: float = 0.0,
        mu: float = 0,
        sigma: float = 1.0,
        reward_noise: float = 1.0,
    ):
        """
        Initializes a k-armed bandit
        :param k: number of arms/possible actions
        :param mu: mean of true action values q_*(a)
        :param sigma: standard deviation of true action values q_*(a)
        :param reward_noise: standard deviation of reward distribution around q_*(a)
        """
        self.k: int = k
        self.epsilon: float = epsilon
        self.mu: float = mu
        self.sigma: float = sigma
        self.reward_noise: float = reward_noise
        self.q_star: Optional[NDArray[np.float64], None] = (
            None  # True reward for each action
        )
        self.q_estimation: Optional[NDArray[np.float64], None] = (
            None  # Reward estimation for each action; updated after each action
        )
        self.action_count: Optional[NDArray[int], None] = (
            None  # Count for how often each action was chosen
        )
        self.best_action: Optional[int, None] = (
            None  # Best action, i.e., action with the empirically highest value
        )
        self.t: Optional[int, None] = None  # Timestep

    def reset(self):
        """
        Reinitialize the internal state of a bandit instance

        Used to resets the instance state at the start of a simulation/run, for example, if we want to try different
        decision strategies
        :return: --
        """
        self.q_star = np.random.normal(loc=self.mu, scale=self.sigma, size=self.k)
        self.q_estimation = np.zeros(self.k)
        self.action_count = np.zeros(self.k)
        self.best_action = np.argmax(self.q_star)
        self.t = 0

    def act(self) -> int:
        """
        Pick an action (i.e., decide which bandit arm to pull).
        Implements a greedy strategy if self.epsilon = 0, epsilon-greedy otherwise
        :return: index of action to take/arm to pull
        """
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.k)

        q_best = np.max(self.q_estimation)  # Greedy selection method
        return np.random.choice(
            np.where(self.q_estimation == q_best)[0]
        )  # Pick a random action if multiple actions have the same value

    def step(self, action: int, method: str = None) -> float:
        """
        Execute an action, i.e., pull the bandit's arm, and update/estimate action values
        :param action: index of action to take
        :param method: method to use for estimating action values. Can be: sample_average
        :return: reward after taking the action
        """
        assert method in [
            "sample_average"
        ], "Please select valid method: sample_average"

        reward = np.random.normal(loc=self.q_star[action], scale=self.reward_noise)
        self.t += 1
        self.action_count[action] += 1

        # Sample averages
        # Update value estimation of action by averaging over the reward we got from taking an action
        #
        # (reward - self.q_estimation[action]) computes the average iteratively, i.e., updates it at each new step,
        # avoiding us to store all rewards across all time stamps
        if method == "sample_average":
            self.q_estimation[action] += (
                reward - self.q_estimation[action]
            ) / self.action_count[action]
        return reward
