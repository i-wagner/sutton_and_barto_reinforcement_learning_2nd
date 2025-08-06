import numpy as np
import pytest

from bandit import Bandit


def create_bandit(k: int = 5, epsilon: float = 0, seed: int = 42) -> Bandit:
    """
    Instantiates a bandit for testing purposes
    :param k: number of arms
    :param epsilon: probability [0, 1] to select random action, instead of highest-value action
    :param seed: Random seed (for reproducibility)
    :return: Instance of bandit
    """
    if seed is not None:
        np.random.seed(seed)
    bandit = Bandit(k=k, epsilon=epsilon)
    bandit.reset()
    return bandit


def test_bandit_initialization():
    """
    Tests whether a bandit is initialized correctly
    :return: --
    """
    bandit = create_bandit()

    assert bandit.q_star.shape == (bandit.k,)
    assert np.all(bandit.q_estimation == 0) and bandit.q_estimation.shape == (bandit.k,)
    assert np.all(bandit.action_count == 0) and bandit.action_count.shape == (bandit.k,)
    assert 0 <= bandit.best_action < bandit.k
    assert bandit.t == 0


@pytest.mark.parametrize("epsilon", [0, 1])
def test_action_sampling(epsilon):
    """
    Tests whether actions are sampled within valid bounds, set by k
    :param epsilon: probability for exploration
    :return: --
    """
    timesteps = 100
    bandit = create_bandit(epsilon=epsilon)
    bandit.q_estimation = np.random.rand(bandit.k)

    actions = [bandit.act() for _ in range(timesteps)]

    assert set(actions).issubset(set(range(bandit.k)))


def test_greedy_sampling():
    """
    Tests whether the greedy sampling strategy consistently picks the highest-value action
    :return: --
    """
    timesteps = 100

    bandit = create_bandit()
    bandit.q_estimation = np.random.rand(bandit.k)
    best_action = np.argmax(bandit.q_estimation)

    actions = [bandit.act() for _ in range(timesteps)]

    assert all(
        action == best_action for action in actions
    )  # Checks if only the highest-value action was chosen


def test_epsilon_greedy_sampling():
    """
    Tests whether random sampling results in non-deterministic action selection
    :return: --
    """
    timesteps = 100

    bandit = create_bandit(epsilon=1)
    bandit.q_estimation = np.random.rand(bandit.k)

    actions = [bandit.act() for _ in range(timesteps)]

    assert (
        len(set(actions)) > 1
    )  # Explores, instead of always picking highest-value action
