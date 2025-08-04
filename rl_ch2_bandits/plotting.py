import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt

def get_max_abs(values: NDArray) -> float:
    """
    Gets the greatest absolut value in a distribution of values

    :param values: distribution of values
    :return: Greatest absolute value in distribution
    """
    return max(abs(values.flatten()))

def figure_2_1(mu: list[float], sigma: float, folder: str = "./plots/fig_2_1.png"):
    """
    Plot distribution of rewards for each arm of a k-armed bandit. Distributions are simulated of, for each arm,
    generating a distribution of rewards for mean "mu" and standard deviation "sigma", and drawing an arbitrary number
    of samples from that distribution. Each draw corresponds to one action taken, and the corresponding reward received
    after the action

    Recreates Figure 2.1 from Sutton & Barto, 2020
    :param mu: Mean reward for each of the k arms of a bandit
    :param sigma: Standard deviation of reward distribution; the same for each arm
    :param folder: Filename and folder where to save figure
    :return: --
    """
    reward_distributions = np.array([np.random.normal(loc=m, scale=sigma, size=2000) for m in mu])
    y_ax_limit = np.ceil(get_max_abs(reward_distributions))

    parts = plt.violinplot(dataset=reward_distributions.transpose(), showmeans=True, showextrema=False)
    plt.axhline(y=0, color='k', linestyle="--", zorder=0)
    plt.ylim(-y_ax_limit, y_ax_limit)
    plt.xticks(np.arange(1, len(mu) + 1))
    plt.xlabel("Action")
    plt.ylabel("Reward distribution")

    # Change style of violin
    for pc in parts['bodies']:
        pc.set_facecolor(np.zeros(3) + 0.50)
        pc.set_edgecolor('black')
    parts["cmeans"].set_edgecolor('k')

    plt.savefig(folder)
    plt.close()
