# Write your experiments in here! You can use the plotting helper functions from the previous assignment if you want.
from typing import Union
import numpy as np
from ShortCutAgents import QLearningAgent, SARSAAgent, ExpectedSARSAAgent
from ShortCutEnvironment import ShortcutEnvironment
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


# Helper classes and functions
class LearningCurvePlot:

    def __init__(self, title=None):
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlabel('Episode')
        self.ax.set_ylabel('Reward')
        if title is not None:
            self.ax.set_title(title)

    def add_curve(self, y, label=None):
        """
        y: vector of average reward results
        label: string to appear as label in plot legend
        """
        if label is not None:
            self.ax.plot(y, label=label)
        else:
            self.ax.plot(y)

    def save(self, name='test.png'):
        """
        name: string for filename of saved figure
        """
        self.ax.legend()
        self.fig.savefig(name, dpi=300)


class ComparisonPlot:

    def __init__(self, title=None):
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlabel('Parameter (exploration)')
        self.ax.set_ylabel('Average reward')
        self.ax.set_xscale('log')
        if title is not None:
            self.ax.set_title(title)

    def add_curve(self, x, y, label=None):
        """
        x: vector of parameter values
        y: vector of associated mean reward for the parameter values in x
        label: string to appear as label in plot legend
        """
        if label is not None:
            self.ax.plot(x, y, label=label)
        else:
            self.ax.plot(x, y)

    def save(self, name='test.png'):
        """
        :param: name - string for filename of saved figure
        """
        self.ax.legend()
        self.fig.savefig(name, dpi=300)


def smooth(y, window, poly=1):
    """
    y: vector to be smoothed
    window: size of the smoothing window
    """
    return savgol_filter(y, window, poly)


def print_greedy_actions(Q):
    greedy_actions = np.argmax(Q, 1).reshape((12, 12))
    print_string = np.zeros((12, 12), dtype=str)
    print_string[greedy_actions == 0] = '^'
    print_string[greedy_actions == 1] = 'v'
    print_string[greedy_actions == 2] = '<'
    print_string[greedy_actions == 3] = '>'
    print_string[np.max(Q, 1).reshape((12, 12)) == 0] = '0'
    line_breaks = np.zeros((12, 1), dtype=str)
    line_breaks[:] = '\n'
    print_string = np.hstack((print_string, line_breaks))
    print(print_string.tobytes().decode('utf-8'))


# Experiment functions
def run_repetition(
        env: ShortcutEnvironment,
        agent: Union[QLearningAgent, SARSAAgent, ExpectedSARSAAgent],
        n_episodes: int,
        n_steps: int
) -> np.array:
    episodes_rewards = np.zeros(n_episodes)
    for i in range(n_episodes):  # for each episode
        cum_reward = 0
        for _ in range(n_steps):  # simulate one run (episode) with max of n_steps steps
            state = env.state()
            action = agent.select_action(state)
            reward = env.step(action)
            cum_reward += reward
            state_prime = env.state()
            if env.done():
                break
            agent.update(state, action, reward, state_prime)
            episodes_rewards[i] = cum_reward
        env.reset()

    return episodes_rewards


def run_repetitions(
        env: ShortcutEnvironment,
        agent_type: str,
        n_repetitions: int,
        n_episodes: int,
        n_steps: int,
        **kwargs
) -> np.array:
    curve = np.zeros(n_episodes)
    agent = None
    for _ in range(n_repetitions):
        if agent_type == "q-learning":
            agent = QLearningAgent(environment, epsilon=kwargs["epsilon"], alpha=kwargs["alpha"])
        elif agent_type == "SARSA":
            pass
        else:
            pass
        repetition_curve = run_repetition(env, agent, n_episodes=n_episodes, n_steps=n_steps)
        env.reset()  # just double-check, it should be also done in the run_repetitions
        curve += repetition_curve
    return curve / n_repetitions


def experiment(
        env: ShortcutEnvironment,
        n_repetitions: int,
        n_episodes: int,
        n_steps: int,
        smoothing_window: int,
        epsilon: float,
        alphas: list,
):
    # Assignment 1 - Q-learning
    QLearn_plot = LearningCurvePlot(title="Q-learning learning curves based on alpha hyperparam.")
    for i in range(len(alphas)):
        alpha = alphas[i]
        curve = run_repetitions(
            env=env,
            agent_type="q-learning",
            n_repetitions=n_repetitions,
            n_episodes=n_episodes,
            n_steps=n_steps,
            epsilon=epsilon,
            alpha=alpha
        )
        QLearn_plot.add_curve(smooth(curve, smoothing_window), f"alpha={alpha}")
    QLearn_plot.save("Q-Learning plot")


if __name__ == "__main__":
    # Setting up the parameters of the experiment
    environment = ShortcutEnvironment()
    n_repetitions = 50
    n_episodes = 100
    n_steps = 100
    alphas = [0.01, 0.1, 0.5, 0.9]
    epsilon = 0.1
    smoothing_window = 31
    # The experiment itself
    experiment(
        env=environment,
        n_repetitions=n_repetitions,
        n_episodes=n_episodes,
        n_steps=n_steps,
        alphas=alphas,
        epsilon=epsilon,
        smoothing_window=smoothing_window
    )
