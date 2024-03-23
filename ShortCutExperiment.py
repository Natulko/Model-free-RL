# Write your experiments in here! You can use the plotting helper functions from the previous assignment if you want.
from typing import Union
import numpy as np
from ShortCutAgents import QLearningAgent, SARSAAgent, ExpectedSARSAAgent
from ShortCutEnvironment import ShortcutEnvironment


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

def experiment(alphas):
    return


if __name__ == "__main__":
    environment = ShortcutEnvironment()
    #Q_agent = QLearningAgent(environment, epsilon=0.1, alpha=0.5)
    rewards = run_repetitions(
        environment,
        agent_type="q-learning",
        n_repetitions=50,
        n_episodes=100,
        n_steps=100,
        epsilon=0.1,
        alpha=0.5
    )
    print(rewards)
    print(rewards.shape)
    environment.render()
