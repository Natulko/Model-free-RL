# Write your experiments in here! You can use the plotting helper functions from the previous assignment if you want.
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


def run_repetitions(n_rep, n_episodes, n_states, n_actions, agent_type, **kwargs):
    # repetitions
    for _ in range(n_rep):
        # initialize environment (automatically resets)
        env = ShortcutEnvironment()
        # initialize agent
        if agent_type == "qlearning":
            agent = QLearningAgent(n_actions, n_states, kwargs["epsilon"], kwargs["alpha"])
        # simulate one run
        for _ in range(n_episodes):
            state = env.state()
            action = agent.select_action(state)
            reward = env.step(action)
            state_prime = env.state()
            if env.done():
                break
            agent.update(state, action, reward, state_prime)


def experiment(alphas):
    return


if __name__ == "__main__":
    run_repetitions(100, 10000, 12*12, 4, "qlearning", epsilon=0.1, alpha = 0.5)
    print('\n')
    print_greedy_actions(np.random.rand(12*12, 4))
