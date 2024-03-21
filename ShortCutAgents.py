import random
import numpy as np

class QLearningAgent(object):

    def __init__(self, n_actions, n_states, epsilon, alpha): # TODO delete alpha??
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = epsilon
        self.alpha = alpha
        self.Q = np.zeros((144, 4))
        pass
        
    def select_action(self, state):
        a = np.argmax(self.Q[state])    # greedy action
        if random.random() < self.epsilon:
            a = random.choice(range(self.n_actions))
        return a

    def update(self, state, action, reward, state_prime):  # TODO delete state_prime??
        # gamma is 1 based on description of the task
        self.Q[state, action] = self.Q[state, action] + self.alpha * (reward + np.max(self.Q[state_prime]) - self.Q[state, action])
        pass

class SARSAAgent(object):

    def __init__(self, n_actions, n_states, epsilon):
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = epsilon
        # TO DO: Add own code
        pass
        
    def select_action(self, state):
        # TO DO: Add own code
        a = random.choice(range(self.n_actions)) # Replace this with correct action selection
        return a
        
    def update(self, state, action, reward):
        # TO DO: Add own code
        pass

class ExpectedSARSAAgent(object):

    def __init__(self, n_actions, n_states, epsilon):
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = epsilon
        # TO DO: Add own code
        pass
        
    def select_action(self, state):
        # TO DO: Add own code
        a = random.choice(range(self.n_actions)) # Replace this with correct action selection
        return a
        
    def update(self, state, action, reward):
        # TO DO: Add own code
        pass