import numpy as np

class Experience(object):
    def __init__(self, state, action, reward, next_state, is_terminal = False):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.is_terminal = is_terminal


class Memory(object):
    """Memory for experience replay"""
    def __init__(self, size):
        self.size = size
        self.memory = []

    @property
    def current_size(self):
        return len(self.memory)

    def clear(self):
        self.memory = []

    def all(self):
        return self.memory

    def add(self, item):
        if len(self.memory) > self.size:
            self.memory.pop()

        self.memory.append(item)

    def sample(self, batch_size):
        if batch_size > len(self.memory):
            return np.random.choice(self.memory, len(self.memory))

        return np.random.choice(self.memory, batch_size)
