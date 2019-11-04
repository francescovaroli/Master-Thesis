from collections import namedtuple
import random

# Taken from
# https://github.com/pytorch/tutorials/blob/master/Reinforcement%20(Q-)Learning%20with%20PyTorch.ipynb

Transition = namedtuple('Transition', ('state', 'action', 'mask', 'next_state',
                                       'reward'))


class Memory(object):
    def __init__(self):
        self.memory = []

    def push(self, *args):
        """Saves a transition."""
        self.memory.append(Transition(*args))

    def sample(self, batch_size=None):
        if batch_size is None:
            return Transition(*zip(*self.memory))
        else:
            random_batch = random.sample(self.memory, batch_size)
            return Transition(*zip(*random_batch))

    def append(self, new_memory):
        self.memory += new_memory.memory

    def __len__(self):
        return len(self.memory)


class ReplayMemory(object):
    def __init__(self, memory_size):
        self.memory = []

    def push(self, *args):
        """Saves a transition."""
        self.memory.append(Transition(*args))

    def sample(self, batch_size=None):
        if batch_size is None:
            return Transition(*zip(*self.memory))
        else:
            random_batch = random.sample(self.memory, batch_size)
            return Transition(*zip(*random_batch))

    def append(self, new_memory):
        self.memory += new_memory.memory

    def __len__(self):
        return len(self.memory)






    def push(self, *args):
        """Saves a transition."""
        iter = args[0]
        if len(self.memory) < iter-1:
            self.memory.append([])
        self.memory[iter].append(Transition(*args[1:]))

    def sample(self, batch_size=None, from_iter=None):
        if from_iter is None:  # merge transitions from all iterations
            from_memory = [self.memory[i][j] for i in range(len(self.memory)) for j in range(len(self.memory[i]))]
        else:
            from_memory = self.memory[from_iter]
        if batch_size is None:
            return Transition(*zip(*from_memory))
        else:
            random_batch = random.sample(from_memory, batch_size)
            return Transition(*zip(*random_batch))

    def append(self, new_memory):
        self.memory += new_memory.memory

    def __len__(self):
        return len(self.memory)
