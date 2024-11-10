from collections import namedtuple
import random

# Define namedtuples for storing states and transitions
State = namedtuple('State', ('obs', 'description', 'inventory'))
Transition = namedtuple('Transition', ('state', 'act', 'reward', 'next_state', 'next_acts', 'done'))
Episode = namedtuple('Episode', ('states', 'acts', 'rewards', 'next_acts', 'dones'))


class ReplayMemory:
    """A simple cyclic buffer for storing transitions or episodes with fixed capacity."""

    def __init__(self, capacity, obj_type=Transition):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.obj_type = obj_type

    def push(self, *args):
        """Stores a transition or episode at the current position in memory."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = self.obj_type(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """Randomly samples a batch of items from memory."""
        return random.sample(self.memory, batch_size)

    def pull_all(self):
        """Retrieves all current items from memory and clears it."""
        current_transitions = self.memory
        self.memory = []
        self.position = 0
        return current_transitions

    def __len__(self):
        return len(self.memory)


class PrioritizedReplayMemory:
    """Replay buffer with prioritized and non-prioritized storage compartments."""

    def __init__(self, capacity, priority_fraction):
        self.priority_fraction = priority_fraction
        self.alpha_capacity = int(capacity * priority_fraction)
        self.beta_capacity = capacity - self.alpha_capacity
        self.alpha_memory = []
        self.beta_memory = []
        self.alpha_position = 0
        self.beta_position = 0

    def clear_alpha(self):
        """Clears the prioritized (alpha) memory compartment."""
        self.alpha_memory = []
        self.alpha_position = 0

    def push(self, transition, is_prior=False):
        """Saves a transition in the prioritized or non-prioritized buffer based on `is_prior`."""
        target_memory, target_capacity, target_position = (
            (self.alpha_memory, self.alpha_capacity, 'alpha_position') if is_prior and self.priority_fraction > 0 else
            (self.beta_memory, self.beta_capacity, 'beta_position')
        )
        # Insert into target memory at the current position
        if len(target_memory) < target_capacity:
            target_memory.append(None)
        target_memory[getattr(self, target_position)] = transition
        setattr(self, target_position, (getattr(self, target_position) + 1) % target_capacity)

    def sample(self, batch_size):
        """Samples a batch of transitions, with a fraction from prioritized memory if available."""
        if self.priority_fraction > 0:
            from_alpha = min(int(batch_size * self.priority_fraction), len(self.alpha_memory))
            from_beta = min(batch_size - from_alpha, len(self.beta_memory))
            res = random.sample(self.alpha_memory, from_alpha) + random.sample(self.beta_memory, from_beta)
        else:
            res = random.sample(self.beta_memory, min(batch_size, len(self.beta_memory)))
        random.shuffle(res)
        return res

    def __len__(self):
        return len(self.alpha_memory) + len(self.beta_memory)
