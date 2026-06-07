from collections import deque, namedtuple
import random

memory = namedtuple('memory','state action next_state reward')

class ReplayBuffer:
    def __init__(self, capacity=10_000, batch_size=32):
        self.buffer = deque(maxlen=capacity)
        self.batch_size = batch_size
    
    def push(self, memory):
        self.buffer.append(memory)
    
    def sample(self):
        return random.sample(self.buffer, self.batch_size)
    
    def __len__(self):
        return len(self.buffer)