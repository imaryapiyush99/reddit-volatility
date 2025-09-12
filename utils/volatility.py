import numpy as np
from collections import deque

class VolatilityTracker:
    def __init__(self, window_size=20):
        self.window = deque(maxlen=window_size)

    def update(self, score):
        self.window.append(score)
        return 0 if len(self.window) < 5 else np.std(self.window)
