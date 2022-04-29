import numpy as np


class Env:
    def __init__(self, probs):
        self.probs = probs

    def round(self, arm_pulled):
        reward = np.random.binomial(n=1, p=self.probs[arm_pulled])
        return reward
