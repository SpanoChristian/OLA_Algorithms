import numpy as np


class Env:
    def __init__(self, probs, prices):
        self.probs = probs
        self.prices = prices

    def round(self, arm_pulled):
        conversion = np.random.binomial(n=1, p=self.probs[arm_pulled])
        reward = conversion*self.prices[arm_pulled]
        return reward
