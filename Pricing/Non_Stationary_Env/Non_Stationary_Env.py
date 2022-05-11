import numpy as np


class NS_Env:
    def __init__(self, n_arms, probs_matrix, horizon):
        self.n_arms = n_arms
        self.probs_matrix = probs_matrix
        self.horizon = horizon
        self.n_changes = len(probs_matrix)
        self.inner_horizon = self.horizon // self.n_changes
        self.t = 0
        self.phase = 0

    def round(self, pulled_arm):
        self.t += 1
        if self.t > (self.phase + 1) * self.inner_horizon:
            self.phase = min(self.phase+1, self.n_changes-1)
        reward = np.random.binomial(n=1, p=self.probs_matrix[self.phase, pulled_arm])
        return reward
    