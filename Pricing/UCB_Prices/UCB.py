import numpy as np
import Learner


class UCB(Learner.Learner):
    def __init__(self, n_arms, prices):
        super(UCB, self).__init__(n_arms)
        self.means = np.zeros(n_arms)
        self.widths = np.array([np.inf for _ in range (n_arms)])
        self.prices = prices

    def act(self):
        idx = np.argmax(self.means + self.widths)
        return idx

    def update(self, arm_pulled, reward):
        reward = reward > 0
        super(UCB, self).update(arm_pulled, reward)
        self.means[arm_pulled] = np.mean(self.reward_per_arm[arm_pulled])
        for idx in range(self.n_arms):
            n = len(self.reward_per_arm[idx])
            if n > 0:
                self.widths[idx] = np.sqrt(2*np.max(self.prices)*np.log(self.t) / n)
            else:
                self.widths[idx] = np.inf