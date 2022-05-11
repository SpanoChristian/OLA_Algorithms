import numpy as np
import Learner


class SW_TS(Learner.Learner):
    def __init__(self, n_arms, window_size):
        super(SW_TS, self).__init__(n_arms)
        self.n_arms = n_arms
        self.window_size = window_size
        self.alphas = np.ones(self.n_arms)
        self.betas = np.ones(self.n_arms)
        self.t = 0

    def reset(self):
        self.__init__(self.n_arms, self.window_size)

    def update(self, pulled_arm, reward):
        super(SW_TS, self).update(pulled_arm, reward)
        for arm_idx in range(self.n_arms):
            n_samples = np.sum(self.pulled[-self.window_size:] == arm_idx)
            if n_samples == 0:
                n_sold = 0
            else:
                n_sold = np.sum(self.reward_per_arm[arm_idx][-n_samples:])
            self.alphas[arm_idx] = n_sold + 1
            self.betas[arm_idx] = n_samples - n_sold + 1
            
    def act(self):
        samples = [np.random.beta(a=self.alphas[i], b=self.betas[i]) for i in range(self.n_arms)]
        return np.argmax(samples)
            