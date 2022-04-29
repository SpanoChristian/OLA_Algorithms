class Learner:
    def __init__(self, n_arms):
        self.t = 0
        self.n_arms = n_arms
        self.rewards = []
        self.reward_per_arm = [[] for _ in range(n_arms)]

    def reset(self):
        self.__init__(self.n_arms, self.prices)

    def act(self):
        pass

    def update(self, arm_pulled, reward):
        self.t += 1
        self.rewards.append(reward)
        self.reward_per_arm[arm_pulled].append(reward)