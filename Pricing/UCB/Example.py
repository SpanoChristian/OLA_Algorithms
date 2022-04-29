import numpy as np
import matplotlib.pyplot as plt

import Environment
import UCB

p = [0.5, 0.3, 0.3, 0.9]

pricing_env = Environment.Env(p)
agent = UCB.UCB(len(p))
T = 1000
opt = np.max(p)
n_experiments = 300

cumulative_regret = 0
tot_cumulative_regret = []

for _ in range(n_experiments):
    instant_regret = []
    agent.reset()
    for t in range(T):
        pulled_arm = agent.act()
        reward = pricing_env.round(pulled_arm)
        agent.update(pulled_arm, reward)
        instant_regret.append(opt - reward)
    cumulative_regret = np.cumsum(instant_regret)
    tot_cumulative_regret.append(cumulative_regret)

mean_regret = np.mean(tot_cumulative_regret, axis=0)
std_dev = np.std(tot_cumulative_regret, axis=0)/np.sqrt(n_experiments)

plt.plot(mean_regret)
plt.fill_between(range(T), mean_regret-std_dev, mean_regret + std_dev, alpha=0.4)
plt.show()
