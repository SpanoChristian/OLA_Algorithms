import numpy as np
import matplotlib.pyplot as plt

import Non_Stationary_Env as nv
import SW_TS as sw_ts

p = np.array([[0.5, 0.1, 0.2, 0.9],
             [0.4, 0.5, 0.2, 0.3],
             [0.3, 0.2, 0.4, 0.5]])

T = 1000
n_arms = p.shape[1]
non_stat_env = nv.NS_Env(n_arms, p, T)
window_size = 4*int(np.sqrt(T))
agent = sw_ts.SW_TS(n_arms, window_size)
opt = np.max(p, axis=1)
n_experiments = 20

cumulative_regret = 0
tot_cumulative_regret = []

for _ in range(n_experiments):
    instant_regret = []
    agent.reset()
    for t in range(T):
        pulled_arm = agent.act()
        reward = non_stat_env.round(pulled_arm)
        agent.update(pulled_arm, reward)
        phase = non_stat_env.phase
        instant_regret.append(opt[phase] - p[phase, pulled_arm])
    cumulative_regret = np.cumsum(instant_regret)
    tot_cumulative_regret.append(cumulative_regret)

mean_regret = np.mean(tot_cumulative_regret, axis=0)
std_dev = np.std(tot_cumulative_regret, axis=0)/np.sqrt(n_experiments)

plt.plot(mean_regret)
plt.fill_between(range(T), mean_regret-std_dev, mean_regret + std_dev, alpha=0.4)
plt.show()
