import numpy as np
import matplotlib.pyplot as plt

import DiabetesTrial as dt

t_max = 48  
n = 100
mu = None

trial = dt.DiabetesTrial(n, t_max, initial_states=dt.get_burned_in_states(n, mu))
for t in range(t_max):
    trial.step_forward_in_time(mu, apply_dropout=True)

n_remaining = trial.engaged_inds.sum()
plt.hist(trial.T_dis[trial.T_dis != np.inf], bins=t_max)
plt.title(f"Disengagement times (n={n}, n_remaining={n_remaining})")
plt.show()

trial.plot_feature_over_time("stress",)
trial.plot_feature_over_time("fatigue")
trial.plot_feature_over_time("glucose", hlines=[80, 120])
trial.plot_feature_over_time("insulin-1")

trial.plot_feature_dist("exercise")
trial.plot_feature_dist("unhealthy_ind")