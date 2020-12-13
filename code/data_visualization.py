import numpy as np
import matplotlib.pyplot as plt

import data_generation as dg

def plot_feature_over_time(feature, trial, num_to_plot=10):
    stress_over_time = trial.S[range(num_to_plot),:,dg.s_idx[feature]]
    fig, ax = plt.subplots()
    ax.plot(stress_over_time.T)
    ax.set_title(f"Patient {feature} over time")
    ax.set_xlabel("time")
    ax.set_ylabel(feature)
    plt.show()
    

t_max = 48  
n = 100
mu = lambda x: np.full_like(x,fill_value=0.3)

trial = dg.DiabetesTrial(n, t_max, initial_states=dg.get_burned_in_states(n, 50, mu))
for t in range(t_max):
    trial.step_forward_in_time(mu, apply_dropout=True)

n_remaining = trial.engaged_inds.sum()
plt.hist(trial.T_dis[trial.T_dis != np.inf], bins=t_max)
plt.title(f"Disengagement times (n={n}, n_remaining={n_remaining})")
plt.show()

plot_feature_over_time("stress", trial)
plot_feature_over_time("fatigue", trial)
plot_feature_over_time("glucose", trial)
plot_feature_over_time("insulin-1", trial)