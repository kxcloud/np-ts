import numpy as np
import matplotlib.pyplot as plt

import data_generation as dg

def plot_feature_over_time(feature, trial, num_to_plot=10, hlines=[]):
    feature_over_time = trial.S[range(num_to_plot),:,trial.s_idx[feature]]
    fig, ax = plt.subplots()
    for value in hlines:
        ax.axhline(y=value, ls="--", lw=1, color="gray")
    ax.plot(feature_over_time.T)

    ax.set_title(f"Patient {feature} over time")
    ax.set_xlabel("time")
    ax.set_ylabel(feature)
    plt.show()

def plot_feature_dist(feature, trial):
    fig, ax = plt.subplots()
    all_feature = trial.S[:,:,trial.s_idx[feature]]
    flattened = all_feature[~np.isnan(all_feature)].flatten()
    ax.hist(flattened, bins=30, density=True)
    ax.set_xlabel(feature)
    ax.set_title(f"Distribution of {feature} across patients, time")
    plt.show()

t_max = 48  
n = 100
mu = None

trial = dg.DiabetesTrial(n, t_max, initial_states=dg.get_burned_in_states(n, mu))
for t in range(t_max):
    trial.step_forward_in_time(mu, apply_dropout=True)

n_remaining = trial.engaged_inds.sum()
plt.hist(trial.T_dis[trial.T_dis != np.inf], bins=t_max)
plt.title(f"Disengagement times (n={n}, n_remaining={n_remaining})")
plt.show()

plot_feature_over_time("stress", trial)
plot_feature_over_time("fatigue", trial)
plot_feature_over_time("glucose", trial, hlines=[80, 120])
plot_feature_over_time("insulin-1", trial)

plot_feature_dist("exercise", trial)
plot_feature_dist("unhealthy_ind", trial)