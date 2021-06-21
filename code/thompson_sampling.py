import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize

import DiabetesTrial as dt
from Gridworld import Gridworld
import learning as l

def get_ts_policy(
        trial, 
        discount,
        state_encoding_value_fn, 
        example_policy,
        policy_penalty=0.01,
        beta_0=None, 
        mc_evaluation=False
    ):  
    value_estimator = l.get_value_estimator(
        trial, 
        discount, 
        state_encoding_value_fn, 
        example_policy.state_encoding,
        bootstrap_weights = np.random.exponential(scale=1,size=trial.n)
    )
    policy_loss = l.get_policy_loss(
        trial, example_policy, value_estimator, policy_penalty=policy_penalty
    )
    
    if beta_0 is None:
        beta_0 = np.zeros_like(example_policy.beta)
    
    t_start = time.time()
    result = minimize(
        # NOTE: Adam mentioned using Powell for the optimizer
        policy_loss, beta_0, method="BFGS", options={'maxiter':5, 'gtol':1e-5, 'disp':True}
    )
    duration = time.time() - t_start
    
    beta_hat = result.x.reshape(beta_0.shape)
    est_opt_val = -result.fun + policy_penalty * np.sum(beta_hat**2)
    
    policy = l.Policy(beta_hat, state_encoding=example_policy.state_encoding)
    info = {
        "duration": duration,
        "value_fn_evals": result.nfev,
        "est_val": est_opt_val
    }
    if mc_evaluation:
        mean, std = l.mc_value_est(policy, discount, trial_type=type(trial))
        info["mc_est_val"] = mean
        info["mc_est_std"] = std
    return policy, info

if __name__ == "__main__":
    t_max = 5
    n = 20
    dropout = False
    trial = dt.DiabetesTrial(n, t_max)
    # trial = Gridworld(n, t_max)
    feature_scaler = l.get_feature_scaler(trial)
    # rbf = l.get_rbf()
    encoding = lambda x : l.add_intercept(feature_scaler(x))
    # encoding = lambda x : np.ones(shape=(x.shape[0], 1))
    # encoding = lambda x: l.add_intercept(rbf(x))
    n_actions = len(trial.action_space)
    beta_0 = np.zeros((trial.infer_encoding_size(encoding), n_actions))
    # beta_0 = np.random.normal(scale=1, size=beta_0.shape)
    policy = l.Policy(beta_0, state_encoding=encoding)    
    disc = 0.95
    
    info_list = []
    for t in range(t_max):
        trial.step_forward_in_time(policy=policy, apply_dropout=dropout)
        policy, info = get_ts_policy(trial, disc, encoding, policy, mc_evaluation=False)
        info_list.append(info)
    
    info = pd.DataFrame(info_list)
    print(f"{trial.n - trial.engaged_inds.sum()} of {trial.n} dropped out of {trial}.")
    print(info)