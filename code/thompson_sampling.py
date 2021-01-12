import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize

import DiabetesTrial as dt
import learning as l

def get_ts_policy(trial, policy, discount, policy_penalty=1):
    bs_weights = np.random.exponential(scale=1,size=trial.n)
    
    _ , policy_loss = l.get_value_estimator(
        trial, policy, discount, bootstrap_weights = bs_weights, 
        policy_penalty=policy_penalty, verbose=True
    )
    
    t_start = time.time()
    result = minimize(
        policy_loss, beta_0, method="BFGS", options={'maxiter':5, 'gtol':1e-3, 'disp':True}
    )
    duration = time.time() - t_start
    
    beta_hat = result.x.reshape(beta_0.shape)
    est_opt_val = -result.fun + policy_penalty * np.sum(beta_hat**2)
    
    policy = l.Policy(beta_hat, state_encoding=policy.state_encoding)
    info = {
        "est_val": est_opt_val, 
        "duration": duration,
        "value_fn_evals": result.nfev
    }
    return policy, info

if __name__ == "__main__":
    t_max = 10
    n = 200
    dropout = True
    trial = dt.DiabetesTrial(n, t_max)
    # trial = Gridworld(n, t_max)
    feature_scaler = l.get_feature_scaler(trial)
    # rbf = get_rbf()
    encoding = lambda x : l.add_intercept(feature_scaler(x))
    # encoding = lambda x : np.ones(shape=(x.shape[0], 1))
    # encoding = add_intercept
    n_actions = len(trial.action_space)
    beta_0 = np.zeros((trial.infer_encoding_size(encoding), n_actions))
    # beta_0 = np.random.normal(scale=1, size=beta_0.shape)
    policy = l.Policy(beta_0, state_encoding=encoding)    
    disc = 0.95
    
    info_list = []
    for t in range(t_max):
        trial.step_forward_in_time(policy=policy, apply_dropout=dropout)
        policy, info = get_ts_policy(trial, policy, discount=disc)
        info_list.append(info)
    
    info = pd.DataFrame(info_list)
    print(f"{trial.n - trial.engaged_inds.sum()} of {trial.n} dropped out of {trial}.")