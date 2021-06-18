import copy

import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy.special import comb
from scipy.optimize import minimize

import learning
import BanditTrial as b_t
import thompson_sampling as t_s

def subset_trial(trial, patient_indices):
    trial_subset = copy.deepcopy(trial)
    trial_subset.n = len(patient_indices)
    trial_subset.S = trial_subset.S[patient_indices]
    trial_subset.A = trial_subset.A[patient_indices]
    trial_subset.A_prob = trial_subset.A_prob[patient_indices]
    trial_subset.R =  trial_subset.R[patient_indices]
    trial_subset.T_dis = trial_subset.T_dis[patient_indices]
    trial_subset.engaged_inds = trial_subset.engaged_inds[patient_indices] 
    return trial_subset 

def split_trial(trial, n_1=None):
    """ Split a trial into n_1 and trial.n - n_1 trials """
    if n_1 is None:
        n_1 = int(trial.n/2)

    trial_1 = subset_trial(trial, range(n_1))
    trial_2 = subset_trial(trial, range(n_1, trial.n))
    return trial_1, trial_2

if __name__ == "__main__":
    
    # SETUP
    t_max = 1
    n = 11
    num_actions = 2
    p = 5
    context_dist = b_t.ContinuousContextDistribution(p, "normal")
    # context_dist = b_t.DiscreteContextDistribution([0,1],[1,0])
    reward_fn = b_t.get_linear_reward_function(matrix_shape=(p,num_actions))
    safety_fn = b_t.get_linear_threshold_function(matrix_shape=(p,num_actions))
    trial = b_t.BanditTrial(
        n, 
        t_max, 
        num_actions=num_actions, 
        context_distribution=context_dist,
        reward_function=reward_fn,
        safety_function=safety_fn
    )
    encoding = lambda x : x
    n_actions = len(trial.action_space)
   
    
    trial = b_t.get_easy_bandit(n=13, t_max=1)
    
    beta_0 = np.zeros((trial.infer_encoding_size(encoding), n_actions))
    unif_policy = learning.Policy(beta_0, state_encoding=encoding)    
    disc = 1
            
    # GENERATE DATA
    trial.step_forward_in_time(policy=unif_policy, apply_dropout=False)

    trial_1, trial_2 = split_trial(trial)
        
    policy, info = t_s.get_ts_policy(trial_1, unif_policy, discount=1, policy_penalty=0.01)
    print(info)

    
    # TODO: 
    # write bandit tests (odd # of patients a problem??)
    # write docs on tests
    
    # test that thompson sampling works (on easy bandit)
    # run TS in loop
    
    # hypothesis test for safety:
    # first, figure out how to test a fixed hypothesis
    # second, figure out how to test multiple




if False:        
    value_estimator, policy_loss = learning.get_value_estimator(
        trial_1, unif_policy, disc, policy_penalty=0.01
    )
    theta_hat = value_estimator(beta_0)

    value_est_at_t0 = np.mean(encoding(trial.S[:,0,:]) @ theta_hat)
    mc_value = trial.get_returns(discount=disc).mean()
    print(f"Est. value fn param: {theta_hat.round(3)}")
    print(f"{'Avg estimated value across starting states:':<50} {value_est_at_t0:8.2f}")
    print(f"{'Uniform policy in-sample MC value estimate:':<50} {mc_value:8.2f}")
    
    result = minimize(policy_loss, beta_0, method="BFGS", options={'gtol':1e-5, 'disp':True})
    
    beta_hat = result.x.reshape(beta_0.shape)
    est_opt_val = -result.fun # Note: doesn't account for penalty.
    est_policy = learning.Policy(beta_hat, state_encoding=encoding)
    
    trials = {}
    policies = {
        "Uniform" : unif_policy,
        "Estimated" : est_policy,
    }
    
    for label, pol in policies.items():
        newtrial = type(trial)(n=5000, t_total=t_max, p=5, num_actions=2) # will need to generalize
        mc_value, mc_std = learning.mc_value_est(
            pol, disc, new_trial=newtrial, apply_dropout=dropout
        )
        trials[label] = newtrial
        printstr = label + " policy out-of-sample MC value estimate:"
        print(f"{printstr:<60} {mc_value:8.2f} ± {mc_std:2.2f}")