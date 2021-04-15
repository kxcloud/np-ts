import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy.special import comb
from scipy.optimize import minimize

import learning
import DiabetesTrial as dt
from Gridworld import Gridworld
from BanditTrial import BanditTrial

class OptimalBanditPolicy():
    
    def act(self, states, apply_encoding=None):
        n = states.shape[0]
        p = states.shape[1]
        num_actions = 2
            
        action_probs = np.zeros((n, num_actions))
        
        # Recreate true param
        random_state = np.random.RandomState(47)
        self.reward_param_linear = random_state.normal(size=(p,num_actions))
        
        reward_per_arm = states @ self.reward_param_linear
        
        for i, potential_rewards in enumerate(reward_per_arm):
            best_a = np.argmax(potential_rewards)
            action_probs[i, best_a] = 1
        return action_probs


if __name__ == "__main__":
    
    ## TEST: contextual bandit
    t_max = 20
    n = 200
    dropout = False
    trial = BanditTrial(n, t_max, p=5, num_actions=2)
    encoding = lambda x : x
    n_actions = len(trial.action_space)
    beta_0 = np.zeros((trial.infer_encoding_size(encoding), n_actions))
    unif_policy = learning.Policy(beta_0, state_encoding=encoding)    
    disc = 0.95
        
    for t in range(t_max):
        trial.step_forward_in_time(policy=unif_policy, apply_dropout=dropout)
        trial.test_indexing()
                
    value_estimator, policy_loss = learning.get_value_estimator(
        trial, unif_policy, disc, policy_penalty=0.01
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
        "Optimal" : OptimalBanditPolicy()
    }
    
    for label, pol in policies.items():
        newtrial = type(trial)(n=5000, t_total=t_max, p=5, num_actions=2) # will need to generalize
        mc_value, mc_std = learning.mc_value_est(
            pol, disc, new_trial=newtrial, apply_dropout=dropout
        )
        trials[label] = newtrial
        printstr = label + " policy out-of-sample MC value estimate:"
        print(f"{printstr:<60} {mc_value:8.2f} ± {mc_std:2.2f}")