import numpy as np
from scipy.special import softmax

import data_generation as dg

np.set_printoptions(precision=3)

class Policy():
    
    def __init__(self, theta_0, state_encoding=None):
        self.theta = theta_0
        
        if state_encoding is None:
            state_encoding = lambda x : x
        self.state_encoding = state_encoding
        
    def act(self, states):
        action_probs = softmax(self.state_encoding(states) @ self.theta, axis=1)
        return action_probs
        
def policy_eval_oracle(policy, n, t_max, mu_burn=None):
    trial = dg.DiabetesTrial(n, t_max, initial_states=dg.get_burned_in_states(n, mu_burn))
    for t in range(t_max):
        trial.step_forward_in_time(policy, apply_dropout=True)
    return trial.get_returns().mean()

N_ACTIONS = 4

t_max = 48  
n = 100
theta_0 = np.zeros((len(dg.feature_names), N_ACTIONS))
mu_burn = Policy(theta_0)

for k in range(0):
    theta_0 = np.random.uniform(size=(len(dg.feature_names), N_ACTIONS))
    avg_return = policy_eval_oracle(Policy(theta_0), n=1000, t_max=t_max)
    print(theta_0)
    print(avg_return)
    
trial = dg.DiabetesTrial(n, t_max, initial_states=dg.get_burned_in_states(n))
for t in range(t_max):
    trial.step_forward_in_time(policy=None, apply_dropout=True)