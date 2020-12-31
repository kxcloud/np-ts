import numpy as np
from scipy.special import softmax
from scipy.optimize import minimize 

import data_generation as dg

np.set_printoptions(precision=3)

class Policy():
    
    def __init__(self, beta_0, state_encoding=None):
        self.beta = beta_0
        
        if state_encoding is None:
            state_encoding = lambda x : x
        self.state_encoding = state_encoding
        
        self.param_size = len(self.beta)
        
    def act(self, states):
        action_probs = softmax(self.state_encoding(states) @ self.beta, axis=1)
        return action_probs
    
    def prob_of(self, state, action):
        """ 
        Convenience function to get probabilities of state action pairs. 
        """
        action_probs = self.act(np.expand_dims(state,axis=0))
        return action_probs[0][int(action)]
        
def policy_eval_oracle(policy, n, t_max, mu_burn=None):
    trial = dg.DiabetesTrial(n, t_max, initial_states=dg.get_burned_in_states(n, mu_burn))
    for t in range(t_max):
        trial.step_forward_in_time(policy, apply_dropout=True)
    return trial.get_returns().mean()


N_ACTIONS = 4

t_max = 48  
n = 100
beta_0 = np.zeros((len(dg.feature_names), N_ACTIONS))
mu_burn = Policy(beta_0)

for k in range(0):
    beta_0 = np.random.uniform(size=(len(dg.feature_names), N_ACTIONS))
    avg_return = policy_eval_oracle(Policy(beta_0), n=1000, t_max=t_max)
    print(beta_0)
    print(avg_return)
    
trial = dg.DiabetesTrial(n, t_max, initial_states=dg.get_burned_in_states(n))
for t in range(t_max):
    trial.step_forward_in_time(policy=None, apply_dropout=True)

policy = Policy(beta_0.copy())    

def get_value_fn_optimizer(trial, policy, discount=0.99):
    """
    Precompute terms used for calculating the mean square Bellman Error, then
    return the objective function.
    """
    total_num_actions_taken = 0
    for i in range(trial.n):
        total_num_actions_taken += trial.num_treatments_applied(i)
    
    all_phi_terms = np.zeros((total_num_actions_taken, policy.param_size), dtype=np.float64)
    all_rewards = np.zeros((total_num_actions_taken), dtype=np.float64)
    term_idx = 0
    for i in range(trial.n):
        last_t = trial.num_treatments_applied(i)
        phi_s_next = None
        for t in range(last_t):
            phi_s = phi_s_next if t > 0 else policy.state_encoding(trial.S[i,t,:])
            phi_s_next = policy.state_encoding(trial.S[i,t+1,:]) if t < trial.T_dis[i] else 0
            
            imp_weight = (
                policy.prob_of(trial.S[i,t,:], trial.A[i,t]) 
                / trial.A_prob[i, t]
            )
            
            all_phi_terms[term_idx, :] = imp_weight * (
                discount * phi_s_next - phi_s
            )
            all_rewards[term_idx] = imp_weight * trial.R[i,t]
            term_idx += 1

    def mean_square_bellman_error(theta):
        return np.sum((all_rewards + all_phi_terms @ theta)**2)/trial.n
    
    return mean_square_bellman_error, all_phi_terms, all_rewards

msbe, all_phi_terms, all_rewards = get_value_fn_optimizer(trial, policy)
bfgs_obj = minimize(msbe, np.zeros(len(dg.feature_names)))
theta_hat1 = bfgs_obj.x

from sklearn.linear_model import LinearRegression
reg = LinearRegression(fit_intercept=False).fit(-all_phi_terms, all_rewards)

theta_hat2 = reg.coef_

print(theta_hat1)
print(theta_hat2)
print(f"Abs total diff: {np.abs(theta_hat1-theta_hat2).sum()}")