import numpy as np
from scipy.special import softmax
from scipy.optimize import minimize 
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression

import data_generation as dg

np.set_printoptions(precision=3)

class Policy():
    
    def __init__(self, beta_0, state_encoding=None):
        self.beta = beta_0
        
        if state_encoding is None:
            state_encoding = lambda x : x
        self.state_encoding = state_encoding
        
        self.param_size = len(self.beta)
        
    def act(self, states, apply_encoding=True):
        if apply_encoding:
            action_probs = softmax(self.state_encoding(states) @ self.beta, axis=1)
        else:
            action_probs = softmax(states @ self.beta, axis=1)
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

def get_optimization_terms(trial, state_encoding, discount):
    """ 
    Get feature matrix, reward vector, and other components for quick value 
    estimation and policy optimization. Includes weighting by behavior policy 
    but NOT by target.
    """
    total_num_actions_taken = 0
    for i in range(trial.n):
        total_num_actions_taken += trial.num_treatments_applied(i)
    
    feature_matrix = np.zeros((total_num_actions_taken, policy.param_size), dtype=np.float64)
    rewards = np.zeros((total_num_actions_taken), dtype=np.float64)
    encoded_states = np.zeros((total_num_actions_taken, policy.param_size))
    actions = np.zeros((total_num_actions_taken), dtype=int)
    term_idx = 0
    for i in range(trial.n):
        last_t = trial.num_treatments_applied(i)
        phi_s_next = None
        for t in range(last_t):
            phi_s = phi_s_next if t > 0 else state_encoding(trial.S[i,t,:])
            phi_s_next = state_encoding(trial.S[i,t+1,:]) if t < trial.T_dis[i] else 0
            
            feature_matrix[term_idx, :] = (
                discount * phi_s_next - phi_s
            ) / trial.A_prob[i, t]
            rewards[term_idx] = trial.R[i,t] / trial.A_prob[i, t]
            encoded_states[term_idx] = phi_s
            actions[term_idx] = int(trial.A[i,t])
            term_idx += 1
    
    return feature_matrix, rewards, encoded_states, actions

def get_policy_probs(policy, encoded_states, actions):
    all_action_probs = policy.act(encoded_states, apply_encoding=False)
    total_num_actions_taken = len(actions)
    selected_action_probs = np.zeros(total_num_actions_taken)
    for idx in range(total_num_actions_taken):
        selected_action_probs[idx] = all_action_probs[idx,actions[idx]]
    return selected_action_probs

def fit_value_fn(trial, policy, discount=0.99, ridge_penalty=0):
    feature_matrix, rewards, encoded_states, actions = get_optimization_terms(
        trial, policy.state_encoding, discount
    )
    
    policy_probs = get_policy_probs(policy, encoded_states, actions)
    
    wt_feature_matrix = policy_probs[:,None] * feature_matrix
    wt_rewards = policy_probs * rewards
    
    reg = Ridge(fit_intercept=False, alpha=ridge_penalty).fit(-wt_feature_matrix, wt_rewards)
    theta_hat2 = reg.coef_
    return theta_hat2
    
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

theta_hat3 = fit_value_fn(trial, policy)

msbe, all_phi_terms, all_rewards = get_value_fn_optimizer(trial, policy)
bfgs_obj = minimize(msbe, np.zeros(len(dg.feature_names)))
theta_hat1 = bfgs_obj.x

reg = LinearRegression(fit_intercept=False).fit(-all_phi_terms, all_rewards)

theta_hat2 = reg.coef_

print(theta_hat1.round(2))
print(theta_hat3.round(2))
print(f"Abs total diff: {np.abs(theta_hat1-theta_hat2).sum()}")