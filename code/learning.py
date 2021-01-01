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
    
    def copy(self):
        return Policy(beta_0=self.beta, state_encoding = self.state_encoding)
        
def policy_eval_oracle(policy, n, t_max, mu_burn=None, discount=0.99):
    trial = dg.DiabetesTrial(n, t_max, initial_states=dg.get_burned_in_states(n, mu_burn))
    for t in range(t_max):
        trial.step_forward_in_time(policy, apply_dropout=True)
        trial.R[:,t] *= discount**t
    return trial.get_returns().mean()


N_ACTIONS = 4

t_max = 48  
n = 10000
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

def get_value_estimator(trial, policy, discount=0.99, ridge_penalty=0):
    feature_matrix, rewards, encoded_states, actions = get_optimization_terms(
        trial, policy.state_encoding, discount
    ) 
    policy = policy.copy()
    
    def value_estimator(policy_param):
        """ 
        Estimate the value for the policy given by policy_param. Returns 
        fitted scikit-learn regression object.
        """
        policy.beta = policy_param
        policy_probs = get_policy_probs(policy, encoded_states, actions)
    
        wt_feature_matrix = policy_probs[:,None] * feature_matrix
        wt_rewards = policy_probs * rewards
    
        reg = Ridge(fit_intercept=False, alpha=ridge_penalty)
        reg.fit(-wt_feature_matrix, wt_rewards)
        return reg
    
    return value_estimator
    
value_estimator = get_value_estimator(trial, policy, ridge_penalty=0)
reg = value_estimator(policy.beta)
theta_hat = reg.coef_

# Why are these different-- model misspecification?
value_est_at_t0 = np.mean(trial.S[:,0,:] @ theta_hat)
mc_value_est = policy_eval_oracle(policy, 1000, t_max=48)