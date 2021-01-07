import numpy as np
from scipy.special import softmax
from scipy.special import comb
from sklearn.linear_model import Ridge

import DiabetesTrial as dt
from Gridworld import Gridworld

np.set_printoptions(precision=3)

def add_intercept(x):
    n_dims = len(x.shape)
    if n_dims == 1:
        return np.array([1,*x])
    if n_dims == 2:
        return np.hstack([np.ones((x.shape[0],1)), x])
    else:
        raise ValueError(f"Unexpected n_dims: {n_dims}.")

def one_hot_y(x):
    if len(x.shape) == 1:
        x = x[None,:]
    encoded_state = np.zeros((x.shape[0],10))
    for item_idx, y in enumerate(x[:,1]):
        encoded_state[item_idx, int(y)] = 1
    return encoded_state

def get_rbf(function=np.exp, centers=[-1,0,1], scale=1):
    # NOTE: change to vector of centers so features are combined?
    def rbf(x):
        num_samples, size = x.shape
        rbf_vals = np.zeros((num_samples, size, len(centers)))
        for k, center in enumerate(centers):
            rbf_vals[:,:,k] = function((np.abs(x-center))/scale)
    
        return rbf_vals.reshape((num_samples, size*len(centers)))
    return rbf

def add_interactions(x):
    num_samples, n_features = x.shape
    x_new = np.zeros((num_samples, n_features + comb(n_features,2, True)))
    x_new[:,:n_features] = x
    interaction_idx = 0
    for j in range(n_features):
        for k in range(j):
            x_new[:, n_features+interaction_idx] = x[:,j] * x[:,k]
            interaction_idx += 1
    return x_new

def get_feature_scaler(trial, t_max=None):
    t_max = t_max or trial.t + 1
    mean_vec = np.zeros(len(trial.feature_names))
    std_vec = np.zeros(len(trial.feature_names))
    for j, feature_name in enumerate(trial.feature_names):
        feature = trial.S[:,:t_max,trial.s_idx[feature_name]]
        mean_vec[j], std_vec[j] = np.nanmean(feature), np.nanstd(feature)
        
    def feature_scaler(x):
        return (x - mean_vec)/std_vec

    return feature_scaler

class Policy():
    
    def __init__(self, beta_0, state_encoding=None):
        self.beta = beta_0
        
        if state_encoding is None:
            state_encoding = lambda x : x
        self.state_encoding = state_encoding
        
        self.param_size = len(self.beta)
        
    def act(self, states, apply_encoding=True):
        if apply_encoding:
            if len(states.shape) == 1:
                states = np.expand_dims(states, axis=0)
            action_probs = softmax(self.state_encoding(states) @ self.beta, axis=1)
        else:
            action_probs = softmax(states @ self.beta, axis=1)
        return action_probs
       
    def copy(self):
        return Policy(beta_0=self.beta, state_encoding = self.state_encoding)
        

def get_optimization_terms(trial, state_encoding, discount):
    """ 
    Get feature matrix, reward vector, and other components for quick value 
    estimation and policy optimization. Includes weighting by behavior policy 
    but NOT by target.
    """
    total_num_actions_taken = 0
    for i in range(trial.n):
        total_num_actions_taken += trial.num_treatments_applied(i)
    
    encoding_size = trial.infer_encoding_size(state_encoding)
    
    feature_matrix = np.zeros((total_num_actions_taken, encoding_size), dtype=np.float64)
    rewards = np.zeros((total_num_actions_taken), dtype=np.float64)
    encoded_states = np.zeros((total_num_actions_taken, encoding_size))
    actions = np.zeros((total_num_actions_taken), dtype=int)
    term_idx = 0
    for i in range(trial.n):
        last_t = trial.num_treatments_applied(i)
        phi_s_next = None
        for t in range(last_t):
            # Note: indexing with None adds back the patient dimension which 
            # would otherwise be lost when selecting patient i.
            phi_s = phi_s_next if t > 0 else state_encoding(trial.S[None,i,t,:])
            phi_s_next = state_encoding(trial.S[None,i,t+1,:]) if t < trial.T_dis[i] else 0
            
            feature_matrix[term_idx, :] = (
                phi_s - discount * phi_s_next
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

def get_value_estimator(
        trial, 
        policy, 
        discount, 
        ridge_penalty=0
    ):
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

        reg = Ridge(alpha=ridge_penalty, fit_intercept=False)
        reg.fit(wt_feature_matrix, wt_rewards)
        return reg
    
    return value_estimator, feature_matrix/4, rewards/4

if __name__ == "__main__":  
    t_max = 20
    n = 300
    dropout = False
    # trial = Gridworld(n, t_max)
    trial = dt.DiabetesTrial(n, t_max)
    feature_scaler = get_feature_scaler(trial)
    rbf = get_rbf()
    encoding = lambda x : add_intercept(feature_scaler(x))
    # encoding = lambda x : np.ones(shape=(x.shape[0], 1))
    # encoding = add_intercept
    n_actions = len(trial.action_space)
    beta_0 = np.zeros((trial.infer_encoding_size(encoding), n_actions))
    # beta_0 = np.random.normal(scale=1, size=beta_0.shape)
    policy = Policy(beta_0, state_encoding=encoding)    
    disc = 0.95

    for t in range(t_max):
        trial.step_forward_in_time(policy=policy, apply_dropout=dropout)
        trial.test_indexing()
        
    value_estimator, w, r = get_value_estimator(trial, policy, discount=disc, ridge_penalty=0)
    reg = value_estimator(policy.beta)
    theta_hat = reg.coef_
    
    # Why are these different-- model misspecification?
    value_est_at_t0 = np.mean(encoding(trial.S[:,0,:]) @ theta_hat)
    t_max2 = t_max*100
    trial2 = dt.DiabetesTrial(n, t_total=t_max2)
    for t in range(t_max2):
        trial2.step_forward_in_time(policy=policy, apply_dropout=dropout)
    mc_value_est = trial.get_returns(discount=disc).mean()
    mc_value_est2 = trial2.get_returns(discount=disc).mean()
    print(f"Est. value fn param: {theta_hat.round(3)}")
    print(f"Avg estimated value across starting states: {value_est_at_t0:10.3f}")
    print(f"Monte Carlo value estimate (t={t_max:4.0f}):        {mc_value_est:10.3f}")
    print(f"Monte Carlo value estimate (t={t_max2:4.0f}):        {mc_value_est2:10.3f}")