import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy.special import comb
from scipy.optimize import minimize

import DiabetesTrial as dt
from Gridworld import Gridworld
from BanditTrial import BanditTrial

pd.options.display.width = 0
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


class SimpleInsulinPolicy():
    
    def act(self, states, apply_encoding=None):
        action_probs = np.zeros((states.shape[0], 4))
        for i, state in enumerate(states):
            glucose = state[0]
            if glucose > 100:
                action_probs[i, 1] = 1
            else:
                action_probs[i, 0] = 1
        return action_probs

def mc_value_est(
        policy, 
        discount,
        apply_dropout,
        new_trial=None, 
        trial_type=None, 
        n=1000,
        t_total=1000, 
    ):
    """ Evaluate a policy by running it on a new trial. """
    
    assert_msg = "Pass either a new trial to run or a trial type to create."
    assert (new_trial is None) + (trial_type is None) == 1, assert_msg
    
    if new_trial is None:
        sim_trial = trial_type(n, t_total)
    else:
        sim_trial = new_trial
        n = new_trial.n
        t_total = new_trial.t_total
        
    for t in range(t_total):
        sim_trial.step_forward_in_time(policy, apply_dropout=apply_dropout)
    
    returns = sim_trial.get_returns(discount=discount)
    mean = returns.mean()
    ci_width = 1.96 * returns.std() / np.sqrt(n)
    return mean, ci_width

def get_policy_probs(policy, encoded_states, actions):
    all_action_probs = policy.act(encoded_states, apply_encoding=False)
    total_num_actions_taken = len(actions)
    selected_action_probs = np.zeros(total_num_actions_taken)
    for idx in range(total_num_actions_taken):
        selected_action_probs[idx] = all_action_probs[idx,actions[idx]]
    return selected_action_probs

def get_value_estimator(
        trial, policy, discount, bootstrap_weights = None,
        policy_penalty=0, verbose=True
    ):
    """ 
    Precompute terms for value function estimation, then return a function
    which takes a policy parameter and returns the estimated value.
    """

    # Precompute as much as possible.
    psi_S = [
        policy.state_encoding(trial.S[i,:trial.last_time_index(i)+1]) 
        for i in range(trial.n)
    ]
    bootstrap_weights = (
        np.ones(trial.n) if bootstrap_weights is None else bootstrap_weights
    )
    p = trial.infer_encoding_size(policy.state_encoding)
    n_actions = [trial.num_treatments_applied(i) for i in range(trial.n)]
    actions = [trial.A[i,:n_actions[i]].astype(int) for i in range(trial.n)]
    
    # The probability-weighed sum of these terms form the A and b in Ax=b.
    matrix_summands = [np.zeros((n_actions[i], p, p)) for i in range(trial.n)]
    vector_summands = [np.zeros((n_actions[i], p)) for i in range(trial.n)]
    for i in range(trial.n):         
        for t in range(n_actions[i]):
            psi_s = psi_S[i][t]
            psi_s_next = psi_S[i][t+1] if t < trial.T_dis[i] else 0
            
            matrix_summands[i][t,:,:] = np.outer(
                (bootstrap_weights[i] / trial.A_prob[i,t]) * psi_s,
                psi_s - discount * psi_s_next
            )
            
            vector_summands[i][t,:] = (trial.R[i,t]/trial.A_prob[i,t]) * psi_s
        
    def value_estimator(policy_param):
        """ 
        Estimate the value for the policy given by policy_param. Returns 
        fitted scikit-learn regression object.
        """
        policy = Policy(beta_0=policy_param)
        feature_matrix = np.zeros((p, p), dtype=float)
        reward_vector = np.zeros(p, dtype=float)
        
        for i in range(trial.n):
            policy_probs = get_policy_probs(policy, psi_S[i], actions[i]) #TODO: vectorize this across patients?
            feature_matrix += np.tensordot(policy_probs, matrix_summands[i], axes=(0,0))
            reward_vector += np.tensordot(policy_probs, vector_summands[i], axes=(0,0))
        theta_hat = np.linalg.solve(feature_matrix/trial.n, reward_vector/trial.n)
        return theta_hat
    
    psi_S_0 = policy.state_encoding(trial.S[:,0,:])
    
    def policy_loss(policy_param):
        theta_hat = value_estimator(policy_param.reshape(policy.beta.shape))
        mean_value = np.mean(psi_S_0 @ theta_hat)
        return -mean_value + policy_penalty * np.sum(policy_param**2)
        
    return value_estimator, policy_loss

if __name__ == "__main__":
    t_max = 48
    n = 200
    dropout = False
    trial = dt.DiabetesTrial(n, t_max)
    # trial = Gridworld(n, t_max)
    # trial = BanditTrial(n, t_max, p=5, num_actions=2)
    feature_scaler = get_feature_scaler(trial)
    rbf = get_rbf()
    encoding = lambda x : add_intercept(feature_scaler(x))
    # encoding = lambda x : np.ones(shape=(x.shape[0], 1))
    # encoding = add_intercept
    n_actions = len(trial.action_space)
    beta_0 = np.zeros((trial.infer_encoding_size(encoding), n_actions))
    # beta_0 = np.random.normal(scale=1, size=beta_0.shape)
    unif_policy = Policy(beta_0, state_encoding=encoding)    
    disc = 0.95
    
    beta_expert = np.zeros_like(beta_0)
    beta_expert[0,1] = 2 # Make insulin more likely in general
    beta_expert[0,[2,3]] = -2 # make encouraging messages unlikely
    beta_expert[1,[1,3]] = 5 # as glucose goes up, increase chance of insulin
    exp_policy = Policy(beta_expert, state_encoding=encoding) 
    
    for t in range(t_max):
        trial.step_forward_in_time(policy=unif_policy, apply_dropout=dropout)
        trial.test_indexing()
    print(f"{trial.n - trial.engaged_inds.sum()} of {trial.n} dropped out of {trial}.")
                
    value_estimator, policy_loss = get_value_estimator(trial, unif_policy, disc, policy_penalty=0.01)
    theta_hat = value_estimator(beta_0)

    value_est_at_t0 = np.mean(encoding(trial.S[:,0,:]) @ theta_hat)
    mc_value = trial.get_returns(discount=disc).mean()
    print(f"Est. value fn param: {theta_hat.round(3)}")
    print(f"{'Avg estimated value across starting states:':<50} {value_est_at_t0:8.2f}")
    print(f"{'Uniform policy in-sample MC value estimate:':<50} {mc_value:8.2f}")
    
    result = minimize(policy_loss, beta_0, method="BFGS", options={'gtol':1e-5, 'disp':True})
    
    beta_hat = result.x.reshape(beta_0.shape)
    est_opt_val = -result.fun # Note: doesn't account for penalty.
    est_policy = Policy(beta_hat, state_encoding=encoding)
    
    # Monte Carlo evaluations
    trials = {}
    policies = {
        "Uniform" : unif_policy,
        "Expert-Parametric" : exp_policy,
        "Expert-Heuristic" : SimpleInsulinPolicy(),
        "Estimated" : est_policy
    }
    
    for label, pol in policies.items():
        newtrial = type(trial)(n=5000, t_total=500)
        mc_value, mc_std = mc_value_est(pol, disc, new_trial=newtrial, apply_dropout=dropout)
        trials[label] = newtrial
        printstr = label + " policy out-of-sample MC value estimate:"
        print(f"{printstr:<60} {mc_value:8.2f} ± {mc_std:2.2f}")
        
    ylim = (dt.gluc_min, dt.gluc_max)
    for label, newtrial in trials.items():
        newtrial.plot_feature(
            "glucose", 
            hlines=[70,80,120,150], 
            t_max=50, 
            ylim=ylim,
            subtitle=f"({label} policy)"
        )

    # print(trials["Estimated"].get_patient_table(0).round(2))
    