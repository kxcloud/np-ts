import os

import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy.special import expit
from scipy.optimize import minimize

import learning
import DiabetesTrial as dt
from Gridworld import Gridworld
import BanditTrial as b_t

project_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
tests_path = os.path.join(project_path,"tests")

TEST_SETTINGS = ["test","discount", "dropout", "behavior_policy", "target_policy"]
COL_NAMES = TEST_SETTINGS + ["estimated_value"]

class HeuristicPolicy():
    
    def __init__(self, num_actions, decision_rule, epsilon=0):
        """ Add epsilon chance of random move. """
        self.num_actions = num_actions
        self.decision_rule = decision_rule
        self.epsilon = epsilon
        
    def act(self, states, apply_encoding=None):
        action_probs = np.full(
            shape=(len(states), self.num_actions), 
            fill_value = self.epsilon/self.num_actions
        )
        
        for i, state in enumerate(states):
            action_probs[i, self.decision_rule(state)] += 1 - self.epsilon
            
        return action_probs
    
class OracleBanditPolicy():
    
    def __init__(self, reward_fn):
        self.reward_fn = reward_fn
    
    def act(self, states, apply_encoding=None):
        reward_per_arm = self.reward_fn(states)
        n, num_actions = reward_per_arm.shape
            
        action_probs = np.zeros((n, num_actions))

        for i, potential_rewards in enumerate(reward_per_arm):
            best_a = np.argmax(potential_rewards)
            action_probs[i, best_a] = 1
        return action_probs


def compare_values(results, err_tol):
    abs_errors = np.abs(results["estimated_value"] - results["anticipated_value"])
    return (abs_errors > err_tol).apply(lambda failed: "x" if failed else "")


def save_test_with_anticipated_values(test_name, results, err_tol=0.05):
    filepath = os.path.join(tests_path, f"{test_name}.csv")
    
    if os.path.exists(filepath):
        old_results = pd.read_csv(filepath)[TEST_SETTINGS+["anticipated_value"]]
        n_anticipated = len(old_results) - old_results["anticipated_value"].isna().sum()
        print(
            f"Reading {n_anticipated} of {len(old_results)} anticipated " 
            f"values from {filepath}."
        )
        results = results.merge(
            old_results,
            on=TEST_SETTINGS, 
            validate="one_to_one"
        )
        results["failed"] = compare_values(results, err_tol=err_tol)
    else:
        results["anticipated_value"] = None
    
    results.to_csv(filepath, index=False)
    print(f"Saved {test_name} results to {filepath}.")

def get_two_state_bandit(n, t_max):
    context_dist = b_t.DiscreteContextDistribution([[0,1],[1,0]])
    
    def reward_fn(contexts):
        return contexts
            
    trial = b_t.BanditTrial(
        n, 
        t_total=t_max, 
        num_actions=2, 
        context_distribution=context_dist,
        reward_function=reward_fn,
        safety_function=None
    )
    return trial
    

def test_two_state_bandit():
    test_name = "2-state bandit"
    trial = get_two_state_bandit(n=1, t_max=1)
    
    encoding = lambda x: x
    
    target_policies = {
        "worst": HeuristicPolicy(len(trial.action_space), lambda x: int(np.argmin(x))),
        "action_0_always": HeuristicPolicy(len(trial.action_space), lambda x: 0),
        "uniform_random": HeuristicPolicy(len(trial.action_space), lambda x: 0, epsilon=1),
        "oracle": OracleBanditPolicy(trial.reward_function)
    }
        
    records = []
    discounts = [0,0.5,1]
    
    behavior_policy = HeuristicPolicy(len(trial.action_space), lambda x: 0, epsilon=0.5)
    behavior_policy_name = "biased"
    
    for dropout_setting in [True,False]:
        trial = get_two_state_bandit(n=2000, t_max=1)
        trial.step_forward_until_end(policy=behavior_policy, apply_dropout=dropout_setting)
    
        for discount in discounts:
            value_estimator = learning.get_value_estimator(
                trial, 
                discount=discount, 
                state_encoding_value_fn=encoding
            )
            
            for target_policy_name, target_policy in target_policies.items():
                theta_hat = value_estimator(target_policy)
                v_est = np.mean(encoding(trial.S[:,0,:]) @ theta_hat)
                record = {
                    "test": test_name,
                    "discount" : discount,
                    "dropout" : dropout_setting,
                    "behavior_policy" : behavior_policy_name,
                    "target_policy" : target_policy_name,
                    "estimated_value" : v_est
                }
                records.append(record)

    results = pd.DataFrame.from_records(records, columns=COL_NAMES)
    results = results.sort_values(by=["dropout","target_policy","discount"])
    return results

def get_two_state_bernoulli_bandit(n, t_max):
    context_dist = b_t.DiscreteContextDistribution([[0,1],[1,0]])
    
    def reward_fn(contexts):
        return contexts*0.8
            
    trial = b_t.BernoulliBandit(
        n, 
        t_total=t_max, 
        num_actions=2, 
        context_distribution=context_dist,
        reward_function=reward_fn,
        safety_function=None
    )
    return trial
    

def test_two_state_bernoulli_bandit():
    test_name = "2-state bernoulli bandit"
    trial = get_two_state_bernoulli_bandit(n=1, t_max=1)
    
    encoding = lambda x: x
    
    target_policies = {
        "worst": HeuristicPolicy(len(trial.action_space), lambda x: int(np.argmin(x))),
        "action_0_always": HeuristicPolicy(len(trial.action_space), lambda x: 0),
        "uniform_random": HeuristicPolicy(len(trial.action_space), lambda x: 0, epsilon=1),
        "oracle": OracleBanditPolicy(trial.reward_function)
    }
        
    records = []
    discounts = [0,0.5,1]
    
    behavior_policy = HeuristicPolicy(len(trial.action_space), lambda x: 0, epsilon=0.5)
    behavior_policy_name = "biased"
    
    for dropout_setting in [True,False]:
        trial = get_two_state_bernoulli_bandit(n=2000, t_max=1)
        trial.step_forward_until_end(policy=behavior_policy, apply_dropout=dropout_setting)
    
        for discount in discounts:
            value_estimator = learning.get_value_estimator(
                trial, 
                discount=discount, 
                state_encoding_value_fn=encoding
            )
            
            for target_policy_name, target_policy in target_policies.items():
                theta_hat = value_estimator(target_policy)
                v_est = np.mean(encoding(trial.S[:,0,:]) @ theta_hat)
                record = {
                    "test": test_name,
                    "discount" : discount,
                    "dropout" : dropout_setting,
                    "behavior_policy" : behavior_policy_name,
                    "target_policy" : target_policy_name,
                    "estimated_value" : v_est
                }
                records.append(record)

    results = pd.DataFrame.from_records(records, columns=COL_NAMES)
    results = results.sort_values(by=["dropout","target_policy","discount"])
    return results

def get_continuous_deterministic_bandit(n, t_max=1):
    p = 3
    context_dist = b_t.ContinuousContextDistribution(p, "normal")
    
    num_actions = 4
    def reward_fn(contexts):
        return contexts @ np.ones(shape=(3,num_actions))
        
    trial = b_t.BanditTrial(
        n, 
        t_total=t_max, 
        num_actions=p, 
        context_distribution=context_dist,
        reward_function=reward_fn,
        safety_function=None
    )
    return trial

def test_continuous_deterministic_bandit():
    test_name = "continuous deterministic bandit"
    trial = get_continuous_deterministic_bandit(n=2000, t_max=1)
    
    rbf = learning.get_rbf()
    encoding = lambda x: learning.add_interactions(rbf(x))
    
    target_policies = {
        "worst": HeuristicPolicy(len(trial.action_space), lambda x: int(np.argmin(x))),
        "action_0_always": HeuristicPolicy(len(trial.action_space), lambda x: 0),
        "uniform_random": HeuristicPolicy(len(trial.action_space), lambda x: 0, epsilon=1),
        "oracle": OracleBanditPolicy(trial.reward_function)
    }
        
    records = []
    discounts = [0,0.5,1]
    
    behavior_policy = HeuristicPolicy(len(trial.action_space), lambda x: 0, epsilon=1)
    behavior_policy_name = "uniform_random"
    
    for dropout_setting in [True, False]:
        trial = get_continuous_deterministic_bandit(n=2000, t_max=1)
        trial.step_forward_until_end(policy=behavior_policy, apply_dropout=dropout_setting)
    
        for discount in discounts:
            value_estimator = learning.get_value_estimator(trial, state_encoding_value_fn=encoding, discount=discount)
            
            for target_policy_name, target_policy in target_policies.items():
                theta_hat = value_estimator(target_policy)
                v_est = np.mean(encoding(trial.S[:,0,:]) @ theta_hat)
                record = {
                    "test": test_name,
                    "discount" : discount,
                    "dropout" : dropout_setting,
                    "behavior_policy" : behavior_policy_name,
                    "target_policy" : target_policy_name,
                    "estimated_value" : v_est
                }
                records.append(record)

    results = pd.DataFrame.from_records(records, columns=COL_NAMES)
    results = results.sort_values(by=["dropout","target_policy","discount"])
    return results


def get_bernoulli_bandit(n, t_max):
    # context_dist = b_t.DiscreteContextDistribution([[0,1],[1,0]])
    p = 3
    context_dist = b_t.ContinuousContextDistribution(p, "normal")

    
    def reward_fn(contexts):
        # Scale to 0, 1
        tmp = contexts-contexts.min(axis=0, keepdims=True)
        return tmp / (tmp.max(axis=0, keepdims=True)+0.1)
            
    trial = b_t.BernoulliBandit(
        n, 
        t_total=t_max, 
        num_actions=p, 
        context_distribution=context_dist,
        reward_function=reward_fn,
        safety_function=None
    )
    return trial
    

def test_bernoulli_bandit():
    test_name = "bernoulli bandit"
    trial = get_bernoulli_bandit(n=2000, t_max=1)
    
    rbf = learning.get_rbf()
    encoding = lambda x: learning.add_interactions(rbf(x))
    
    target_policies = {
        "worst": HeuristicPolicy(len(trial.action_space), lambda x: int(np.argmin(x))),
        "action_0_always": HeuristicPolicy(len(trial.action_space), lambda x: 0),
        "uniform_random": HeuristicPolicy(len(trial.action_space), lambda x: 0, epsilon=1),
        "oracle": OracleBanditPolicy(trial.reward_function)
    }
        
    records = []
    discounts = [0,0.5,1]
    
    behavior_policy = HeuristicPolicy(len(trial.action_space), lambda x: 0, epsilon=1)
    behavior_policy_name = "uniform_random"
    
    for dropout_setting in [True, False]:
        trial = get_bernoulli_bandit(n=2000, t_max=1)
        trial.step_forward_until_end(policy=behavior_policy, apply_dropout=dropout_setting)
    
        for discount in discounts:
            value_estimator = learning.get_value_estimator(trial, state_encoding_value_fn=encoding, discount=discount)
            
            for target_policy_name, target_policy in target_policies.items():
                theta_hat = value_estimator(target_policy)
                v_est = np.mean(encoding(trial.S[:,0,:]) @ theta_hat)
                record = {
                    "test": test_name,
                    "discount" : discount,
                    "dropout" : dropout_setting,
                    "behavior_policy" : behavior_policy_name,
                    "target_policy" : target_policy_name,
                    "estimated_value" : v_est
                }
                records.append(record)

    results = pd.DataFrame.from_records(records, columns=COL_NAMES)
    results = results.sort_values(by=["dropout","target_policy","discount"])
    return results

if __name__ == "__main__":    
    # res = test_two_state_bandit()
    # save_test_with_anticipated_values("2_state_bandit", res)

    # res = test_two_state_bernoulli_bandit()
    # save_test_with_anticipated_values("2_state_bernoulli_bandit", res)
    
    
    res = test_continuous_deterministic_bandit()
    save_test_with_anticipated_values("2_continuous_bandit", res)
    
    # bern_bandit = get_bernoulli_bandit(10,1)
    
    # res2 = test_bernoulli_bandit()
    # save_test_with_anticipated_values("bernoulli_bandit", res2)
    # Next test: hand-coded bandit that is more complicated, but reward function is understood
    
    # Add Diabetes trial fixed policy value est to this test suite
    

if __name__ == "__main__" and False:
    
    ## TEST: contextual bandit
    n = 5
    t_max = 1
    num_actions = 2
    p = 5
    context_dist = b_t.ContinuousContextDistribution(p, "normal")
    # context_dist = DiscreteContextDistribution([0,1],[1,0])
    reward_fn = b_t.get_linear_reward_function(matrix_shape=(p,num_actions))
    trial = b_t.BanditTrial(
        n, 
        t_max, 
        num_actions=num_actions, 
        context_distribution=context_dist,
        reward_function=reward_fn,
    )
    
    dropout=False
    encoding = lambda x : x
    n_actions = len(trial.action_space)
    beta_0 = np.zeros((trial.infer_encoding_size(encoding), n_actions))
    unif_policy = learning.Policy(beta_0, state_encoding=encoding)    
    disc = 1
        
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
        "Optimal" : OracleBanditPolicy(reward_fn)
    }
    
    for label, pol in policies.items():
        newtrial = type(trial)(
            n=50000, 
            t_total=t_max, 
            num_actions=num_actions, 
            context_distribution=context_dist,
            reward_function=reward_fn
        ) # will need to generalize
        mc_value, mc_std = learning.mc_value_est(
            pol, disc, new_trial=newtrial, apply_dropout=dropout
        )
        trials[label] = newtrial
        printstr = label + " policy out-of-sample MC value estimate:"
        print(f"{printstr:<60} {mc_value:8.2f} ± {mc_std:2.2f}")