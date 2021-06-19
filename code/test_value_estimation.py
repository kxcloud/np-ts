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

def test_value_estimation(
    trial_class,
    trial_constructor_args,
    n,
    t_total,
    target_policy_dict,
    monte_carlo_n=200,
    monte_carlo_t=300,
    error_tolerance=0.1,
    state_encoding_value_fn_dict=None,
    behavior_policy_dict=None,
    discount_list=None,
    dropout_list=None
):
    """ 
    Compare estimating equation value estimates with monte carlo estimates
    from simulation. Returns a Pandas DataFrame. 
    """
    if state_encoding_value_fn_dict is None:
        state_encoding_value_fn_dict = {"identity": lambda x: x}
    
    if behavior_policy_dict is None:
        behavior_policy_dict = {"uniform": None}
    
    if dropout_list is None:
        dropout_list = [True, False]
        
    if discount_list is None:
        discount_list = [0, 0.5, 1]

    # Get Monte Carlo estimates
    monte_carlo_results = {}
    for dropout_setting in dropout_list:
        for target_policy_name, target_policy in target_policy_dict.items():
            trial = trial_class(
                n=monte_carlo_n, 
                t_total=monte_carlo_t,
                **trial_constructor_args
            )
            trial.step_forward_until_end(
                policy=target_policy, apply_dropout=dropout_setting
            )
            for discount in discount_list:
                mc_value = trial.get_returns(discount=discount).mean()
                monte_carlo_results[(dropout_setting, target_policy, discount)] = mc_value
                
    records = []    
    # Outer loop: generate data
    for behavior_policy_name, behavior_policy in behavior_policy_dict.items():
        for dropout_setting in dropout_list:
            trial = trial_class(n=n, t_total=t_total, **trial_constructor_args)
            trial.step_forward_until_end(
                policy=behavior_policy, 
                apply_dropout=dropout_setting
            )
    
            # Middle loop: create value estimator
            for discount in discount_list:
                for encoding_name, encoding in state_encoding_value_fn_dict.items():
                    value_estimator = learning.get_value_estimator(
                        trial, 
                        state_encoding_value_fn=encoding, 
                        discount=discount
                    )
            
                    # Inner loop: estimate values and record data
                    for target_policy_name, target_policy in target_policy_dict.items():
                        theta_hat = value_estimator(target_policy)
                        v_est = np.mean(encoding(trial.S[:,0,:]) @ theta_hat)
                        mc_value = monte_carlo_results[(dropout_setting, target_policy, discount)]
                        
                        test_status = ""
                        if discount == 1 and dropout_setting == False:
                            test_status = "n/a"
                        elif abs(v_est - mc_value) <= error_tolerance:
                            test_status = "pass"
                        else:
                            test_status = "FAIL"
                    
                        record = {
                            "trial_type" : trial_class.__name__,
                            "discount" : discount,
                            "dropout" : dropout_setting,
                            "state_encoding" : encoding_name,
                            "behavior_policy" : behavior_policy_name,
                            "target_policy" : target_policy_name,
                            "estimated_value" : v_est,
                            "monte_carlo_value" : mc_value,
                            "status" : test_status
                        }
                        records.append(record)

    results = pd.DataFrame.from_records(records)
    results = results.sort_values(by=["dropout","target_policy","discount"])
    return results
    
if __name__ == "__main__":
    
    context_dist = b_t.DiscreteContextDistribution([[0,1],[1,0]])
    
    def reward_fn(contexts):
        return contexts
    
    params = {
        "num_actions": 2,
        "context_distribution" : context_dist,
        "reward_function" : reward_fn,
        "safety_function": None
    }
    
    target_policies = {
        "worst": HeuristicPolicy(2, lambda x: int(np.argmin(x))),
        "action_0_always": HeuristicPolicy(2, lambda x: 0),
        "uniform_random": HeuristicPolicy(2, lambda x: 0, epsilon=1),
        "oracle": OracleBanditPolicy(reward_fn)
    }
        
    a = test_value_estimation(b_t.BanditTrial, params, n=2000, t_total=1, target_policy_dict=target_policies)
    
    def reward_fn(contexts):
        return contexts * 0.8
    
    params["reward_function"] = reward_fn
    
    b = test_value_estimation(b_t.BernoulliBandit, params, n=2000, t_total=1, target_policy_dict=target_policies)
    
    