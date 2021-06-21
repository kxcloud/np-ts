import copy

import numpy as np

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
    trial_subset.safety = trial_subset.safety[patient_indices]
    return trial_subset 

def split_trial(trial, n_1=None):
    """ Split a trial into n_1 and trial.n - n_1 trials """
    if n_1 is None:
        n_1 = int(trial.n/2)

    trial_1 = subset_trial(trial, range(n_1))
    trial_2 = subset_trial(trial, range(n_1, trial.n))
    return trial_1, trial_2

if __name__ == "__main__":
    # CONTEXTUAL BANDIT WITH SAFETY
    # In this setting, a context X contains two bits:
    #    x[0] is an indicator for which of two arms has a reward;
    #    x[1] is an indicator for whether it is safe to pull arm 1.
    # (Arm 0 is always safe to pull.)
    
    context_space = [[0,0],[0,1],[1,0],[1,1]]
    context_dist = b_t.DiscreteContextDistribution(context_space)
    
    def reward_fn(contexts):
        reward_per_arm = np.zeros(shape=(len(contexts), 2))
        for i, context in enumerate(contexts):
            if context[0]:
                reward_per_arm[i,0] = 1
            else:
                reward_per_arm[i,1] = 1
        return reward_per_arm
        
    def safety_fn(contexts):
        safety_per_arm = np.zeros(shape=(len(contexts), 2))
        for i, context in enumerate(contexts):
            if context[1]:
                safety_per_arm[i,:] = 1
            else:
                safety_per_arm[i,0] = 1
        return safety_per_arm
        
    n_actions = 2
    bandit_params = {
        "t_total" : 1,
        "num_actions" : n_actions,
        "context_distribution" : context_dist,
        "reward_function" : reward_fn,
        "safety_function" : safety_fn
    }
    
    trial = b_t.BanditTrial(n=200, **bandit_params)
    trial.step_forward_in_time(policy=None, apply_dropout=True)
    trial_1, trial_2 = split_trial(trial)
    
    encoding = lambda x : x   
    beta_0 = np.zeros((trial.infer_encoding_size(encoding), n_actions))
    unif_policy = learning.Policy(beta_0, state_encoding=encoding)    
        
    num_ts_samples = 10
    
    policies = []
    infos = []
    for _ in range(num_ts_samples):
        policy, info = t_s.get_ts_policy(trial_1, 1, encoding, unif_policy, policy_penalty=0.01)
        policies.append(policy)
        infos.append(info)

    policy.act(np.array(context_space)) # check action probs

    # NEXT: hypothesis test safety of each test

    # Why are tests failing?

    # Steps:
    # 1. Set up basic bandit problem with safety (done)
    # 2. Split data in half (done)
    # 3. Apply TS (on rewards) to generate policies on first half (done)
    # 4. Apply B-H (on safety) with dependence correction on second half
    # 5. Check actual safety of policies

    # Next: non-sample split test
