import numpy as np

import Trial as trial


class ContextDistribution():
    
    def __init__(self):
        raise NotImplementedError
    
    def sample(self, n):
        raise NotImplementedError
        
class DiscreteContextDistribution(ContextDistribution):
    
    def __init__(self, context_space):
        """ 
        The context_space should be a list of numpy arrays of the same shape.
        """
        self.context_space = context_space
        self.num_contexts = len(self.context_space)
        self.p = len(self.context_space[0])
        
        for context in self.context_space:
            assert self.p == len(context)
        
    def sample(self, n):
        context_sample = np.empty(shape=(n,self.p))
        context_indices = np.random.choice(self.num_contexts, n)
        for i, context_idx in enumerate(context_indices):
            context_sample[i,:] = self.context_space[context_idx]
        return context_sample
        
class ContinuousContextDistribution(ContextDistribution):
    
    def __init__(self, p, dist_name):
        self.sample_method = getattr(np.random, dist_name)
        self.p = p
        
    def sample(self, n):
        return self.sample_method(size=(n, self.p))
        
        
def get_linear_reward_function(reward_matrix=None, matrix_shape=None):

    if reward_matrix is None:
        reward_matrix = np.random.normal(size=matrix_shape)
    
    def reward_function(context):
        return context @ reward_matrix
    
    return reward_function

def get_linear_threshold_function(reward_matrix=None, matrix_shape=None):

    if reward_matrix is None:
        reward_matrix = np.random.normal(size=matrix_shape)
    
    def reward_function(context):
        return ((context @ reward_matrix) > 0).astype(int)
    
    return reward_function

class BanditTrial(trial.Trial):
    
    def __init__(
            self, 
            n, 
            t_total, 
            num_actions,
            context_distribution,
            reward_function,
            safety_function=None
        ):
        self.context_distribution = context_distribution
        self.reward_function = reward_function      
        self.safety_function = safety_function

        self.feature_names = [f"X_{j}" for j in range(context_distribution.p)]
        self.action_space = [f"A_{j}" for j in range(num_actions)]
        
        super().__init__(n, t_total, initial_states=None, compute_extras=True)
        
        if self.safety_function is not None:
            self.safety = self.R.copy()
        
    def generate_initial_states(self):
        context = self.context_distribution.sample(self.n)
        return context
    
    def _apply_dropout(self):
        """ In a bandit, everyone drops out after the last action. """
        if self.t == self.t_total - 1:
            self.engaged_inds[:] = False
            self.T_dis[:] = self.t
        
    def _apply_state_transition(self):
        context = self.context_distribution.sample(self.engaged_inds.sum())
        self.S[self.engaged_inds,self.t+1,:] = context

    def _compute_rewards(self):
        # NOTE: we index directly instead of using self.get_S because
        # we want to get include patients who disengaged.
        contexts = self.S[:, self.t, :]
        actions = self.A[:, self.t] 
        
        reward_per_arm = self.reward_function(contexts)
                
        for i in range(self.n):
            self.R[i, self.t] = reward_per_arm[i,int(actions[i])]

        if self.safety_function is not None:
            safety_per_arm = self.safety_function(contexts)
            for i in range(self.n):
                self.safety[i, self.t] = safety_per_arm[i,int(actions[i])]
                
def get_easy_bandit(n, t_max, num_actions=2):
    context_dist = DiscreteContextDistribution([[0,1],[1,0]])
    
    def reward_fn(contexts):
        return contexts
            
    trial = BanditTrial(
        n, 
        t_total=t_max, 
        num_actions=num_actions, 
        context_distribution=context_dist,
        reward_function=reward_fn,
        safety_function=None
    )
    return trial
    
if __name__ == "__main__":
    t_total = 10
    trial = get_easy_bandit(n=10, t_max=t_total)

    mu = None
    for _ in range(t_total):
        trial.step_forward_in_time(mu, apply_dropout=True)
        trial.test_indexing()

if __name__ == "__main__" and False:
    n = 5
    t_max = 10
    num_actions = 2
    p = 5
    context_dist = ContinuousContextDistribution(p, "normal")
    # context_dist = DiscreteContextDistribution([0,1],[1,0])
    reward_fn = get_linear_reward_function(matrix_shape=(p,num_actions))
    safety_fn = get_linear_threshold_function(matrix_shape=(p,num_actions))
    trial = BanditTrial(
        n, 
        t_max, 
        num_actions=num_actions, 
        context_distribution=context_dist,
        reward_function=reward_fn,
        safety_function=safety_fn
    )
    mu = None
    for t in range(t_max):
        trial.step_forward_in_time(mu, apply_dropout=True)
        trial.test_indexing()
    # trial.plot_feature_over_time("y")
        