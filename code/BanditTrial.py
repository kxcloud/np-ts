import numpy as np
from scipy.special import expit

import Trial as trial

class BanditTrial(trial.Trial):
    
    def __init__(
            self, 
            n, 
            t_total, 
            p,
            num_actions,
            initial_states=None, 
            compute_extras=True
        ):
        self.feature_names = [f"X_{j}" for j in range(p)]
        self.action_space = [f"A_{j}" for j in range(num_actions)]
        
        # The context times the reward param equals the reward per arm.
        # Choose based on a fixed random state so it is the same across
        # environment creations.
        random_state = np.random.RandomState(47)
        self.reward_param_linear = random_state.normal(size=(p,num_actions))
        super().__init__(n, t_total, initial_states=None, compute_extras=True)
        
    def generate_initial_states(self):
        context = np.random.normal(size=(self.n,self.p))
        return context
    
    def _apply_state_transition(self):
        context = np.random.normal(size=(self.n,self.p))
        self.S[:,self.t+1,:] = context

    def _compute_rewards(self):
        """ Penalty for distance to target point """
        reward_per_arm = self.S[:,self.t,:] @ self.reward_param_linear
        
        actions = self.get_A()
        
        for i in range(self.n):
           self.R[i,self.t] = reward_per_arm[i, int(actions[i])]

        
if __name__ == "__main__":
    n = 5
    t_max = 10
    mu = None
    trial = BanditTrial(n, t_max, p=5, num_actions=2)
    for t in range(t_max):
        trial.step_forward_in_time(mu, apply_dropout=False)
        trial.test_indexing()
    # trial.plot_feature_over_time("y")
        