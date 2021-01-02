import numpy as np

import Trial as trial

direction_dict = {"N" : (0,1), "E": (1,0), "S": (0,-1), "W": (-1,0)}

class Gridworld(trial.Trial):
    feature_names = ["x", "y"]
    action_space = ["N","E","S","W"]
    
    def __init__(
            self, 
            n, 
            t_total, 
            grid_shape = (10,10),
            target_loc = (0,0),
            initial_states=None, 
            compute_extras=True
        ):
        self.grid_shape = grid_shape
        self.target_loc = target_loc
        super().__init__(n, t_total, initial_states=None, compute_extras=True)
        
    
    def generate_initial_states(self):
        x_loc = np.random.randint(self.grid_shape[0], size=(self.n,1))
        y_loc = np.random.randint(self.grid_shape[1], size=(self.n,1))
        return np.hstack([x_loc, y_loc])
        
    def _apply_state_transition(self):
        # NOTE: assumes x, y in first two features are "x" and "y".
        S = self.S[self.engaged_inds,self.t,0:2] 
        A = self.get_A()
        
        S_next = S.copy()
        
        for i, (s,a) in enumerate(zip(S,A)):
            direction = self.action_space[int(a)]
            S_next[i,:] = s + direction_dict[direction]
        
        # Stay in bounds.
        S_next[:,self.s_idx["x"]] = (
            S_next[:,self.s_idx["x"]].clip(0,self.grid_shape[0]-1)
        )
        S_next[:,self.s_idx["y"]] = (
            S_next[:,self.s_idx["y"]].clip(0,self.grid_shape[1]-1)
        )
        
        self.set_S("x", S_next[:,0])
        self.set_S("y", S_next[:,1])

    def _compute_rewards(self):
        """ Penalty for distance to target point """
        x_loc, y_loc = self.get_S("x",self.t+1), self.get_S("y",self.t+1)
        
        euclidean_distance = np.sqrt(
            (x_loc - self.target_loc[0])**2 + (y_loc - self.target_loc[1])**2
        )
        self.R[self.engaged_inds,self.t] = 10-euclidean_distance.round(2)
        
if __name__ == "__main__":
    n = 5
    t_max = 10
    mu = None
    trial = Gridworld(n, t_max)
    for t in range(t_max):
        trial.step_forward_in_time(mu, apply_dropout=False)
        