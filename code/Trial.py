import numpy as np
import pandas as pd

def random_choice_idx(p):
    """ 
    Basically np.random.choice with different probabilities per item.
    """
    num_patients = p.shape[0]
    rand_per_item = np.random.uniform(size=(num_patients,1))
    action_index = (np.cumsum(p, axis=1) < rand_per_item).argmin(axis=1)
    
    action_prob = np.zeros(num_patients)
    for i in range(num_patients):
        action_prob[i] = p[i, action_index[i]]
    return action_index, action_prob

class Trial:
    """ 
    A generic object that simulates the progression of patients in a clinical 
    trial. Each patient is an instance of a Markov Decision Process.
    """
    feature_names = None 
    action_space = None

    def __init__(
            self, 
            n, 
            t_total,
            initial_states=None, 
            compute_extras=True
        ):
        self.n = n
        self.p = len(self.feature_names)
        self.t_total = t_total
        self.S = np.full((n, t_total + 1, self.p), np.nan) # States (patient features)
        self.A = np.full((n, t_total), np.nan) # Action index
        self.T_dis = np.full(n, np.inf) # Per-patient disengagement times
        
        # Save compute during burn-in by skipping calculation of utility.
        if compute_extras:
            self.R = np.full((n, t_total), np.nan) # Observed utilities
            self.A_prob = np.full((n, t_total), np.nan) # Action selection probabilities
        else: 
            self.R = None
            self.A_prob = None
        
        if initial_states is None:
           initial_states = self.generate_initial_states()
        self.S[:,0,:] = initial_states
    
        # Store a feature name -> feature index map for quick lookup.
        self.s_idx = { 
            feature_name : j 
            for j, feature_name in enumerate(self.feature_names)
        }
    
        # These attributes store values that change w.r.t. the current time.
        self.t = 0
        self.engaged_inds = np.full(n, True)
    
    def __repr__(self):
        return "{}(n={}, t_total={}, num_engaged={}, t={})".format(
            self.__class__.__name__, 
            self.n, self.t_total, np.sum(self.engaged_inds), self.t
        )
    
    def generate_initial_states(self):
        raise NotImplementedError()
    
    def check_time(self, t):
        if t is None:
            t = self.t
        elif t < 0:
            print(f"Warning: time index {t} is negative.")
        return t
            
    def get_S(self, feature_name, t=None):
        """ 
        Convenience function for getting the value of engaged patient features
        at the CURRENT time step.
        """
        t = self.check_time(t)
        return self.S[self.engaged_inds, t, self.s_idx[feature_name]]
        
    def set_S(self, feature_name, feature_value):
        """ 
        Convenience function for setting the value of engaged patient features
        at the NEXT time step.
        """
        self.S[self.engaged_inds, self.t+1, self.s_idx[feature_name]] = feature_value
    
    def get_A(self, t=None):
        t = self.check_time(t)
        return self.A[self.engaged_inds, t]
    
    def get_A_indicators(self, t=None):
        """ 
        Return array of action indicators, where the (i,j)-entry is the 
        indicator that patient i received treatment type j at time t.
        """
        t = self.check_time(t)
        A = self.get_A(t)
        num_remaining = np.sum(self.engaged_inds)
        num_action_types = len(self.action_space[0])
        A_indicators = np.zeros((num_remaining, num_action_types), dtype=int)
        for i, action_idx in enumerate(A):
            A_indicators[i,:] = self.action_space[int(action_idx)]
        return A_indicators
    
    def set_A(self, action_index):
        self.A[self.engaged_inds, self.t] = action_index
    
    def _compute_actions(self, policy=None):
        if policy is None:
            num_remaining = np.sum(self.engaged_inds)
            n_actions = len(self.action_space)
            all_action_probs = np.ones((num_remaining, n_actions)) / n_actions
        else:
            s_prev = self.S[self.engaged_inds, self.t, :]
            all_action_probs = policy.act(s_prev)
        
        action, action_prob = random_choice_idx(all_action_probs)
            
        if self.A_prob is not None:
            self.A_prob[self.engaged_inds, self.t] = action_prob
            
        self.set_A(action)  

    def _apply_dropout(self):
        raise NotImplementedError()
        
    def _apply_state_transition(self):
        raise NotImplementedError()

    def _compute_rewards(self):
        raise NotImplementedError()

    def step_forward_in_time(self, policy, apply_dropout):
        if self.t >= self.t_total:
            raise RuntimeError(f"The trial is over; t={self.t}.")
        
        self._compute_actions(policy) 
    
        if apply_dropout:
            self._apply_dropout()        
        
        self._apply_state_transition()
        
        if self.R is not None:
            self._compute_rewards()
        self.t += 1
    
    def get_returns(self):
        """ Return per-patient total rewards. """
        return np.nansum(self.R, axis=1)
    
    def last_time_index(self, i):
        """ 
        Return the index of the last time point patient i was observed. This 
        is denoted "T" in the paper. Here, indexing starts at 0 instead of 1,
        so T=7 implies the patient was observed at 8 time points.
        """
        return int(min(self.t, self.T_dis[i]))
    
    def num_treatments_applied(self, i):
        """ 
        Return the number of actions taken / treatments applied to patient i.
        Disengaged patients had an action applied after their last observed 
        state, whereas patients observed at the current time have not yet
        had an action applied.
        """ 
        last_time_index = self.last_time_index(i)
        if last_time_index == self.T_dis[i]:
            return last_time_index + 1
        else:
            return last_time_index
 
    def get_patient_table(self, i):
        """ Return observations for patient i as a DataFrame. """
        data = self.S[i,:self.last_time_index(i) + 1]
        return pd.DataFrame(data, columns = self.feature_names)