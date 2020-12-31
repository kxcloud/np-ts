import numpy as np
import pandas as pd
from scipy.special import expit

np.set_printoptions(precision=3)

# Feature values at time t describe the time period (t-1,t]
feature_names = [
    "stress",
    "fatigue",
    "unhealthy_ind",
    "unhealthy_ind-1",
    "calories",
    "calories-1",
    "calories-2",
    "calories-3",
    "exercise",
    "exercise-1",
    "glucose",
    "insulin-1"
]
s_idx = { feature_name : j for j, feature_name in enumerate(feature_names)}
a_idx = {"insulin" : 0, "message" : 1}

action_space = [
    np.array([0,0]),
    np.array([1,0]),
    np.array([0,1]),
    np.array([1,1])
]

gluc_mean = 100
gluc_std = 5.5
gluc_state_effect = {
   "glucose": 0.9,
   "calories" : 0.1,
   "calories-1" : 0.1,
   "exercise": -0.01,
   "exercise-1": -0.01,
   "insulin-1": -4
}
gluc_act_effect = -2
gluc_min, gluc_max = (50, 250)

stress_mean = 4.5
stress_std = 2
stress_min, stress_max = (0, 15)

def random_choice_vec(a, p):
    num_patients = p.shape[0]
    action_shape = a[0].shape
        
    rand_per_item = np.random.uniform(size=(num_patients,1))
    action_index = (np.cumsum(p, axis=1) < rand_per_item).argmin(axis=1)
    
    #TODO: vectorize this.
    action_array = np.zeros((num_patients, *action_shape), dtype=int)
    prob_array = np.zeros((num_patients, 1), dtype=float)
    for i, act_idx in enumerate(action_index):
        action_array[i,:] = a[act_idx]
        prob_array[i] = p[i, act_idx]
    return action_array
    
class DiabetesTrial:
    def __init__(self, n, t_total, initial_states=None, compute_extras=True):
        self.n = n
        self.p = len(feature_names)
        self.t_total = t_total
        self.S = np.full((n, t_total + 1, self.p), np.nan) # States (patient features)
        self.A = np.full((n, t_total, 2), np.nan) # Actions (insulin x messaging)
        self.T_dis = np.full(n, np.inf) # Per-patient disengagement times
        
        # Save compute during burn-in by skipping calculation of utility.
        if compute_extras:
            self.R = np.full((n, t_total), np.nan) # Observed utilities
            self.A_prob = np.full((n, t_total), np.nan) # Action selection probabilities
        else: 
            self.R = None
            self.A_prob = None
        
        if initial_states is None:
            initial_states = [stress_mean, 0, 0, 0, 0, 0, 0, 0, 0, 0, gluc_mean, 0]
        self.S[:,0,:] = initial_states
        
        # These attributes store values that change w.r.t. the current time.
        self.t = 0
        self.engaged_inds = np.full(n, True)
    
    def __repr__(self):
        return "DiabetesTrial(n={}, t_total={}, num_engaged={}, t={})".format(
            self.n, self.t_total, np.sum(self.engaged_inds), self.t
        )
    
    def get_S(self, feature_name, t=None):
        """ 
        Convenience function for getting the value of engaged patient features
        at the CURRENT time step.
        """
        if t is None:
            t = self.t
        return self.S[self.engaged_inds, t, s_idx[feature_name]]
        
    def set_S(self, feature_name, feature_value):
        """ 
        Convenience function for setting the value of engaged patient features
        at the NEXT time step.
        """
        self.S[self.engaged_inds, self.t+1, s_idx[feature_name]] = feature_value
    
    def get_A(self, action_name, t=None):
        if t is None:
            t = self.t
        return self.A[self.engaged_inds, t, a_idx[action_name]]
        
    def set_A(self, action_name, action_value):
        self.A[self.engaged_inds, self.t, a_idx[action_name]] = action_value
    
    def _apply_dropout(self):
        dropout_probs = 0.2*expit(self.get_S("stress")-8) + 0.8 * 0.02
        dropout_inds = np.random.binomial(1, dropout_probs)
        
        engaged_inds_before = self.engaged_inds.copy()
        self.engaged_inds[self.engaged_inds] = np.logical_not(dropout_inds)

        # NOTE: this is slightly inefficient because it compares all patients.
        just_disengaged = self.engaged_inds < engaged_inds_before
        self.T_dis[just_disengaged] = self.t
    
    def _compute_actions(self, policy=None):
        if policy is None:
            num_remaining = np.sum(self.engaged_inds)
            probs = 0.3
            actions = np.random.binomial(1, p=probs, size=(num_remaining, 2))
        else:
            s_prev = self.S[self.engaged_inds, self.t, :]
            all_action_probs = policy.act(s_prev)
            actions, probs = random_choice_vec(action_space, all_action_probs)
            
        if self.A_prob is not None:
            self.A_prob[self.engaged_inds, self.t] = probs
            
        self.set_A("insulin", actions[:, a_idx["insulin"]])
        self.set_A("message", actions[:, a_idx["message"]])
    
    def _apply_state_transition(self):
        n_engaged = np.sum(self.engaged_inds)
        
        prev_fatigue = self.get_S("fatigue")
        message_ind = self.get_A("message")
        stress = (
            0.1*stress_mean + 0.9*self.get_S("stress") 
            - 2 * message_ind * (1-prev_fatigue)
            + np.random.normal(0, stress_std, size=n_engaged)
        ).clip(stress_min, stress_max).round(1)
          
        fatigue = (
            0.95*prev_fatigue**2 
            + 0.4*message_ind
            + np.random.normal(0,0.1, size=n_engaged)
        ).clip(0,1).round(2)
        
        exercise = (
            (np.random.binomial(1, 0.2, size=n_engaged) 
             * np.random.normal(819, 10, size=n_engaged)).clip(min=0)
          + (np.random.binomial(1, 0.4, size=n_engaged) 
             * np.random.normal(31, 5, size=n_engaged)).clip(min=0)
        )
        
        expit_stress = expit((self.get_S("stress") - 7)/4) 
                
        eat_ind = np.random.binomial(1, 0.1 + 0.3*expit_stress, size=n_engaged)
        unhealthy_snack_ind = (
            eat_ind * np.random.binomial(1, expit_stress, size=n_engaged)
        )
        
        calories = (
            eat_ind * np.random.normal(60, 10, size=n_engaged)
            + unhealthy_snack_ind * np.random.normal(90, 10, size=n_engaged)
        ).clip(min=0)
        
        prev_insulin = 0 if self.t == 0 else self.get_A("insulin",t=self.t-1)
        
        glucose = gluc_mean * (1-gluc_state_effect["glucose"])
        for feature_name, coefficient in gluc_state_effect.items():
            glucose += coefficient * self.get_S(feature_name)
        glucose += (
            gluc_act_effect * self.get_A("insulin") 
            + np.random.normal(0, gluc_std, size=n_engaged)
        )
        glucose = glucose.clip(gluc_min, gluc_max)
        
        self.set_S("stress", stress)
        self.set_S("fatigue", fatigue)
        self.set_S("unhealthy_ind", unhealthy_snack_ind)
        self.set_S("unhealthy_ind-1", self.get_S("unhealthy_ind"))
        self.set_S("calories", calories)
        self.set_S("calories-1", self.get_S("calories"))
        self.set_S("calories-2", self.get_S("calories-1"))
        self.set_S("calories-3", self.get_S("calories-2"))
        self.set_S("exercise", exercise)
        self.set_S("exercise-1", self.get_S("exercise"))
        self.set_S("glucose", glucose)
        self.set_S("insulin-1", prev_insulin)
        
    def _compute_rewards(self):
        """ 
        Composite reward for exercise and avoiding unhealthy eating. Small
        penalty for disengaging.
        """
        self.R[self.engaged_inds, self.t] = (
            self.get_S("exercise", self.t+1) / 800 
            - self.get_S("unhealthy_ind", self.t+1)
        )
        
        self.R[self.T_dis == self.t, self.t] = -0.5
        
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
        return pd.DataFrame(data, columns = feature_names)
    
def get_burned_in_states(n, mu_burn=None, t_burn=50):
    trial_burn = DiabetesTrial(n=n, t_total=t_burn, compute_extras=False)
    for t in range(t_burn):
        trial_burn.step_forward_in_time(mu_burn, apply_dropout=False)
    initial_states = trial_burn.S[:,-1,:]
    return initial_states

def test_indexing(trial):
    """ 
    Run a few tests on active trial to make sure states and actions are 
    counted correctly. 
    """
    for i in range(trial.n):
        num_states_observed = trial.last_time_index(i) + 1
        assert (~np.isnan(trial.S[i,:,0])).sum() == num_states_observed, (
            f"Number of states observed for patient {i} must match actual "
            "state array."    
        )
        num_actions_observed = trial.num_treatments_applied(i)
        assert (~np.isnan(trial.A[i,:,0])).sum() == num_actions_observed, (
            f"Number of treatments applied for patient {i} must match actual "
            "action array."
        )        

if __name__ == "__main__":
    t_max = 15  
    n = 20
    mu = None
    
    trial = DiabetesTrial(n, t_max, initial_states=get_burned_in_states(n))
    for t in range(t_max):
        trial.step_forward_in_time(mu, apply_dropout=True)
        test_indexing(trial)
        