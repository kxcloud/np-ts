import numpy as np
from scipy.special import expit

np.set_printoptions(precision=3)

# State values at time t describe the period time (t-1,t]
feature_names = [
    "stress",
    "fatigue",
    "unhealthy_ind",
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
p = len(feature_names)

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
stress_std = 3
stress_min, stress_max = (0, 15)

class DiabetesTrial:
    def __init__(self, n, t_total):
        self.n = n
        self.t_total = t_total
        self.S = np.full((n, t_total+1, p), np.nan) # States (patient features)
        self.A = np.full((n, t_total, 2), np.nan) # Actions (insulin x messaging)
        self.T_dis = np.full(n, np.inf) # Per-patient disengagement times
        self.R = np.full((n, t_total), np.nan) # Observed utilities
        
        self.S[:,0,:] = [stress_mean, 0, 0, 0, 0, 0, 0, 0, 0, gluc_mean, 0]
        
        # These attributes store values that change w.r.t. the current time.
        self.t = 0
        self.engaged_inds = np.full(n, True)
    
    def get_S(self, feature_name):
        """ 
        Convenience function for getting the value of engaged patient features
        at the CURRENT time step.
        """
        return self.S[self.engaged_inds, self.t, s_idx[feature_name]]
        
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
        dropout_probs = 0.2*expit(self.get_S("stress")-8)+ 0.8 * 0.02
        dropout_inds = np.random.binomial(1, dropout_probs)
        
        engaged_inds_before = self.engaged_inds.copy()
        self.engaged_inds[self.engaged_inds] = np.logical_not(dropout_inds)

        # NOTE: this is slightly inefficient because it compares all patients.
        just_disengaged = self.engaged_inds < engaged_inds_before
        self.T_dis[just_disengaged] = self.t
    
    def _compute_actions(self, policy):
        s_prev = self.S[self.engaged_inds, self.t, :]
        actions = np.random.binomial(1, policy(s_prev))
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
        ).clip(stress_min, stress_max)
          
        fatigue = (0.95*prev_fatigue**2 + 0.4*message_ind).clip(0,1)
        
        exercise = (
            (np.random.binomial(1, 0.2, size=n_engaged) 
             * np.random.normal(819, 10, size=n_engaged)).clip(min=0)
          + (np.random.binomial(1, 0.4, size=n_engaged) 
             * np.random.normal(31, 5, size=n_engaged)).clip(min=0)
        )
        
        expit_stress = expit(self.get_S("stress") - 7)
                
        eat_ind = np.random.binomial(1, 0.1 + 0.3*expit_stress, size=n_engaged)
        unhealthy_snack_ind = (
            eat_ind * np.random.binomial(1, 0.5*expit_stress, size=n_engaged)
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
        self.set_S("calories", calories)
        self.set_S("calories-1", self.get_S("calories"))
        self.set_S("calories-2", self.get_S("calories-1"))
        self.set_S("calories-3", self.get_S("calories-2"))
        self.set_S("exercise", exercise)
        self.set_S("exercise-1", self.get_S("exercise"))
        self.set_S("glucose", glucose)
        self.set_S("insulin-1", prev_insulin)
        
    def _compute_rewards(self):
        pass
        
    def step_forward_in_time(self, policy, apply_dropout=True):
        if self.t >= self.t_total:
            raise RuntimeError(f"The trial is over; t_prev={self.t_prev}.")
        
        if apply_dropout:
            self._apply_dropout()    
            
        self._compute_actions(policy)            
        self._apply_state_transition()
        self._compute_rewards()
        self.t += 1
                
trial = DiabetesTrial(n=4, t_total=5)
for i in range(trial.t_total):
    trial.step_forward_in_time(lambda x: np.full_like(x,fill_value=0.3))