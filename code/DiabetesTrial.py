import numpy as np
from scipy.special import expit

import Trial as trial
        
a_idx = {"insulin" : 0, "message" : 1}

gluc_mean = 100
gluc_std = 5.5
gluc_state_effect = {
   "glucose": 0.95,
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

class DiabetesTrial(trial.Trial):
    # Feature values at time t describe the time period (t-1,t]
    feature_names = [
        "glucose",
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
        "insulin-1",
    ]
    
    # Define what action indexes refer to.
    action_space = [
        np.array([0,0]),
        np.array([1,0]),
        np.array([0,1]),
        np.array([1,1])
    ]
    
    def __init__(self, n, t_total, initial_states=None, compute_extras=True, 
            burn_in_policy=None, burn_in_steps=100
        ):
        self.burn_in_policy = burn_in_policy
        if initial_states is None:
            self.burn_in_steps = burn_in_steps
        else:
            self.burn_in_steps = "Already burned in."
        super().__init__(n, t_total, 
            initial_states=initial_states, compute_extras=compute_extras)
    
    def generate_initial_states(self):
        initial_states = [gluc_mean, stress_mean, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        trial_burn = DiabetesTrial(n=self.n, t_total=self.burn_in_steps, 
            initial_states=initial_states, compute_extras=False
        )
        for t in range(self.burn_in_steps):
            trial_burn.step_forward_in_time(
                self.burn_in_policy, apply_dropout=False
            )
        new_initial_states = trial_burn.S[:,-1,:]
        return new_initial_states
    
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
    
    def _apply_dropout(self):
        dropout_probs = 0.2*expit(self.get_S("stress")-8) + 0.8 * 0.02
        dropout_inds = np.random.binomial(1, dropout_probs)
        
        engaged_inds_before = self.engaged_inds.copy()
        self.engaged_inds[self.engaged_inds] = np.logical_not(dropout_inds)

        # NOTE: this is slightly inefficient because it compares all patients.
        just_disengaged = self.engaged_inds < engaged_inds_before
        self.T_dis[just_disengaged] = self.t
    
    def _apply_state_transition(self):
        n_engaged = np.sum(self.engaged_inds)
        
        action_inds = self.get_A_indicators()
        message_ind = action_inds[:, a_idx["message"]]
        insulin_ind = action_inds[:, a_idx["insulin"]]
           
        prev_fatigue = self.get_S("fatigue")
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
                
        eat_ind = np.random.binomial(1, 0.4 + 0.3*expit_stress, size=n_engaged)
        unhealthy_snack_ind = (
            eat_ind * np.random.binomial(1, expit_stress, size=n_engaged)
        )
        
        calories = (
            eat_ind * np.random.normal(60, 10, size=n_engaged)
            + unhealthy_snack_ind * np.random.normal(90, 10, size=n_engaged)
        ).clip(min=0)
        
        glucose = gluc_mean * (1-gluc_state_effect["glucose"])
        for feature_name, coefficient in gluc_state_effect.items():
            glucose += coefficient * self.get_S(feature_name)
        glucose += (
            gluc_act_effect * insulin_ind 
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
        self.set_S("insulin-1", insulin_ind)
        
    def _compute_rewards(self):
        """ 
        Composite reward for exercise and avoiding unhealthy eating. Small
        penalty for disengaging.
        """
        glucose = self.get_S("glucose")
        
        # Rewards for glucose levels from V-learning simulations.
        self.R[self.engaged_inds, self.t] = (
            # self.get_S("exercise", self.t+1) / 800 
            # - self.get_S("unhealthy_ind", self.t+1) 
            - 2*(glucose < 70)
            - (glucose < 80)
            - (glucose > 120)
            - (glucose > 150)
        )
        
        self.R[self.T_dis == self.t, self.t] = -0.5
        
    def get_patient_table(self, i):
        """ Return observations for patient i as a DataFrame. """
        table = super().get_patient_table(i)
        
        num_actions = self.num_treatments_applied(i)
        A = self.A[i, :num_actions].astype(int)
        A_prob = self.A_prob[i, :num_actions]
        R = self.R[i, :self.last_time_index(i)+1]
        
        insulin = np.zeros(self.last_time_index(i)+1, dtype=int)
        message = np.zeros(self.last_time_index(i)+1, dtype=int)
        a_prob = np.full(self.last_time_index(i)+1, np.nan, dtype=float)
        r = np.full(self.last_time_index(i)+1, np.nan, dtype=float)
        
        for t, (action, action_prob) in enumerate(zip(A, A_prob)):
            indicators = self.action_space[action]
            insulin[t] = indicators[a_idx["insulin"]]
            message[t] = indicators[a_idx["message"]]
            a_prob[t] = action_prob
            r[t] = R[t]
        
        table["a:insulin"] = insulin
        table["a:message"] = message
        table["p"] = a_prob
        table["r"] = r
        return table

if __name__ == "__main__":
    t_max = 15  
    n = 20
    mu = None
    
    trial = DiabetesTrial(n, t_max)
    for t in range(t_max):
        trial.step_forward_in_time(mu, apply_dropout=True)
        trial.test_indexing()        