import numpy as np
import pandas as pd
np.set_printoptions(precision=3)
pd.set_option('precision', 2)

import cProfile, pstats, io
from pstats import SortKey

# Evolution of glucose level
alpha = np.array([0.9, 0.1, 0.1, -0.01, -0.01, -2, -4])
gluc_mean = 100
gluc_min, gluc_max = (50, 250)
patient_features = [
    "glucose", 
    "food", 
    "exercise", 
    "glucose-1", 
    "food-1", 
    "food-2", 
    "food-3",
    "exercise-1", 
    "insulin"
]


def u(s1, a, s2):
    n = s1.shape[0]
    gluc_penalty = [-3,-1,0,-1,-2]
    gluc_cutoff  = [70,80,120,150]
    
    both_glucose = s2[:,[0,3]]
    glucose_idx = np.digitize(both_glucose, gluc_cutoff)
    r = np.zeros(n, dtype=int)
    for i in range(glucose_idx.shape[0]):
        r[i] = gluc_penalty[glucose_idx[i,0]] + gluc_penalty[glucose_idx[i,1]]
    return r

def positive(x):
    return x if x > 0 else 0

def initialize_patient(t_burn, t_max):
    patient = pd.DataFrame(
        index=range(-t_burn, t_max), 
        columns = patient_features,
        data = np.zeros((t_burn+t_max, 9))
    )
    patient.loc[-t_burn, "glucose"] = gluc_mean
    return patient

def set_next_state(patient, t, policy, apply_dropout=False):
    """ 
    Return randomly sampled next state based on:
        s   - current state (as Pandas series)
        p_a - probability of insulin assignment
    """
    if apply_dropout and np.random.uniform() < 0.1:
        return None
    
    s = patient.loc[t]
    
    insulin = np.random.binomial(1, policy(s))
    food = positive(np.random.binomial(1, 0.2) * np.random.normal(78, 10))
    light_exercise = positive(np.random.binomial(1, 0.4) * np.random.normal(31, 5))
    moderate_exercise = positive(np.random.binomial(1, 0.2) * np.random.normal(819, 10))
    
    glucose = (
        gluc_mean * (1-alpha[0]) + alpha[0] * s["glucose"]
        + alpha[1] * s["food"] + alpha[2] * s["food-1"]
        + alpha[3] * s["exercise"] + alpha[4] * s["exercise-1"]
        + alpha[5] * insulin + alpha[6] * s["insulin"]
        + np.random.normal(0, 5.5)
    ).clip(gluc_min, gluc_max)
    
    patient.loc[t+1] = [
        glucose, 
        food,
        light_exercise + moderate_exercise,
        s["glucose"],
        s["food"],
        s["food-1"],
        s["food-2"],
        s["exercise"],
        insulin
    ]

n= 30 
t_max = 5
mu = lambda x : 0.3
t_burn = 50
mu_burn = lambda x : 0.3

p = 8 # of state variables

pr = cProfile.Profile()
pr.enable()

patients = [initialize_patient(t_burn, t_max) for i in range(n)]
for t in range(-t_burn, t_max):
    burned_in = t > 0
    policy = mu if burned_in else mu_burn
    for i in range(n):
        # Check if the patient has dropped out.
        if t-1 in patients[i].index:
            set_next_state(patients[i], t, policy, apply_dropout=burned_in)

pr.disable()
            
s = io.StringIO()
sortby = SortKey.CUMULATIVE
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats()
print(s.getvalue())
        
        
        