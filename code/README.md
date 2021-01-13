# NP-TS project code
Code for application of the Neyman-Pearson Thompson Sampling algorithm to simulated data. Overviews of existing scripts are given below.


### Trial.py
Defines the `Trial` class, a generic representation of a clinical trial, or more generally, a collection of instances of a Markov Decision Processes. States, actions, rewards, and disengagement times are stored in arrays `S`, `A`, `R`, `T_dis` respectively. The dynamics of a `Trial` are defined by the `step_forward_in_time` method. To create your own clinical trial, you can subclass the Trial object and implement the methods in the step forward method, defining (i) the dropout mechanism via `_apply_dropout()`, (ii) the state transition dynamics via `_apply_state_transitions()`, and (iii) the reward function via `_compute_rewards()`.

### DiabetesTrial.py
Defines the `DiabetesTrial` class, a simulation of a clinical trial where patient blood sugar must be managed while helping them lose weight. Most MDP parameters are given fixed definitions at the top of the script. Others are defined inline in `_apply_state_transition`.

### learning.py
Includes feature construction, a lightweight `Policy` class (which stores a state encoding and parameter vector), value estimation by solving a linear system, and policy optimization that wraps value estimation.

### thompson_sampling.py
Preliminary implementation of frequentist Thompson Sampling. At each time step, multiplier bootstrap weights are drawn and applied patientwise to inform a bootstrap draw of a value function. Then, an optimal policy is estimated with respect to that value function and applied to all patients in the trial.
