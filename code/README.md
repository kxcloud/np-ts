# NP-TS project code
Code for application of the Neyman-Pearson Thompson Sampling algorithm to simulated data. Overviews of existing scripts are given below.


### Trial.py ###
Defines the `Trial` class, a generic representation of a clinical trial, or more generally, a collection of instances of a Markov Decision Processes. States, actions, rewards, and disengagement times are stored in arrays `S`, `A`, `R`, `T_dis` respectively. The dynamics of a `Trial` are defined by the `step_forward_in_time` method. To create your own clinical trial, you can subclass the Trial object and implement the methods in the step forward method, defining (i) the dropout mechanism via `_apply_dropout()`, (ii) the state transition dynamics via `_apply_state_transitions()`, and (iii) the reward function via `_compute_rewards()`.

### data_generation.py
Defines the `DiabetesTrial` class, a simulation of a clinical trial where patient blood sugar must be managed while helping them lose weight. Most MDP parameters are given fixed definitions at the top of the script. Others are defined inline in `_apply_state_transition`.

### data_visualization.py
Functions for convenient visualization of simulated data from a `DiabetesTrial`.

### learning.py
Efficient estimation of the value function for a fixed policy via `get_value_estimator`.
