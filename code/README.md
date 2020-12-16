# NP-TS project code
Code for application of the Neyman-Pearson Thompson Sampling algorithm to simulated data. Overviews of existing scripts are given below.

### data_generation.py
Defines the `DiabetesTrial` class, a stateful object which holds fixed-size Numpy arrays of states, actions, rewards, and disengagement times representing multiple instances of a Markov Decision Process (MDP). Key methods for a `trial` include
* `trial.step_forward_in_time(policy, *)` - simulates one time step of the trial based on action probabilities given by a policy. Look at the method definition to see the order of operations applied in the simulation.
* `trial._apply_state_transition()` - defines the transition dynamics of the MDP. All operations are vectorized over engaged patients.
* `get_S`, `set_S` - convenience functions for selecting and assigning to the state array that automatically slice out disengaged patients, select the appropriate time step, and accept a feature *label* (instead of index) as arguments. See the definition of the state transition method for usage.

Most MDP parameters are given fixed definitions at the top of the script. Others are defined inline in `_apply_state_transition`.

### data_visualization.py
Functions for convenient visualization of simulated data from a `DiabetesTrial`.
