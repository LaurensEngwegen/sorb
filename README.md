# Search on the Replay Buffer

### Laurens Engwegen, Dazhuo Qiu

Repository for our implementation of, experimentation with and contributions to [Search on the Replay Buffer (SoRB)](https://arxiv.org/pdf/1906.05253.pdf) for the seminar course Advanced Deep Reinforcement Learning, Leiden University (2021-2022).

We build upon SoRB by introducing a clustering approach to construct a replay buffer that contains states that are evenly distributed over the state space.

<p align="center">
  <img width="700" height="400" src="https://github.com/LaurensEngwegen/sorb/blob/main/figures/graphical_explanation.PNG">
</p>

### Experimentation
Our experiments can be reproduced by running `main.py` after which various options can be specified:
* `--experiments`: Type of experiments to reproduce, with a choice between,
    - distance: Reproduction of experiment from the original paper with different distances between start and goal.
    - kmeansdistance: Distance experiment in which our clustering approach is compared to the default approach.
    - maxdist: Optimization of the MaxDist parameter.
    - kmeansbuffersize: Experiment with different values for the replay buffer size.
    - upsampling: Experiment with different upsampling factors for the construction of the replay buffer for different sizes of the replay buffer.
    - kmeanssamebuffer: Experiment where the original replay buffer is clustered into fractions of its original size.
    - additionsamebuffer: Addition of kmeanssamebuffer by means of tuning the MaxDist parameter
* `--environments`: Enivornment(s) to experiment with (default: FourRooms).
* `--max_search_steps`: Value of the MaxDist parameter (default: 8).
* `--resize_factor`: Factor to scale the environment with to increase its size (default: 10).
* `--visualize`: Boolean that indicates whether or not to show starting state, goal state, replay buffer and waypoints.

All other parameters, e.g. concerning the goal-conditioned agent, must be changed in the raw code. Those parameters were, however, fixed to their current values during experimentation.

The original paper can be found [here](https://arxiv.org/pdf/1906.05253.pdf), which includes a [link](http://bit.ly/rl_search) to the code published by the authors.
