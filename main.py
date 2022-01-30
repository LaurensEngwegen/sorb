from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from environments import *
from experiments import *
from agent import *
from train import *
from visualizations import *
from graphsearch import *

import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--experiments', default=['kmeansdistance',
											'kmeansbuffersize',
											'distance',
											'maxdist',
											'upsampling',
											'kmeanssamebuffer',
											'additionsamebuffer'
											], nargs='+', help="Experiment list")
parser.add_argument('--environments', default=['FourRooms'], nargs='+', help="Environment list")
parser.add_argument('--max_search_steps', default=8, type=int, help="MaxDist parameter")
parser.add_argument('--resize_factor', default=10, type=int, help="Inflate the environment to increase the difficulty")
parser.add_argument('--visualize', default=False, type=bool, help="Visualization of algorithm steps")

args = parser.parse_args()

experiments = args.experiments
environments = args.environments
max_search_steps = args.max_search_steps
resize_factor = args.resize_factor
visualize = args.visualize

# Fixed parameters
n_experiments = 100 # Test repetitions
max_duration = 300 # Max. episode length during testing
train_iterations = 1000000 # Nr. of training iterations
max_episode_steps = 20 # Max. episode length during training (goal-cond.)
min_distance = 10 # Min. distance to goal
max_distance = 120 # Max. distance to goal

experiment_repetitions = 3

results_folder = 'results'
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

# Run all the experiments
for run in range(1, experiment_repetitions+1):
    for env_name in environments:
        # Create environments
        tf_env = env_load_fn(env_name, max_episode_steps, resize_factor=resize_factor, terminate_on_timeout=False)
        eval_tf_env = env_load_fn(env_name, max_episode_steps, resize_factor=resize_factor, terminate_on_timeout=True)
        # Visualize a start, goal and replay buffer (with and without using clustering)
        if visualize:
            visualize_start_goal(seed=42, eval_tf_env=eval_tf_env)
            visualize_replaybuffer(seed=42, eval_tf_env=eval_tf_env, kmeans=False)
            visualize_replaybuffer(seed=42, eval_tf_env=eval_tf_env, kmeans=True)
        # Create agent
        agent = UvfAgent(
                tf_env.time_step_spec(),
                tf_env.action_spec(),
                max_episode_steps=max_episode_steps,
                use_distributional_rl=True,
                ensemble_size=3)
        # Train agent
        train_eval(
                agent,
                tf_env,
                eval_tf_env,
                initial_collect_steps=1000,
                eval_interval=1000,
                num_eval_episodes=10,
                num_iterations=train_iterations,
        )
        # Visualize the waypoints used between a start and goal
        if visualize:
            visualize_search_path(seed=42, eval_tf_env=eval_tf_env, kmeans=False)
            visualize_search_path(seed=42, eval_tf_env=eval_tf_env, kmeans=True)
        # Experiment
        if experiments[0] != 'False':
            experimenter = Experimenter(experiments,
                                        env_name, 
                                        eval_tf_env, 
                                        agent, 
                                        max_duration, 
                                        max_search_steps, 
                                        min_distance, 
                                        max_distance,
                                        run, 
                                        n_experiments, 
                                        results_folder,
                                        call_print_func=True)
        