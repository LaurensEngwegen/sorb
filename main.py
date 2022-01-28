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

parser=argparse.ArgumentParser()
parser.add_argument('--n_experiments', default=100, type=int, help="Number of experiemnts")
parser.add_argument('--max_duration', default=300, type=int, help="Number of examples per class in the support set")
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
parser.add_argument('--train_iterations', default=1000000, type=int, help="Training iterations")
parser.add_argument('--max_episode_steps', default=20, type=int, help="Maximum episode steps")
parser.add_argument('--resize_factor', default=10, type=int, help="Inflate the environment to increase the difficulty")
parser.add_argument('--min_distance', default=10, type=int, help="Minimum distance between sampled (start, goal) states")
parser.add_argument('--max_distance', default=120, type=int, help="Maximum distance between sampled (start, goal) states")

args = parser.parse_args()
n_experiments = args.n_experiments
max_duration = args.max_duration
experiments = args.experiments
environments = args.environments
max_search_steps = args.max_search_steps
train_iterations = args.train_iterations
max_episode_steps = args.max_episode_steps
resize_factor = args.resize_factor
min_distance = args.min_distance
max_distance = args.max_distance

results_folder = 'results'
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

# Run all the experiments 3 times
for run in range(1,4):
    for env_name in environments:
        # Create environments
        tf_env = env_load_fn(env_name, max_episode_steps, resize_factor=resize_factor, terminate_on_timeout=False)
        eval_tf_env = env_load_fn(env_name, max_episode_steps, resize_factor=resize_factor, terminate_on_timeout=True)
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

        print(f'\nStarting experiments in environment: {env_name}\n')
        
        for exp in experiments:

            # Pickles will contain a dictionary consisting of:
            # Keys corresponding to the conditions (i.e. distances, replay buffer sizes, etc.)
            # that will contain list(s) with number of steps it took to complete the task
                # 0 -> Task was not completed within max_duration
                # None -> No path was found between start and goal

            if exp == 'kmeansdistance':
                results = kmeans_distance_exp(eval_tf_env, agent, max_search_steps, n_experiments, max_duration)
            elif exp == 'kmeansbuffersize':
                results = kmeans_buffersize_exp(eval_tf_env, agent, min_distance, max_distance, max_search_steps, n_experiments, max_duration)
            elif exp == 'distance':
                results = distance_exp(eval_tf_env, agent, max_search_steps, n_experiments, max_duration)
            elif exp == 'maxdist':
                results = maxdist_exp(eval_tf_env, agent, min_distance, max_distance, n_experiments, max_duration)
            elif exp == 'upsampling':
                results = kmeans_upsampling_exp(eval_tf_env, agent, min_distance, max_distance, max_search_steps, n_experiments, max_duration)
            elif exp == 'kmeanssamebuffer':
                results = kmeans_same_buffer_exp(eval_tf_env, agent, max_search_steps, n_experiments, max_duration)
            elif exp == 'additionsamebuffer':
                maxsearchsteps = [4, 6, 10, 12, 15, 20]
                for max_search_steps in maxsearchsteps:
                    results = addition_same_buffer_exp(eval_tf_env, agent, max_search_steps, n_experiments, max_duration)
            
            # Save results dict in pickle
            save_path = os.path.join(results_folder, 
                                    f'results_{exp}_{env_name}_run{run}.pkl')
            with open(save_path, 'wb') as f:
                pkl.dump(results, f)
