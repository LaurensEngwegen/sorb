from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from environments import *
from agent import *
from train import *
from visualizations import *
from graphsearch import *

from tqdm import tqdm
import numpy as np

def rollout(seed, eval_tf_env, agent, search_policy, use_search=1):
    np.random.seed(seed)
    ts = eval_tf_env.reset()
    step_counter = 0
    for _ in range(eval_tf_env.pyenv.envs[0]._duration):
        if ts.is_last():
            break
        if use_search:
            action = search_policy.action(ts)
        else:
            action = agent.policy.action(ts)
        ts = eval_tf_env.step(action)
        step_counter += 1
    if step_counter < eval_tf_env.pyenv.envs[0]._duration:
        steps = step_counter
    else:
        steps = 0
    return steps

def print_results(title, n_steps, n_experiments):
    print(f'\n{title}')
    successes = np.count_nonzero(n_steps)
    if successes > 0:
        rate = successes/n_experiments
        print(f'\tSuccess rate: {rate}')
        print(f'\tAverage number of steps to reach goal: {sum(n_steps)/successes}')
    else:
        print(f'Success rate: 0.0')



def distance_exp(n_experiments, max_duration):
    tf.compat.v1.reset_default_graph()

    max_episode_steps = 20
    resize_factor = 5 # Inflate the environment to increase the difficulty.

    tf_env = env_load_fn(env_name, max_episode_steps, resize_factor=resize_factor, terminate_on_timeout=False)
    eval_tf_env = env_load_fn(env_name, max_episode_steps, resize_factor=resize_factor, terminate_on_timeout=True)

    agent = UvfAgent(
            tf_env.time_step_spec(),
            tf_env.action_spec(),
            max_episode_steps=max_episode_steps,
            use_distributional_rl=True,
            ensemble_size=3)
    
    train_eval(
			agent,
			tf_env,
			eval_tf_env,
			initial_collect_steps=1000,
			eval_interval=1000,
			num_eval_episodes=10,
			num_iterations=100000,
	)

    # Initialize search policy
    replay_buffer_size = 1000
    rb_vec = fill_replay_buffer(eval_tf_env, replay_buffer_size=replay_buffer_size)
    agent.initialize_search(rb_vec, max_search_steps=7)
    search_policy = SearchPolicy(agent, rb_vec, open_loop=True)

    distances = [10, 20, 40, 60]
    for distance in distances:
        print(f'\nDistance set to {distance}')
        search = [1, 0]
        steps = [[], []]
        for _ in tqdm(range(n_experiments)):
            seed = np.random.randint(0, 1000000)
            eval_tf_env.pyenv.envs[0]._duration = max_duration
            eval_tf_env.pyenv.envs[0].gym.set_sample_goal_args(
                prob_constraint=1.0,
                min_dist=distance,
                max_dist=distance)
            for use_search in search:
                steps[use_search].append(rollout(seed, eval_tf_env, agent, search_policy, use_search))
        print_results('SEARCH', steps[1], n_experiments)
        print_results('NO SEARCH', steps[0], n_experiments)

def maxdist_exp(n_experiments, max_duration):
    tf.compat.v1.reset_default_graph()

    max_episode_steps = 20
    resize_factor = 5 # Inflate the environment to increase the difficulty.

    tf_env = env_load_fn(env_name, max_episode_steps, resize_factor=resize_factor, terminate_on_timeout=False)
    eval_tf_env = env_load_fn(env_name, max_episode_steps, resize_factor=resize_factor, terminate_on_timeout=True)

    agent = UvfAgent(
            tf_env.time_step_spec(),
            tf_env.action_spec(),
            max_episode_steps=max_episode_steps,
            use_distributional_rl=True,
            ensemble_size=3)

    train_eval(
			agent,
			tf_env,
			eval_tf_env,
			initial_collect_steps=1000,
			eval_interval=1000,
			num_eval_episodes=10,
			num_iterations=100000,
	)

    max_dists = [7, 9, 11, 13, 15]
    steps_results = {dist: [] for dist in max_dists}
    
    print(f'\tExperiments with MaxDists: {max_dists}')

    for _ in tqdm(range(n_experiments)):
        seed = np.random.randint(0, 1000000)
        eval_tf_env.pyenv.envs[0]._duration = max_duration
        eval_tf_env.pyenv.envs[0].gym.set_sample_goal_args(
            prob_constraint=1.0,
            min_dist=10,
            max_dist=60)
        for max_dist in max_dists:
            # Initialize search policy
            replay_buffer_size = 1000
            rb_vec = fill_replay_buffer(eval_tf_env, replay_buffer_size=replay_buffer_size)
            agent.initialize_search(rb_vec, max_search_steps=max_dist)
            search_policy = SearchPolicy(agent, rb_vec, open_loop=True)
            steps_results[max_dist].append(rollout(seed, eval_tf_env, agent, search_policy))
    for dist in max_dists:
        print_results(f'MaxDist = {dist}', steps_results[dist], n_experiments)



n_experiments = 100
max_duration = 300

environments = ['FourRooms', 'Maze6x6']

for env_name in environments:
    print(f'\nStarting experiments in environment: {env_name}\n')
    distance_exp(n_experiments, max_duration)
    maxdist_exp(n_experiments, max_duration)
