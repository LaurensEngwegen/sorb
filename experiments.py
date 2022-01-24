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
import pickle as pkl

def rollout(seed, eval_tf_env, agent, search_policy, use_search=1):
    np.random.seed(seed)
    ts = eval_tf_env.reset()
    step_counter = 0
    for _ in range(eval_tf_env.pyenv.envs[0]._duration):
        if ts.is_last():
            break
        if use_search:
            try:
                action = search_policy.action(ts)
            except:
                step_counter = eval_tf_env.pyenv.envs[0]._duration + 1 # To identify this case
                # print('Error: no path found from start to goal')
                break
        else:
            action = agent.policy.action(ts)
        ts = eval_tf_env.step(action)
        step_counter += 1
    if step_counter < eval_tf_env.pyenv.envs[0]._duration:
        steps = step_counter
    elif step_counter == eval_tf_env.pyenv.envs[0]._duration + 1:
        steps = None
    else:
        steps = 0
    return steps

def print_results(title, n_steps, n_experiments):
    print(f'\n{title}')
    successes = np.count_nonzero(n_steps)
    total_steps = 0
    n_nones = 0
    for steps in n_steps:
        if steps is not None:
            total_steps += steps
        else:
            n_nones += 1

    if successes > 0:
        rate = successes/n_experiments
        print(f'\tSuccess rate: {rate}')
        print(f'\tAverage number of steps to reach goal: {total_steps/successes}')
    else:
        print(f'\tSuccess rate: 0.0')
    print(f'\tNumber of times no path was found between start and goal: {n_nones}/{n_experiments}')

# Default experiments of different distance from original paper
def distance_exp(eval_tf_env, agent, max_search_steps, n_experiments, max_duration, call_print_function=True):
    # Initialize search policy
    replay_buffer_size = 1000
    rb_vec = fill_replay_buffer(eval_tf_env, replay_buffer_size=replay_buffer_size)
    agent.initialize_search(rb_vec, max_search_steps=max_search_steps)
    search_policy = SearchPolicy(agent, rb_vec, open_loop=True)

    # distances = [10, 20, 40, 60] # For resize factor 5
    distances = [30, 60, 90, 120]
    results = dict()
    for distance in distances:
        print(f'\nDistance set to {distance}')
        search = {'search': 1, 'no search': 0}
        steps = {'search': [], 'no search': []}
        for i in tqdm(range(n_experiments)):
            seed = i # To ensure same start and goal states for different conditions
            eval_tf_env.pyenv.envs[0]._duration = max_duration
            eval_tf_env.pyenv.envs[0].gym.set_sample_goal_args(
                prob_constraint=1.0,
                min_dist=distance,
                max_dist=distance)
            for use_search in search:
                steps[use_search].append(rollout(seed, eval_tf_env, agent, search_policy, search[use_search]))
        if call_print_function:
            print_results('SEARCH', steps['search'], n_experiments)
            print_results('NO SEARCH', steps['no search'], n_experiments)
        results[distance] = steps
    return results

# Experiments of maximum distance
def maxdist_exp(eval_tf_env, agent, min_distance, max_distance, n_experiments, max_duration, call_print_function=True):
    max_dist_params = [4,6,8,10,12,14]
    kmeans = {'k-means': 1, 'default': 0}
    results = dict()
    print(f'\nExperiments with MaxDists: {max_dist_params}')

    for max_dist_param in max_dist_params:
        print(f'\nMaxDist set to {max_dist_param}')
        steps = {'k-means': [], 'default': []}
        for use_kmeans in kmeans:
            # Initialize search policy
            rb_vec = fill_replay_buffer(eval_tf_env, replay_buffer_size=1000, use_kmeans=kmeans[use_kmeans], upsampling_factor=10)
            agent.initialize_search(rb_vec, max_search_steps=max_dist_param)
            search_policy = SearchPolicy(agent, rb_vec, open_loop=True)
            for i in tqdm(range(n_experiments)):
                seed = i # To ensure same start and goal states for different conditions
                eval_tf_env.pyenv.envs[0]._duration = max_duration
                eval_tf_env.pyenv.envs[0].gym.set_sample_goal_args(
                    prob_constraint=1.0,
                    min_dist=min_distance,
                    max_dist=max_distance)
                steps[use_kmeans].append(rollout(seed, eval_tf_env, agent, search_policy))
        if call_print_function:
            print_results('KMEANS', steps['k-means'], n_experiments)
            print_results('DEFAULT', steps['default'], n_experiments)
        results[max_dist_param] = steps
    return results

# Experiments of different distance with k-means clustering
def kmeans_distance_exp(eval_tf_env, agent, max_search_steps, n_experiments, max_duration, call_print_function=True):
    replay_buffer_size = 1000
    # distances = [10, 20, 40, 60] # For resize factor 5
    distances = [30, 60, 90, 120]
    kmeans = {'k-means': 1, 'default': 0}
    results = dict()
    for distance in distances:
        print(f'\nDistance set to {distance}')
        steps = {'k-means': [], 'default': []}
        for use_kmeans in kmeans:
            rb_vec = fill_replay_buffer(eval_tf_env, replay_buffer_size=replay_buffer_size, use_kmeans=kmeans[use_kmeans])
            agent.initialize_search(rb_vec, max_search_steps=max_search_steps)
            search_policy = SearchPolicy(agent, rb_vec, open_loop=True)
            for i in tqdm(range(n_experiments)):
                seed = i # To ensure same start and goal states for different conditions
                eval_tf_env.pyenv.envs[0]._duration = max_duration
                eval_tf_env.pyenv.envs[0].gym.set_sample_goal_args(
                    prob_constraint=1.0,
                    min_dist=distance,
                    max_dist=distance)
                steps[use_kmeans].append(rollout(seed, eval_tf_env, agent, search_policy))
        if call_print_function:
            print_results('KMEANS', steps['k-means'], n_experiments)
            print_results('DEFAULT', steps['default'], n_experiments)
        results[distance] = steps
    return results

# Experiments of different reply buffer size with k-means clustering
def kmeans_buffersize_exp(eval_tf_env, agent, min_distance, max_distance, max_search_steps, n_experiments, max_duration, call_print_function=True):
    replay_buffer_sizes = [250, 500, 750, 1000, 1250]
    kmeans = {'k-means': 1, 'default': 0}
    results = dict()
    for replay_buffer_size in replay_buffer_sizes:
        print(f'\nReplay buffer size: {replay_buffer_size}')
        steps = {'k-means': [], 'default': []}
        for use_kmeans in kmeans:
            rb_vec = fill_replay_buffer(eval_tf_env, replay_buffer_size=replay_buffer_size, use_kmeans=kmeans[use_kmeans])
            agent.initialize_search(rb_vec, max_search_steps=max_search_steps)
            search_policy = SearchPolicy(agent, rb_vec, open_loop=True)
            for i in tqdm(range(n_experiments)):
                seed = i # To ensure same start and goal states for different conditions
                eval_tf_env.pyenv.envs[0]._duration = max_duration
                eval_tf_env.pyenv.envs[0].gym.set_sample_goal_args(
                    prob_constraint=1.0,
                    min_dist=min_distance,
                    max_dist=max_distance)
                steps[use_kmeans].append(rollout(seed, eval_tf_env, agent, search_policy))
        if call_print_function:
            print_results('KMEANS', steps['k-means'], n_experiments)
            print_results('DEFAULT', steps['default'], n_experiments)
        results[replay_buffer_size] = steps
    return results

# Experiments of different upsampling factor of reply buffer size with k-means clustering
def kmeans_upsampling_exp(eval_tf_env, agent, min_distance, max_distance, max_search_steps, n_experiments, max_duration, call_print_function=True):
    replay_buffer_sizes = [100, 250, 500, 750, 1000]
    upsampling_factors = [1, 5, 10, 50, 100]
    results = dict()
    for replay_buffer_size in replay_buffer_sizes:
        print(f'\nReplay buffer size: {replay_buffer_size}')
        steps = {i: [] for i in upsampling_factors}
        for index, upsampling_factor in enumerate(upsampling_factors):
            rb_vec = fill_replay_buffer(eval_tf_env, replay_buffer_size=replay_buffer_size, use_kmeans=True, upsampling_factor=upsampling_factor)
            agent.initialize_search(rb_vec, max_search_steps=max_search_steps)
            search_policy = SearchPolicy(agent, rb_vec, open_loop=True)
            for i in tqdm(range(n_experiments)):
                seed = i # To ensure same start and goal states for different conditions
                eval_tf_env.pyenv.envs[0]._duration = max_duration
                eval_tf_env.pyenv.envs[0].gym.set_sample_goal_args(
                    prob_constraint=1.0,
                    min_dist=min_distance,
                    max_dist=max_distance)
                steps[upsampling_factor].append(rollout(seed, eval_tf_env, agent, search_policy))
        if call_print_function:
            print_results('UPSAMPLING: 1', steps[1], n_experiments)
            print_results('UPSAMPLING: 5', steps[5], n_experiments)
            print_results('UPSAMPLING: 10', steps[10], n_experiments)
            print_results('UPSAMPLING: 50', steps[50], n_experiments)
            print_results('UPSAMPLING: 100', steps[100], n_experiments)
        # Dictionary: keys = replay buffer size, values = dictionary with keys=upsampling_factor and values=list with nr of steps
        results[replay_buffer_size] = steps
    return results

# Experiments of different clustering fractions with same reply buffer size with k-means clustering
def kmeans_same_buffer_exp(eval_tf_env, agent, max_search_steps, n_experiments, max_duration, call_print_function=True):
    fractions = [0.1, 0.25, 0.5, 0.75, 1.0]
    results = dict()
    for fraction in fractions:
        print(f'\nFraction rate: {fraction}')
        steps = {'k-means': []}

        rb_vec = fill_replay_buffer(eval_tf_env, use_same=True, fraction=fraction)
        agent.initialize_search(rb_vec, max_search_steps=max_search_steps)
        search_policy = SearchPolicy(agent, rb_vec, open_loop=True)
        for i in tqdm(range(n_experiments)):
            seed = i # To ensure same start and goal states for different conditions
            eval_tf_env.pyenv.envs[0]._duration = max_duration
            eval_tf_env.pyenv.envs[0].gym.set_sample_goal_args(
                prob_constraint=1.0,
                min_dist=10,
                max_dist=120)
            steps['k-means'].append(rollout(seed, eval_tf_env, agent, search_policy))
        if call_print_function:
            print_results('KMEANS', steps['k-means'], n_experiments)
        # Dictionary with keys = fractions, values = dict with key=k-means and values=list with nr of steps
        results[fraction] = steps
    return results

# Expeiremnts of additional same reply buffer size experiments, which explore the effect of different max search steps
def addition_same_buffer_exp(eval_tf_env, agent, max_search_steps, n_experiments, max_duration, call_print_function=True):
    fractions = [0.1, 0.25]
    results = dict()
    for fraction in fractions:
        print(f'\nFraction rate: {fraction}, max_search_steps: {max_search_steps}')
        steps = {'k-means': []}

        rb_vec = fill_replay_buffer(eval_tf_env, use_same=True, fraction=fraction)
        agent.initialize_search(rb_vec, max_search_steps=max_search_steps)
        search_policy = SearchPolicy(agent, rb_vec, open_loop=True)
        for i in tqdm(range(n_experiments)):
            seed = i # To ensure same start and goal states for different conditions
            eval_tf_env.pyenv.envs[0]._duration = max_duration
            eval_tf_env.pyenv.envs[0].gym.set_sample_goal_args(
                prob_constraint=1.0,
                min_dist=10,
                max_dist=120)
            steps['k-means'].append(rollout(seed, eval_tf_env, agent, search_policy))
        if call_print_function:
            print_results('KMEANS', steps['k-means'], n_experiments)
        # Dictionary with keys = fractions, values = dict with key=k-means and values=list with nr of steps
        results[fraction] = steps
    return results
