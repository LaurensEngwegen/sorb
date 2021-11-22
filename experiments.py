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

def create_agent(tf_env, eval_tf_env, max_episode_steps, use_distr_rl=True, ensemble_size=3, train_iterations=100000):
    new_agent = UvfAgent(
                tf_env.time_step_spec(),
                tf_env.action_spec(),
                max_episode_steps=max_episode_steps,
                use_distributional_rl=use_distr_rl,
                ensemble_size=ensemble_size)
    train_eval(
			new_agent,
			tf_env,
			eval_tf_env,
			initial_collect_steps=1000,
			eval_interval=1000,
			num_eval_episodes=10,
			num_iterations=train_iterations,
	)        
    return new_agent

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
        search = [1, 0]
        steps = [[], []]
        for i in tqdm(range(n_experiments)):
            seed = i # To ensure same start and goal states for different conditions
            eval_tf_env.pyenv.envs[0]._duration = max_duration
            eval_tf_env.pyenv.envs[0].gym.set_sample_goal_args(
                prob_constraint=1.0,
                min_dist=distance,
                max_dist=distance)
            for use_search in search:
                steps[use_search].append(rollout(seed, eval_tf_env, agent, search_policy, use_search))
        if call_print_function:
            print_results('SEARCH', steps[1], n_experiments)
            print_results('NO SEARCH', steps[0], n_experiments)
        results[distance] = steps
    return results

def maxdist_exp(eval_tf_env, agent, min_distance, max_distance, n_experiments, max_duration, call_print_function=True):
    max_dists = [8,10,12,14]
    steps_results = {dist: [] for dist in max_dists}
    
    print(f'\tExperiments with MaxDists: {max_dists}')

    for i in tqdm(range(n_experiments)):
        seed = i # To ensure same start and goal states for different conditions
        eval_tf_env.pyenv.envs[0]._duration = max_duration
        eval_tf_env.pyenv.envs[0].gym.set_sample_goal_args(
            prob_constraint=1.0,
            min_dist=min_distance,
            max_dist=max_distance)
        for max_dist in max_dists:
            # Initialize search policy
            replay_buffer_size = 1000
            rb_vec = fill_replay_buffer(eval_tf_env, replay_buffer_size=replay_buffer_size)
            agent.initialize_search(rb_vec, max_search_steps=max_dist)
            search_policy = SearchPolicy(agent, rb_vec, open_loop=True)
            steps_results[max_dist].append(rollout(seed, eval_tf_env, agent, search_policy))
    if call_print_function:
        for dist in max_dists:
            print_results(f'MaxDist = {dist}', steps_results[dist], n_experiments)
    return steps_results

def kmeans_distance_exp(eval_tf_env, agent, max_search_steps, n_experiments, max_duration, call_print_function=True):
    replay_buffer_size = 1000

    # distances = [10, 20, 40, 60] # For resize factor 5
    distances = [30, 60, 90, 120]
    kmeans = [1, 0]
    results = dict()
    for distance in distances:
        print(f'\nDistance set to {distance}')
        steps = [[], []]
        for use_kmeans in kmeans:
            rb_vec = fill_replay_buffer(eval_tf_env, replay_buffer_size=replay_buffer_size, use_kmeans=use_kmeans)
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
            print_results('KMEANS', steps[1], n_experiments)
            print_results('DEFAULT', steps[0], n_experiments)
        results[distance] = steps
    return results

def kmeans_buffersize_exp(eval_tf_env, agent, min_distance, max_distance, max_search_steps, n_experiments, max_duration, call_print_function=True):
    replay_buffer_sizes = [250, 500, 750, 1000, 1250]
    kmeans = [1, 0]
    results = dict()
    for replay_buffer_size in replay_buffer_sizes:
        print(f'\nReplay buffer size: {replay_buffer_size}')
        steps = [[], []]
        for use_kmeans in kmeans:
            rb_vec = fill_replay_buffer(eval_tf_env, replay_buffer_size=replay_buffer_size, use_kmeans=use_kmeans)
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
            print_results('KMEANS', steps[1], n_experiments)
            print_results('DEFAULT', steps[0], n_experiments)
        results[replay_buffer_size] = steps
    return results

def kmeans_upsampling_exp(eval_tf_env, agent, min_distance, max_distance, max_search_steps, n_experiments, max_duration, call_print_function=True):
    replay_buffer_sizes = [100, 250, 500, 750, 1000]
    upsampling_factors = [5, 10, 50, 100]
    results = dict()
    for replay_buffer_size in replay_buffer_sizes:
        print(f'\nReplay buffer size: {replay_buffer_size}')
        steps = [[], [], [], []]
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
                steps[index].append(rollout(seed, eval_tf_env, agent, search_policy))
        if call_print_function:
            print_results('KMEANS', steps[1], n_experiments)
            print_results('DEFAULT', steps[0], n_experiments)
        # Dictionary: keys = replay buffer size, values = lists with nr of steps for different upscaling factors
        results[replay_buffer_size] = steps
    return results


n_experiments = 100
max_duration = 300
experiments = ['upsampling', 'kmeansbuffersize', 'maxdist']
environments = ['FourRooms', 'Maze6x6']

max_search_steps = 10 # MaxDist parameter
train_iterations = 1000000
max_episode_steps = 20
resize_factor = 10 # Inflate the environment to increase the difficulty.
min_distance = 10
max_distance = 120

for env_name in environments:
    # Create environments
    tf_env = env_load_fn(env_name, max_episode_steps, resize_factor=resize_factor, terminate_on_timeout=False)
    eval_tf_env = env_load_fn(env_name, max_episode_steps, resize_factor=resize_factor, terminate_on_timeout=True)
    # Create and train agent
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
            num_iterations=train_iterations,
    )

    for exp in experiments:
        print(f'\nStarting experiments in environment: {env_name}\n')

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
            results = kmeans_upsampling_exp(eval_tf_env, agent, min_distance, max_distance, n_experiments, max_duration)


        with open(f'results_{exp}_{env_name}_resize{resize_factor}_trainiters{int(train_iterations/1000)}k.pkl', 'wb') as f:
            pkl.dump(results, f)
