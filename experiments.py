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

def distance_exp(eval_tf_env, agent, search_policy, goal_distance, max_duration):
    seed = np.random.randint(0, 1000000)
    
    eval_tf_env.pyenv.envs[0]._duration = max_duration
    eval_tf_env.pyenv.envs[0].gym.set_sample_goal_args(
        prob_constraint=1.0,
        min_dist=goal_distance-1,
        max_dist=goal_distance+1)

    search = [1, 0]
    steps = [0, 0]
    for use_search in search:
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
            steps[use_search] = step_counter

    return steps

def print_success(search_steps, nosearch_steps, n_experiments):
    print(f'\nSEARCH')
    if len(search_steps) > 0:
        print(f'\tSuccess rate: {len(search_steps)/n_experiments}')
        print(f'\tAverage number of steps to reach goal: {sum(search_steps)/len(search_steps)}')
    else:
        print(f'Success rate: 0.0')
    print(f'\nNO SEARCH')
    if len(nosearch_steps) > 0:
        print(f'\tSuccess rate: {len(nosearch_steps)/n_experiments}')
        print(f'\tAverage number of steps to reach goal: {sum(nosearch_steps)/len(nosearch_steps)}\n')
    else:
        print(f'Success rate: 0.0')

environments = ['FourRooms', 'Maze6x6']

for env_name in environments:
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
			num_iterations=30000,
	)

    # Initialize search policy
    replay_buffer_size = 1000
    rb_vec = fill_replay_buffer(eval_tf_env, replay_buffer_size=replay_buffer_size)
    agent.initialize_search(rb_vec, max_search_steps=7)
    search_policy = SearchPolicy(agent, rb_vec, open_loop=True)

    print(f'Starting experiments in environment: {env_name}')

    max_duration = 300
    n_experiments = 100
    goal_distances = [10, 20, 40, 60]
    
    for goal_distance in goal_distances:
        search_steps = []
        nosearch_steps = []
        print(f'{n_experiments} experiments, with max. {max_duration} timesteps and distance to goal = {goal_distance}')
        for _ in tqdm(range(n_experiments)):
            steps = distance_exp(eval_tf_env, agent, search_policy, goal_distance, max_duration)
            search_steps.append(steps[1])
            nosearch_steps.append(steps[0])
        print_success(search_steps, nosearch_steps, n_experiments)
