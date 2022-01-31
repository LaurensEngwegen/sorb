from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from environments import *
from agent import *
from train import *
from visualizations import *
from graphsearch import *

import os
from tqdm import tqdm
import numpy as np
import pickle as pkl

class Experimenter():
    def __init__(self, experiments, env_name, eval_tf_env, agent, max_duration, max_search_steps, min_distance, 
                 max_distance, run, n_experiments, results_folder='results', call_print_func=True):
        self.eval_tf_env = eval_tf_env
        self.agent = agent
        self.max_duration = max_duration
        self.max_search_steps = max_search_steps
        self.original_maxsearchsteps = 10
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.n_experiments = n_experiments
        self.call_print_func = call_print_func
        
        print(f'\nStarting experiments in environment: {env_name}\n')

        for exp in experiments:
            if exp == 'kmeansdistance':
                results = self.kmeans_distance_exp()
            elif exp == 'kmeansbuffersize':
                results = self.kmeans_buffersize_exp()
            elif exp == 'distance':
                results = self.distance_exp()
            elif exp == 'maxdist':
                results = self.maxdist_exp()
            elif exp == 'upsampling':
                results = self.kmeans_upsampling_exp()
            elif exp == 'kmeanssamebuffer':
                results = self.kmeans_same_buffer_exp()
            
            elif exp == 'additionsamebuffer':

                # TODO: put the different maxsearchsteps experiments into addition_same_buffer_exp function
                maxsearchsteps = [4, 6, 10, 12, 15, 20]
                for max_search_steps in maxsearchsteps:
                    results = self.addition_same_buffer_exp()
            
            # Save results dict in pickle
            save_path = os.path.join(results_folder, 
                                    f'results_{exp}_{env_name}_run{run}.pkl')
    
            # Pickles will contain a dictionary consisting of:
            # Keys corresponding to the conditions (i.e. distances, replay buffer sizes, etc.)
            # that will contain list(s) with number of steps it took to complete the task
                # 0 -> Task was not completed within max_duration
                # None -> No path was found between start and goal                   
            with open(save_path, 'wb') as f:
                pkl.dump(results, f)

    def rollout(self, seed, search_policy, use_search=1):
        np.random.seed(seed)
        ts = self.eval_tf_env.reset()
        step_counter = 0
        for _ in range(self.eval_tf_env.pyenv.envs[0]._duration):
            if ts.is_last():
                break
            if use_search:
                try:
                    action = search_policy.action(ts)
                except:
                    step_counter = self.eval_tf_env.pyenv.envs[0]._duration + 1 # To identify this case
                    # print('Error: no path found from start to goal')
                    break
            else:
                action = self.agent.policy.action(ts)
            ts = self.eval_tf_env.step(action)
            step_counter += 1
        # Return step_counter if goal is reached within duration
        if step_counter < self.eval_tf_env.pyenv.envs[0]._duration:
            steps = step_counter
        # Return None if no path was found from start to goal
        elif step_counter == self.eval_tf_env.pyenv.envs[0]._duration + 1:
            steps = None
        # Return 0 otherwise (i.e. if goal was not reached within time limit)
        else:
            steps = 0
        return steps

    def print_results(self, title, n_steps, ):
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
            rate = successes/self.n_experiments
            print(f'\tSuccess rate: {rate}')
            print(f'\tAverage number of steps to reach goal: {total_steps/successes}')
        else:
            print(f'\tSuccess rate: 0.0')
        print(f'\tNumber of times no path was found between start and goal: {n_nones}/{self.n_experiments}')

    # Default experiments of different distance from original paper
    def distance_exp(self):
        # Initialize search policy
        replay_buffer_size = 1000
        rb_vec = fill_replay_buffer(self.eval_tf_env, replay_buffer_size=replay_buffer_size)
        # Use non-optimized maxsearchsteps
        self.agent.initialize_search(rb_vec, max_search_steps=self.original_maxsearchsteps)
        search_policy = SearchPolicy(self.agent, rb_vec, open_loop=True)

        # distances = [10, 20, 40, 60] # For resize factor 5
        distances = [30, 60, 90, 120]
        results = dict()
        for distance in distances:
            print(f'\nDistance set to {distance}')
            search = {'search': 1, 'no search': 0}
            steps = {'search': [], 'no search': []}
            for i in tqdm(range(self.n_experiments)):
                seed = i # To ensure same start and goal states for different conditions
                self.eval_tf_env.pyenv.envs[0]._duration = self.max_duration
                self.eval_tf_env.pyenv.envs[0].gym.set_sample_goal_args(
                    prob_constraint=1.0,
                    min_dist=distance,
                    max_dist=distance)
                for use_search in search:
                    steps[use_search].append(self.rollout(seed, search_policy, search[use_search]))
            if self.call_print_func:
                self.print_results('SEARCH', steps['search'])
                self.print_results('NO SEARCH', steps['no search'])
            results[distance] = steps
        return results

    # Experiments of maximum distance
    def maxdist_exp(self):
        max_dist_params = [4,6,8,10,12,14]
        kmeans = {'k-means': 1, 'default': 0}
        results = dict()
        print(f'\nExperiments with MaxDists: {max_dist_params}')

        for max_dist_param in max_dist_params:
            print(f'\nMaxDist set to {max_dist_param}')
            steps = {'k-means': [], 'default': []}
            for use_kmeans in kmeans:
                # Initialize search policy
                rb_vec = fill_replay_buffer(self.eval_tf_env, replay_buffer_size=1000, use_kmeans=kmeans[use_kmeans], upsampling_factor=10)
                self.agent.initialize_search(rb_vec, max_search_steps=max_dist_param)
                search_policy = SearchPolicy(self.agent, rb_vec, open_loop=True)
                for i in tqdm(range(self.n_experiments)):
                    seed = i # To ensure same start and goal states for different conditions
                    self.eval_tf_env.pyenv.envs[0]._duration = self.max_duration
                    self.eval_tf_env.pyenv.envs[0].gym.set_sample_goal_args(
                        prob_constraint=1.0,
                        min_dist=self.min_distance,
                        max_dist=self.max_distance)
                    steps[use_kmeans].append(self.rollout(seed, search_policy))
            if self.call_print_func:
                self.print_results('KMEANS', steps['k-means'])
                self.print_results('DEFAULT', steps['default'])
            results[max_dist_param] = steps
        return results

    # Experiments of different distance with k-means clustering
    def kmeans_distance_exp(self):
        replay_buffer_size = 1000
        # distances = [10, 20, 40, 60] # For resize factor 5
        distances = [30, 60, 90, 120]
        kmeans = {'k-means': 1, 'default': 0}
        results = dict()
        for distance in distances:
            print(f'\nDistance set to {distance}')
            steps = {'k-means': [], 'default': []}
            for use_kmeans in kmeans:
                rb_vec = fill_replay_buffer(self.eval_tf_env, replay_buffer_size=replay_buffer_size, use_kmeans=kmeans[use_kmeans])
                self.agent.initialize_search(rb_vec, max_search_steps=self.max_search_steps)
                search_policy = SearchPolicy(self.agent, rb_vec, open_loop=True)
                for i in tqdm(range(self.n_experiments)):
                    seed = i # To ensure same start and goal states for different conditions
                    self.eval_tf_env.pyenv.envs[0]._duration = self.max_duration
                    self.eval_tf_env.pyenv.envs[0].gym.set_sample_goal_args(
                        prob_constraint=1.0,
                        min_dist=distance,
                        max_dist=distance)
                    steps[use_kmeans].append(self.rollout(seed, search_policy))
            if self.call_print_func:
                self.print_results('KMEANS', steps['k-means'])
                self.print_results('DEFAULT', steps['default'])
            results[distance] = steps
        return results

    # Experiments of different reply buffer size with k-means clustering
    def kmeans_buffersize_exp(self):
        replay_buffer_sizes = [250, 500, 750, 1000, 1250]
        kmeans = {'k-means': 1, 'default': 0}
        results = dict()
        for replay_buffer_size in replay_buffer_sizes:
            print(f'\nReplay buffer size: {replay_buffer_size}')
            steps = {'k-means': [], 'default': []}
            for use_kmeans in kmeans:
                rb_vec = fill_replay_buffer(self.eval_tf_env, replay_buffer_size=replay_buffer_size, use_kmeans=kmeans[use_kmeans])
                self.agent.initialize_search(rb_vec, max_search_steps=self.max_search_steps)
                search_policy = SearchPolicy(self.agent, rb_vec, open_loop=True)
                for i in tqdm(range(self.n_experiments)):
                    seed = i # To ensure same start and goal states for different conditions
                    self.eval_tf_env.pyenv.envs[0]._duration = self.max_duration
                    self.eval_tf_env.pyenv.envs[0].gym.set_sample_goal_args(
                        prob_constraint=1.0,
                        min_dist=self.min_distance,
                        max_dist=self.max_distance)
                    steps[use_kmeans].append(self.rollout(seed, search_policy))
            if self.call_print_func:
                self.print_results('KMEANS', steps['k-means'])
                self.print_results('DEFAULT', steps['default'])
            results[replay_buffer_size] = steps
        return results

    # Experiments of different upsampling factor of reply buffer size with k-means clustering
    def kmeans_upsampling_exp(self):
        replay_buffer_sizes = [100, 250, 500, 750, 1000]
        upsampling_factors = [1, 5, 10, 50, 100]
        results = dict()
        for replay_buffer_size in replay_buffer_sizes:
            print(f'\nReplay buffer size: {replay_buffer_size}')
            steps = {i: [] for i in upsampling_factors}
            for index, upsampling_factor in enumerate(upsampling_factors):
                rb_vec = fill_replay_buffer(self.eval_tf_env, replay_buffer_size=replay_buffer_size, use_kmeans=True, upsampling_factor=upsampling_factor)
                self.agent.initialize_search(rb_vec, max_search_steps=self.max_search_steps)
                search_policy = SearchPolicy(self.agent, rb_vec, open_loop=True)
                for i in tqdm(range(self.n_experiments)):
                    seed = i # To ensure same start and goal states for different conditions
                    self.eval_tf_env.pyenv.envs[0]._duration = self.max_duration
                    self.eval_tf_env.pyenv.envs[0].gym.set_sample_goal_args(
                        prob_constraint=1.0,
                        min_dist=self.min_distance,
                        max_dist=self.max_distance)
                    steps[upsampling_factor].append(self.rollout(seed, search_policy))
            if self.call_print_func:
                for j in upsampling_factors:
                    self.print_results(f'UPSAMPLING: {j}', steps[j])
            # Dictionary: keys = replay buffer size, values = dictionary with keys=upsampling_factor and values=list with nr of steps
            results[replay_buffer_size] = steps
        return results

    # Experiments of different clustering fractions with same reply buffer size with k-means clustering
    def kmeans_same_buffer_exp(self):
        fractions = [0.1, 0.25, 0.5, 0.75, 1.0]
        results = dict()
        for fraction in fractions:
            print(f'\nFraction rate: {fraction}')
            steps = {'k-means': []}

            rb_vec = fill_replay_buffer(self.eval_tf_env, use_same=True, fraction=fraction)
            self.agent.initialize_search(rb_vec, max_search_steps=self.max_search_steps)
            search_policy = SearchPolicy(self.agent, rb_vec, open_loop=True)
            for i in tqdm(range(self.n_experiments)):
                seed = i # To ensure same start and goal states for different conditions
                self.eval_tf_env.pyenv.envs[0]._duration = self.max_duration
                self.eval_tf_env.pyenv.envs[0].gym.set_sample_goal_args(
                    prob_constraint=1.0,
                    min_dist=10,
                    max_dist=120)
                steps['k-means'].append(self.rollout(seed, search_policy))
            if self.call_print_func:
                self.print_results('KMEANS', steps['k-means'])
            # Dictionary with keys = fractions, values = dict with key=k-means and values=list with nr of steps
            results[fraction] = steps
        return results

    # Expeiremnts of additional same reply buffer size experiments, which explore the effect of different max search steps
    def addition_same_buffer_exp(self):
        fractions = [0.1, 0.25, 0.5, 0.75, 1.0]
        results = dict()
        for fraction in fractions:
            print(f'\nFraction rate: {fraction}, max_search_steps: {self.max_search_steps}')
            steps = {'k-means': []}

            rb_vec = fill_replay_buffer(self.eval_tf_env, use_same=True, fraction=fraction)
            self.agent.initialize_search(rb_vec, max_search_steps=self.max_search_steps)
            search_policy = SearchPolicy(self.agent, rb_vec, open_loop=True)
            for i in tqdm(range(self.n_experiments)):
                seed = i # To ensure same start and goal states for different conditions
                self.eval_tf_env.pyenv.envs[0]._duration = self.max_duration
                self.eval_tf_env.pyenv.envs[0].gym.set_sample_goal_args(
                    prob_constraint=1.0,
                    min_dist=10,
                    max_dist=120)
                steps['k-means'].append(self.rollout(seed, search_policy))
            if self.call_print_func:
                self.print_results('KMEANS', steps['k-means'])
            # Dictionary with keys = fractions, values = dict with key=k-means and values=list with nr of steps
            results[fraction] = steps
        return results
