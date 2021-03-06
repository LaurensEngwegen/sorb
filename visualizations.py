from graphsearch import fill_replay_buffer, SearchPolicy

import matplotlib.pyplot as plt
import numpy as np
import tqdm


def plot_walls(walls):
	walls = walls.T
	(height, width) = walls.shape
	for (i, j) in zip(*np.where(walls)):
		x = np.array([j, j+1]) / float(width)
		y0 = np.array([i, i]) / float(height)
		y1 = np.array([i+1, i+1]) / float(height)
		plt.fill_between(x, y0, y1, color='grey')
	plt.xlim([0, 1])
	plt.ylim([0, 1])
	plt.xticks([])
	plt.yticks([])

def get_rollout(tf_env, policy, seed=None):
	np.random.seed(seed)	# Use the same task for both policies.
	obs_vec = []
	waypoint_vec = []
	ts = tf_env.reset()
	goal = ts.observation['goal'].numpy()[0]
	for _ in tqdm.tnrange(tf_env.pyenv.envs[0]._duration):
		obs_vec.append(ts.observation['observation'].numpy()[0])
		action = policy.action(ts)
		waypoint_vec.append(ts.observation['goal'].numpy()[0])
		ts = tf_env.step(action)
		if ts.is_last():
			break
	obs_vec.append(ts.observation['observation'].numpy()[0])
	obs_vec = np.array(obs_vec)
	waypoint_vec = np.array(waypoint_vec)
	return obs_vec, goal, waypoint_vec

def visualize_search_vs_nosearch(eval_tf_env, agent, search_policy, rb_vec, distance=None):
	seed = np.random.randint(0, 1000000)

	if distance is None:
		difficulty = 0.7
		max_goal_dist = eval_tf_env.pyenv.envs[0].gym.max_goal_dist
		eval_tf_env.pyenv.envs[0].gym.set_sample_goal_args(
			prob_constraint=1.0,
			min_dist=max(0, max_goal_dist * (difficulty - 0.05)),
			max_dist=max_goal_dist * (difficulty + 0.05))
	else:
		eval_tf_env.pyenv.envs[0].gym.set_sample_goal_args(
			prob_constraint=1.0,
			min_dist=distance,
			max_dist=distance)

	plt.figure(figsize=(12, 5))
	for col_index in range(2):
		title = 'no search' if col_index == 0 else 'search'
		plt.subplot(1, 2, col_index + 1)
		plot_walls(eval_tf_env.pyenv.envs[0].env.walls)
		use_search = (col_index == 1)
		np.random.seed(seed)
		ts = eval_tf_env.reset()
		goal = ts.observation['goal'].numpy()[0]
		start = ts.observation['observation'].numpy()[0]
		obs_vec = []
		for _ in range(eval_tf_env.pyenv.envs[0]._duration):
			if ts.is_last():
				break
			obs_vec.append(ts.observation['observation'].numpy()[0])
			if use_search:
				try:
					action = search_policy.action(ts)
				except:
					print(f'No path found from start to goal')
			else:
				action = agent.policy.action(ts)

			ts = eval_tf_env.step(action)
		obs_vec = np.array(obs_vec)

		plt.plot(obs_vec[:, 0], obs_vec[:, 1], 'b-o', alpha=0.3)
		plt.scatter([obs_vec[0, 0]], [obs_vec[0, 1]], marker='+',
					color='red', s=200, label='start')
		plt.scatter([obs_vec[-1, 0]], [obs_vec[-1, 1]], marker='+',
					color='green', s=200, label='end')
		plt.scatter([goal[0]], [goal[1]], marker='*',
					color='green', s=200, label='goal')
		
		plt.title(title, fontsize=24)
		if use_search:
			waypoint_vec = [start]
			for waypoint_index in search_policy._waypoint_vec:
				waypoint_vec.append(rb_vec[waypoint_index])
			waypoint_vec.append(goal)
			waypoint_vec = np.array(waypoint_vec)

			plt.plot(waypoint_vec[:, 0], waypoint_vec[:, 1], 'y-s', alpha=0.3, label='waypoint')
			plt.legend(loc='lower left', bbox_to_anchor=(-0.8, -0.15), ncol=4, fontsize=16)
	plt.show()

def visualize_start_goal(seed, eval_tf_env, distance=60):
	eval_tf_env.pyenv.envs[0].gym.set_sample_goal_args(
		prob_constraint=1.0,
		min_dist=distance,
		max_dist=distance)

	np.random.seed(seed)
	
	ts = eval_tf_env.reset()
	goal = ts.observation['goal'].numpy()[0]
	start = ts.observation['observation'].numpy()[0]

	plt.figure(figsize=(10,10))
	plot_walls(eval_tf_env.pyenv.envs[0].env.walls)
	plt.scatter([start[0]], [start[1]], marker='+', color='red', s=200, label='start')
	plt.scatter([goal[0]], [goal[1]], marker='*', color='green', s=200, label='goal')
	plt.show()

def visualize_replaybuffer(seed, eval_tf_env, kmeans, distance=60):
	eval_tf_env.pyenv.envs[0].gym.set_sample_goal_args(
		prob_constraint=1.0,
		min_dist=distance,
		max_dist=distance)

	np.random.seed(seed)
	ts = eval_tf_env.reset()
	goal = ts.observation['goal'].numpy()[0]
	start = ts.observation['observation'].numpy()[0]
	rb_vec = fill_replay_buffer(eval_tf_env, replay_buffer_size=500, upsampling_factor=10, use_kmeans=kmeans)

	plt.figure(figsize=(10,10))
	plot_walls(eval_tf_env.pyenv.envs[0].env.walls)
	plt.scatter([start[0]], [start[1]], marker='+', color='red', s=200, label='start')
	plt.scatter([goal[0]], [goal[1]], marker='*', color='green', s=200, label='goal')
	plt.scatter(*rb_vec.T, alpha=0.5)
	plt.show()

def visualize_search_path(seed, eval_tf_env, agent, kmeans, with_steps, distance=60):
	# Only works if a path is found between start and goal
	# Otherwise no waypoints will be defined in search_policy._waypoint_vec
	if kmeans:
		# Visualize only 500 points in RB
		rb_vec = fill_replay_buffer(eval_tf_env, replay_buffer_size=750, upsampling_factor=10, use_kmeans=kmeans)
	else:
		# Visualize the standard 1000 points in RB
		rb_vec = fill_replay_buffer(eval_tf_env, use_kmeans=kmeans)
	agent.initialize_search(rb_vec, max_search_steps=8)
	search_policy = SearchPolicy(agent, rb_vec, open_loop=True)

	eval_tf_env.pyenv.envs[0].gym.set_sample_goal_args(
		prob_constraint=1.0,
		min_dist=distance,
		max_dist=distance)

	np.random.seed(seed)
	ts = eval_tf_env.reset()
	goal = ts.observation['goal'].numpy()[0]
	start = ts.observation['observation'].numpy()[0]
	obs_vec = []

	for _ in range(eval_tf_env.pyenv.envs[0]._duration):
		if ts.is_last():
			break
		obs_vec.append(ts.observation['observation'].numpy()[0])
		# if use_search:
		try:
			action = search_policy.action(ts)
		except:
			# print('Error: no path found from start to goal')
			break
		# else:
		# 	action = agent.policy.action(ts)
		ts = eval_tf_env.step(action)
	obs_vec = np.array(obs_vec)

	# Get waypoints
	waypoint_vec = [start]
	for waypoint_index in search_policy._waypoint_vec:
		waypoint_vec.append(rb_vec[waypoint_index])
	waypoint_vec.append(goal)
	waypoint_vec = np.array(waypoint_vec)

	plt.figure(figsize=(10,10))
	plot_walls(eval_tf_env.pyenv.envs[0].env.walls)
	if with_steps:
		plt.plot(obs_vec[:, 0], obs_vec[:, 1], 'b-o', alpha=0.3)
	plt.scatter([start[0]], [start[1]], marker='+', color='red', s=200, label='start')
	plt.scatter([goal[0]], [goal[1]], marker='*', color='green', s=200, label='goal')
	# plt.scatter(*rb_vec.T, alpha=0.5)
	plt.plot(waypoint_vec[:, 0], waypoint_vec[:, 1], 'y-s', alpha=0.3, label='waypoint')
	plt.legend(loc='lower left', bbox_to_anchor=(-0.8, -0.15), ncol=4, fontsize=16)
	plt.show()

