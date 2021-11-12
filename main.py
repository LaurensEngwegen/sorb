from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from environments import *
from agent import *
from train import *
from visualizations import *
from graphsearch import *

# Run this cell before training on a new environment!
tf.compat.v1.reset_default_graph()

# If you change the environment parameters below, make sure to run
# tf.reset_default_graph() in the cell above before training.
max_episode_steps = 20
env_name = 'FourRooms'	# Choose one of the environments shown above. 
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

# visualize_naive_rollouts(eval_tf_env, agent)
distance = 10
stop = ''
while stop != 'q':
	if stop == 'c':
		distance = int(input('New distance: '))
	visualize_rollouts(eval_tf_env, agent, search_policy, rb_vec, distance)
	stop = input('Input q to quit, c to change distance, anything else for another rollout: ')

