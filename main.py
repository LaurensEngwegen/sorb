from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from environments import *
from agent import *
from train import *
from visualizations import *

from tf_agents.environments import gym_wrapper
from tf_agents.environments import tf_py_environment

		
def env_load_fn(environment_name,
				 max_episode_steps=None,
				 resize_factor=1,
				 gym_env_wrappers=(GoalConditionedPointWrapper,),
				 terminate_on_timeout=False):
	"""Loads the selected environment and wraps it with the specified wrappers.

	Args:
		environment_name: Name for the environment to load.
		max_episode_steps: If None the max_episode_steps will be set to the default
			step limit defined in the environment's spec. No limit is applied if set
			to 0 or if there is no timestep_limit set in the environment's spec.
		gym_env_wrappers: Iterable with references to wrapper classes to use
			directly on the gym environment.
		terminate_on_timeout: Whether to set done = True when the max episode
			steps is reached.

	Returns:
		A PyEnvironmentBase instance.
	"""
	gym_env = PointEnv(walls=environment_name,
										 resize_factor=resize_factor)
		
	for wrapper in gym_env_wrappers:
		gym_env = wrapper(gym_env)
	env = gym_wrapper.GymWrapper(
			gym_env,
			discount=1.0,
			auto_reset=True,
	)

	if max_episode_steps > 0:
		if terminate_on_timeout:
			env = wrappers.TimeLimit(env, max_episode_steps)
		else:
			env = NonTerminatingTimeLimit(env, max_episode_steps)

	return tf_py_environment.TFPyEnvironment(env)



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


start = time.time()
# train_eval(
# 		agent,
# 		tf_env,
# 		eval_tf_env,
# 		initial_collect_steps=1000,
# 		eval_interval=1000,
# 		num_eval_episodes=10,
# 		num_iterations=30000,
# )
print(f'Training took {round((time.time()-start)/60, 2)} minutes')

visualize_rollouts(eval_tf_env, agent)

