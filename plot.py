import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl

def get_success_rates(results, conditions):
    # Will contain success rate (for each of the labels: a list with for every condition the success rate)
    successes = []
    # Calculate success rates and store them in successes
    for i, condition in enumerate(range(len(conditions))):
        successes.append([])
        for key in results.keys():
            # Multiple conditions
            if len(conditions) > 1:
                n_experiments = len(results[key][0])
                cond_successes = np.count_nonzero(results[key][condition])
                if cond_successes > 0:
                    successes[i].append(cond_successes/n_experiments)
                else:
                    successes[i].append(0)
            # Only one condition
            else:
                n_experiments = len(results[key])
                cond_successes = np.count_nonzero(results[key])
                if cond_successes > 0:
                    successes[i].append(cond_successes/n_experiments)
                else:
                    successes[i].append(0)
    return successes, n_experiments

def plot_successrate(results, env, conditions, xlabel):
    success_rates, n_experiments = get_success_rates(results, conditions)
    
    labels = [key for key in results.keys()]
    x = [] # Used for the label locations
    x.append(np.arange(len(labels)))
    width = 0.15  # The width of the bars
    plt.rcParams.update({'font.size': 15})

    fig, ax = plt.subplots()
    for i in range(len(conditions)):
        if i>0:
            x.append([j+width for j in x[i-1]])
        ax.bar(x[i], success_rates[i], width, label=conditions[i])
    ax.set_ylim([0,1])
    ax.set_ylabel('Success rate')
    ax.set_xlabel(xlabel)
    ax.set_title(f'Success rate in {env}')
    ax.set_xticks(x[0]+(len(conditions)-1)*(width/2))
    ax.set_xticklabels(labels)
    ax.grid(alpha=0.5)
    ax.legend()
    fig.tight_layout()
    plt.show()

def print_nones(results, conditions):
    for i in range(len(conditions)):
        print(f'Condition: {conditions[i]}')
        for paramvalue in results.keys():
            n_nones = sum(x is None for x in results[paramvalue][i])
            print(f'Parameter value: {paramvalue},\tNones = {n_nones}')

def print_avg_steps(results, conditions):
    for i in range(len(conditions)):
        print(f'Condition: {conditions[i]}')
        for paramvalue in results.keys():
            n_steps = 0
            successes = 0
            for j in range(len(results[paramvalue][i])):
                if results[paramvalue][i][j] is not None and results[paramvalue][i][j] > 0:
                    n_steps += results[paramvalue][i][j]
                    successes += 1
            if successes > 0:
                print(f'Average number of steps to reach goal: {n_steps/successes}')

def plot_experiments(experiments, env, resize_factor, training_iters):
    plot_args = {
        'distance': {'conditions': ['no search', 'search'], 'xlabel': 'Distance to goal'}, 
        'kmeansdistance': {'conditions': ['default', 'k-means'], 'xlabel': 'Distance to goal'},
        'kmeansbuffersize': {'conditions': ['default', 'k-means'], 'xlabel': 'Replay buffer size'},
        'maxdist': {'conditions': ['default', 'k-means'], 'xlabel': 'MaxDist parameter'},
        'upsampling': {'conditions': ['1', '5', '10', '50', '100'], 'xlabel': 'Replay buffer size'}
        }
    for exp in experiments:
        with open(f'results/results_{exp}_{env}_resize{resize_factor}_trainiters{int(training_iters/1000)}k.pkl', 'rb') as f:
            results = pkl.load(f)
        plot_successrate(results, env, **plot_args[exp])
        # print(f'\nCounting Nones for experiment "{exp}" in environment {env}')
        # print_nones(results, plot_args[exp]['conditions'])
        print_avg_steps(results, plot_args[exp]['conditions'])

envs = ['FourRooms']
resize_factor = 10
training_iters = 1000000
# Possible experiments: 'distance', 'kmeansdistance', 'kmeansbuffersize', 'maxdist', 'upsampling'
experiments = ['upsampling']

for env in envs:
    plot_experiments(experiments, env, resize_factor, training_iters)


