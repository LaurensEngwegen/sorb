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
    width = 0.2  # The width of the bars

    fig, ax = plt.subplots()
    for i in range(len(conditions)):
        if i>0:
            x.append([j+width for j in x[i-1]])
        ax.bar(x[i], success_rates[i], width, label=conditions[i])
    ax.set_ylim([0,1])
    ax.set_ylabel('Success rate')
    ax.set_xlabel(xlabel)
    ax.set_title(f'Performance on {env} over {n_experiments} evaluations')
    ax.set_xticks(x[0]+(len(conditions)-1)*(width/2))
    ax.set_xticklabels(labels)
    ax.grid(alpha=0.5)
    ax.legend()
    fig.tight_layout()
    plt.show()

def plot_experiments(experiments, env, resize_factor, training_iters):
    plot_args = {
        'distance': {'conditions': ['no search', 'search'], 'xlabel': 'Distance to goal'}, 
        'kmeansdistance': {'conditions': ['default', 'k-means'], 'xlabel': 'Distance to goal'},
        'kmeansbuffersize': {'conditions': ['default', 'k-means'], 'xlabel': 'Replay buffer size'},
        'maxdist': {'conditions': ['default', 'k-means'], 'xlabel': 'MaxDist parameter'},
        'upsampling': {'conditions': ['5', '10', '50', '100'], 'xlabel': 'Replay buffer size'}
        }
    for exp in experiments:
        with open(f'results/results_{exp}_{env}_resize{resize_factor}_trainiters{int(training_iters/1000)}k.pkl', 'rb') as f:
            results = pkl.load(f)
        plot_successrate(results, env, **plot_args[exp])


env = 'FourRooms'
resize_factor = 10
training_iters = 1000000
# Possible experiments: 'distance', 'kmeansdistance', 'kmeansbuffersize', 'maxdist', 'upsampling'
experiments = ['upsampling']

plot_experiments(experiments, env, resize_factor, training_iters)
