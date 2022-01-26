import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl

def get_success_rates(results):
    successes = {}
    # Count number of successes for all parameter values and conditions
    for param_value, conditions in results.items():
        successes[param_value] = {}
        for condition in conditions:
            n_experiments = len(results[param_value][condition])
            # Non-zeros (and non-Nones) indicate a success
            n_successes = np.count_nonzero(results[param_value][condition])
            if n_successes > 0:
                successes[param_value][condition] = n_successes/n_experiments
            else:
                successes[param_value][condition] = 0
    return successes

def get_nones(results):
    all_nones = {}
    # Count number of Nones for all parameter values and conditions
    for param_value, conditions in results.items():
        all_nones[param_value] = {}
        for condition in conditions:
            # Count Nones
            n_nones = sum(x is None for x in results[param_value][condition])
            all_nones[param_value][condition] = n_nones
    return all_nones

def plot_successrate(success_rates, env, labels, conditions, xlabel):
    # Average over experiments
    avg_sr = np.average(success_rates, 0)
    std_sr = np.std(success_rates, 0)

    x = [] # Used for the label locations
    x.append(np.arange(len(labels)))
    width = 0.15  # The width of the bars
    plt.rcParams.update({'font.size': 15})
    # Plot
    fig, ax = plt.subplots()
    for i in range(len(conditions)):
        if i>0:
            x.append([j+width for j in x[i-1]])
        ax.bar(x[i], avg_sr[:,i], width=width, label=conditions[i])
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

def print_nones(all_n_nones, exp, env, param_values, conditions):
    avg_nones = np.average(all_n_nones, 0)
    print(f'\nCounting Nones for experiment "{exp}" in environment {env}')
    print('(i.e. number of times no path was found to connect start and end goal)')
    for i, param_value in enumerate(param_values):
        print(f'\nParameter value: {param_value}')
        for j, condition in enumerate(conditions):
            print(f'\tCondition: {condition}\t(avg.) nones = {avg_nones[i,j]}')

def print_avg_steps(results, conditions):
    # Not used in current version...
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

def show_results(experiments, env, resize_factor, training_iters):
    xlabels = {
        'distance': 'Distance to goal', 
        'kmeansdistance': 'Distance to goal',
        'kmeansbuffersize': 'Replay buffer size',
        'maxdist': 'MaxDist parameter',
        'upsampling': 'Replay buffer size'
        }
    for exp in experiments:
        all_succ_rates = [[], [], []]
        all_n_nones = [[], [], []]
        for i in range(1, 4):
            with open(f'results/results_{exp}_{env}_run{i}.pkl', 'rb') as f:
                results = pkl.load(f)
            # Calculate succes rates
            success_rates = get_success_rates(results)
            n_nones = get_nones(results)
            # Convert dictionaries to lists for easily averaging over experiments
            param_values = [key for key in results]
            conditions = [key for key in results[param_values[0]]]
            for j, param in enumerate(param_values):
                all_succ_rates[i-1].append([])
                all_n_nones[i-1].append([])
                for condition in conditions:
                    all_succ_rates[i-1][j].append(success_rates[param][condition])
                    all_n_nones[i-1][j].append(n_nones[param][condition])
        # Plot (averaged) success rates
        plot_successrate(all_succ_rates, env, param_values, conditions, xlabels[exp])
        # Print (averaged) number of Nones
        print_nones(all_n_nones, exp, env, param_values, conditions)

        # print_avg_steps(results, plot_args[exp]['conditions'])


envs = ['FourRooms']
resize_factor = 10
training_iters = 1000000
# Possible experiments: 'distance', 'kmeansdistance', 'kmeansbuffersize', 'maxdist', 'upsampling'
experiments = ['distance', 'maxdist']

for env in envs:
    show_results(experiments, env, resize_factor, training_iters)



