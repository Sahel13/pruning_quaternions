import os
import pandas as pd
import matplotlib.pyplot as plt


def load_pruned_data(base_dir, num_trials, retrain=True):
    """
    Function to load data from pruning experiments.
    """
    comb_table = None

    for model in ['real', 'quat']:
        first_file = None
        for trial in range(num_trials):
            trial += 1
            dir_path = os.path.join(base_dir, f'Trial {trial}', model)

            if retrain:
                acc_file = pd.read_csv(
                    os.path.join(dir_path, 'acc_data_retrain.csv')
                )
            else:
                acc_file = pd.read_csv(
                    os.path.join(dir_path, 'acc_data.csv')
                )

            if trial == 1:
                first_file = acc_file
            else:
                first_file = pd.concat((first_file, acc_file['Accuracy']),
                                       axis=1)

        # Get mean and standard deviation.
        mean = first_file.iloc[:, 1:].mean(axis=1)
        std = first_file.iloc[:, 1:].std(axis=1)

        # Combine the data.
        table = first_file['Sparsity']
        table = pd.concat((table, mean, std), axis=1)

        # Combine results for both models.
        if model == 'real':
            col_names = ['Sparsity (R)', 'Mean (R)', 'Std (R)']
            table.columns = col_names
            comb_table = table
        else:
            col_names = ['Sparsity (Q)', 'Mean (Q)', 'Std (Q)']
            table.columns = col_names
            comb_table = pd.concat((comb_table, table), axis=1)

    return comb_table


def plot_results(base_dir, num_trials, retrain: bool):
    """
    Function to plot data from pruning experiments.
    """
    data = load_pruned_data(base_dir, num_trials, retrain)

    plt.figure(figsize=(10, 5))
    plt.errorbar(x=data['Sparsity (R)'], y=data['Mean (R)'],
                 yerr=data['Std (R)'], label='Real')
    plt.errorbar(x=data['Sparsity (Q)']*0.25, y=data['Mean (Q)'],
                 yerr=data['Std (Q)'], label='Quat')

    plt.xscale('log')
    plt.xlabel('Percentage of weights left.')
    plt.ylabel('Mean accuracy +- std')

    if retrain:
        last_words = 'when retrained.'
    else:
        last_words = 'during pruning.'

    plt.title(f'Accuracy vs sparsity {last_words}')
    plt.legend()
    plt.show()
