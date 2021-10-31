import os
import pandas as pd
import matplotlib.pyplot as plt


class ProcessResults():
    def __init__(self, base_dir, num_trials: int, retrain: bool):
        self.base_dir = base_dir
        self.num_trials = num_trials
        self.retrain = retrain
        self.data = self.load_pruned_data()

    def load_pruned_data(self):
        """
        Function to load data from pruning experiments.
        """
        data = {}

        for model in ['real', 'quat']:
            first_file = None
            for trial in range(self.num_trials):
                trial += 1
                dir_path = os.path.join(self.base_dir, f'Trial {trial}', model)

                if self.retrain:
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
                    agg_len = first_file.shape[0]
                    curr_len = acc_file.shape[0]
                    if curr_len > agg_len:
                        first_file = pd.concat(
                            (first_file, acc_file.iloc[:agg_len]['Accuracy']),
                            axis=1
                        )
                    elif curr_len < agg_len:
                        first_file = pd.concat(
                            (first_file.iloc[:curr_len], acc_file['Accuracy']),
                            axis=1
                        )
                    else:
                        first_file = pd.concat(
                            (first_file, acc_file['Accuracy']), axis=1
                        )

            # Get mean and standard deviation.
            mean = first_file.iloc[:, 1:].mean(axis=1)
            std = first_file.iloc[:, 1:].std(axis=1)

            # Combine the data.
            table = first_file['Sparsity']
            table = pd.concat((table, mean, std), axis=1)

            # Rename columns
            if model == 'real':
                col_names = ['Sparsity (R)', 'Mean (R)', 'Std (R)']
            else:
                col_names = ['Sparsity (Q)', 'Mean (Q)', 'Std (Q)']

            table.columns = col_names
            data[model] = table

        return data

    def plot_results(self):
        """
        Function to plot data from pruning experiments.
        """
        plt.figure(figsize=(10, 5))
        plt.errorbar(
            x=self.data['real']['Sparsity (R)'],
            y=self.data['real']['Mean (R)'],
            yerr=self.data['real']['Std (R)'],
            label='Real'
        )

        plt.errorbar(
            x=self.data['quat']['Sparsity (Q)']*0.25,
            y=self.data['quat']['Mean (Q)'],
            yerr=self.data['quat']['Std (Q)'],
            label='Quaternion'
        )

        plt.xscale('log')
        plt.xlabel('Percentage of weights left.')
        plt.ylabel('Mean accuracy +- std')

        if self.retrain:
            last_words = 'when retrained.'
        else:
            last_words = 'during pruning.'

        plt.title(f'Accuracy vs sparsity {last_words}')
        plt.legend()
        plt.show()

    def save_model(self):
        for model in ['real', 'quat']:
            file_path = os.path.join(self.base_dir, f'{model}_data.csv')
            self.data[model].to_csv(file_path, index=False)
