import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()


class ProcessResults():
    def __init__(self, base_dir, num_trials: int):
        self.base_dir = base_dir
        self.num_trials = num_trials

    def _get_mean_std(self, rel_path, file_name: str, retrain: bool,
                      x: int = 0, y: int = -1):
        """
        Function to combine the data from mutliple trials.
        rel_path is the relative path from inside the model
        specific folder to the file with name file_name.
        file_name should not include the csv extension.
        """
        data = {}

        for model in ['real', 'quat']:
            first_file = None
            for trial in range(self.num_trials):
                trial += 1
                dir_path = os.path.join(self.base_dir, f'Trial {trial}', model,
                                        rel_path)

                # If data does not exist for a particular trial,
                # return none.
                if not os.path.isdir(dir_path):
                    return None

                if retrain:
                    acc_file = pd.read_csv(
                        os.path.join(dir_path, f'{file_name}_retrain.csv')
                    )
                else:
                    acc_file = pd.read_csv(
                        os.path.join(dir_path, f'{file_name}.csv')
                    )

                if trial == 1:
                    first_file = pd.concat(
                        (acc_file.iloc[:, x], acc_file.iloc[:, y]), axis=1
                    )
                else:
                    agg_len = first_file.shape[0]
                    curr_len = acc_file.shape[0]
                    if curr_len > agg_len:
                        first_file = pd.concat(
                            (first_file, acc_file.iloc[:agg_len, y]),
                            axis=1
                        )
                    elif curr_len < agg_len:
                        first_file = pd.concat(
                            (first_file.iloc[:curr_len], acc_file.iloc[:, y]),
                            axis=1
                        )
                    else:
                        first_file = pd.concat(
                            (first_file, acc_file.iloc[:, y]), axis=1
                        )

            # Get mean and standard deviation.
            mean = first_file.iloc[:, 1:].mean(axis=1)
            std = first_file.iloc[:, 1:].std(axis=1)

            # Combine the data.
            table = first_file.iloc[:, x]
            table = pd.concat((table, mean, std), axis=1)

            data[model] = table

        return data

    def save_model(self):
        for model in ['real', 'quat']:
            file_path = os.path.join(self.base_dir, f'{model}_data.csv')
            self.data[model].to_csv(file_path, index=False)

    def train_log(self, level: int):
        """
        Function to load the train log for Q and R
        where level is the pruning level (n = 0 for no pruning).
        """
        retrain = True
        if level == 0:
            retrain = False

        data = self._get_mean_std(f'Level {level}', 'logger', retrain)

        for model in ['real', 'quat']:
            table = data[model]

            if model == 'real':
                col_names = ['Epoch (R)', 'Mean (R)', 'Std (R)']
            else:
                col_names = ['Epoch (Q)', 'Mean (Q)', 'Std (Q)']

            table.columns = col_names
            data[model] = table

        return data

    def plot_train_log(self):
        """
        Function to plot data from pruning experiments.
        """
        print('Accuracy vs training epochs.')

        data = self.train_log(0)

        plt.figure(figsize=(10, 7))
        plt.errorbar(
            x=data['real']['Epoch (R)'],
            y=data['real']['Mean (R)'],
            yerr=data['real']['Std (R)'],
            label='Real'
        )

        plt.errorbar(
            x=data['quat']['Epoch (Q)'],
            y=data['quat']['Mean (Q)'],
            yerr=data['quat']['Std (Q)'],
            label='Quaternion'
        )

        plt.xlabel('Number of training epochs.')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()

    def plot_lth(self):
        pruning_iterations = 2

        for model in ['real', 'quat']:
            print(f'Results for {model}')

            plt.figure(figsize=(10, 7))
            for iter in range(pruning_iterations + 1):
                data = self.train_log(iter)[model]

                if data is None:
                    break

                plt.errorbar(
                    x=data.iloc[:, 0],
                    y=data.iloc[:, 1],
                    # yerr=data.iloc[:, 2],
                    label=iter
                )

            plt.xlabel('Number of epochs.')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.show()

    def spar_acc_data(self, retrain: bool):
        """
        Function to load sparsity vs accuracy data.
        """
        data = self._get_mean_std('', 'acc_data', retrain)

        for model in ['real', 'quat']:
            table = data[model]

            if model == 'real':
                col_names = ['Sparsity (R)', 'Mean (R)', 'Std (R)']
            else:
                col_names = ['Sparsity (Q)', 'Mean (Q)', 'Std (Q)']

            table.columns = col_names
            data[model] = table

        return data

    def plot_spar_acc(self, retrain: bool = True):
        """
        Function to plot data from pruning experiments.
        """
        data = self.spar_acc_data(retrain)

        if retrain:
            last_words = 'when retrained.'
        else:
            last_words = 'during pruning.'
        print(f'Accuracy vs sparsity {last_words}')

        plt.figure(figsize=(10, 7))
        plt.errorbar(
            x=data['real']['Sparsity (R)'],
            y=data['real']['Mean (R)'],
            yerr=data['real']['Std (R)'],
            label='Real'
        )

        plt.errorbar(
            x=data['quat']['Sparsity (Q)']*0.25,
            y=data['quat']['Mean (Q)'],
            yerr=data['quat']['Std (Q)'],
            label='Quaternion'
        )

        plt.xscale('log')
        plt.xlabel('Percentage of weights left.')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()
