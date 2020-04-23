from matplotlib import pyplot as plt
import numpy as np

def plot_convergence_from_lr_time(learning_rates, list_of_methods, list_of_labels):
    colors = ['g', 'r']
    color_labels = ['^', 'o']
    plt.figure(figsize = (3.5,2.5))
    for method, label, color, col_lab in zip(list_of_methods, list_of_labels, colors, color_labels):
        mean    = np.mean(method, axis = 0)
        std     = np.std(method, axis = 0)   
        plt.loglog(learning_rates, mean, color+col_lab, label = label)
        plt.loglog(learning_rates, mean, color+':')
        plt.fill_between(learning_rates, mean-std, mean+std, color=color, alpha=0.05)
        plt.grid(True,which="both", linestyle='--', linewidth=0.4)
        # plt.grid()
        plt.xlabel('Learning rate')
        plt.ylabel('Iterations to converge')
        plt.legend()
    plt.tight_layout()
    plt.show()

TARGET_ERROR = 0.001
problem = 'Logreg' # 'Softmax'

data = np.load(f'{problem}_mnist_iter_err{TARGET_ERROR}_raw.npz')

LEARNING_RATES, SPL_N_iter, SGD_N_iter = data['LEARNING_RATES'], data['SPL_N_iter'], data['SGD_N_iter']

plot_convergence_from_lr_time(LEARNING_RATES, [SPL_N_iter, SGD_N_iter], ['Splitting','SGD'])
plt.savefig(f'{problem}_mnist_iter_err{TARGET_ERROR}.pdf')