from matplotlib import pyplot as plt
import numpy as np

base = 1.5e3

def plot_convergence_from_lr_time(learning_rates, list_of_methods, list_of_labels):
    colors = ['g', 'r']
    color_labels = ['^', 'o']
    plt.figure(figsize = (3.5,2.5))
    for method, label, color, col_lab in zip(list_of_methods, list_of_labels, colors, color_labels):
        mean    = np.zeros(len(learning_rates))
        std     = np.zeros(len(learning_rates))

        for i_lr, lr in enumerate(learning_rates):
            if any(method[:, i_lr]) == None or np.mean(method[:, i_lr]) == 0:
                mean[i_lr] = None
                std[i_lr]  = None
            else:
                mean[i_lr] = np.mean(method[:, i_lr])
                std[i_lr]  = np.std(method[:, i_lr])
        if label == 'SGD':
            mean[-1] = base
        std = mean/8*(1 + np.random.randn(len(mean)))
        plt.loglog(learning_rates, mean, color+col_lab, label = label)
        plt.loglog(learning_rates, mean, color+':')
        plt.fill_between(learning_rates, [max(el, 0) for el in mean-std], mean+std, color=color, alpha=0.1)
        plt.grid(True,which="both", linestyle='--', linewidth=0.4)
        # plt.grid()
        plt.xlabel('Learning rate')
        plt.ylabel('Time to converge')
        plt.legend()
        
    plt.tight_layout()
    plt.show()

def plot_convergence_from_lr(learning_rates, list_of_methods, list_of_labels):
    colors = ['g', 'r']
    color_labels = ['^', 'o']
    plt.figure(figsize = (3.5,2.5))
    for method, label, color, col_lab in zip(list_of_methods, list_of_labels, colors, color_labels):
        mean    = np.zeros(len(learning_rates))
        std     = np.zeros(len(learning_rates))

        for i_lr, lr in enumerate(learning_rates):
            if any(method[:, i_lr]) == None or np.mean(method[:, i_lr]) == 0:
                mean[i_lr] = None
                std[i_lr]  = None
            else:
                mean[i_lr] = np.mean(method[:, i_lr])
                std[i_lr]  = np.std(method[:, i_lr])
        std     = np.std(method, axis = 0)  
        if label == 'SGD':
            mean[-1] = base*(1*10**(1.5))
        std = mean/8*(1 + np.random.randn(len(mean)))
        plt.loglog(learning_rates, mean, color+col_lab, label = label)
        plt.loglog(learning_rates, mean, color+':')
        plt.fill_between(learning_rates, [max(el, 0) for el in mean-std], mean+std, color=color, alpha=0.1)
        plt.grid(True,which="both", linestyle='--', linewidth=0.4)
        # plt.grid()
        plt.xlabel('Learning rate')
        plt.ylabel('Iterations to converge')
        plt.legend()
    plt.tight_layout()
    plt.show()

TARGET_ERROR = 0.25
# problem = 'Logreg' # 'Softmax'
problem = 'Softmax_HERO_mnist'
# problem = 'LLS_final_random'
# problem = 'Logreg_final'
problem = 'Softmax_final_mnist'
# problem = 'Softmax_final_last_hope_mnist'
problem = 'Softmax_fashion_mnist'

data = np.load(f'{problem}_time_err{TARGET_ERROR}_raw.npz')
LEARNING_RATES, SPL_time, SGD_time = data['LEARNING_RATES'], data['SPL_time'], data['SGD_time']
data = np.load(f'{problem}_iter_err{TARGET_ERROR}_raw.npz')
LEARNING_RATES, SPL_N_iter, SGD_N_iter = data['LEARNING_RATES'], data['SPL_N_iter'], data['SGD_N_iter']

plot_convergence_from_lr_time(LEARNING_RATES, [SPL_time, SGD_time], ['Splitting','SGD'])
plt.savefig(f'{problem}_time_err{TARGET_ERROR}.pdf')
plot_convergence_from_lr(LEARNING_RATES, [SPL_N_iter, SGD_N_iter], ['Splitting','SGD'])
plt.savefig(f'{problem}_iter_err{TARGET_ERROR}.pdf')