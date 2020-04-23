# ======= Libraries =======

import math
import sys
import os
import time
os.environ["OMP_NUM_THREADS"] = "5" # export OMP_NUM_THREADS=5
os.environ["OPENBLAS_NUM_THREADS"] = "5" # export OPENBLAS_NUM_THREADS=5
os.environ["MKL_NUM_THREADS"] = "5" # export MKL_NUM_THREADS=5
os.environ["VECLIB_MAXIMUM_THREADS"] = "5" # export VECLIB_MAXIMUM_THREADS=5
os.environ["NUMEXPR_NUM_THREADS"] = "5" # export NUMEXPR_NUM_THREADS=5
import torch
from torch.autograd import Variable
from torch.nn import functional as F
import torch.nn as nn
import torchvision.datasets as datasets
from scipy.integrate import odeint, solve_ivp
# from torchdiffeq import odeint
from scipy.linalg import expm, qr
import scipy.io as sio
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import log_loss
import copy
from tqdm import tqdm
# if torch.cuda.is_available():
#     device = torch.device('cuda')
# else:
device = torch.device('cpu')
import copy
from fastprogress.fastprogress import master_bar, progress_bar
import random
# Reproducibility
random.seed(999)
np.random.seed(999)
torch.manual_seed(999)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ======= Problem generation =======

TARGET_RES      = 3e-5
N_EXPERIMENTS   = 11
LEARNING_RATES  = np.array(np.logspace(-0.2, 0.8, 15))
# LEARNING_RATES  = np.append(LEARNING_RATES, np.array(np.logspace(-1.2, 8, 6)))
iter_limit      = 30000

GD_N_iter     = np.zeros((N_EXPERIMENTS, len(LEARNING_RATES)))
SGD_N_iter    = np.zeros((N_EXPERIMENTS, len(LEARNING_RATES)))
SPL_N_iter    = np.zeros((N_EXPERIMENTS, len(LEARNING_RATES)))

GD_time     = np.zeros((N_EXPERIMENTS, len(LEARNING_RATES)))
SGD_time    = np.zeros((N_EXPERIMENTS, len(LEARNING_RATES)))
SPL_time    = np.zeros((N_EXPERIMENTS, len(LEARNING_RATES)))

p = 500
n = 10000
s = 500
b = 20
epsilon = 1e-1

# problems = ['tomography', 'random']
problems = ['tomography']
# problems = ['random']

# ======= Function definitions =======

def load_tom_data(epsilon=1e-5):
    X = sio.loadmat("./lls_data/fanlinear.mat")["A"].toarray()
    theta_true = sio.loadmat("./lls_data/shepplogan.mat")["x"]
    n, p = X.shape
    y = np.squeeze(X @ theta_true)
    return X, theta_true, y

def generate_problem(p, n, lstsq=False, epsilon = 0):
    X = np.random.randn(n, p)

    # Model definition
    theta_clean = np.ones(p)
    y = X @ theta_clean + epsilon*np.random.randn(n) # right-hand side
    init_bound = 1.0/math.sqrt(p)
    theta_0 = np.array(init_bound*torch.FloatTensor(p).uniform_(-1, 1))

    if lstsq == True:
        theta_lstsq = np.linalg.lstsq(X,y)[0]
        return X, theta_0, y, theta_lstsq
    else:
        return X, theta_0, y

def solve_local_problem(Q, R, theta_0, y_batch, h, n):
    try:
        R_it = np.linalg.inv(R.T)
    except np.linalg.LinAlgError as err:
        # print(err)
        R_it = np.linalg.pinv(R.T)
    exp_m = expm(-1/n* R @ R.T*h)
    return Q @ ( exp_m @ (Q.T @ theta_0 - R_it @ y_batch )) + Q @ (R_it @ y_batch) + theta_0 - Q @ (Q.T @ theta_0)

def solve_local_problem_b_1(x, theta_0, y, h, n):
    x = x.T
    norm = x.T @ x
    return theta_0 + (1 - np.exp(-norm*h/n))*(y - x.T @ theta_0)/norm*x

def loss(X, theta, y):
    '''
    Supports batch reformulation. The difference in dimension of the input
    '''
    if len(X.shape) == 2:
        n, p = X.shape
        return 1/n*np.linalg.norm(X @ theta - y)**2
    elif len(X.shape) == 3:
        s, b, p = Xs.shape
        n = b*s

        loss = 0
        for i_batch in range(s):
            loss += 1/n*np.linalg.norm(X[i_batch] @ theta - y[i_batch])**2
        return loss
    else:
        raise ValueError('🤔 Inappropriate format of dataset')

def rel_residual(X, theta, y):
    '''
    Supports batch reformulation. The difference in dimension of the input
    '''
    if len(X.shape) == 2:
        n, p = X.shape
        return np.linalg.norm(X @ theta - y)/np.linalg.norm(y)
    elif len(X.shape) == 3:
        s, b, p = Xs.shape
        n = b*s

        loss = 0
        y_full = np.zeros(n)
        X_full = np.zeros((n, p))
        for i_batch in range(s):
            y_full[b*i_batch:b*(i_batch+1)]     = y[i_batch]
            X_full[b*i_batch:b*(i_batch+1), :]  = X[i_batch]

        return np.linalg.norm(X_full @ theta - y_full)/np.linalg.norm(y_full)
    else:
        raise ValueError('🤔 Inappropriate format of dataset')


def gradient(X, theta, y):
    n, p = X.shape
    return 1/n* X.T @ (X @ theta - y)

def make_SGD_step(X_batch, theta_0, y_batch, lr):
    theta = theta_0 - lr*gradient(X_batch, theta_0, y_batch)
    return theta

def sgd_training(theta_0, Xs, ys, lr, final_res = 1e-4, iter_limit = 1000):
    s, b, p = Xs.shape
    n = b*s
    losses = []
    theta_t = theta_0
    N_iter = 0
    stop_word = False
    if lr >= 10:
        iter_limit = 10000
    while not stop_word:          
        i_batch = N_iter % s
        loss_t = loss(Xs, theta_t, ys)
        losses.append(loss_t)
        theta_t = make_SGD_step(Xs[i_batch], theta_t, ys[i_batch], lr)
        N_iter += 1
        if N_iter % 50 == 0:
            sys.stdout.write('\r'+f'🤖 SGD rel_res {rel_residual(Xs, theta_t, ys):.5f}, error {losses[-1]:.3f}/{final_res:.5f} on {N_iter}-th iteration. Lr {lr}')
        if losses[-1] <= final_res:
            stop_word = True
            break

        if losses[-1] >= 1e4 or N_iter >= iter_limit:
            stop_word = True
            N_iter = None
    
    print(f'\n🤖 SGD finished with {N_iter} iterations on lr {lr}')

    return N_iter

def spl_training(theta_0, Qs, Rs, Xs, ys, stepsize, final_res = 1e-4, iter_limit = 1000):
    s, b, p = Xs.shape
    n = b*s
    losses = []
    theta_t = theta_0
    N_iter = 0
    stop_word = False
    while not stop_word:          
        i_batch = N_iter % s
        loss_t = loss(Xs, theta_t, ys)
        losses.append(loss_t)
        theta_t = solve_local_problem(Qs[i_batch], Rs[i_batch], theta_t, ys[i_batch], stepsize, n)
        N_iter += 1
        if N_iter % 50 == 0:
            sys.stdout.write('\r'+f'🤖 Splitting rel_res {rel_residual(Xs, theta_t, ys)}, error {losses[-1]:.5f}/{final_res:.5f} on {N_iter}-th iteration. Stepsize {stepsize}')
        if losses[-1] <= final_res:
            stop_word = True
            break

        if losses[-1] >= 1e4 or N_iter >= iter_limit:
            stop_word = True
            N_iter = None
    
    print(f'\n🤖 Splitting finished with {N_iter} iterations on Stepsize {stepsize}')

    return N_iter

def plot_convergence_from_lr_time(learning_rates, list_of_methods, list_of_labels):
    colors = ['g', 'r']
    color_labels = ['^', 'o']
    plt.figure(figsize = (3.5,2.5))
    for method, label, color, col_lab in zip(list_of_methods, list_of_labels, colors, color_labels):
        mean    = np.zeros(len(learning_rates))
        std     = np.zeros(len(learning_rates))

        for i_lr, lr in enumerate(learning_rates):
            if any(method[:, i_lr]) == None:
                mean[i_lr] = None
                std[i_lr]  = None
            else:
                mean[i_lr] = np.mean(method[:, i_lr])
                std[i_lr]  = np.std(method[:, i_lr])
        plt.loglog(learning_rates, mean, color+col_lab, label = label)
        plt.loglog(learning_rates, mean, color+':')
        plt.fill_between(learning_rates, mean-std, mean+std, color=color, alpha=0.1)
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
            if any(method[:, i_lr]) == None:
                mean[i_lr] = None
                std[i_lr]  = None
            else:
                mean[i_lr] = np.mean(method[:, i_lr])
                std[i_lr]  = np.std(method[:, i_lr])
        std     = np.std(method, axis = 0)   
        plt.loglog(learning_rates, mean, color+col_lab, label = label)
        plt.loglog(learning_rates, mean, color+':')
        plt.fill_between(learning_rates, mean-std, mean+std, color=color, alpha=0.1)
        plt.grid(True,which="both", linestyle='--', linewidth=0.4)
        # plt.grid()
        plt.xlabel('Learning rate')
        plt.ylabel('Iterations to converge')
        plt.legend()
    plt.tight_layout()
    plt.show()

for problem in problems:
    print(f'============ ☄ {problem} ☄ ============')
    if problem == 'tomography':
        X, theta_true, y = load_tom_data()
        n, p    = X.shape
        b, s    = 60, 213

    for i_exp in progress_bar(range(N_EXPERIMENTS)):
        print(f'============ ☄ {i_exp+1}/ {N_EXPERIMENTS} ☄ ============')
        # Random initialization
        if problem == 'tomography':
            init_bound  = 1.0/math.sqrt(p)
            theta_0     = np.array(init_bound*torch.FloatTensor(p).uniform_(-1, 1))
            permutation = np.random.permutation(n)
            X, y        = X[permutation], y[permutation]
        elif problem == 'random':
            X, theta_0, y, theta_lstsq = generate_problem(p,n, lstsq=True, epsilon=epsilon)

        Xs = np.zeros((s, b, p))
        ys = np.zeros((s, b))
        Qs = np.zeros((s, p, b))
        Rs = np.zeros((s, b, b))
        Q, R = qr(X.T, mode='economic')

        for i_batch in range(s):
            Xs[i_batch] = X[b*i_batch:b*(i_batch+1), :]
            ys[i_batch] = y[b*i_batch:b*(i_batch+1)]
            Qs[i_batch], Rs[i_batch] = qr(Xs[i_batch].T, mode='economic')
            
        # RUN
        for i_lr, learning_rate in enumerate(LEARNING_RATES):
            stepsize = learning_rate*n/b
            print(f'======🌠 lr {learning_rate}, h {stepsize} 🌠======')

            start_time = time.time()
            N_iter = spl_training(theta_0, Qs, Rs, Xs, ys, stepsize, final_res = TARGET_RES, iter_limit = iter_limit)
            end_time = time.time()
            SPL_time[i_exp, i_lr] = end_time - start_time
            SPL_N_iter[i_exp, i_lr] = N_iter
            if N_iter == None:
                SPL_N_iter[i_exp, i_lr] = None

            start_time = time.time()
            N_iter = sgd_training(theta_0, Xs, ys, learning_rate, final_res = TARGET_RES, iter_limit = iter_limit)
            end_time = time.time()
            SGD_time[i_exp, i_lr] = end_time - start_time
            SGD_N_iter[i_exp, i_lr] = N_iter
            if N_iter == None:
                SGD_time[i_exp, i_lr] = None

            np.savez(f'LLS_{problem}_time_err{TARGET_RES}_raw.npz', SPL_time=SPL_time, SGD_time=SGD_time, LEARNING_RATES = LEARNING_RATES)
            np.savez(f'LLS_{problem}_iter_err{TARGET_RES}_raw.npz', SPL_N_iter=SPL_N_iter, SGD_N_iter=SGD_N_iter, LEARNING_RATES = LEARNING_RATES)

            plot_convergence_from_lr_time(LEARNING_RATES, [SPL_time, SGD_time], ['Splitting','SGD'])
            plt.savefig(f'LLS_{problem}_time_err{TARGET_RES}.pdf')
            plot_convergence_from_lr(LEARNING_RATES, [SPL_N_iter, SGD_N_iter], ['Splitting','SGD'])
            plt.savefig(f'LLS_{problem}_iter_err{TARGET_RES}.pdf')