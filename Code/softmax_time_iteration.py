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

TARGET_ERROR    = 0.25
N_EXPERIMENTS   = 10
LEARNING_RATES  = np.array(np.logspace(-2, 1.5, 10))[::-1]
# LEARNING_RATES  = np.append(LEARNING_RATES, np.array(np.logspace(1.5, 2.5, 6)))
iter_limit      = 300000

GD_time     = np.zeros((N_EXPERIMENTS, len(LEARNING_RATES)))
SGD_time    = np.zeros((N_EXPERIMENTS, len(LEARNING_RATES)))
SPL_time    = np.zeros((N_EXPERIMENTS, len(LEARNING_RATES)))

GD_N_iter     = np.zeros((N_EXPERIMENTS, len(LEARNING_RATES)))
SGD_N_iter    = np.zeros((N_EXPERIMENTS, len(LEARNING_RATES)))
SPL_N_iter    = np.zeros((N_EXPERIMENTS, len(LEARNING_RATES)))

# Problem generation
batch_size = 64
number_of_classes = 10


# ======= Function definitions =======

def load_batched_data(batch_size=50, shuffle = True, qr_mode = False, number_of_classes = 3):
    '''
    Load batches of MNIST data.

    Output: X_trains - s_train batches of training data, 
            y_trains - s_train batches of labels,
            X_test - test points
            y_test - test labels
    X_trains: torch.array of shape (s_train,batch_size,*X_train[0].shape),
        where 
        s_train - the number of batches, 
        batch_size - batch size
        *X_train[0].shape - shape of the dataset point;

    y_trains: torch.array of shape (s_train, K, batch_size),
        where
        K - the number of classes in the problem;

    X_test: torch.array of shape (n_test,*X_train[0].shape),
        where
        n_test - the number of test points;

    y_test: torch.array of shape (K, n_test);
    '''
    trainset = datasets.FashionMNIST('./fashion_mnist_data/', download=True, train=True)
    X_train = trainset.train_data.to(dtype=torch.float)/255
    y_train = trainset.train_labels
    mask    = y_train < number_of_classes
    X_train = X_train[mask]
    y_train = y_train[mask]
    X_train.resize_(len(X_train), *X_train[0].view(-1).shape)
    y_train.view(-1).long()

    if shuffle == True:
        shuffling = torch.randperm(len(y_train))
        X_train = X_train[shuffling]
        y_train = y_train[shuffling]

    # Download and load the test data
    testset = datasets.FashionMNIST('./fashion_mnist_data/', download=True, train=False)
    X_test = testset.test_data.to(dtype=torch.float)/255
    y_test = testset.test_labels
    mask   = y_test < number_of_classes
    X_test = X_test[mask]
    y_test = y_test[mask]
    X_test.resize_(len(X_test), *X_test[0].view(-1).shape)
    y_test.view(-1).long()

    if shuffle == True:
        shuffling = torch.randperm(len(y_test))
        X_test = X_test[shuffling].to(device)
        y_test = y_test[shuffling]

    n_train = len(y_train)
    n_test  = len(y_test)

    s_train = int(n_train/batch_size)   # Number of training batches

    K           = number_of_classes 
    X_trains    = torch.zeros((s_train, batch_size, *X_train[0].view(-1).shape), requires_grad=False).to(device)
    y_trains    = torch.zeros((s_train, K, batch_size), requires_grad=False, dtype=torch.int64).to(device)
    if qr_mode:
        Qs      = torch.zeros((s_train, *X_train[0].view(-1).shape, batch_size), requires_grad=False).to(device)
        Rs      = torch.zeros((s_train, batch_size, batch_size), requires_grad=False).to(device)
        print('🤖QR started')

    y_test_one_hot = torch.zeros((n_test, K))
    y_test_one_hot[np.arange(n_test), y_test] = 1
    y_test_one_hot = y_test_one_hot.t()

    for i in range(s_train):
        X_trains[i] = X_train[batch_size*i:batch_size*(i+1), :]
        batch_lbls  = y_train[batch_size*i:batch_size*(i+1)]
        y_batch_one_hot = torch.zeros((batch_size, K))
        y_batch_one_hot[np.arange(batch_size), batch_lbls] = 1
        y_trains[i] = y_batch_one_hot.t()
        if qr_mode:
            Qs[i], Rs[i] = torch.qr(X_trains[i].t())      

    if qr_mode:
        print('✅QR computed')
        return X_trains, y_trains, X_test, y_test_one_hot, Qs, Rs
    else:
        return X_trains, y_trains, X_test, y_test_one_hot

def load_batched_data_cifar(batch_size=50, shuffle = True, qr_mode = False, number_of_classes = 3):
    '''
    Load batches of MNIST data.

    Output: X_trains - s_train batches of training data, 
            y_trains - s_train batches of labels,
            X_test - test points
            y_test - test labels
    X_trains: torch.array of shape (s_train,batch_size,*X_train[0].shape),
        where 
        s_train - the number of batches, 
        batch_size - batch size
        *X_train[0].shape - shape of the dataset point;

    y_trains: torch.array of shape (s_train, K, batch_size),
        where
        K - the number of classes in the problem;

    X_test: torch.array of shape (n_test,*X_train[0].shape),
        where
        n_test - the number of test points;

    y_test: torch.array of shape (K, n_test);
    '''
    trainset = datasets.CIFAR10('./cifar_data/', download=True, train=True)
    X_train = torch.from_numpy(trainset.train_data).to(dtype=torch.float)/255
    y_train = torch.Tensor(trainset.train_labels)
    mask    = y_train < number_of_classes
    X_train = X_train[mask]
    y_train = y_train[mask]
    X_train.resize_(len(X_train), *X_train[0].view(-1).shape)
    y_train.view(-1).long()

    if shuffle == True:
        shuffling = torch.randperm(len(y_train))
        X_train = X_train[shuffling]
        y_train = y_train[shuffling]

    # Download and load the test data
    testset = datasets.CIFAR10('./cifar_data/', download=True, train=False)
    X_test = torch.from_numpy(testset.test_data).to(dtype=torch.float)/255
    y_test = torch.Tensor(testset.test_labels)
    mask   = y_test < number_of_classes
    X_test = X_test[mask]
    y_test = y_test[mask]
    X_test.resize_(len(X_test), *X_test[0].view(-1).shape)
    y_test.view(-1).long()

    if shuffle == True:
        shuffling = torch.randperm(len(y_test))
        X_test = X_test[shuffling].to(device)
        y_test = y_test[shuffling]

    n_train = len(y_train)
    n_test  = len(y_test)

    s_train = int(n_train/batch_size)   # Number of training batches

    K           = number_of_classes 
    X_trains    = torch.zeros((s_train, batch_size, *X_train[0].view(-1).shape), requires_grad=False).to(device)
    y_trains    = torch.zeros((s_train, K, batch_size), requires_grad=False, dtype=torch.int64).to(device)
    if qr_mode:
        Qs      = torch.zeros((s_train, *X_train[0].view(-1).shape, batch_size), requires_grad=False).to(device)
        Rs      = torch.zeros((s_train, batch_size, batch_size), requires_grad=False).to(device)
        print('🤖QR started')

    y_test_one_hot = torch.zeros((n_test, K))
    y_test_one_hot[np.arange(n_test), y_test.long()] = 1
    y_test_one_hot = y_test_one_hot.t()

    for i in range(s_train):
        X_trains[i] = X_train[batch_size*i:batch_size*(i+1), :]
        batch_lbls  = y_train[batch_size*i:batch_size*(i+1)]
        y_batch_one_hot = torch.zeros((batch_size, K))
        y_batch_one_hot[np.arange(batch_size), batch_lbls.long()] = 1
        y_trains[i] = y_batch_one_hot.t()
        if qr_mode:
            Qs[i], Rs[i] = torch.qr(X_trains[i].t())      

    if qr_mode:
        print('✅QR computed')
        return X_trains, y_trains, X_test, y_test_one_hot, Qs, Rs
    else:
        return X_trains, y_trains, X_test, y_test_one_hot

def softmax_numpy(X):
    return np.array([np.exp(x)/sum(np.exp(x)) for x in X.T]).T

class CrossEntropyLoss_one_hot(nn.CrossEntropyLoss):
    '''
    Slightly modified version of the original CrossEntropyLoss in order to
    handle one-hot encodings
    '''
    def forward(self, input, target):
        target = torch.squeeze(torch.max(target, 1, keepdim=True)[1])
        return F.cross_entropy(input, target, weight=self.weight,
                                ignore_index=self.ignore_index, reduction=self.reduction)

def full_problem_from_batches(Xs, ys):
    s_train, batch_size, p = Xs.shape
    s_train, K, batch_size = ys.shape
    X = torch.zeros(s_train*batch_size, p)
    y = torch.zeros(K, s_train*batch_size)
    for i_batch in range(s_train):
        X[batch_size*i_batch:batch_size*(i_batch+1), :] = Xs[i_batch]
        y[:, batch_size*i_batch:batch_size*(i_batch+1)] = ys[i_batch]
    return X, y

def model_init(model, parameters_tensor):
    new_model = copy.deepcopy(model)
    for parameter in new_model.parameters():
        parameter.data = parameters_tensor.clone().to(device)
        # We won't update bias during the training, since they are not affect the model predictions
        break
    return new_model

class LogisticRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(int(input_dim), int(output_dim))
    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        return F.softmax(self.linear(x), dim=1)

def load_model(X_test, y_test):
    '''
    Returns logistic regression model
    Which is just single linear layer with flattening at the beginning and softmax at the end
    '''
    input_dim = X_test[0].numel()
    K, n_test = y_test.shape

    model = LogisticRegression(input_dim, K)
    return model

def gradient_flow_euler_training(theta_0, X_trains, y_trains,  X_test, y_test, lr, model, final_error = 0.2, epochs_limit = 1000):
    X, y        = full_problem_from_batches(X_trains, y_trains)
    X, y, X_test, y_test = X.float().to(device), y.float().to(device), X_test.to(device), y_test.to(device)
    model = model.to(device)
    n_train, p  = X.shape
    K, n_test   = y_test.shape
    thetas      = []
    losses_train    = []
    errors_train    = []
    losses_test     = []
    errors_test     = []
    criterion       = CrossEntropyLoss_one_hot()
    theta_t         = theta_0
    model = model_init(model, theta_0.T)
    stop_word = False
    N_epochs = 0
    while not stop_word:  
        N_epochs += 1     
        model.zero_grad()
        # Forward pass
        y_pred = model(X)
        loss = criterion(y_pred, y.T)
        # Metrics
        model.eval()
        thetas.append(theta_t)
        losses_train.append(loss.data)
        pred_labels         = torch.max(y_pred, 1, keepdim=True)[1]
        true_labels         = torch.max(y.T, 1, keepdim=True)[1]
        train_acc           = true_labels.eq(pred_labels.data).sum().to(dtype=torch.float)/len(true_labels)
        errors_train.append(1 - train_acc) 
        y_pred_test         = model(X_test)
        loss_test           = criterion(y_pred_test, y_test.T)
        losses_test.append(loss_test.data)
        pred_labels_test    = torch.max(y_pred_test, 1, keepdim=True)[1]
        true_labels_test    = torch.max(y_test.T, 1, keepdim=True)[1]
        test_acc            = true_labels_test.eq(pred_labels_test.data).sum().to(dtype=torch.float)/len(true_labels_test)
        errors_test.append(1 - test_acc)
        sys.stdout.write('\r'+f'🤖 GD error {errors_test[-1]:.3f}/{final_error:.3f} on {N_epochs}-th iteration. Lr {lr}')
        if errors_test[-1] <= final_error or N_epochs >= epochs_limit:
            stop_word = True
            break
        # Backward pass 
        model.train()
        loss.backward()
        for parameter in model.parameters():
            parameter.data = parameter.data - lr*parameter.grad.data
            theta_t = np.array((parameter.data.T).cpu())
            break
            
    model.eval()
    thetas.append(theta_t)
    losses_train.append(loss.data)
    pred_labels         = torch.max(y_pred, 1, keepdim=True)[1]
    true_labels         = torch.max(y.T, 1, keepdim=True)[1]
    train_acc           = true_labels.eq(pred_labels.data).sum().to(dtype=torch.float)/len(true_labels)
    errors_train.append(1 - train_acc) 
    y_pred_test = model(X_test)
    loss_test   = criterion(y_pred_test, y_test.T)
    losses_test.append(loss_test.data)
    pred_labels_test    = torch.max(y_pred_test, 1, keepdim=True)[1]
    true_labels_test    = torch.max(y_test.T, 1, keepdim=True)[1]
    test_acc            = true_labels_test.eq(pred_labels_test.data).sum().to(dtype=torch.float)/len(true_labels_test)
    errors_test.append(1 - test_acc)
    
    print(f'\n🤖 GD finished with {N_epochs} iterations on lr {lr}')

    return N_epochs, thetas, losses_train,losses_test, errors_train, errors_test

def sgd_training(theta_0, X_trains, y_trains,  X_test, y_test, lr, model, final_error = 0.2, iter_limit = 1000):
    X, y        = full_problem_from_batches(X_trains, y_trains)
    X, y, X_test, y_test = X.float().to(device), y.float().to(device), X_test.to(device), y_test.to(device)
    model = model.to(device)
    s_train, batch_size, p = X_trains.shape
    n_train, p  = X.shape
    K, n_test   = y_test.shape
    thetas      = []
    losses_train    = []
    errors_train    = []
    losses_test     = []
    errors_test     = []
    criterion       = CrossEntropyLoss_one_hot()
    theta_t         = theta_0
    model = model_init(model, theta_0.t())
    stop_word = False
    N_iter = 0
    if lr >= 1:
        iter_limit = 30000
    if lr >= 10:
        iter_limit = 1000
    while not stop_word:          
        i_batch = N_iter % s_train

        if i_batch % 1 == 0:
            # Evaluation pass
            model.eval()
            y_pred = model(X)
            loss = criterion(y_pred, y.t())
            thetas.append(theta_t)
            losses_train.append(loss.data)
            pred_labels     = torch.max(y_pred, 1, keepdim=True)[1]
            true_labels     = torch.max(y.t(), 1, keepdim=True)[1]
            train_acc       = true_labels.eq(pred_labels.data).sum().to(dtype=torch.float)/len(true_labels)
            errors_train.append(1 - train_acc) 
            y_pred_test = model(X_test)
            loss_test   = criterion(y_pred_test, y_test.t())
            losses_test.append(loss_test.data)
            pred_labels_test    = torch.max(y_pred_test, 1, keepdim=True)[1]
            true_labels_test    = torch.max(y_test.t(), 1, keepdim=True)[1]
            test_acc            = true_labels_test.eq(pred_labels_test.data).sum().to(dtype=torch.float)/len(true_labels_test)
            errors_test.append(1 - test_acc)
            sys.stdout.write('\r'+f'🤖 SGD error {errors_test[-1]:.3f}/{final_error:.3f} on {N_iter}-th iteration. Lr {lr}')
            if errors_test[-1] <= final_error:
                stop_word = True
                break

            if N_iter >= iter_limit:
                N_iter = None
                print(f'\n🤖 SGD Failed on lr {lr}')
                return N_iter, thetas, losses_train,losses_test, errors_train, errors_test

        # Backward pass
        model.train()
        model.zero_grad()
        # Forward pass
        y_pred = model(X_trains[i_batch])
        loss = criterion(y_pred, y_trains[i_batch].t())
        loss.backward()
        for parameter in model.parameters():
            parameter.data = parameter.data - lr*parameter.grad.data
            theta_t = np.array((parameter.data.t()).cpu())
            break
        N_iter += 1

    
    print(f'\n🤖 SGD finished with {N_iter} iterations on lr {lr}')

    return N_iter, thetas, losses_train,losses_test, errors_train, errors_test

def make_splitting_step(theta_0, Q, R, y, h, n_train):
    p, K = theta_0.shape
    p, batch_size = Q.shape
    Q, R, y = np.array(Q), np.array(R), np.array(y)
    h_seq = [0, h]
    theta_0 = np.array(theta_0)   
    H_0 = np.array(Q.T @ theta_0)
    H_0_vec = H_0.ravel('F')
    H_h = np.zeros((batch_size, K))

    def rhs_vec(H, t):
        H = H.reshape((batch_size, K), order='F')
        rhs = -1/n_train * R@(softmax_numpy(H.T@R) - y).T
        return rhs.ravel('F')
        
    H_h_vec = odeint(rhs_vec, H_0_vec, h_seq)[-1]
    H_h = H_h_vec.reshape((batch_size, K), order='F')

    theta = Q@(H_h - H_0) + theta_0
    return torch.from_numpy(theta)

def spl_training(theta_0, Qs, Rs, X_trains, y_trains,  X_test, y_test, stepsize, model, final_error = 0.2, iter_limit = 1000):
    X, y        = full_problem_from_batches(X_trains, y_trains)
    X, y, X_trains, y_trains, X_test, y_test, model = X.float().to(device), y.float().to(device), X_trains.float().to(device), y_trains.float().to(device), X_test.float().to(device), y_test.float().to(device), model.to(device)
    s_train, batch_size, p = X_trains.shape
    n_train, p  = X.shape
    K, n_test   = y_test.shape
    thetas      = []
    losses_train    = []
    errors_train    = []
    losses_test     = []
    errors_test     = []
    criterion       = CrossEntropyLoss_one_hot()
    theta_t         = theta_0.to(device)
    model = model_init(model, theta_0.t())
    stop_word = False
    N_iter = 0
    if stepsize >= 1000:
        iter_limit = 1000
    while not stop_word:
        i_batch = N_iter % s_train

        if i_batch % 1 == 0:      
            # Evaluation pass
            model.eval()
            y_pred = model(X)
            loss = criterion(y_pred, y.t())
            thetas.append(theta_t)
            losses_train.append(loss.data)
            pred_labels         = torch.max(y_pred, 1, keepdim=True)[1]
            true_labels         = torch.max(y.t(), 1, keepdim=True)[1]
            train_acc           = true_labels.eq(pred_labels.data).sum().to(dtype=torch.float)/len(true_labels)
            errors_train.append(1 - train_acc) 
            y_pred_test         = model(X_test)
            loss_test           = criterion(y_pred_test, y_test.t())
            losses_test.append(loss_test.data)
            pred_labels_test    = torch.max(y_pred_test, 1, keepdim=True)[1]
            true_labels_test    = torch.max(y_test.t(), 1, keepdim=True)[1]
            test_acc            = true_labels_test.eq(pred_labels_test.data).sum().to(dtype=torch.float)/len(true_labels_test)
            errors_test.append(1 - test_acc)
            sys.stdout.write('\r'+f'🤖 Splitting error {errors_test[-1]:.3f}/{final_error:.3f} on {N_iter}-th iteration. Stepsize {stepsize}')
            if errors_test[-1] <= final_error:
                stop_word = True
                break

            if N_iter >= iter_limit:
                N_iter = None
                print(f'\n🤖 Splitting Failed on Stepsize {stepsize}')
                return N_iter, thetas, losses_train,losses_test, errors_train, errors_test


        # Backward pass
        model.train()
        theta_t = make_splitting_step(theta_t.cpu(), Qs[i_batch].cpu(), Rs[i_batch].cpu(), y_trains[i_batch].cpu(), stepsize, n_train).to(dtype=torch.float)
        model = model_init(model, theta_t.t())
        N_iter += 1  

    print(f'\n🤖 Splitting finished with {N_iter} iterations on Stepsize {stepsize}')

    return N_iter, thetas, losses_train,losses_test, errors_train, errors_test

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


X_trains, y_trains, X_test, y_test, Qs, Rs = load_batched_data(batch_size=batch_size, qr_mode = True, number_of_classes=number_of_classes)
s_train, batch_size, p = X_trains.shape # Yes, here we have bs the same as input parameter in the previous line.
n_train, K, n_test = s_train*batch_size, *y_test.shape

print('🐱 Data loaded')

model = load_model(X_test, y_test)
print('🐱 Model loaded')

for i_exp in progress_bar(range(N_EXPERIMENTS)):
    print(f'============ ☄ {i_exp+1}/ {N_EXPERIMENTS} ☄ ============')
    # Random initialization
    init_bound = 1.0/math.sqrt(p)
    theta_0 = init_bound*torch.FloatTensor(p, K).uniform_(-1, 1)
    
    # RUN
    for i_lr, learning_rate in enumerate(LEARNING_RATES):
        stepsize = learning_rate*n_train/batch_size
        print(f'======🌠 lr {learning_rate}, h {stepsize} 🌠======')
        
        # N_iter, thetas, losses_train,losses_test, errors_train, errors_test = \
        #     gradient_flow_euler_training(theta_0, X_trains, y_trains,  X_test, y_test, learning_rate, model, final_error = TARGET_ERROR)
        # GD_time[i_exp, i_lr] = N_iter

        start_time = time.time()
        N_iter, thetas, losses_train,losses_test, errors_train, errors_test = \
            spl_training(theta_0,  Qs, Rs, X_trains, y_trains,  X_test, y_test, stepsize, model, final_error = TARGET_ERROR, iter_limit=iter_limit)
        end_time = time.time()
        SPL_time[i_exp, i_lr] = end_time - start_time
        SPL_N_iter[i_exp, i_lr] = N_iter
        if N_iter == None:
            SPL_N_iter[i_exp, i_lr] = None

        start_time = time.time()
        N_iter, thetas, losses_train,losses_test, errors_train, errors_test = \
            sgd_training(theta_0, X_trains, y_trains,  X_test, y_test, learning_rate, model, final_error = TARGET_ERROR, iter_limit=iter_limit)
        end_time = time.time()
        SGD_time[i_exp, i_lr] = end_time - start_time
        SGD_N_iter[i_exp, i_lr] = N_iter

        if N_iter == None:
            SGD_time[i_exp, i_lr] = None

        np.savez(f'Softmax_fashion_mnist_iter_err{TARGET_ERROR}_raw.npz', SPL_N_iter=SPL_N_iter, SGD_N_iter=SGD_N_iter, LEARNING_RATES = LEARNING_RATES)
        np.savez(f'Softmax_fashion_mnist_time_err{TARGET_ERROR}_raw.npz', SPL_time=SPL_time, SGD_time=SGD_time, LEARNING_RATES = LEARNING_RATES)

        plot_convergence_from_lr_time(LEARNING_RATES, [SPL_time, SGD_time], ['Splitting','SGD'])
        plt.savefig(f'Softmax_fashion_mnist_time_err{TARGET_ERROR}.pdf')
        plot_convergence_from_lr(LEARNING_RATES, [SPL_N_iter, SGD_N_iter], ['Splitting','SGD'])
        plt.savefig(f'Softmax_fashion_mnist_iter_err{TARGET_ERROR}.pdf')