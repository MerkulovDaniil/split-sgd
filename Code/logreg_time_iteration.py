# ======= Function definitions =======

def sigmoid(x):
    '''
    Calculates element-wise sigmoid function
    Parameters
    ----------
    x : array-like of floats
        Input vector (scalar)
    Returns
    -------
    sigma(x) : array-like of floats
        1/(1 + exp(-x_i)) for each x_i in x
    '''
    if np.isscalar(x):
        return 1/(1 + np.exp(-x))
    else:
        return np.array([1/(1 + np.exp(-x_i)) for x_i in x])

def make_splitting_step(Q, R, theta_0, y, h, n):
    h_seq = [0, h]
    eta_0 = Q.T@theta_0
    def rhs(eta, t):
        return -1/n * R@(sigmoid(R.T @ eta) - np.array(y))
    eta_h = odeint(rhs, eta_0, h_seq)[-1]

    theta = Q@(eta_h - eta_0) + theta_0
    return theta

def load_batched_data(batch_size=50, shuffle = True, qr_mode = False, number_of_classes = 2):
    '''
    Load batches of MNIST data.

    Output: X_trains - m_train batches of training data, 
            y_trains - m_train batches of labels,
            X_test - test points
            y_test - test labels
    X_trains: torch.array of shape (m_train,batch_size,*X_train[0].shape),
        where 
        m_train - the number of batches, 
        batch_size - batch size
        *X_train[0].shape - shape of the dataset point;

    y_trains: torch.array of shape (m_train, K, batch_size),
        where
        K - the number of classes in the problem;

    X_test: torch.array of shape (n_test,*X_train[0].shape),
        where
        n_test - the number of test points;

    y_test: torch.array of shape (K, n_test);
    '''
    trainset = datasets.MNIST('./mnist_data/', download=True, train=True)
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
    testset = datasets.MNIST('./mnist_data/', download=True, train=False)
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

    m_train = int(n_train/batch_size)   # Number of training batches

    K           = number_of_classes 
    X_trains    = torch.zeros((m_train, batch_size, *X_train[0].view(-1).shape), requires_grad=False).to(device)
    y_trains    = torch.zeros((m_train, batch_size), requires_grad=False).to(device)
    if qr_mode:
        Qs      = torch.zeros((m_train, *X_train[0].view(-1).shape, batch_size), requires_grad=False).to(device)
        Rs      = torch.zeros((m_train, batch_size, batch_size), requires_grad=False).to(device)
        print('🤖QR started')

    for i in range(m_train):
        X_trains[i] = X_train[batch_size*i:batch_size*(i+1), :]
        y_trains[i] = y_train[batch_size*i:batch_size*(i+1)]
        if qr_mode:
            Qs[i], Rs[i] = torch.qr(X_trains[i].t())      
    
    if qr_mode:
        print('✅QR computed')
        return X_trains, y_trains, X_test, y_test, Qs, Rs
    else:
        return X_trains, y_trains, X_test, y_test

class LogisticRegression(torch.nn.Module):
     def __init__(self):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(p, 1)
     def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred

def full_problem_from_batches(Xs, ys):
    m_train, batch_size, p = Xs.shape
    X = torch.zeros(m_train*batch_size, p)
    y = torch.zeros(m_train*batch_size)
    for i_batch in range(m_train):
        X[batch_size*i_batch:batch_size*(i_batch+1), :] = Xs[i_batch]
        y[batch_size*i_batch:batch_size*(i_batch+1)]    = ys[i_batch]
    return X, y

def model_init(model, parameters_tensor):
    new_model = copy.deepcopy(model)
    for parameter in new_model.parameters():
        parameter.data = parameters_tensor.clone().to(device)
        # We won't update bias during the training, since they are not affect the model predictions
        break
    return new_model

def gradient_flow_euler_training(theta_0, X_trains, y_trains,  X_test, y_test, lr, model, total_time):
    X, y        = full_problem_from_batches(X_trains, y_trains)
    X, y, X_test, y_test = X.float().to(device), y.float().to(device), X_test.to(device), y_test.to(device)
    model = model.to(device)
    n_train, p  = X.shape
    n_test      = len(y_test)
    thetas      = []
    losses_train    = []
    errors_train    = []
    losses_test     = []
    errors_test     = []
    criterion       = torch.nn.BCELoss()
    theta_t         = theta_0
    model = model_init(model, theta_0.T)
    N_steps = int(total_time/lr)
    for i_step in range(N_steps):    
        model.zero_grad()
        # Forward pass
        y_pred = model(X)
        loss = criterion(y_pred, y)
        # Metrics
        model.eval()
        thetas.append(theta_t)
        losses_train.append(loss.data)
        pred_labels         = torch.max(y_pred)[1]
        true_labels         = torch.max(y)[1]
        train_acc           = true_labels.eq(pred_labels.data).sum().to(dtype=torch.float)/len(true_labels)
        errors_train.append(1 - train_acc) 
        y_pred_test         = model(X_test)
        loss_test           = criterion(y_pred_test, y_test)
        losses_test.append(loss_test.data)
        pred_labels_test    = torch.max(y_pred_test)[1]
        true_labels_test    = torch.max(y_test)[1]
        test_acc            = true_labels_test.eq(pred_labels_test.data).sum().to(dtype=torch.float)/len(true_labels_test)
        errors_test.append(1 - test_acc)
        sys.stdout.write('\r'+f'🤖 GD error {errors_test[-1]:.3f}/{final_error:.3f} on {i_step}-th iteration. Lr {lr}')
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
    pred_labels         = torch.max(y_pred)[1]
    true_labels         = torch.max(y)[1]
    train_acc           = true_labels.eq(pred_labels.data).sum().to(dtype=torch.float)/len(true_labels)
    errors_train.append(1 - train_acc) 
    y_pred_test = model(X_test)
    loss_test   = criterion(y_pred_test, y_test)
    losses_test.append(loss_test.data)
    pred_labels_test    = torch.max(y_pred_test)[1]
    true_labels_test    = torch.max(y_test)[1]
    test_acc            = true_labels_test.eq(pred_labels_test.data).sum().to(dtype=torch.float)/len(true_labels_test)
    errors_test.append(1 - test_acc)
    
    print(f'\n🤖 GD finished with {N_steps} iterations on lr {lr}')

    return thetas, losses_train, losses_test, errors_train, errors_test

def sgd_training(theta_0, X_trains, y_trains,  X_test, y_test, lr, model, N_epochs):
    X, y        = full_problem_from_batches(X_trains, y_trains)
    X, y, X_test, y_test = X.float().to(device), y.float().to(device), X_test.to(device), y_test.to(device)
    model = model.to(device)
    m_train, batch_size, p = X_trains.shape
    n_train, p  = X.shape
    n_test   = len(y_test)
    thetas      = []
    losses_train    = []
    errors_train    = []
    losses_test     = []
    errors_test     = []
    criterion       = torch.nn.BCELoss()
    theta_t         = theta_0
    model = model_init(model, theta_0.t())
    for i_epoch in range(N_epochs):          
        i_batch = i_epoch % m_train

        if i_batch % 1 == 0:
            # Evaluation pass
            model.eval()
            y_pred = model(X)
            loss = criterion(y_pred, y)
            thetas.append(theta_t)
            losses_train.append(loss.data)
            pred_labels     = torch.squeeze(y_pred >= 0.5).float()
            train_acc       = y.eq(pred_labels.data).sum().to(dtype=torch.float)/len(pred_labels)
            errors_train.append(1 - train_acc) 
            y_pred_test = model(X_test)
            loss_test   = criterion(y_pred_test, y_test.float())
            losses_test.append(loss_test.data)
            pred_labels_test    = torch.squeeze(y_pred_test >= 0.5).long()
            test_acc            = y_test.eq(pred_labels_test.data).sum().to(dtype=torch.float)/len(y_pred_test)
            errors_test.append(1 - test_acc)
            sys.stdout.write('\r'+f'🤖 SGD error {errors_test[-1]:.3f}/{final_error:.3f} on {N_iter}-th iteration. Lr {lr}')

        # Backward pass
        model.train()
        model.zero_grad()
        # Forward pass
        y_pred = model(X_trains[i_batch])
        loss = criterion(y_pred, y_trains[i_batch])
        loss.backward()
        for parameter in model.parameters():
            parameter.data = parameter.data - lr*parameter.grad.data
            theta_t = np.array((parameter.data.t()).cpu())
            break
    
    print(f'\n🤖 SGD finished with {N_epochs} iterations on lr {lr}')

    return thetas, losses_train,losses_test, errors_train, errors_test

def make_splitting_step(theta_0, Q, R, y, h, n):
    h_seq = [0, h]
    Q, R, theta_0 = np.array(Q), np.array(R), np.array(theta_0)
    eta_0, theta_0 = np.squeeze(Q.T@theta_0), np.squeeze(theta_0)
    def rhs(eta, t):
        return -1/n * R@(sigmoid(R.T @ eta) - np.array(y))
    eta_h = odeint(rhs, eta_0, h_seq)[-1]

    theta = Q@(eta_h - eta_0) + theta_0
    return torch.from_numpy(theta).reshape(p, 1)

def spl_training(theta_0, Qs, Rs, X_trains, y_trains,  X_test, y_test, stepsize, N_spl_steps, model):
    X, y        = full_problem_from_batches(X_trains, y_trains)
    X, y, X_trains, y_trains, X_test, y_test, model = X.float().to(device), y.float().to(device), X_trains.float().to(device), y_trains.float().to(device), X_test.float().to(device), y_test.float().to(device), model.to(device)
    m_train, batch_size, p = X_trains.shape
    n_train, p  = X.shape
    n_test      = len(y_test)
    thetas      = []
    losses_train    = []
    errors_train    = []
    losses_test     = []
    errors_test     = []
    criterion       = torch.nn.BCELoss()
    theta_t         = theta_0.to(device)
    model = model_init(model, theta_0.t())
    for i_step in range(N_spl_steps):         
        i_batch = i_step % m_train

        if i_batch % 1 == 0:      
            # Evaluation pass
            model.eval()
            y_pred = model(X)
            loss = criterion(y_pred, y)
            thetas.append(theta_t)
            losses_train.append(loss.data)
            pred_labels     = torch.squeeze(y_pred >= 0.5).float()
            train_acc       = y.eq(pred_labels.data).sum().to(dtype=torch.float)/len(pred_labels)
            errors_train.append(1 - train_acc) 
            y_pred_test = model(X_test)
            loss_test   = criterion(y_pred_test, y_test)
            losses_test.append(loss_test.data)
            pred_labels_test    = torch.squeeze(y_pred_test >= 0.5).float()
            test_acc            = y_test.eq(pred_labels_test.data).sum().to(dtype=torch.float)/len(y_pred_test)
            errors_test.append(1 - test_acc)
            sys.stdout.write('\r'+f'🤖 Splitting error {errors_test[-1]:.3f}/{final_error:.3f} on {i_step}-th iteration. Stepsize {stepsize}')

        # Backward pass
        model.train()
        theta_t = make_splitting_step(theta_t.cpu(), Qs[i_batch].cpu(), Rs[i_batch].cpu(), y_trains[i_batch].cpu(), stepsize, n_train).to(dtype=torch.float)
        model = model_init(model, theta_t.t())
        N_iter += 1  

    print(f'\n🤖 Splitting finished with {N_spl_steps} iterations on Stepsize {stepsize}')

    return thetas, losses_train, losses_test, errors_train, errors_test

def plot_continuous_time_logreg(times, losses_trains, losses_tests, errors_trains, errors_tests, labels, N_epochs, title = 'LogReg. MNIST 0,1'):
    colors = ['r', 'g', 'b']
    color_labels = ['^', 'o', '-']
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(2, 2, figsize = (7,7))
    fig.suptitle(title)
    for time, losses_train, losses_test, errors_train, errors_test, label, col, col_lab in zip(times, losses_trains, losses_tests, errors_trains, errors_tests, labels, colors, color_labels):
        ax1.semilogy(time, losses_train, col+col_lab, label = label)
        ax1.semilogy(time, losses_train, col+':')
        ax2.semilogy(time, errors_train, col+col_lab, label = label)
        ax2.semilogy(time, errors_train, col+':')
        ax3.semilogy(time, losses_test,  col+col_lab, label = label)
        ax3.semilogy(time, losses_test,  col+':')
        ax4.semilogy(time, errors_test,  col+col_lab, label = label)
        ax4.semilogy(time, errors_test,  col+':')
    ax1.grid(True,which="both", linestyle='--', linewidth=0.4)
    ax1.set_title('Train loss')
    ax1.set_xlabel('t')
    ax2.grid(True,which="both", linestyle='--', linewidth=0.4)
    ax2.set_title('Train error')
    ax2.set_xlabel('t')
    ax3.grid(True,which="both", linestyle='--', linewidth=0.4)
    ax3.set_title('Test loss')
    ax3.set_xlabel('t')
    ax4.grid(True,which="both", linestyle='--', linewidth=0.4)
    ax4.set_title('Test error')
    ax4.set_xlabel('t')
    plt.legend()
    fig.tight_layout()
    # plt.savefig(title + '.pdf')
    plt.show()