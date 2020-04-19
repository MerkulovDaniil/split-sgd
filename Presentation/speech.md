Hello, my name is Daniil Merkulov and this is short presentation of our work.

========= 1 =========

A lot of practical problems arising in machine learning require minimization of a finite sample
average which can be written in the following form

Gradient Descent algorithm is a great tool to analyze behavior of this optimization problem in a discrete time. 

One of the well known stochastic version of this algorithm implies switch from the full gradient to its mini-batch sampling over random b summands

Gradient descent method can be considered as an Euler discretization of the ordinary differential equation (ODE) of the form of the gradient flow

In continuous time, SGD if often analyzed by introducing noise into the right-hand side. However, for a real dataset, the distribution of the noise obtained by replacing the full gradient by its minibatch variant is not known and can be different for different problems. 

Instead, we propose a new view on the SGD as a first-order splitting scheme, thus shedding a new light on SGD-type algorithms. 

========= 2 =========

To illustrate the idea of ODE splitting we will consider simplest example of the ODE problem with right-hand side, which contains only 2 summands

Here we split right-hand side into primitive summands and solve each problem separately with reinitialization. 

This approach could be very useful in a lot of practical problems (for example - splitting into linear and non-linear parts, convection + diffusion parts and etc)

Thus, the first-order approximation could be written as a combination of all local solutions

========= 3 =========

It is interesting to look at how the pure splitting scheme corresponds to the SGD approach. For this purpose, we consider an illustrative example of Gradient Flow equation with 2 summands (say, its SGD, running only on 2 data points with unit batch size)

The table describes the correspondence between splitting scheme for discretized Gradient Flow ODE and single epoch of SGD

Thus, we can conclude, that one epoch of SGD is just the splitting scheme for the discretized Gradient Flow ODE with 2 · h step size (m · h in case of m batches) in terms of continuous time approximation. Given information about the Euler scheme limitation (first-order accuracy, stability issues), we propose to solve each local problem more precisely.

========= 4 =========

Consider finite-sum problem

Formulate corresponding ODE

Apply first order splitting and solve local problems more precisely, than Euler.

Compare with SGD approximation in the same continuous time, which involves setting m times larger stepsize of splitting scheme, where m is the total number of minibatches.

We have tested three types of problems, where we can explicitly write batch gradients and corresponding local ODE problems. 

========= 5 =========

It is very important, that we can reduce dimensionality of the dynamic system via QR decomposition of each batch data matrix and following substitution.

While the original dynamics goes through the theta vector of dimension p, which could be really bigger, than the batch size (especially in large scale optimization applications)

It is crucial to consider dynamics of vector eta which is b-dimensional.

Here is the list of local ODE for chosen problems. For the linear least squares we can write down effective analytical solution, while other non-linear ODEs will be solved using Runge-Kutta solvers from scipy's odeint for Python
========= 6 =========

Here is the total algorithm of splitting optimization. Firstly, we go to the epoch cycle for k. Than, we iterate over the mini batches and solve local initial value problems for all the small ODEs using solvers, which was mentioned above.

========= 7 =========

Let us briefly describe experimental setup. Random linear system with gaussian noise of the sizes 10000 times 500, batch size 20.
Lack of point of one algorithm on the
graph means reaching the limit of iterations without achieving the termination rule.  

Real linear system based on tomography data with batch size 60. Relative error 10^(-3) was used as a stopping criterion.

binary logistic regression on the MNIST 0,1 dataset.  The size of the batch for presented figures is 50. Test error 0.1% was used as the stopping criterion.

Softmax regression was tested on the FashionMNIST dataset with batch size 64 and test error 25% as a stopping criterion

========= 8 =========

As it is expected, SGD diverges starting from some value of learning rate, which is specific for each
problem. While we can see comparative robustness of the proposed splitting optimization approach

As we can see, more precise integration of local problems provide us with a robustness to the stepsize choosing. For the softmax problem we can even see better performance in time consumption.