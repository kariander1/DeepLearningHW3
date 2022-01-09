r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers

part1_q1 = r"""
**Your answer:**
1. A. A fully-connected layer can be depicted as follows:

$Y=XW$

Since $batch\_size=64$, $in\_features=1024$ and $out\_features=512$, we would expect the following dimensions:

$Y_{64\times512}=X_{64\times1024}W_{1024\times512}$

Ans so the following Jacobian would result in:

$dim\left\{ \frac{\partial Y_{64\times512}}{\partial X_{64\times1024}}\right\} =64\times512\times64\times1024$

B. Let us break down the partial derivatives. Each element in Y is defined as a matrix multiplication, so:

$Y_{ij}=\sum_{k=1}^{1024}X_{ik}W_{kj}$

Therefore:

$\frac{\partial Y_{ij}}{\partial X}=\begin{pmatrix}\frac{\partial Y_{ij}}{\partial X_{11}}\\
\\
\frac{\partial Y_{ij}}{\partial X_{i1}} &  & \frac{\partial Y_{ij}}{\partial X_{ik}}\\
\\
\frac{\partial Y_{ij}}{\partial X_{64,1}} &  &  &  & \frac{\partial Y_{ij}}{\partial X_{64,1024}}
\end{pmatrix},\frac{\partial Y_{ij}}{\partial X_{mn}}\neq0\mid m=i$

$\frac{\partial Y_{ij}}{\partial X}=\begin{pmatrix}- & - & \overline{0} & - & -\\
- & - & \overline{0} & - & -\\
W_{i1} & \dots & W_{ik} & \dots & W_{i,1024}\\
- & - & \overline{0} & - & -\\
- & - & \overline{0} & - & -
\end{pmatrix}$

We produced a sparse matrix where only one row may be non-zero, and that is for each i,j, there fore we get many sparse matrix in 4D.

C. As derieved in class, we can express the partial derievatives as a sum of the following (as $\frac{\partial Y_{ij}}{\partial X}$ are sparse) 

$\boxed{\delta X=\frac{\partial L}{\partial X}=\frac{\partial Y}{\partial X}\frac{\partial L}{\partial Y}=\frac{\partial Y}{\partial X}\delta Y=\sum_{i=1}^{64}\frac{\partial Y_{i}}{\partial X}\left(\delta Y_{i}\right)=\left(\delta Y\right)\left(W^{T}\right)}$

2. A. In the same fashion from 1:

$dim\left\{ \frac{\partial Y_{64\times512}}{\partial W_{1024\times512}}\right\} =64\times512\times1024\times512$

B. in the same way in 1 we get a sparse matrix where a column may be non-zero for each i,j:

$\frac{\partial Y_{ij}}{\partial W}=\begin{pmatrix}| & | & X_{1j} & | & |\\
| & | & \vdots & | & |\\
\overline{0} & \overline{0} & X_{kj} & \overline{0} & \overline{0}\\
| & | & \vdots & | & |\\
| & | & X_{64,j} & | & |
\end{pmatrix}$

C. The same can be applied here as in previous question:

$\boxed{\delta W=\frac{\partial L}{\partial W}=\frac{\partial L}{\partial Y}\frac{\partial Y}{\partial X}=\delta Y\frac{\partial Y}{\partial X}=\sum_{i=1}^{64}\left(\delta Y_{i}\right)\frac{\partial Y_{i}}{\partial X}=\left(X\right)^{T}\left(\delta Y\right)}$

"""

part1_q2 = r"""
**Your answer:**
The backpropagation isn't required for training with descent based optimizers since it is only a method of improving performance
and calculating efficiently the partial gradients of the network by utilizing (his-majesty) the chain-rule.
Without backpropagation we can calculate the numerical gradient of the whole network which is a lot slower, but an alternative.

"""
# ==============
# Part 2 answers


def part2_overfit_hp():
    wstd, lr, reg = 0, 0, 0
    # TODO: Tweak the hyperparameters until you overfit the small dataset.
    # ====== YOUR CODE: ======
    wstd, lr, reg = 0.09, 0.05, 0.03
    # ========================
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_optim_hp():
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = 0, 0, 0, 0, 0

    # TODO: Tweak the hyperparameters to get the best results you can.
    # You may want to use different learning rates for each optimizer.
    # ====== YOUR CODE: ======
    wstd, lr_momentum, lr_vanilla, lr_rmsprop, reg = 1e-1, 3e-3, 1.5e-2, 1.75e-4, 1e-2
    # ========================
    return dict(
        wstd=wstd,
        lr_vanilla=lr_vanilla,
        lr_momentum=lr_momentum,
        lr_rmsprop=lr_rmsprop,
        reg=reg,
    )


def part2_dropout_hp():
    wstd, lr, = 0, 0
    # TODO: Tweak the hyperparameters to get the model to overfit without
    # dropout.
    # ====== YOUR CODE: ======
    wstd, lr, = 0.1, 0.001
    # ========================
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""
**Your answer:**

(1+2). The graphs seem to fit what we expected to see. We got the following results:

$\begin{array}{ccccc}
\text{Dropout} & \text{Train Loss} & \text{Train Accuracy} & \text{Test Loss} & \text{Test Accuracy}\\
0 & 0.399 & 94\% & 2.901 & 22.5\%\\
0.4 & 0.914 & 69.2\% & 2.143 & 29.2\%\\
0.8 & 1.876 & 31.2\% & 2.115 & 25.5\%
\end{array}$

*Dropout=0* With a dropout of zero, we expect to get the classic MLP model we constructed earlier in this assignment 
which overfitted the train data. It can be seen that the train loss and accuracy were the highest while using this configuraiton,
and the test_loss at some point around iteration #15 started to get even higher, which indicates the overfitting.


*Dropout=0.4* In this configuration we received the best results test-wise.
Dropout makes the training noisier making nodes take more responsibility for the inputs with bigger probability or vice-versa.
This may lead to less situations where nodes adapt together to mitigate mistakes from previous layers, and thus make the model more robust.
 
*Dropout=0.8* In this case we drop-out to many nodes while training and underfit the data as input which reflect in the low
training loss.

"""

part2_q2 = r"""
**Your answer:**
Yes it is possible with the cross-entropy function.
As the cross-entropy penalizes when correct classes differ from 1 and the incorrect classes to 0, we might get a situation
where we actually were able to predict a label with a probability just above 0.5, however the incorrect classes were also not
so so far from 0. 

E.G:

2 Labels, and 2 inputs, both are labeled 0.

$Y_{prob}^{1}=\begin{bmatrix}1 & 0\\
0.49 & 0.51
\end{bmatrix}\to\text{X1 is predicted label 0, and X2 predicted 1, 50% acc}$

$Y_{prob}^{2}=\begin{bmatrix}0.51 & 0.49\\
0.51 & 0.49
\end{bmatrix}\to\text{X1 is predicted label 0, and X2 predicted 0, 100% acc but loss increases}$

"""

part2_q3 = r"""
**Your answer:**
1. The gradient descent is a method of how to solve an optimization problem utilizing gradients of the target function.
The backpropagation is a method of easily calculating the gradients of the model with the chain-rule, and is not a optimization
solving tool by itself.

2. Stochastic gradient descent is an approximation of the gradient from a single data point rather than calculating
 the gradient of all data points like in GD. In both of SGD and GD we update a set of parameters in an iterative manner.
 In GD we would run through all of our samples before even updating a single parameter while in SGD we would use only one sample
 (or a subset of samples if using mini-batches) to update a parameter right away. 
 This results in different intermediate weight updates across iterations.  
 We would expect from SGD to converge faster than GD, but probably increase the approximation error relative to SGD.
 
3. We would expect from SGD to avert some of the saddle-points and local-minimas that are dominant in GD as that changing the samples we always 
 optimize a different error-surface, thus those problematic points (saddle, local minimas etc.) will be different.
Moreover, SGD as explained above would run faster and thus is feasible when datasets get larger.

4. A. This approach would not produce the same results as GD since calculating loss on mini-batches and summing them is not
equivalent to calculating the loss on the whole dataset.
This is due the fact that our model uses non-linearities on our input batches (activations such as RELU, tanh, sigmoid)
and so summing two different input won't result in the sum of the two outputs in the general case.

Defining $\varphi$ to be a model or a non-linearity:

$\varphi(X_{1}+X_{2})\neq\varphi(X_{1})+\varphi(X_{2})$

B. This approach may save the memory required to load the inputs themselves into the memory, however the results of each
forward pass is still stored in memory so that the accumulation of the total loss would be possible (results such as the
batch loss and intermediate batches losses). Therefore at some point these results will take up all the memory available.
 
"""
# ==============

# ==============
# Part 3 answers


def part3_rnn_hyperparams():
    hypers = dict(
        batch_size=0,
        seq_len=0,
        h_dim=0,
        n_layers=0,
        dropout=0,
        learn_rate=0.0,
        lr_sched_factor=0.0,
        lr_sched_patience=0,
    )
    # TODO: Set the hyperparameters to train the model.
    # ====== YOUR CODE: ======
    hypers['batch_size'] = 256
    hypers['seq_len'] = 64
    hypers['h_dim'] = 750
    hypers['n_layers'] = 3
    hypers['dropout'] = 0.48
    hypers['learn_rate'] = 0.0011
    hypers['lr_sched_factor'] = 0.5
    hypers['lr_sched_patience'] = 2
    # ========================
    return hypers


def part3_generation_params():
    start_seq = ""
    temperature = 0.0001
    # TODO: Tweak the parameters to generate a literary masterpiece.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return start_seq, temperature


part3_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q4 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""
# ==============
