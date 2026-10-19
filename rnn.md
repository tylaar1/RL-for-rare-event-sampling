## Adding recurrence

The excursion problem we studied has the Markovian property, meaning that the state at time $t$ provides all the information needed for the policy to make its next decision, this is not a typical case as there could be for example some momentum term in the dynamics that the model could need to account for which isn't apparent without the history. More generally, self-interacting and partially observed dynamics are often seen in statistical physics problems and could be solved via the same reinforcement learning algorithms we already covered by adding recurrence into our neural networks. A simple modification to the excursion problem to create a partially observed Markov state is to remove our system's knowledge of the current time, $t$. This requires our neural network to build a representation of the time to allow it to pick the best action at every state. 

## Recurrence in neural nets

Traditional neural networks assume that all inputs and outputs are independent (or in the Markov case can be made independent provided enough information is provided). Recurrent Neural Networks (RNNs) instead work with sequential information, by utilising  hidden states they build up a representation of the prior sequence to inform decision making.

The simplest form of RNNs (known as vanilla RNNs) build up a single hidden state which tracks the sequence history. It takes the previous hidden state and the current ‘actual’ state as input and outputs logits which can be used to make predictions. The hidden state and logit are controlled by weighting matrices (U,W,V) which are learned over time, by gradient descent, with gradients computed via backpropagation. Below we provide the list of equations used at each timestep to calculate the hidden state and make a prediction, note that here $h$ refers to the hidden state, and the inputs (which could be considered analogous to the state in reinforcement learning) are denoted with $x$.

At each timestep $t$, the hidden state and logit are

$$
\begin{gather}
z_t = U x_t + W h_{t-1} + b, \tag{1} \\
h_t = \tanh(z_t), \tag{2} \\
o_t = V h_t + c, \tag{3} \\
\pi(a_t = 1 \mid x_{1:t}) = \sigma(o_t), \tag{4}
\end{gather}
$$

where $h_0 = 0$, $\sigma$ is the sigmoid function, and the same weights $(U, W, V, b, c)$ are reused at every timestep.


Because states not only impact their own returns but also impact later actions and by extension later returns we cannot simply treat each timestep individually. This should make sense to readers with a background in RL, where actions affect future states and must be accounted for accordingly, to account for this in RL we calculate the return to go, which is the (discounted) sum of all future rewards, to estimate the total value of a transition to another state. RNNs have the added challenge of passing on key information about the state, to do this we need Backpropagation through time (BPTT).

BPTT firstly requires the gradient of the loss with respect to the output. Here the exact form of the loss is not provided for the sake of universality, however we use the loss derived in {...}.

$$
\begin{gather}
\delta^{o}_t \equiv \frac{\partial \mathcal{L}}{\partial o_t}  \tag{7}
\end{gather}
$$

The hidden state affects both the current logit and all later hidden states, so the gradient with respect to $h_t$ has two contributions:

$$
\begin{gather}
\delta^{h}_t \equiv \frac{\partial \mathcal{L}}{\partial h_t} = V^\top \delta^{o}_t + W^\top \delta^{z}_{t+1}, \tag{8} \\
\delta^{z}_t \equiv \frac{\partial \mathcal{L}}{\partial z_t} = \left(1 - h_t^{\,2}\right) \odot \delta^{h}_t, \tag{9}
\end{gather}
$$

with $\delta^{z}_{T+1} = 0$. This is evaluated backwards from $t = T$ to $t = 1$. Because the weights are shared across timesteps, their gradients are sums over the whole episode:

$$
\begin{gather}
\frac{\partial \mathcal{L}}{\partial V} = \sum_{t=1}^{T} \delta^{o}_t \, h_t^\top, \tag{10} \\
\frac{\partial \mathcal{L}}{\partial c} = \sum_{t=1}^{T} \delta^{o}_t, \tag{11} \\
\frac{\partial \mathcal{L}}{\partial W} = \sum_{t=1}^{T} \delta^{z}_t \, h_{t-1}^\top, \tag{12} \\
\frac{\partial \mathcal{L}}{\partial U} = \sum_{t=1}^{T} \delta^{z}_t \, x_t^\top, \tag{13} \\
\frac{\partial \mathcal{L}}{\partial b} = \sum_{t=1}^{T} \delta^{z}_t. \tag{14}
\end{gather}
$$


<!--
\iffalse

The source of vanishing and exploding gradients is the Jacobian linking distant timesteps,

$$
\begin{gather}
\frac{\partial h_t}{\partial h_k} = \prod_{j=k+1}^{t} \mathrm{diag}\!\left(1 - h_j^{\,2}\right) W, \tag{15}
\end{gather}
$$

a product of $t-k$ matrices, which shrinks or grows exponentially with $t-k$.
\end{comment}
\fi
-->


While understanding BPTT is a useful tool and can be used to analytically check results, in practice most modern machine learning libraries use an automatic differentiation (autodiff) package (for example I used Enzyme.jl when implementing the RNNs discussed here). Autodiff works by creating a computational graph over the forwards pass to backpropagate through and while it can have some difficulties if non differentiable functions are used during the forwards pass, navigating these is considered preferable in many cases to implementing complex BPTT by hand.  

Vanilla RNNs struggle to cope with longer sequences due to vanishing/exploding gradients and for complex problems have largely been replaced by Long Short Term Memory (LSTM) cells and transformers. Transformers use attention rather than a hidden state, allowing them to trade off slower outputs at inference time for massively parallelisable training. LSTM on the other hand extends the hidden state concept from Vanilla RNNS, splitting memory into a short term and long term state and using a clever gating system to control information flow between them. 

While there are many variations on LSTMs (including GRUs, which are sometimes considered to be seperate from LSTM) the traditional LSTM takes in the current (input) state $x_t$ and the long $c_{t-1}$ and short term $h_{t-1}$ states. It passes the current and short term states through input (16), forget (17) and output (18) gates, as the purpose of these gates is to determine how much information passes through rather than to carry information themselves, a sigmoid function squashes the values in these gates to be between 0 and 1. The current and short term gate also combine together to make a candidate for addition to the long term memory (19). These functions (with the exception of the output) then combine (20) to generate a new long term memory. This long term memory is then combined with the output gate to create the new short term memory (21). The long term and short term memory are then carried over to the next timestep and a prediction is generated from the short term memory if required (22,23).


$$
\begin{gather}
i_t = \sigma\!\left(U_i x_t + W_i h_{t-1} + b_i\right) \tag{16} \\
f_t = \sigma\!\left(U_f x_t + W_f h_{t-1} + b_f\right) \tag{17} \\
q_t = \sigma\!\left(U_q x_t + W_q h_{t-1} + b_q\right) \tag{18} \\
g_t = \tanh\!\left(U_g x_t + W_g h_{t-1} + b_g\right) \tag{19} \\
c_t = f_t \odot c_{t-1} + i_t \odot g_t. \tag{20} \\
h_t = q_t \odot \tanh(c_t), \tag{21} \\
o_t = V h_t + c, \tag{22} \\
\pi(a_t = 1 \mid x_{1:t}) = \sigma(o_t). \tag{23}
\end{gather}
$$

We can stack reccurent cells to learn more complex features in the same way that we would for a neural network. Intermediate cells each contain their own $h^{(i)}_t$ and $c^{(i)}_t$ but recieve $h^{(i-1)}_t$ as an input from the previous state rather than $x_t$. Each layer also has its own set of learned parameters which are shared across all timesteps. Only the final layer produces an output $o_t$.

## Results

### A Noisy KL estimate

Previously we were able to calculate the KL divergence for large T by making use of its markovian properties - because every action depended only on the current state and not the history we could re use probabilities calculated at previous states to perform computation in $\mathcal{O}(T^2)$. Because this recurrent method builds up independant hidden states each individual path must be realised to perform the full calculation, this comes at a cost of $\mathcal{O}(2^T)$ (which can be halved to $\mathcal{O}(2^{T-1})$ by using a depth first tree). 

This scaling makes accessing direct KL calculations for large T unrealistic, thankfully our reward formulation means that the returns provide a Monte Carlo estimate for this number.
- Use estimated KL from returns rather than direct calc - verified at low T (seperate graph for this inc error bars)

### Setup

- LSTM, RNN (if get results) params etc




### Actual results

- Graphs
- compare to other NN result (not rigarous)

### Discussion/conclusion

- apple paper
- try on problems inacessible to neural networks



