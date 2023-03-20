# Modeling/Forecasting the Probability of Recession w/ a Regime-switching model + Bayesian Neural Net

## Goal
- Construct a "draft" model for modeling and forecasting recession w/ a regime-switching model that incorporates a bayesian neural net in the transition probability.
- An application of the Bayesian Endogenous Markov Regime-switching Multivariate Regression in [Kim and Kang (2022)](https://academic.oup.com/jfec/article/20/3/391/5909218).

## Model

The endogenous (Markov) regime-switching multivariate regression model posits, for $t \in \mathbb{Z}$, the DGP of the observable random vectors $(\textbf{y}_t, \textbf{x}_t)$ as
```math
\underset{j \times 1}{\textbf{y}_t} = \underset{j \times 1}{\textbf{m}(s_t)} + \underset{j \times k}{\textbf{B}(s_t)} ~ \underset{k \times 1}{\textbf{x}_t} + \underset{j \times j}{\textbf{L}(s_t)} ~ \underset{j \times 1}{\textbf{e}_t};
\qquad
\textbf{L}(s_t):\text{lower triangular w/ positive diagonals},
\quad
s_t \in S:=\{1,...,N\}.
```

This is endogenous in the sense that we additionally suppose
```math
\underset{(N-1 + j) \times 1}{\begin{bmatrix} \textbf{z}_t \\ \textbf{e}_t \end{bmatrix}}
\stackrel{\text{iid}}{\sim}
\mathcal{N} \left(
\begin{bmatrix} \textbf{z} \\ \textbf{e} \end{bmatrix} \mid
\begin{bmatrix} f_s(O_{t-1}) \\ \textbf{0}_j \end{bmatrix},
\begin{bmatrix} \textbf{S}_z & \textbf{R} \\ \textbf{R}' & \textbf{I}_j \end{bmatrix}
\right),
\qquad
s_t = f_{\textbf{z}}(\textbf{z}_t).
```

In other words, we are considering a joint distribution, at $t$, of the measurement error $\textbf{e}\_t$ and some (continuous) underlying process $\textbf{z}\_t$ that governs the regime-switching dynamics via some map $f\_{\textbf{z}} : \mathbb{R}^{N-1} \rightarrow S$.

(Note the obvious that $\textbf{S}\_z \in \mathbb{R}^{(N-1) \times (N-1)}$ is positive definite while $\textbf{R} \in \mathbb{R}^{(N-1) \times j}$ is necessarily not.)

The unobserved quantitites of interest are the
- parameters $(\textbf{m}(s), \textbf{B}(s), \text{vech}(\textbf{L}(s))$ for $s \in S$ and the
- discrete latent process $(s_t)_{t\in\mathbb{Z}}$.

## An Extension

In [Kim and Kang (2022)](https://academic.oup.com/jfec/article/20/3/391/5909218), $f\_s(O_{t-1}) : S \rightarrow \mathbb{R}^{N-1}$ is set to some constant $\textbf{g}(s\_{t-1})$ that "switches" depending on the realization of the state variable at the previous time period.

Here, I consider $O\_{t-1}=\textbf{x}\_{t-1}$ (as a covariate at $t-1$) and $f\_s$ to be parameterized as a feedforward neural network: that is,
```math
f_s = f_1 \circ \dots \circ f_{L-1} \circ f_L  : \mathbb{R}^{N-1} \rightarrow \mathbb{R}^{N-1}
```
w/ appropriate intermediate dimensions. The activation function in $f_L$ maps to $\mathbb{R}^{N-1}$.

This setting equates two notably novel and appealing features of the model:
1. explicitly allowing for nonlinearities in the transition probability.
2. the state variable process $(s_t)\_{t \in \mathbb{Z}}$ is Markovian but in a different sense; we used to have
```math
\mathbb{P}(s_t = i \mid s_{t-1} = j, s_{t-2} = k, \textbf{x}_{t-1}) = \mathbb{P}(s_t = i \mid s_{t-1} = j) = p_{j \rightarrow i}
```
due to the specification $f\_s(O_{t-1}) = \textbf{g}(s\_{t-1})$, whereas now we have
```math
\mathbb{P}(s_t = i \mid s_{t-1} = j, s_{t-2} = k, \textbf{x}_{t-1}) = \mathbb{P}(s_t = i \mid \textbf{x}_{t-1}),
```
which additionally became a function of $\textbf{x}\_{t-1}$ (alterable), and the mapping rule is nonlinear (to be learnt from the observations). This is considerably more useful for the purpose for forecasting.

## In Play

The above model is estimated under the following simplistic settings.

- A bivaraite measurement (as VAR)
```math
\textbf{y}_t = \textbf{m}(s_t) + \textbf{B}(s_t) ~ \textbf{y}_{t-1} + \textbf{L}(s_t) ~ \textbf{e}_t;
\qquad
\textbf{y}_t = \begin{bmatrix} \text{\color{green}{INDUSTRIAL PRODUCTION INDEX}}_t \\ \text{\color{green}{UNEMPLOYMENT RATE}}_t \end{bmatrix}.
```

- "Deep" 2-state (Exogenous) Regime-switches
```math
\underset{(1 + 2) \times 1}{\begin{bmatrix} z_t \\ \textbf{e}_t \end{bmatrix}}
\stackrel{\text{iid}}{\sim}
\mathcal{N} \left(
\begin{bmatrix} z \\ \textbf{e} \end{bmatrix} \mid
\begin{bmatrix} \text{FFNN}(\textbf{x}_{t-1}) \\ \textbf{0}_j \end{bmatrix},
\begin{bmatrix} \textbf{S}_z & \textbf{0} \\ \textbf{0}' & \textbf{I}_j \end{bmatrix}
\right),
\qquad
s_t = \mathbb{I}\{z_t \geq 0\} + 1 \in S := \{1, 2\},
\qquad
\textbf{x}_t = \begin{bmatrix} \text{CPI}_t \\ \text{\color{green}{INDUSTRIAL PRODUCTION INDEX}}_t \\ \text{FED FUNDS RATE}_t \\ \text{FFR 30D FUTURES}_t \\ \text{\color{green}{UNEMPLOYMENT RATE}}_t \end{bmatrix}.
```

- Take $j \in S$. By construction,
```math
\mathbb{P}(s_{t} = 1 \mid s_{t-1} = j, \textbf{x}_{t-1}) = \mathbb{P}(z_t < 0 \mid \textbf{x}_{t-1}) = F_{\color{yellow}{z_t}}
```
