# A Sampling Approach to Time-varying Sharpe-optimal Portfolio Allocation w/ Bayesian Multivariate GARCH

## Goal
- Sampling from the "posterior" of optimal portfolios

## Data
- Daily returns of S&P500, N225, and FTSE100 from 2022-12-29 to 2023-03-09.

## Models
### #1: IID Multivariate Gaussian
- Consider the conventional "iid"-model which posits the DGP of returns as a purely i.i.d. process. That is,

```math
\textbf{y}_t = (y_{1,t}, ..., y_{k,t})' \stackrel{\text{iid}}{\sim} \mathcal{N}(\textbf{y} \mid \textbf{m}, \textbf{S}); \quad t\in\mathbb{Z}.
```

- Given some (long position) weight
```math
\textbf{w} \in W := \{ (w_1,...,w_k)' : [w_1,...,w_k \in \mathbb{R}_{\geq 0}] \wedge [\sum_{i=1}^k w_i = 1] \},
```
the portfolio (at $t$) is defined as $\textbf{w}'\textbf{y}_t$. It follows that the portfolio
  - expected return $\mathbb{E}_{\textbf{y}_t}[\textbf{w}'\textbf{y}_t] = \textbf{w}'\textbf{m}$;
  - variance $\mathbb{V}_{\textbf{y}_t} [\textbf{w}'\textbf{y}_t] = \textbf{w}' \mathbb{V}[\textbf{y}_t] \textbf{w} = \textbf{w}'\textbf{S}\textbf{w}$.

To obtain the Sharpe-optimal portfolio, one solves for the optimization problem
```math
\textbf{w}^* = \text{argmax}_{\textbf{w} \in W} ~ {\textbf{w}'\textbf{m} \over \textbf{w}'\textbf{S}\textbf{w}},
```
using the sample analogue of $(\textbf{m},\textbf{S})$. Here, a generative model is assumed with the following additional structure:
```math
\theta = (\textbf{m}, \textbf{S})' \sim p(\theta) \stackrel{\text{d}}{=} p(\textbf{m}) p(\textbf{S}),
```
where $\textbf{t}_0$ is a hyperparemeter and $p(\textbf{S})$ is a positive semi-definite matrix-valued prior distribution.

The goal is to inspect the posterior distribution of $\theta$ given $\textbf{Y}:=(\textbf{y}_1,...,\textbf{y}_T)$, that is
```math
p(\theta \mid \textbf{Y}) \propto p(\textbf{Y} \mid \theta) p(\theta),
```
as a means of inspecting the posterior distribution of the Sharpe-optimal portfolio, as follows.

![model1](https://user-images.githubusercontent.com/46773720/224541902-7f5205eb-9472-42f5-b1ee-415dadfa3378.png)

### 2: Multivariate GARCH (BEKK)
- Upon the previous generative model, the model of BEKK specifies the deterministic evolution of the conditional variance.
- The generative model is
```math
\begin{align}
\textbf{y}_t &= \textbf{m} + \underbrace{\textbf{H}_t^{1/2} \textbf{z}_t}_{=\textbf{e}_t}; \qquad
\textbf{z}_t \stackrel{\text{iid}}{\sim} \mathcal{N}(\textbf{z} \mid \textbf{0}_k, \textbf{I}_k), \\
\textbf{H}_t &= \textbf{C} + \textbf{A}\textbf{e}_{t-1}\textbf{e}_{t-1}'\textbf{A}' + \textbf{B}\textbf{H}_{t-1}\textbf{B}'.
\end{align}
```
  - $\textbf{C}$ is a positive definite matrix.
- With appropriate priors in hand, the time-varying sharpe-optimal portfolio is constructed from the MCMC output as follows.

![model2](https://user-images.githubusercontent.com/46773720/224542127-7bb2ae04-e0d0-4648-b91a-327385f09780.gif)

### 3: Multivariate GARCH (BEKK) w/ Time-varying Expected Returns

- Additionally posit that the expected returns are time-varying.
- The generative model is
```math
\begin{align}
\textbf{y}_t &= \textbf{m} + \underbrace{\textbf{H}_t^{1/2} \textbf{z}_t}_{=\textbf{e}_t}; \qquad
\textbf{z}_t \stackrel{\text{iid}}{\sim} \mathcal{N}(\textbf{z} \mid \textbf{0}_k, \textbf{I}_k), \\
\textbf{H}_t &= \textbf{C} + \textbf{A}\textbf{e}_{t-1}\textbf{e}_{t-1}'\textbf{A}' + \textbf{B}\textbf{H}_{t-1}\textbf{B}' \\
\textbf{m}_t &= \textbf{m}_{t-1} + \textbf{w}_t; \qquad \textbf{w}_t \stackrel{\text{iid}}{\sim} \mathcal{N}(\textbf{w} \mid \textbf{0}_k, \sigma^2_{m} \textbf{I}_k).
\end{align}
```
- Similarly, I can "sample" the time-varying Sharpe-optimal portfolio as follows.

![model3_ef](https://user-images.githubusercontent.com/46773720/224542282-1784e82a-9f70-4a3e-ad32-24216266296b.gif)

- A time-series visualization of "posterior Sharpe-optimal portfolios".

![model3_sample](https://user-images.githubusercontent.com/46773720/224542258-90dbb5f5-580f-4804-ad97-38e91ab39b1f.png)
