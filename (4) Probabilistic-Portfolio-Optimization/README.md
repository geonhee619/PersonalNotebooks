# A Sampling Approach to Time-varying Sharpe-optimal Portfolio Allocation w/ Bayesian Multivariate GARCH

## Goal
- Sampling from the "posterior" of optimal portfolios

## Data
- Daily returns of S&P500, N225, and FTSE100 from 2022-12-29 to 2023-03-09.

## Model #1: IID Multivariate Gaussian
- Consider the conventional "iid"-model which posits the DGP of returns as a purely i.i.d. process. That is, for some $T \in \mathbb{N}$,

```math
\textbf{y}_t = (y_{1,t}, ..., y_{k,t})' \stackrel{\text{iid}}{\sim} \mathcal{N}(\textbf{y} \mid \textbf{m}, \textbf{S}); \quad t=1,...,T.
```

- Given some (long position) weight
```math
\textbf{w} \in W := \{ (w_1,...,w_k)' : [w_1,...,w_k \in \mathbb{R}_{\geq 0}] \wedge [\sum_{i=1}^k w_i = 1] \},
```
the portfolio (at $t$) is defined as $f(\textbf{w})=\textbf{w}'\textbf{y}_t$. It follows that the portfolio
   - expected return $\mathbb{E}_{\textbf{y}_t}[f(\textbf{w})] = \textbf{w}'\textbf{m}$;
   - variance $\mathbb{V}_{\textbf{y}_t} [f(\textbf{w})] = \textbf{w}' \mathbb{V}[\textbf{y}_t] \textbf{w} = \textbf{w}'\textbf{S}\textbf{w}$.

- To obtain the Sharpe-optimal portfolio, the typical approach is to use, for instance, the maximum likelihood estimate of $(\textbf{m},\textbf{S})$, $(\hat{\textbf{m}},\hat{\textbf{S}})$, and solve for the optimization problem
```math
\textbf{w}^* = \text{argmax}_{\textbf{w} \in W} ~ {\textbf{w}' \hat{\textbf{m}} \over \sqrt{\textbf{w}' \hat{\textbf{S}} \textbf{w}}},
```
using the sample analogue of $(\textbf{m},\textbf{S})$.

- The approach here is to suppose a generative model with the following additional structure:
```math
\theta = (\textbf{m}', \text{vec}(\textbf{S})')' \sim p(\theta) \stackrel{\text{d}}{=} p(\textbf{m}) p(\textbf{S}),
```
where $p(\textbf{S})$ is a prior distribution over positive semi-definite matrices.

- The goal is to inspect the posterior quantitites over $\textbf{w}^*$ via the posterior distribution of $\theta$ given $\textbf{Y}:=(\textbf{y}_1,...,\textbf{y}_T)$, that is
```math
p(\theta \mid \textbf{Y}) \propto p(\textbf{Y} \mid \theta) p(\theta).
```

- For each Monte Carlo sample from the posterior distribution, $(\textbf{m},\textbf{S})^{(r)} \sim p(\theta \mid \textbf{Y}); (r=1,...,R)$, the same optimization problem is solved for:
```math
(\textbf{w}^*)^{(r)} = \text{argmax}_{\textbf{w} \in W} ~ {\textbf{w}' \textbf{m}^{(r)} \over \sqrt{\textbf{w}' \textbf{S}^{(r)} \textbf{w}}}.
```
- The $\textcolor{blue}{\text{efficient frontier --}}$ and the $\textcolor{red}{\text{Sharpe-optimal portfolio ●}}$ is visualized below.

![model1](https://user-images.githubusercontent.com/46773720/224541902-7f5205eb-9472-42f5-b1ee-415dadfa3378.png)

## Model #2: Multivariate GARCH (BEKK of Engel and Kroner, 1995)
- Upon the previous generative model, the model of BEKK specifies the deterministic evolution of the conditional variance.
- The generative model, for $t=1,...,T$, is
```math
\begin{align}
\textbf{y}_t &= \textbf{m} + \underbrace{\textbf{H}_t^{1/2} \textbf{z}_t}_{=\textbf{e}_t}; \qquad
\textbf{z}_t \stackrel{\text{iid}}{\sim} \mathcal{N}(\textbf{z} \mid \textbf{0}_k, \textbf{I}_k), \\
\textbf{H}_t &= \textbf{C} + \textbf{A}\textbf{e}_{t-1}\textbf{e}_{t-1}'\textbf{A}' + \textbf{B}\textbf{H}_{t-1}\textbf{B}',
\end{align}
```
and appropriate priors on $(\textbf{H}_1, \textbf{C}, \textbf{B}, \textbf{A})$. $\textbf{C}$ is positive definite.
- The Sharpe-optimal portfolio with time-variance is constructed from the MCMC output by solving the same optimization:
```math
(\textbf{w}^*_t)^{(r)} = \text{argmax}_{\textbf{w} \in W} ~ {\textbf{w}' \textbf{m}^{(r)} \over \sqrt{\textbf{w}' \textbf{H}_t^{(r)} \textbf{w}}},
\quad \text{for} \quad t=1,...,T.
```
- This is visualized as follows.

![model2](https://user-images.githubusercontent.com/46773720/224542127-7bb2ae04-e0d0-4648-b91a-327385f09780.gif)

## Model #3: Multivariate GARCH (BEKK) w/ Time-varying Expected Returns

- Additionally posit that the expected returns are time-varying.
- The generative model, for $t=1,...,T$, is
```math
\begin{align}
\textbf{y}_t &= \textbf{m} + \underbrace{\textbf{H}_t^{1/2} \textbf{z}_t}_{=\textbf{e}_t}; \qquad
\textbf{z}_t \stackrel{\text{iid}}{\sim} \mathcal{N}(\textbf{z} \mid \textbf{0}_k, \textbf{I}_k), \\
\textbf{H}_t &= \textbf{C} + \textbf{A}\textbf{e}_{t-1}\textbf{e}_{t-1}'\textbf{A}' + \textbf{B}\textbf{H}_{t-1}\textbf{B}' \\
\textbf{m}_t &= \textbf{m}_{t-1} + \textbf{w}_t; \qquad \textbf{w}_t \stackrel{\text{iid}}{\sim} \mathcal{N}(\textbf{w} \mid \textbf{0}_k, \sigma^2_{m} \textbf{I}_k).
\end{align}
```
- Similarly, I can "sample" the time-varying Sharpe-optimal portfolio via
```math
(\textbf{w}^*_t)^{(r)} = \text{argmax}_{\textbf{w} \in W} ~ {\textbf{w}' \textbf{m}_t^{(r)} \over \sqrt{\textbf{w}' \textbf{H}_t^{(r)} \textbf{w}}},
\quad \text{for} \quad t=1,...,T.
```

![model3_ef](https://user-images.githubusercontent.com/46773720/224542282-1784e82a-9f70-4a3e-ad32-24216266296b.gif)

- A time-series visualization of "posterior Sharpe-optimal portfolios".

![model3_sample](https://user-images.githubusercontent.com/46773720/224542258-90dbb5f5-580f-4804-ad97-38e91ab39b1f.png)

### Further Extensions
- The expected-return dynamics need not be a Gaussian random-walk.
- The volatility dynamics need not be BEKK, although they are very convenient.
- They need not be linear.
- The optimization objective need not be the Sharpe-ratio, although they are intuitive.
- Innovations need not be Normal and symmetric.

# References

- Engle, R. F. and Kroner, K. F. (1995). "Multivariate Simultaneous Generalized Arch", Econometric Theory, 11(1), 122-150.
