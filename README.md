# A Compilation of Personal Scribbles in Stats

## (1) Modeling the Japanese Yield Curve
 - Model: Dynamic Nelson-Siegel + Stochastic Volatility
 - Raw Data: Historical JGB interest rate from MoF (see https://www.mof.go.jp/english/policy/jgbs/reference/interest_rate/qa.htm).
 - Data: End-of-month rate from 2007-Nov to 2023-Jan

![posterior_viz](https://user-images.githubusercontent.com/46773720/218244358-b4f642c8-5d7d-49b5-9d34-3985b38cd47a.gif)

## (2) Visualizing Hamiltonian Monte Carlo
- Setting: Sampling from a 2D-Gaussian
- Symplectic integration and HMC from scratch

![leapfrog](https://user-images.githubusercontent.com/46773720/219874506-0ba6258b-0987-42aa-9313-a0e0c9b50c3f.gif)

## (3) Visualizing Variational Inference
- Setting: Univariate Bayesian Linear Regression with known variance
- With Adam (Flux) and Automatic differentiation (ReverseDiff)

![VI](https://user-images.githubusercontent.com/46773720/223634760-fe691ec8-2b9c-4441-9193-f004b1de9638.gif)

## (4) "Sampling" Time-varying Sharpe-optimal Portfolio w/ Bayesian Multivariate GARCH
- Goal: A framework to quantify the epistemic uncertainty of optimal portfolios
- Approach: Mean-Variance & Bayesian BEKK (+ Time-varying latent return process)

![model1](https://user-images.githubusercontent.com/46773720/224539761-dfe4c099-207e-4f70-81d9-67a6edbe69bf.png)

## (5) Modeling/Forecasting the Probability of Recession w/ a Regime-switching model + Bayesian Neural Net
- Model: Multivariate Autoregressive 2-state Regime-switching model (on Industrial Production Total Index & Unemployment Rate) w/ nonlinear transition probabilities modeled via Bayesian Neural Nets.
- Data: FRED-MD (very "basic" variables: Industrial Production Total Index, Unemployment Rate, CPI, Fed Funds Rate) + 30 Day Fed Funds Rate Futures.

![ts2](https://user-images.githubusercontent.com/46773720/226235458-4fa70a1f-d894-4dde-b418-b171f9254f90.gif)

## (6) Stochastic volatility model based clustering of multivariate time-series
- Model: A stochastic volatility model with Dirichlet Process mixture in-mean.

![download](https://github.com/geonhee619/PersonalNotebooks/assets/46773720/2843bcf5-7d81-4997-8031-584f59c44016)
