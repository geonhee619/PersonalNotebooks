# Hamiltonian Monte Carlo from scratch

## Toy Problem: Drawing from a 2D-Gaussian
- Suppose $\textbf{x} = (x_1,x_2)' \sim \pi(\textbf{x}) = \mathcal{N}(\textbf{x} \mid \textbf{m}, \textbf{S})$ where
- $\textbf{m} = (5,5)',$
- $\textbf{S}: \sigma_1^2 = \sigma_2^2 = 1 ~ \text{and} ~ \rho_{1,2}=3/4.$

![MvGaussian](https://user-images.githubusercontent.com/46773720/219849093-be28ae7a-b84a-426f-8c09-b5fce9a6fee3.png)

## Gradients of the Log Density as a vector field

- The information about the gradient is required to derive the potential energy in the Hamiltonian defined below.
![LogDensity](https://user-images.githubusercontent.com/46773720/219849205-b94b91ca-30bf-4c03-a4d9-97296fae2b77.png)

## Vizualising Leapfrog Integration

- The space $\textbf{x}$ is extended to $(\textbf{x},\eta)$ where $\eta \sim p(\eta)$ is the momentum.
- Then, the dynamics of $(\textbf{x},\eta)$ is determined via the Hamiltonian $H(\eta, \textbf{x}) = -\text{log}~p(\eta, \textbf{x})$.
- An absolutely useful property is that proposals along the Hamiltonian dynamics are always accepted due to energy conservation (so long as there are no numerical errors): $dH/dt = 0$. In the Metropolis-Hastings step, the acceptance probability is $1$.

![leapfrog](https://user-images.githubusercontent.com/46773720/219849784-f7aa6f1a-6ab6-46b7-84c3-7f77fbdcfc4f.gif)

- As per standard practice, I assume $\eta \sim \mathcal{N}(\eta \mid \textbf{0}_2, \text{diag} (s_1^2, s_2^2))$.
- Consequently, the kinetic energy $K(\eta \mid \textbf{x}) = K(\eta) = (\eta_1^2/s_1^2 + \eta_2^2/s_2^2)/2$, and also have the vector $\partial K / \partial \eta$ which is required by the Leapfrog integrator.

## One-step of HMC

- With some initial condition, the dynamics is numerically solved for some steps ahead. The destination is our proposal.
- The acceptance probability is evaluated with the standard M-H step.

![dynamics2](https://user-images.githubusercontent.com/46773720/219851332-eaeb5c2d-e083-43eb-84bf-9bb359602763.png)

## Monte Carlo Samples

- With multiple steps, we obtain monte carlo samples from the target distribution.

![result](https://user-images.githubusercontent.com/46773720/219851605-f1883764-756d-4b35-b6bd-1232bc397fb5.png)
