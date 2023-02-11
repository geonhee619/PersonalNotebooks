# Modeling the Japanese Yield Curve
 - Model: Dynamic Nelson-Siegel + Stochastic Volatility
 - Raw Data: Historical JGB interest rate from MoF (see https://www.mof.go.jp/english/policy/jgbs/reference/interest_rate/qa.htm).
 - Data: End-of-month rate from 2007-Nov to 2023-Jan

# Some Visualizations
## Raw Daily Data (1974-09-24 to 2023-01-12)

![data_viz](https://user-images.githubusercontent.com/46773720/218245947-555cee3b-cd3f-4e0f-8e00-85d319d16f30.gif)

## Vizualising the Posterior Mean(s)

### Evolution
- Solid: Posterior mean
- Dashed: Sample posterior samples
- Region: Visualized time-varying volatility (Note: shaded region is between the 2.5th- and 97.5th-percentile of $N(0,15\text{exp}(\text{log-volatility}_t))$; the absolute width is exaggerated for visibility.)
![posterior_viz](https://user-images.githubusercontent.com/46773720/218244358-b4f642c8-5d7d-49b5-9d34-3985b38cd47a.gif)

### Latest (as of 2023-Feb)
- Solid: Posterior mean
- Dashed: Sample posterior samples
- Region: Visualized time-varying volatility (Note: shaded region is between the 2.5th- and 97.5th-percentile of $N(0,15\text{exp}(\text{log-volatility}_t))$; the absolute width is exaggerated for visibility.)
![latest_viz](https://user-images.githubusercontent.com/46773720/218244686-c29862d6-d9f0-4361-9a54-b356f6d94263.png)

### Factors
- Solid: Posterior mean
- Region: 95% credible interval
![factors](https://user-images.githubusercontent.com/46773720/218245784-31ce6fda-5ab7-466b-97bf-b01f8753e759.png)

### An Example of a Prior Sample
![unconditional_viz](https://user-images.githubusercontent.com/46773720/218246286-5790b438-1c5b-475e-96a0-79be456914c8.gif)
