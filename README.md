# Project 24: Extreme Learning Machine with L1 Minimisation

This project implements the model shown in the assignment:

- `(M)` Extreme Learning Machine with one hidden layer, fixed random hidden weights `W1`, and output weights `w`
- `(A1)` Iteratively Reweighted Least Squares (IRLS) for `L1` minimisation
- `(A2)` Smoothed-gradient minimisation of a pseudo-Huber approximation to the `L1` objective

The code relies only on `numpy` and `matplotlib`.

## Problem setup

The regression model is

`y = w sigma(W1 x)`

where:

- `W1` is a fixed random matrix for the hidden layer
- `sigma` is an elementwise activation function
- `w` is learned by minimising `||Xw - y||1`, with `X` the ELM feature matrix

The default experiment uses:

- `sigmoid` activation
- `60` hidden units
- hidden-weight scale `5.0`
- `120` synthetic 1D samples
- Gaussian noise plus `12%` outliers

## Solvers

### (A1) IRLS

IRLS solves a sequence of weighted least-squares problems. At each step it uses the thresholded diagonal weights

`|(Xw_k - y)_i|^(-1/2)`

so residuals close to zero do not create unstable weights.

### (A2) Smoothed Gradient

The `L1` objective is approximated with the pseudo-Huber smoothing

`f_mu(w) = sum_i (sqrt(r_i^2 + mu^2) - mu)`

with residuals `r = Xw - y`. The implementation uses a continuation strategy: solve the smoother problem first, then progressively decrease `mu` while warm-starting each stage from the previous solution.

## Running the project

Run the canonical entrypoint:

```bash
python3 project24.py
```

This prints the metric summary and saves the figure to `project24_results.png`.

`main.py` is kept as a compatibility wrapper and runs the same pipeline.

## Default run summary

With the default configuration, the important expected behavior is:

- `IRLS` achieves the lowest training `L1` objective
- `Smoothed Gradient` improves on the plain `L2` baseline
- the figure shows that `L1` methods are less distorted by outliers than `L2`

After running the script, update the table below only if you intentionally change the configuration:

| Method | Objective |
| --- | ---: |
| IRLS | `44.494` |
| Smoothed Gradient | `49.865` |
| L2 baseline | `51.523` |

## Output figure

The generated figure contains three panels:

- regression fit on noisy data with highlighted outliers
- convergence history of the true `L1` objective
- residual distribution for `L2`, `IRLS`, and Smoothed Gradient

## Interpretation

The experiment is designed so that the `L2` baseline reacts strongly to outliers, while `L1`-based training is more robust. In the default run, `IRLS` gives the strongest fit in terms of the `L1` training objective, and the smoothed-gradient solver also improves on the `L2` baseline while staying within the assignment constraint of using a manually implemented optimisation method.
