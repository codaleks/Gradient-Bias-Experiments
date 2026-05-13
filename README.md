# Gradient Bias Experiments

Small PyTorch notebooks for studying training dynamics in a deep scalar linear network.

## Contents

- `experiment_regression.ipynb` compares optimizers on the same regression problem.
- `pyproject.toml` and `uv.lock` define the Python environment.

## Regression Experiment

The notebook trains an 8-layer linear network with one scalar neuron per layer:

```text
x -> Linear(1, 1) -> ... -> Linear(1, 1) -> y
```

The target is the simple regression problem:

```text
y = 2x
```

The loss is fixed to MSE. The experiment varies only the optimizer:

- full-batch gradient descent (`GD`)
- mini-batch `SGD`
- mini-batch `SGD` with Polyak momentum
- mini-batch `Adam`

For each optimizer, the notebook plots:

- train MSE over iterations
- sharpness, defined as the largest eigenvalue of the full-batch MSE Hessian with respect to the scalar layer weights

## Setup

Install dependencies with `uv`:

```powershell
uv sync
```

Then open the notebook with the project environment selected:

```powershell
uv run jupyter lab
```

If you prefer running the notebook non-interactively:

```powershell
uv run jupyter nbconvert --execute --to notebook --inplace experiment_regression.ipynb
```

## Notes

- The experiment is deterministic: the notebook sets a fixed random seed.
- The stochastic optimizers use mini-batches sampled from the same 64 training points.
- The repository does not require external data for the regression notebook.
