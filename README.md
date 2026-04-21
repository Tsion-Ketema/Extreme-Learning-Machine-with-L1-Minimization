# ELM with L1 Minimization via IRLS

## Project structure

```
elm_project/
├── core/
│   ├── __init__.py
│   ├── activations.py     # tanh, relu, sigmoid
│   ├── elm.py             # ELM hidden layer: W1 generation, X = sigma(Input @ W1.T)
│   └── solvers.py         # IRLS algorithm (A1), L1 loss, L2 baseline
├── dataset/
│   ├── ML-CUP25-TR.csv    # training data (500 samples, 12 features, 4 targets)
│   └── ML-CUP25-TS.csv    # blind test data (1000 samples, no targets)
├── utils/
│   ├── __init__.py
│   ├── data.py            # load_cup, load_cup_test, train_test_split
│   └── experiments.py     # 4 experiment functions (no plotting)
├── viz/
│   ├── __init__.py
│   └── plots.py           # figure generation from experiment results
├── figures/               # generated figures (created automatically)
├── main.py                # entry point — run this
└── README.md
```

## File descriptions

| File                   | What it does                                                                                                                                                                                               |
| ---------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `core/activations.py`  | Activation functions: `tanh`, `relu`, `sigmoid`                                                                                                                                                            |
| `core/elm.py`          | `random_hidden_weights()` generates W1. `build_feature_matrix()` computes X = sigma(Input @ W1.T). `predict()` computes X @ w. `predict_all_targets()` stacks 4 predictions into the final (n x 4) matrix. |
| `core/solvers.py`      | `irls()` runs Algorithm A1 and returns w and convergence history. `l1_loss()` evaluates the objective. `l2_baseline()` computes the OLS solution for comparison.                                           |
| `utils/data.py`        | `load_cup()` loads and standardizes training CSV. `load_cup_test()` loads blind test. `train_test_split()` splits into train/test.                                                                         |
| `utils/experiments.py` | Four experiment functions returning result dicts (no plotting inside).                                                                                                                                     |
| `viz/plots.py`         | Takes result dicts and writes PDF figures to `./figures/`.                                                                                                                                                 |
| `main.py`              | Loads data, runs all 4 experiments, prints results, saves all figures.                                                                                                                                     |

## How to run

```bash
# from the project root
python main.py
```

Figures are written to `./figures/`.

## Shape summary

```
Input  : (500 x 12)          — raw data, 500 samples, 12 features
W1     : (m x 12)            — random, frozen after generation
X      : (500 x m)           — feature matrix, 500 rows x m neurons
w      : (m x 1)             — output weights, one vector per target
y_hat  : (500 x 1)           — predictions for ONE target
Y_hat  : (500 x 4)           — all 4 targets stacked column by column
```

The (500 x 4) final matrix is built by running IRLS 4 times and stacking:

```python
Y_hat = np.column_stack([X @ w1, X @ w2, X @ w3, X @ w4])
```

## Key parameters

| Parameter    | Default | Description                                       |
| ------------ | ------- | ------------------------------------------------- |
| `m`          | 100     | Number of hidden neurons                          |
| `eps`        | 1e-6    | IRLS threshold (prevents division by zero)        |
| `tol`        | 1e-6    | Convergence tolerance on weight change            |
| `max_iter`   | 500     | Maximum IRLS iterations                           |
| `activation` | tanh    | Hidden-layer activation (tanh / relu / sigmoid)   |
| `TEST_FRAC`  | 0.1     | Fraction of training data held out for evaluation |
