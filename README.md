# Running dlp-a1: DLP Assignment 1 

Reproduces: (1) a NumPy-only MLP with manual backprop, verified against PyTorch; (2) an
activation-function study (sigmoid/tanh/ReLU/leaky ReLU); (3) a cross-entropy vs. MSE
comparison for classification, plus an MLP regression baseline on California Housing.

## 1. Environment

```bash
pip install numpy pandas matplotlib scikit-learn torch
```


## 2. Data

- **Fashion-MNIST** (Parts 1–3, classification): CSV version, used as
  `fashion-mnist_train.csv` / `fashion-mnist_test.csv`. The notebook expects them at
  `/kaggle/input/datasets/zalando-research/fashionmnist/`. If running outside Kaggle,
  download from https://www.kaggle.com/datasets/zalando-research/fashionmnist and update
  the two `pd.read_csv(...)` paths in the first data-loading cell.
- **California Housing** (Part 3, regression): fetched automatically via
  `sklearn.datasets.fetch_california_housing()` — no manual download needed, requires
  internet access on first run.

## 3. Running

Run all cells top to bottom, in order — later cells depend on variables defined earlier
(`x_train`, `w1`/`b1`/`w2`/`b2`, `model_ce`, etc.). Rough breakdown:

| Cells | Section |
|---|---|
| 0–8 | Load Fashion-MNIST, split (80/20 stratified), inspect class balance |
| 9–23 | **Part 1** — NumPy MLP (784→64 ReLU→10 softmax), manual forward/backward/update, trained on first 5,000 samples for 20 epochs, gradient-checked against an equivalent PyTorch model on the same batch/weights |
| 24–35 | **Part 2** — 2-hidden-layer PyTorch MLP (784→128→64→10), trained separately with sigmoid/tanh/ReLU/leaky ReLU, validation loss compared, first-layer gradient magnitude and dead-ReLU % reported |
| 36–46 | **Part 3** — same architecture trained with cross-entropy vs. MSE-on-one-hot-targets (identical initialization via `torch.manual_seed(42)` before each), plus a separate MLP regression run on California Housing |

Global seed: `np.random.seed(42)`, `random.seed(42)`; PyTorch models are separately
seeded with `torch.manual_seed(42)` right before construction where a fair side-by-side
comparison matters (Part 3's CE vs. MSE classifiers).

## 4. Expected results (from the reference run)

- **Part 1 gradient check:** max abs difference vs. PyTorch ≈ `1e-9`–`1e-10` for both
  weight matrices — confirms the manual backward pass is correct.
- **Part 2 dead-ReLU check:** ≈ 0.78% of first-hidden-layer ReLU units output zero for
  every sample in the validation batch.
- **Part 3 classification:** CE test accuracy ≈ 13.4%, MSE test accuracy ≈ 5.9% — both
  near chance (10 classes). This is a training-budget artifact, not a bug: each "epoch"
  here is one **full-batch** gradient step (5,000 samples per update), so 20 epochs is
  only 20 total weight updates. That's also why the CE-vs-MSE loss plot looks like two
  flat lines — 20 steps barely move the loss on either curve, and CE (~2.3 scale) and MSE
  (~0.1 scale) are compressed onto the same y-axis, exaggerating the flatness further. To
  get a meaningful comparison, increase the number of gradient steps (many more full-batch
  epochs, e.g. 300–500, or switch to mini-batch SGD) before drawing conclusions from the
  accuracy numbers.
- **Part 3 regression (California Housing):** MSE ≈ 0.728, RMSE ≈ 0.853, MAE ≈ 0.660
  (target units of $100,000s). Features and target are both standardized with a scaler
  fit on the training set only; predictions are inverse-transformed back to real units
  before computing metrics.
