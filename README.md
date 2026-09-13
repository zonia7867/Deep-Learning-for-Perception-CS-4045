# Running dlp-a1: DLP Assignment 1

Reproduces: (1) a NumPy-only MLP with manual backprop, verified against PyTorch; (2) an
activation-function study (sigmoid/tanh/ReLU/leaky ReLU); (3) a cross-entropy vs. MSE
comparison for classification, plus an MLP regression baseline on California Housing;
(4) an optimiser comparison (SGD, SGD+momentum, RMSProp, Adam); (5) a deliberately
overfitted model; (6) a regularisation study (L2, L1, dropout, batchnorm, early stopping,
augmentation, more data); (7) hyperparameter tuning via random search + 5-fold CV, with a
final held-out test evaluation.

## 1. Environment

```bash
pip install numpy pandas matplotlib scikit-learn torch scipy
```

`scipy` is only needed for Part 6's data augmentation step (`scipy.ndimage.rotate`); it
usually installs already as a scikit-learn dependency, but it's listed explicitly here in
case of a clean environment.

## 2. Data

- **Fashion-MNIST** (Parts 1–7, classification): CSV version, used as
  `fashion-mnist_train.csv` / `fashion-mnist_test.csv`. The notebook expects them at
  `/kaggle/input/datasets/zalando-research/fashionmnist/`. If running outside Kaggle,
  download from https://www.kaggle.com/datasets/zalando-research/fashionmnist and update
  the two `pd.read_csv(...)` paths in the first data-loading cell. Parts 4–7 reuse the
  same `x_train`/`y_train`/`x_test`/`y_test` arrays loaded in Section 2 — no separate
  download needed.
- **California Housing** (Part 3, regression): fetched automatically via
  `sklearn.datasets.fetch_california_housing()` — no manual download needed, requires
  internet access on first run. On Kaggle this requires phone-verifying your account and
  enabling the Internet toggle in notebook settings; if that isn't available,
  `sklearn.datasets.load_diabetes()` is a drop-in, no-internet-required substitute.

## 3. Running

Run all cells top to bottom, in order — later cells depend on variables defined earlier
(`x_train`, `w1`/`b1`/`w2`/`b2`, `model_ce`, `X_small`, `y_small`, `model_big`, etc.).
Rough breakdown (Part 4 onward are approximate cell numbers — count may shift by a few
depending on how cells were split/merged when added):

| Cells | Section |
|---|---|
| 0–8 | Load Fashion-MNIST, split (80/20 stratified), inspect class balance |
| 9–23 | **Part 1** — NumPy MLP (784→64 ReLU→10 softmax), manual forward/backward/update, trained on first 5,000 samples for 20 epochs, gradient-checked against an equivalent PyTorch model on the same batch/weights |
| 24–35 | **Part 2** — 2-hidden-layer PyTorch MLP (784→128→64→10), trained separately with sigmoid/tanh/ReLU/leaky ReLU, validation loss compared, first-layer gradient magnitude and dead-ReLU % reported |
| 36–47 | **Part 3** — same architecture trained with cross-entropy vs. MSE-on-one-hot-targets (identical initialization via `torch.manual_seed(42)` before each), plus a separate MLP regression run on California Housing |
| 48–60 | **Part 4** — same 784→128→64→10 architecture trained full-batch with four optimisers (SGD, SGD+momentum, RMSProp, Adam); same shared learning rate first, then a per-optimiser learning-rate search; final tuned comparison table and loss-curve plot |
| 61–69 | **Part 5** — training set reduced to 2,000 samples, network widened to four 512-unit hidden layers, trained to 100% train accuracy to force overfitting; train/val loss plotted with the divergence point marked |
| 70–97 | **Part 6** — starting from Part 5's overfit model, applies L2 weight decay, L1 penalty, dropout, batch normalisation, early stopping, data augmentation (flip + rotation), and more training data (10k/20k samples) one at a time; results table plus gap-vs-strength plots for L2 and dropout, including extreme settings (L2=0.1, dropout=0.9) to show over-regularisation |
| 98–114 | **Part 7** — random search over 12 configurations (learning rate, hidden width, dropout), scored via 5-fold CV on a 5,000-sample subset; best configuration retrained on the full training set with early stopping; single evaluation on the held-out test set (accuracy, macro precision/recall/F1, confusion matrix) |

Global seed: `np.random.seed(42)`, `random.seed(42)`; PyTorch models are separately
seeded with `torch.manual_seed(42)` right before construction wherever a fair
side-by-side comparison matters (Part 3's CE vs. MSE classifiers, Part 4's optimiser
runs, Part 5's overfit model, Part 6's per-method models, Part 7's CV folds and final
retrain).

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
- **Part 4 optimiser comparison:** with a shared learning rate (0.01), SGD lagged well
  behind the others (61.4% val. acc. vs. 81–82.5%). After tuning the learning rate per
  optimiser (SGD 0.3, SGD+momentum 0.1, RMSProp/Adam 0.005) and extending to 1,500
  full-batch epochs, none of the four reached the 85% val. accuracy target — all
  plateaued between 81–82.5%, with Adam highest (82.5%) and most stable (fewest loss
  spikes). This is reported as an honest negative result rather than forced, since it
  reflects the limits of a 5,000-sample full-batch setup.
- **Part 5 forced overfitting:** train accuracy reaches 100% on the 2,000-sample subset
  while validation accuracy stays at 80.8%, a 19.2-point generalisation gap. Validation
  loss reaches its minimum at epoch 67 and rises steadily afterward — high variance, not
  high bias, since train accuracy is high throughout.
- **Part 6 regularisation study:** early stopping (patience 15, stopped at epoch 83)
  gave the best trade-off, cutting the gap to 9.15% at near-baseline validation accuracy
  (81.3%). More training data (20,000 samples) gave the lowest gap overall (6.9%) and the
  highest validation accuracy (87.2%), though it isn't a regularisation method in the same
  sense. At extreme strengths (L2 lambda=0.1, dropout=0.9) both train and validation
  accuracy collapse to near-chance (~10%) — the gap shrinks but only because the model can
  no longer learn, illustrating over-regularisation rather than improvement.
- **Part 7 hyperparameter tuning:** random search over 12 configurations, scored by
  5-fold CV on a 5,000-sample subset, selected hidden width 512, learning rate ≈0.0027,
  dropout ≈0.21 (mean CV accuracy 85.0%, std 0.0039). Retrained on the full training set
  with early stopping (stopped at epoch 196), the final model reaches 89.43% test
  accuracy (macro precision 0.895, macro recall 0.894, macro F1 0.894) — a 2.65
  percentage-point improvement over a clean, untuned Part 2-architecture baseline
  (86.78%), trained separately here since Part 2 itself did not report a test accuracy.
