# Chunk 0: Foundations — Worked Numeric Examples + Runnable Code

All numbers below were computed and verified with actual Python (numpy/scipy/sklearn) — not hand-estimated. Every project has a matching `.py` file (see the attached files) that runs fully offline and prints these exact results, with comments explaining *why* each line exists.

---

## Topic 1: Linear Algebra — 10 Worked Examples

Setup: `a = [2, 3, 5]`, `b = [1, 4, 2]`, `A = [[1,2],[3,4]]`, `B = [[5,6],[7,8]]`

| # | Operation | Input | Output |
|---|---|---|---|
| 1 | Vector addition | `a + b` | `[3, 7, 7]` |
| 2 | Vector subtraction | `a - b` | `[1, -1, 3]` |
| 3 | Scalar multiplication | `3 * a` | `[6, 9, 15]` |
| 4 | Element-wise multiplication | `a * b` | `[2, 12, 10]` |
| 5 | Dot product | `a · b` | `(2×1)+(3×4)+(5×2) = 24` |
| 6 | Vector magnitude (L2 norm) | `‖a‖` | `√(4+9+25) = √38 = 6.1644` |
| 7 | Cosine similarity | `cos(a,b)` | `24 / (6.164 × 4.583) = 0.8496` |
| 8 | Matrix addition | `A + B` | `[[6,8],[10,12]]` |
| 9 | Matrix multiplication | `A @ B` | `[[19,22],[43,50]]` — check: `C[0][0] = 1×5 + 2×7 = 19` ✓ |
| 10 | Eigenvalues | `M=[[2,0],[0,3]]` | eigenvalues `[2, 3]`, eigenvectors `[[1,0],[0,1]]` |

**Bonus — SVD** of `X=[[3,1],[1,3]]`: singular values `[4, 2]`, confirming `X` stretches space by factor 4 along one direction and 2 along the perpendicular direction — this is exactly the mechanism behind the recommender project below.

### Project: Matrix Factorization Recommender (see `matrix_factorization_project.py`)
Builds a synthetic sparse user-movie ratings matrix, factors it via SVD into user/movie "taste" vectors, and uses **dot products** between those vectors to predict ratings for unrated movies and recommend the best one. Verified output:
```
Observed (sparse) ratings matrix (0 = unrated):
[[2.3 0.  4.1 2.5 0.  0.3]
 [0.  0.2 0.  0.  1.1 0. ]
 ...]

User 0's learned taste vector: [-5.048  1.421]
Unrated movies and predicted scores: {1: 0.31, 4: 0.48}
--> Recommend movie #4 to user 0 (highest predicted score)

sklearn TruncatedSVD user factors (matches, up to sign/scale):
[[ 5.048 -1.421] ...]
```
The by-hand SVD result and sklearn's `TruncatedSVD` agree (sign flips are mathematically fine — SVD directions are unique only up to sign). Run it yourself: `python3 matrix_factorization_project.py`.

---

## Topic 2: Probability & Statistics — 10 Worked Examples

| # | Concept | Input | Output |
|---|---|---|---|
| 1 | Normal PDF | `N(0,1)` at `x=0` | `0.3989` (peak of the bell curve) |
| 2 | Normal CDF | `P(X ≤ 1.96)` for `N(0,1)` | `0.9750` (the classic 95% CI cutoff) |
| 3 | Binomial probability | `P(7 heads in 10 flips, p=0.5)` | `0.1172` |
| 4 | Bayes — medical test | `P(disease)=0.01`, test 99% accurate, tested positive | `P(disease|+) = 0.5000` (only 50%, not 99%!) |
| 5 | Bayes — spam word | `P(spam)=0.4`, `P('free'|spam)=0.6`, `P('free'|ham)=0.1` | `P(spam|'free') = 0.8000` |
| 6 | MLE — coin bias | flips `[1,1,1,0,1,0,1,1,0,1]` | `MLE p̂ = 7/10 = 0.7` |
| 7 | Mean & variance | `[4,8,6,5,3,7]` | `mean=5.5, var=2.9167, std=1.7078` |
| 8 | Bias-variance (synthetic) | true fn `y=3x+2` + noise | underfit MSE=`73.22`, good-fit MSE=`2.41`, "overfit" train MSE=`0.55` |
| 9 | Cross-entropy loss | true label=1, pred=0.8 vs pred=0.2 | loss=`0.2231` vs loss=`1.6094` (wrong-but-confident is punished hard) |
| 10 | Poisson | `P(exactly 5 events | rate=3)` | `0.1008` |

**Why example 4 matters most for interviews:** even with a 99%-accurate test, a rare disease (1% base rate) means a positive result only implies a 50% real chance of having it — false positives from the huge healthy population dilute the signal. This exact reasoning pattern (base-rate correction) shows up in fraud detection, spam filtering, and anomaly detection at every MAANG company.

### Project: Naive Bayes Spam Classifier From Scratch (see `naive_bayes_project.py`)
Computes `P(word|spam)` and `P(word|ham)` directly from word counts (MLE), then applies Bayes' theorem in log-space to classify new emails. Verified output:
```
Prior P(spam) = 0.500, Prior P(ham) = 0.500

'free prize claim now' -> spam  (P(spam)=0.9952)
'team meeting agenda for tomorrow' -> ham  (P(spam)=0.0691)
'cheap pills for sale' -> spam  (P(spam)=0.6966)

sklearn MultinomialNB predictions (matches exactly):
'free prize claim now' -> spam
'team meeting agenda for tomorrow' -> ham
'cheap pills for sale' -> spam
```
The by-hand version and `sklearn.naive_bayes.MultinomialNB` agree on every prediction. Run it: `python3 naive_bayes_project.py`.

---

## Topic 3: Optimization / Gradient Descent — 10-Step Worked Trace

Fitting `y = w·x` (true `w=3`) to data `X=[1,2,3,4]`, `y=[3,6,9,12]`, learning rate `0.05`, starting at `w=0`:

| iter | w | predictions | loss (MSE) | gradient |
|---|---|---|---|---|
| 0 | 0.0000 | [0,0,0,0] | 67.5000 | -45.0000 |
| 1 | 2.2500 | [2.25,4.5,6.75,9] | 4.2188 | -11.2500 |
| 2 | 2.8125 | [2.81,5.62,8.44,11.25] | 0.2637 | -2.8125 |
| 3 | 2.9531 | [2.95,5.91,8.86,11.81] | 0.0165 | -0.7031 |
| 4 | 2.9883 | [2.99,5.98,8.96,11.95] | 0.0010 | -0.1758 |
| 5 | 2.9971 | [3,5.99,8.99,11.99] | 0.0001 | -0.0439 |
| 6 | 2.9993 | [3,6,9,12] | 0.0000 | -0.0110 |
| 7 | 2.9998 | [3,6,9,12] | 0.0000 | -0.0027 |
| 8 | 3.0000 | [3,6,9,12] | 0.0000 | -0.0007 |
| 9 | 3.0000 | [3,6,9,12] | 0.0000 | -0.0002 |

**Learning rate too high (`lr=0.5`) — diverges:**
`w` goes `0 → 22.5 → -123.75 → 826.9 → -5352.2 → 34811.7` — loss explodes into the billions within 5 steps. This is exactly what "the model isn't training, loss is NaN" looks like in a real training run.

**Learning rate too low (`lr=0.001`) — crawls:**
After 5 steps `w` has only reached `0.218` (target is `3`) — would take thousands of iterations to converge.

### Project: Gradient Descent From Scratch + Momentum (see `gradient_descent_project.py`)
Fits `y = wx + b` (true `w=4, b=7`) using hand-written gradient descent, then compares against a momentum-based version and sklearn's closed-form solution. Verified output:
```
=== Vanilla Gradient Descent ===
epoch 0    loss=868.5123   w=3.3730   b=0.5456
epoch 160  loss=7.6155     w=4.2455   b=5.1645
Learned: w=4.151, b=5.795  (true values: w=4, b=7)

sklearn (closed-form / normal equation): w=3.718, b=8.693
```
(Both approaches land in the same neighborhood — the small remaining gap versus the "true" 4/7 is expected: only 200 epochs and noisy data. Run longer / lower noise to see it converge tighter.) Run it: `python3 gradient_descent_project.py`.

---

## Topic 4: Overfitting, Regularization, Cross-Validation — Worked Examples

**Overfitting as model complexity increases** (polynomial degree fit to noisy quadratic data):

| degree | train MSE | validation MSE |
|---|---|---|
| 1 | 0.8790 | 36.8968 |
| 2 | 0.5430 | 3.8443 |
| 5 | 0.4305 | 4,313.9964 |
| 10 | 0.2906 | 4,267,887,529.16 |
| 15 | 0.1113 | 24,145,688,541,908,532.0 |

Train error keeps dropping smoothly — but validation error explodes past degree 5. **That gap is overfitting, quantified.**

**L1 (Lasso) vs L2 (Ridge)** on 5 features where only features 0 and 3 actually matter (true weights `[5,0,0,3,0]`):
```
True weights:  [5, 0, 0, 3, 0]
Lasso weights: [4.285, -0.0, 0.0, 2.357, -0.0]   <- zeroes out the irrelevant features
Ridge weights: [4.914, -0.01, -0.016, 3.018, -0.032]  <- shrinks all, none exactly zero
```

**Fixing the degree-10 overfit model with regularization:**
```
No regularization      val_MSE = 4,267,887,529.16
Ridge (L2)              val_MSE = 26,883.42
Lasso (L1)              val_MSE = 12.27       <- best fix here
```

**5-fold cross-validation** picking the right polynomial degree reliably:
```
degree=1   mean CV MSE = 3.8486 (+/- 0.6996)
degree=2   mean CV MSE = 0.7264 (+/- 0.4802)   <- best: lowest AND most stable
degree=5   mean CV MSE = 0.9317 (+/- 0.4213)
degree=10  mean CV MSE = 3.6601 (+/- 2.7837)   <- high variance across folds = red flag
```

### Project: Overfitting Diagnosis & Fix (see `overfitting_diagnosis_project.py`)
Reproduces all three tables above in one runnable script with inline comments explaining each step — deliberately overfits, then applies regularization and cross-validation as fixes, printing before/after numbers at each stage. Run it: `python3 overfitting_diagnosis_project.py`.

---

## How to run these yourself
All four project files are plain Python (`numpy`, `scipy`, `scikit-learn` — no internet/dataset download required, everything uses synthetic data generated in-script):
```bash
pip install numpy scipy scikit-learn
python3 matrix_factorization_project.py
python3 naive_bayes_project.py
python3 gradient_descent_project.py
python3 overfitting_diagnosis_project.py
```
Each script prints intermediate steps so you can watch the math happen line-by-line rather than just seeing a final answer.

---

**Next up, same format (10 verified numeric examples + tested runnable project code) for Chunk 1: Linear/Logistic Regression, Decision Trees, Random Forest, Gradient Boosting, Feature Engineering, and Evaluation Metrics. Say the word and I'll continue.**
