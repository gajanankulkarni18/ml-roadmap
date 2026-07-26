"""
PROJECT: Diagnose and Fix Overfitting
Concept demonstrated: bias-variance tradeoff, L1/L2 regularization, k-fold cross-validation
Runs fully offline with synthetic data.
"""
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error

np.random.seed(42)

# ---------------------------------------------------------
# STEP 1: Create data with a true quadratic relationship + noise
# ---------------------------------------------------------
X = np.linspace(-3, 3, 30).reshape(-1, 1)
y = 0.5 * X.ravel()**2 + np.random.normal(0, 1, 30)  # true pattern: y = 0.5x^2 + noise

X_train, X_val = X[:20], X[20:]
y_train, y_val = y[:20], y[20:]

# ---------------------------------------------------------
# STEP 2: Fit increasingly complex polynomial models and watch the
# train/validation error gap widen -- this IS overfitting, visualized numerically
# ---------------------------------------------------------
print("=== Step 1: Show overfitting as model complexity increases ===")
print(f"{'degree':<8}{'train_MSE':<15}{'val_MSE':<15}")
for deg in [1, 2, 5, 10, 15]:
    # Build polynomial features by hand: [x, x^2, x^3, ... x^deg]
    Xp_train = np.hstack([X_train**d for d in range(1, deg + 1)])
    Xp_val = np.hstack([X_val**d for d in range(1, deg + 1)])

    model = LinearRegression().fit(Xp_train, y_train)
    train_err = mean_squared_error(y_train, model.predict(Xp_train))
    val_err = mean_squared_error(y_val, model.predict(Xp_val))
    print(f"{deg:<8}{train_err:<15.4f}{val_err:<15.4f}")
print("--> Notice: train error keeps dropping, but val error explodes past degree ~5.")
print("    That gap is the overfitting signal.")

# ---------------------------------------------------------
# STEP 3: Fix #1 -- Regularization (L1 vs L2) on a high-degree model
# Regularization adds a penalty for large weights directly into the loss function,
# discouraging the model from fitting noise with huge coefficients.
# ---------------------------------------------------------
print("\n=== Step 2: Fix with regularization (degree=10 model) ===")
deg = 10
Xp_train = np.hstack([X_train**d for d in range(1, deg + 1)])
Xp_val = np.hstack([X_val**d for d in range(1, deg + 1)])

plain = LinearRegression().fit(Xp_train, y_train)
ridge = Ridge(alpha=5.0).fit(Xp_train, y_train)     # L2 penalty
lasso = Lasso(alpha=0.5, max_iter=5000).fit(Xp_train, y_train)  # L1 penalty

for name, m in [("No regularization", plain), ("Ridge (L2)", ridge), ("Lasso (L1)", lasso)]:
    val_err = mean_squared_error(y_val, m.predict(Xp_val))
    print(f"{name:<22} val_MSE = {val_err:.4f}")
print("--> Regularization pulls validation error back down by discouraging huge weights.")

# ---------------------------------------------------------
# STEP 4: Fix #2 -- k-fold cross-validation for a more reliable complexity choice
# Instead of trusting ONE train/val split, average error across 5 different splits.
# ---------------------------------------------------------
print("\n=== Step 3: Use cross-validation to pick the right degree reliably ===")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for deg in [1, 2, 5, 10]:
    Xp_full = np.hstack([X**d for d in range(1, deg + 1)])
    scores = cross_val_score(LinearRegression(), Xp_full, y, cv=kf, scoring='neg_mean_squared_error')
    print(f"degree={deg:<3} mean CV MSE = {-np.mean(scores):.4f} (+/- {np.std(scores):.4f})")
print("--> The degree with the lowest, most STABLE cross-validated error is the right complexity.")
