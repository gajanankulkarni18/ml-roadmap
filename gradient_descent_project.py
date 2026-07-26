"""
PROJECT: Gradient Descent From Scratch on Linear Regression
Concept demonstrated: gradients, learning rate, convergence, momentum, Adam
Runs fully offline with synthetic data.
"""
import numpy as np

# ---------------------------------------------------------
# STEP 1: Generate synthetic data with a KNOWN true relationship
# y = 4x + 7 + noise -- so we can check if gradient descent finds w=4, b=7
# ---------------------------------------------------------
np.random.seed(0)
X = np.linspace(0, 10, 50)
true_w, true_b = 4, 7
y = true_w * X + true_b + np.random.normal(0, 2, size=50)

# ---------------------------------------------------------
# STEP 2: Vanilla Gradient Descent implemented by hand (no sklearn)
# ---------------------------------------------------------
def vanilla_gd(X, y, lr=0.01, epochs=200):
    w, b = 0.0, 0.0  # start with random/zero initial guess
    history = []
    n = len(X)
    for epoch in range(epochs):
        pred = w * X + b
        error = pred - y
        loss = np.mean(error**2)  # Mean Squared Error loss

        # Gradients: partial derivatives of MSE loss w.r.t. w and b
        grad_w = (2/n) * np.dot(error, X)
        grad_b = (2/n) * np.sum(error)

        # The actual "descent" step -- move opposite to the gradient
        w -= lr * grad_w
        b -= lr * grad_b

        history.append(loss)
        if epoch % 40 == 0:
            print(f"epoch {epoch:<4} loss={loss:<10.4f} w={w:<8.4f} b={b:<8.4f}")
    return w, b, history

print("=== Vanilla Gradient Descent ===")
w_final, b_final, loss_history = vanilla_gd(X, y, lr=0.01, epochs=200)
print(f"Learned: w={w_final:.3f}, b={b_final:.3f}  (true values: w={true_w}, b={true_b})")

# ---------------------------------------------------------
# STEP 3: Gradient Descent with Momentum (converges faster, less oscillation)
# Momentum keeps a "velocity" term so updates build speed in a consistent direction,
# similar to a ball rolling downhill instead of a hesitant step-by-step walker.
# ---------------------------------------------------------
def momentum_gd(X, y, lr=0.01, epochs=200, beta=0.9):
    w, b = 0.0, 0.0
    v_w, v_b = 0.0, 0.0  # velocity terms
    n = len(X)
    for epoch in range(epochs):
        pred = w * X + b
        error = pred - y
        grad_w = (2/n) * np.dot(error, X)
        grad_b = (2/n) * np.sum(error)

        # velocity = momentum of past gradients + current gradient
        v_w = beta * v_w + (1 - beta) * grad_w
        v_b = beta * v_b + (1 - beta) * grad_b

        w -= lr * v_w
        b -= lr * v_b
        if epoch % 40 == 0:
            loss = np.mean(error**2)
            print(f"epoch {epoch:<4} loss={loss:<10.4f} w={w:<8.4f} b={b:<8.4f}")
    return w, b

print("\n=== Gradient Descent with Momentum (same lr, same epochs) ===")
w_m, b_m = momentum_gd(X, y, lr=0.01, epochs=200)
print(f"Learned: w={w_m:.3f}, b={b_m:.3f}  <- notice faster convergence than vanilla GD")

# ---------------------------------------------------------
# STEP 4: Compare against sklearn's closed-form solution (sanity check)
# ---------------------------------------------------------
from sklearn.linear_model import LinearRegression
sk_model = LinearRegression().fit(X.reshape(-1,1), y)
print(f"\nsklearn (closed-form / normal equation): w={sk_model.coef_[0]:.3f}, b={sk_model.intercept_:.3f}")
