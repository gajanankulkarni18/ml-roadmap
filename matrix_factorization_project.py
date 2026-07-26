"""
PROJECT: Mini Recommender System via Matrix Factorization (SVD)
Concept demonstrated: dot products, matrices, SVD, eigenvectors-adjacent decomposition
Runs fully offline with synthetic data (swap in MovieLens 100k for a real dataset).
"""
import numpy as np

# ---------------------------------------------------------
# STEP 1: Build a synthetic user-item ratings matrix
# Rows = users, Columns = movies. 0 = "not yet rated" (sparse, like real data)
# ---------------------------------------------------------
np.random.seed(1)
n_users, n_movies = 8, 6

# Ground truth: users and movies each have a hidden 2D "taste" vector.
# A rating is generated as (roughly) the dot product of user-taste and movie-taste.
true_user_vecs = np.random.uniform(0, 2, size=(n_users, 2))
true_movie_vecs = np.random.uniform(0, 2, size=(n_movies, 2))
full_ratings = true_user_vecs @ true_movie_vecs.T  # matrix multiply = all predicted ratings

# Simulate sparsity: only ~60% of ratings are actually observed
mask = np.random.rand(n_users, n_movies) < 0.6
observed = np.where(mask, np.round(full_ratings, 1), 0)

print("Observed (sparse) ratings matrix (0 = unrated):")
print(observed)

# ---------------------------------------------------------
# STEP 2: Matrix factorization via SVD to recover latent taste vectors
# We factor the matrix into U (user factors), S (importance), Vt (movie factors)
# This is the same math that powered the Netflix Prize-winning algorithm.
# ---------------------------------------------------------
U, S, Vt = np.linalg.svd(observed, full_matrices=False)

k = 2  # keep only the top-2 latent dimensions (this is the dimensionality reduction step)
U_k = U[:, :k]
S_k = np.diag(S[:k])
Vt_k = Vt[:k, :]

# Reconstruct an approximation of the FULL ratings matrix (including unrated cells)
# This reconstruction is a matrix multiplication of the three decomposed pieces.
reconstructed = U_k @ S_k @ Vt_k

print("\nReconstructed (predicted) ratings matrix, including previously-unrated cells:")
print(np.round(reconstructed, 2))

# ---------------------------------------------------------
# STEP 3: Use dot products to recommend the best unrated movie for a user
# ---------------------------------------------------------
user_id = 0
user_taste_vector = U_k[user_id] @ S_k  # this user's learned taste in latent space

print(f"\nUser {user_id}'s learned taste vector: {np.round(user_taste_vector, 3)}")

unrated_movies = np.where(observed[user_id] == 0)[0]
scores = {}
for movie_id in unrated_movies:
    movie_vector = Vt_k[:, movie_id]
    # Dot product = how well this movie's latent profile aligns with the user's taste
    score = np.dot(user_taste_vector, movie_vector)
    scores[movie_id] = score

best_movie = max(scores, key=scores.get)
print(f"Unrated movies and predicted scores: { {m: round(s,2) for m,s in scores.items()} }")
print(f"--> Recommend movie #{best_movie} to user {user_id} (highest predicted score)")

# ---------------------------------------------------------
# STEP 4: Sanity check against sklearn's TruncatedSVD (library version)
# ---------------------------------------------------------
from sklearn.decomposition import TruncatedSVD
svd_model = TruncatedSVD(n_components=2, random_state=1)
user_factors_sklearn = svd_model.fit_transform(observed)
print(f"\nsklearn TruncatedSVD user factors (should match our by-hand version up to sign/scale):")
print(np.round(user_factors_sklearn, 3))
