"""
PROJECT: Naive Bayes Spam Classifier From Scratch
Concept demonstrated: Bayes' Theorem, MLE (word-frequency probabilities), log-probabilities
Runs fully offline with a small synthetic email dataset.
"""
import numpy as np
from collections import Counter

# ---------------------------------------------------------
# STEP 1: Tiny labeled dataset (in real life: SMS Spam Collection / Enron dataset)
# ---------------------------------------------------------
emails = [
    ("free money now claim prize", "spam"),
    ("win free lottery cash prize", "spam"),
    ("free viagra cheap pills buy now", "spam"),
    ("claim your free prize today", "spam"),
    ("meeting scheduled for tomorrow at noon", "ham"),
    ("please review the attached document", "ham"),
    ("lunch with the team on friday", "ham"),
    ("project deadline moved to next week", "ham"),
]

# ---------------------------------------------------------
# STEP 2: Train -- this is literally computing MLE probabilities from word counts
# P(word | spam) = count(word in spam emails) / total words in spam emails
# ---------------------------------------------------------
def train_naive_bayes(emails):
    spam_words, ham_words = [], []
    for text, label in emails:
        (spam_words if label == "spam" else ham_words).extend(text.split())

    spam_counts = Counter(spam_words)
    ham_counts = Counter(ham_words)
    vocab = set(spam_words + ham_words)

    n_spam_total = len(spam_words)
    n_ham_total = len(ham_words)
    n_spam_docs = sum(1 for _, l in emails if l == "spam")
    n_ham_docs = sum(1 for _, l in emails if l == "ham")

    # Priors: P(spam), P(ham) -- how common is each class overall
    prior_spam = n_spam_docs / len(emails)
    prior_ham = n_ham_docs / len(emails)

    return {
        "spam_counts": spam_counts, "ham_counts": ham_counts, "vocab": vocab,
        "n_spam_total": n_spam_total, "n_ham_total": n_ham_total,
        "prior_spam": prior_spam, "prior_ham": prior_ham,
    }

model = train_naive_bayes(emails)
print(f"Prior P(spam) = {model['prior_spam']:.3f}, Prior P(ham) = {model['prior_ham']:.3f}")

# ---------------------------------------------------------
# STEP 3: Predict -- apply Bayes' theorem using log-probabilities
# (log-space avoids numerical underflow when multiplying many small probabilities)
# Laplace smoothing (+1) handles words never seen in a class -> avoids P(word)=0
# ---------------------------------------------------------
def predict(text, model):
    vocab_size = len(model["vocab"])
    log_prob_spam = np.log(model["prior_spam"])
    log_prob_ham = np.log(model["prior_ham"])

    for word in text.split():
        # P(word|spam) with Laplace (+1) smoothing
        p_word_given_spam = (model["spam_counts"].get(word, 0) + 1) / (model["n_spam_total"] + vocab_size)
        p_word_given_ham = (model["ham_counts"].get(word, 0) + 1) / (model["n_ham_total"] + vocab_size)
        log_prob_spam += np.log(p_word_given_spam)
        log_prob_ham += np.log(p_word_given_ham)

    # Normalize back to actual probabilities for interpretability
    max_log = max(log_prob_spam, log_prob_ham)
    prob_spam = np.exp(log_prob_spam - max_log)
    prob_ham = np.exp(log_prob_ham - max_log)
    prob_spam_norm = prob_spam / (prob_spam + prob_ham)

    return "spam" if log_prob_spam > log_prob_ham else "ham", prob_spam_norm

test_emails = [
    "free prize claim now",
    "team meeting agenda for tomorrow",
    "cheap pills for sale",
]
print("\n=== Predictions on new emails ===")
for email in test_emails:
    label, prob = predict(email, model)
    print(f"'{email}' -> {label}  (P(spam)={prob:.4f})")

# ---------------------------------------------------------
# STEP 4: Sanity check against sklearn's MultinomialNB
# ---------------------------------------------------------
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

texts, labels = zip(*emails)
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(texts)
sk_model = MultinomialNB().fit(X_train, labels)

X_test = vectorizer.transform(test_emails)
sk_preds = sk_model.predict(X_test)
print("\n=== sklearn MultinomialNB predictions (should match by-hand version) ===")
for email, pred in zip(test_emails, sk_preds):
    print(f"'{email}' -> {pred}")
