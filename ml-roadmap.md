# Machine Learning Roadmap — Staff SWE Track

Built around your existing prep: MAANG-style Staff SWE interviews, hands-on codelabs with open-source tools, and your in-progress "will it rain tomorrow" classifier (scikit-learn/XGBoost → Gradio on HF Spaces → Streamlit consumer app).

This isn't a "learn ML from zero" course — it assumes strong SWE fundamentals and optimizes for **system-design depth + real deployment experience**, since that's what Staff-level ML/infra interviews actually probe.

---

## Chunk 0: Math & ML Foundations Refresh (3–5 days)
Goal: enough intuition to reason about model behavior, not derive proofs.
- Linear algebra: vectors, matrices, dot products, eigenvalues (why PCA/SVD work)
- Probability & stats: distributions, Bayes' theorem, MLE, bias-variance tradeoff
- Optimization: gradient descent, convexity, loss landscapes, learning rate intuition
- Core vocabulary: overfitting/underfitting, regularization (L1/L2), train/val/test splits, cross-validation

**Resource pattern:** 3Blue1Brown (linear algebra + neural nets series) + StatQuest for intuition, skip textbook derivations unless something breaks your mental model.

---

## Chunk 1: Classical ML — Build Everything By Hand (1–2 weeks)
You've already started this with your rain classifier. Extend it deliberately:
- Linear/logistic regression from scratch (numpy only, no sklearn) — understand what `.fit()` is actually doing
- Decision trees → Random Forest → Gradient Boosting (XGBoost/LightGBM) — know *why* boosting beats bagging on tabular data
- Feature engineering: encoding, scaling, handling missing data, leakage traps (a classic Staff-level "what's wrong with this pipeline" question)
- Evaluation: precision/recall/F1, ROC-AUC, calibration, and *when accuracy lies to you*

**Tie-in:** Extend your rain classifier — add a from-scratch logistic regression baseline, compare against your XGBoost model, and write up why boosting wins on this dataset.

---

## Chunk 2: Deep Learning Fundamentals (2 weeks)
- Feedforward nets, backprop intuition (not full derivation)
- CNNs — just enough for image use cases if relevant to your target teams
- RNNs/LSTMs — mostly historical context now, but interviewers still reference them
- Transformers — spend real time here: attention mechanism, positional encoding, encoder/decoder vs decoder-only
- Framework: PyTorch (industry default) — train one model end-to-end by hand, no copy-paste tutorials

**Project:** Fine-tune a small pretrained model (e.g., DistilBERT) on a text classification task, log experiments with Weights & Biases or MLflow.

---

## Chunk 3: LLMs & Modern NLP Stack (2 weeks)
This is where your prep should get MAANG-specific.
- Tokenization, embeddings, context windows
- Pretraining vs fine-tuning vs RLHF vs instruction tuning (know the differences cold)
- Parameter-efficient fine-tuning: LoRA/QLoRA — do this hands-on, it's cheap on free-tier compute
- Retrieval-Augmented Generation (RAG): vector DBs (FAISS/Chroma), chunking strategies, retrieval quality tradeoffs
- Open-source stack: Hugging Face Transformers, `peft`, `bitsandbytes`, LangChain/LlamaIndex (know when *not* to use them too)

**Project:** Build a RAG app over a document set, deploy it — this is a very common "build something real" interview take-home pattern.

---

## Chunk 4: MLOps & Deployment (1–2 weeks)
You're already touching this via Gradio/HF Spaces/Streamlit — go deeper into the production concerns interviewers care about:
- Model serving patterns: batch vs online inference, latency/throughput tradeoffs
- Containerization (Docker) and basic orchestration concepts
- Monitoring: data drift, model drift, feature store concepts
- CI/CD for ML: versioning data, models, and code (DVC, MLflow model registry)
- Cost/scale tradeoffs: quantization, distillation, caching strategies for LLM serving

**Tie-in:** Add monitoring/logging to your rain classifier's Streamlit app — track prediction distribution over time, simulate drift.

---

## Chunk 5: ML System Design (Interview-Focused) (2–3 weeks, ongoing)
This is the highest-leverage chunk for Staff interviews specifically.
- Practice the standard framework: requirements → data → features → model choice → training → serving → monitoring → iteration
- Classic problems to work through end-to-end: recommendation system, fraud detection, search ranking, feed ranking, ad click prediction, LLM-powered feature (e.g., search summarization)
- For each: be ready to discuss offline vs online metrics, A/B testing design, feedback loops, and failure modes at scale
- Study how real MAANG systems are architected (engineering blogs from Meta, Netflix, Uber, Google are gold here)

**Practice format:** Do 1–2 mock ML system design problems per week, timeboxed to 45 min, then compare your approach against a published case study.

---

## Suggested Pacing
| Week | Focus |
|---|---|
| 1 | Chunk 0 + start Chunk 1 |
| 2–3 | Finish Chunk 1, extend rain classifier project |
| 4–5 | Chunk 2 (deep learning + PyTorch project) |
| 6–7 | Chunk 3 (LLM/RAG project) |
| 8–9 | Chunk 4 (deployment/MLOps, wire into existing app) |
| 10+ | Chunk 5, ongoing mock system design practice alongside interview scheduling |

Adjust pace to how much time/week you actually have — the LLM and system design chunks (3 & 5) are the ones worth over-indexing on given your target role.
