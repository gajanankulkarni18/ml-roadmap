# End-to-End ML Infrastructure: A Practical Guide
### Worked example: Credit Card Fraud Detection (tabular, imbalanced classification)
### Built entirely with open-source tools, runnable on a laptop / free-tier compute

---

## 0. How to use this document

This is written for someone prepping for ML Engineer / MLE interviews who wants
both the mental model *and* proof of hands-on execution. Every number in this
document (PR-AUC, latency, PSI values) came from actually running the code in
`fraud-detection-mlops/`, not from a hypothetical. Re-run any script yourself:

```bash
cd fraud-detection-mlops
pip install -r requirements.txt
python src/data_gen.py       # 1. generate synthetic dirty data
python src/sanitize.py       # 2. clean it
python src/train.py          # 3. train baseline + challenger
python src/evaluate.py       # 4. slice-based evaluation
python src/optimize.py       # 5. Optuna tuning + ONNX export + latency bench
python -m uvicorn src.serve:app --reload   # 6. serve it
python src/monitor.py        # 7. drift detection
pytest tests/ -v             # tests
```

The problem we picked — flag fraudulent card transactions in real time — is
a good interview-prep vehicle because it forces you to deal with **every**
hard part of production ML at once: severe class imbalance, a business
metric that isn't accuracy, low-latency serving requirements, adversarial
drift (fraud patterns actively change), and a strict cost asymmetry between
false positives (annoyed customer) and false negatives (fraud loss).

---

## 1. System architecture, end to end

```
┌──────────────┐   ┌───────────────┐   ┌───────────────┐   ┌──────────────────┐
│  Data source  │──▶│  Sanitization  │──▶│Feature engine │──▶│   Training loop   │
│ (raw events)  │   │ (validate,PII, │   │ (features.py, │   │ (train.py: LR     │
│               │   │  dedup, nulls) │   │  shared code) │   │  baseline + XGB)  │
└──────────────┘   └───────────────┘   └───────┬───────┘   └─────────┬─────────┘
                                                 │                     │
                                                 │ SAME feature code   ▼
                                                 │              ┌─────────────┐
                                                 │              │  Evaluation │
                                                 │              │(slice-based)│
                                                 │              └──────┬──────┘
                                                 │                     ▼
                                                 │              ┌─────────────┐
                                                 │              │ Optimization│
                                                 │              │(Optuna+ONNX)│
                                                 │              └──────┬──────┘
                                                 ▼                     ▼
                                          ┌─────────────────────────────────┐
                                          │      Serving (FastAPI+ONNX)      │
                                          │  /predict  /health  /metrics     │
                                          └────────────┬─────────────────────┘
                                                        │ prediction log
                                                        ▼
                                          ┌─────────────────────────────────┐
                                          │   Monitoring (PSI / KS drift)    │
                                          │   → alert / trigger retrain      │
                                          └─────────────────────────────────┘
```

The single design decision that matters most here: **feature-engineering
code lives in one file (`features.py`) and is imported by both the training
script and the serving script.** Training/serving skew — where the offline
pipeline computes a feature slightly differently than the online pipeline —
is one of the most common causes of "the model was great in the notebook,
terrible in prod." Structurally preventing it (shared code, not
re-implemented logic) beats testing for it after the fact.

### Open-source tool map (what we used, and the mainstream alternative at each stage)

| Stage | Tool used here | Also common in industry | Notes |
|---|---|---|---|
| Data validation | hand-rolled + pandas | **Great Expectations**, **Pandera**, **Soda Core** | At real scale, use a schema-as-code validator so checks are declarative and testable, not buried in scripts |
| Experiment tracking | JSON-lines ledger (hand-rolled) | **MLflow**, **Weights & Biases**, **Neptune** | MLflow is the de facto OSS standard: tracking server + model registry + serving in one project |
| Model training | scikit-learn, XGBoost | + **LightGBM**, **CatBoost**; **PyTorch/TensorFlow** for deep learning | For tabular data, gradient-boosted trees usually beat deep learning on accuracy *and* cost |
| Hyperparameter tuning | **Optuna** | **Ray Tune**, **Hyperopt**, scikit-learn `GridSearchCV`/`RandomizedSearchCV` | Optuna's TPE sampler + pruning is more sample-efficient than grid/random search |
| Model optimization/export | **ONNX Runtime** | **TensorRT** (NVIDIA GPU), **OpenVINO** (Intel CPU), **TFLite** (mobile/edge) | ONNX is the portable middle ground: train in any framework, run anywhere |
| Serving framework | **FastAPI** + Uvicorn | **BentoML**, **Ray Serve**, **TorchServe**, **Seldon Core**, **KServe** (K8s-native), **Triton Inference Server** (multi-framework, GPU-optimized) | FastAPI is "build it yourself"; BentoML/Seldon/KServe give you packaging, autoscaling, canary routing out of the box |
| Containerization | Docker | — | Standard |
| Metrics/monitoring | **Prometheus** client + custom PSI/KS | **Evidently AI**, **WhyLabs**, **Arize**, **Fiddler** | Dedicated ML-monitoring tools add automatic drift dashboards, alerting, and feature-attribution drift on top of what we hand-rolled |
| Orchestration (not built here, but expected knowledge) | — | **Airflow**, **Dagster**, **Prefect**, **Kubeflow Pipelines** | Chains data_gen → sanitize → train → evaluate → deploy as a scheduled/triggered DAG |
| Feature store (not built here, at bigger scale) | — | **Feast**, **Tecton** | Solves training/serving skew at organizational scale — a shared, versioned feature registry instead of one shared Python file |

---

## 2. Data: generation, sanitization, processing

### 2.1 The dataset

`data_gen.py` synthesizes 60,480 transactions with a **1.97% fraud rate**
(realistic for card fraud) and deliberately injects the data-quality
problems every real pipeline has to survive:

- **Missing values** in 4 columns (0.5–4% each), some MCAR, some structurally sparse
- **480 exact duplicate rows** (simulating upstream retry/double-send)
- **Sentinel/garbage values**: `amount = -999.0` (buggy upstream default), `distance = 9,999,999 km` (broken GPS sentinel)
- **Inconsistent categorical casing/whitespace** (`"ELECTRONICS  "` vs `"electronics"`)
- **Label noise**: 0.5% of labels deliberately flipped, simulating analyst mislabeling in the fraud-review queue

Fraud isn't driven by one obvious column — it's a mix of interacting signals
(odd hour + high amount + card-not-present + far from home + risky merchant
category), which is what makes the modeling problem realistic instead of
trivially separable.

### 2.2 Sanitization (`sanitize.py`)

Ran on the raw 60,480-row dataset, in order:

1. **Type coercion** — force numeric columns through `pd.to_numeric(errors="coerce")` so silently-corrupted strings become `NaN` instead of crashing downstream code or, worse, being silently cast wrong.
2. **PII masking** — the raw card number is *never* retained past this stage. We keep `sha256(PAN)[:12] + last 4 digits` — enough for fraud-investigation lookups and stable joins, none of the liability of storing a real PAN. This is the same pattern (deterministic pseudonymization, not encryption you'd have to manage keys for) used in real PCI-DSS-scoped systems for anything downstream of the payment processor.
3. **Deduplication** on the business key (`txn_id`), not a blind `drop_duplicates()` on all columns — removed 480 rows.
4. **Categorical normalization** — strip/lowercase, then validate against an allow-list; anything invalid becomes `"unknown"` rather than silently keeping a garbage category the model would treat as meaningful.
5. **Domain-informed outlier handling** — this is the part people get wrong by reaching for a blind IQR/z-score clip:
   - Negative amounts (40 rows) are impossible → set to null, not clipped to 0 (0 is a real value, meaning something different)
   - Distances > 20,000 km (15 rows) are a known GPS-sentinel bug, not a legitimate transaction → set to null
   - Legitimately large-but-real amounts are **winsorized** at the 99.5th percentile (cap ≈ $429) rather than dropped, since whale transactions are still real signal, just shouldn't have unbounded leverage on the loss function
6. **Missing-value imputation** — column-specific median imputation, never a blanket `fillna(0)`, because 0 is meaningful for `amount` and `distance_from_home_km` and would silently bias the fraud signal (fraud tends to happen far from home; imputing 0 would make missing rows look suspiciously "close to home").
7. **Row-level validity gate** — anything still broken after the above (e.g., an hour outside 0–23) gets dropped, not silently kept.

Result: **60,480 → 60,000 clean rows**, with a machine-readable sanitization
report (`artifacts/sanitization_report.json`) logged for auditability — in a
regulated domain like fraud/credit, you need to be able to answer "what did
you do to the data" months later, not just trust that the code did the right
thing.

**Why sanitization code lives in its own module, importable at serving
time too**: `serve.py` calls `mask_card_number()` from `sanitize.py`
directly. A production request never sees different masking logic than
training data did — same reasoning as the feature-code sharing above.

### 2.3 What "data processing" means beyond cleaning (feature engineering, `features.py`)

- `amount_log = log1p(amount)` — transaction amounts are heavily right-skewed; log-transform makes the distribution far more tractable for linear models and slightly helps tree splits too
- `is_night` — binary flag for hours 0–5, since fraud clusters at odd hours (captures a nonlinearity a raw hour feature wouldn't give a linear model)
- `is_high_amount` — flag for being above the 95th percentile, a manually-engineered interaction proxy
- One-hot encoding for categoricals (`merchant_category`, `country`) via `ColumnTransformer` + `OneHotEncoder(handle_unknown="ignore")` — the `handle_unknown` setting matters: a category never seen in training (say, a new country) must not crash inference in production, it should degrade gracefully to "no signal from this feature" instead
- `StandardScaler` on numeric features — required for the logistic regression baseline (scale-sensitive), harmless for XGBoost (scale-invariant), so applying it once for both keeps the preprocessing pipeline uniform

## 3. Creating a model from scratch (`train.py`)

### 3.1 Why two models, not one

We train a **Logistic Regression baseline** and an **XGBoost challenger**
side by side and pick a winner by metric, not by assumption. This is
standard practice: the baseline (a) sanity-checks that the problem is even
learnable with simple linear structure, (b) gives an interpretable
reference point (coefficients you can explain to a fraud-ops team), and (c)
is a floor — if your fancy model can't beat logistic regression, something
is wrong.

### 3.2 Handling severe class imbalance (1.97% positive rate)

Two different mechanisms, one per model family, because they need different fixes:

- **Logistic Regression**: `class_weight="balanced"` — reweights the loss inversely proportional to class frequency, so the rare class isn't drowned out.
- **XGBoost**: `scale_pos_weight = (negatives / positives) ≈ 49.6` — same idea, but XGBoost's own hyperparameter for it, which interacts better with its tree-building objective than post-hoc resampling would.

Alternatives not used here but worth knowing for interviews: **SMOTE**
(synthetic oversampling of the minority class — risky for fraud because
naively interpolating between two fraud examples in feature space can create
unrealistic synthetic points), random undersampling of the majority class
(simpler, throws away real data), and **focal loss** (down-weights
easy-to-classify examples during training, popular in vision, less common
for tabular GBMs). `scale_pos_weight`/`class_weight` are almost always the
right first move for tabular imbalance — reach for SMOTE only if that's not enough.

### 3.3 Train/val/test split — and the leakage trap

We use a stratified 70/15/15 split here (keeps the ~2% fraud rate
consistent across all three sets). **In a real deployment this must be a
chronological split** (train on transactions before date X, validate/test
on transactions after) — fraud patterns evolve over time, and a random
split lets the model implicitly "see the future" (e.g., a fraud ring's
pattern appears in both train and test because rows were shuffled, so the
model gets credit for detecting a pattern it would never have seen yet in
production). This is one of the most common "the offline metric was great,
production was mediocre" bugs in fraud/recommendation/forecasting systems,
and it's worth naming explicitly in an interview even when the toy dataset
doesn't have real timestamps to demonstrate it.

### 3.4 Actual results from this run

| Model | Val PR-AUC | Val ROC-AUC | Precision @ threshold | Recall @ threshold | Train time |
|---|---|---|---|---|---|
| Logistic Regression (baseline) | 0.7111 | 0.8579 | 0.992 | 0.657 | 0.17s |
| XGBoost (untuned challenger) | **0.7288** | 0.8544 | 0.920 | 0.708 | 1.37s |

**Why PR-AUC, not ROC-AUC, decides the winner**: with a 98/2 class split,
ROC-AUC can look deceptively good even for a mediocre model, because the
false-positive rate denominator (all the easy true negatives) is huge.
PR-AUC is sensitive to exactly the thing that's hard here — precision among
the positive predictions — and is the standard metric for severe imbalance
(fraud, rare disease detection, intrusion detection all use it for this
reason).

**Threshold selection**: rather than the default 0.5 cutoff, we scan the
full precision-recall curve and pick the threshold that maximizes F1 on
validation data (`best_threshold()` in `train.py`). In a real system this
threshold is a *business decision*, not just a statistical one — fraud ops
would set it based on the $ cost of a false negative (fraud loss) vs a false
positive (customer friction / support cost), which is usually asymmetric
and not captured by F1 at all. F1 is a reasonable default when you don't
have those cost numbers yet.

**Held-out test set (touched exactly once, after model selection)**:

```
model:                    xgboost_challenger
PR-AUC:                   0.7373
ROC-AUC:                  0.8765
precision @ threshold:    0.9685
recall @ threshold:       0.6949
false positives:          4   (out of ~9,000 test transactions)
false negatives:          54
```

XGBoost wins and becomes the champion model.

### 3.5 Experiment tracking

Every run — both models, both val and test evaluations — is appended to
`artifacts/experiment_ledger.jsonl` with the model name, all metrics, a
hash of the exact data file used, and timing. This is a minimal stand-in
for **MLflow**: same principle (every run is reproducible and comparable
later), fewer moving parts. At team scale you'd want the real thing — MLflow
gives you a UI, a model registry with staged promotion (`staging` →
`production`), and artifact storage, instead of grepping a JSON-lines file.
## 4. Deployment: how, and which strategy

### 4.1 Serving pattern chosen: online, synchronous, REST

Fraud detection is a hard-latency-budget, hard-real-time problem — the
prediction has to come back before the payment authorization completes
(typically a budget of tens of milliseconds end-to-end, of which the model
gets a slice). That rules out batch scoring and points straight at a
**synchronous online model server**, which is what `serve.py` (FastAPI +
ONNX Runtime) implements.

Contrast with other serving patterns you should know for an interview:

| Pattern | When to use it | Example |
|---|---|---|
| **Batch scoring** | Predictions consumed later, not on the critical path of a user action | Nightly churn-risk scores written to a data warehouse |
| **Online synchronous (request/response)** | Prediction needed immediately to complete a user-facing action | Fraud check during checkout, ad ranking |
| **Online asynchronous (streaming/event-driven)** | Prediction feeds a downstream process, latency budget is looser (seconds, not ms) | Content moderation on an uploaded video, feature computation pipelines on Kafka |
| **Embedded/on-device** | No network round-trip acceptable, or offline capability required | Keyboard next-word prediction, mobile fraud pre-screen |

### 4.2 Model-serving *runtime* choice: why ONNX Runtime here, and the landscape

| Runtime | Best fit | Tradeoff |
|---|---|---|
| **ONNX Runtime** (used here) | CPU-bound tabular/classical-ML serving, cross-framework portability | Not the fastest possible option for any single framework, but works everywhere and has zero vendor lock-in |
| **TensorRT** | NVIDIA GPU inference, deep learning | Fastest on NVIDIA hardware, but GPU-only and NVIDIA-only |
| **Triton Inference Server** | Serving many models/frameworks behind one server, GPU fleets, dynamic batching | Heavier operational footprint; overkill for a single tabular model |
| **TorchServe / TF Serving** | Single-framework deep learning shops | Simpler if you're 100% PyTorch or 100% TF; less useful if you mix frameworks |
| **vLLM / TGI** | LLM-specific serving (continuous batching, paged KV-cache) | Not relevant here — this is for autoregressive generation, not classification, but it's the answer if the interview question shifts to "how would you serve an LLM" |

**Why ONNX Runtime specifically was worth it here, with real numbers**: we
benchmarked native XGBoost `.predict_proba()` against the ONNX-exported,
ONNX-Runtime-executed version, single-row inference, CPU, 200 reps:

```
Native XGBoost:   p50 = 0.200 ms   p95 = 0.358 ms   p99 = 0.445 ms
ONNX Runtime:     p50 = 0.010 ms   p95 = 0.011 ms   p99 = 0.028 ms
Speedup:          ~20x at p50
Parity check:     max |native_prob - onnx_prob| = 0.000000  (exact match)
```

The ~20x gap comes from ONNX Runtime's graph-level optimizations
(operator fusion, a lighter-weight execution path than XGBoost's Python/C++
`predict_proba` call for single-row requests) — this matters a lot at fraud
scale, where you're paying this cost per-transaction, continuously, forever.
The parity check is not optional: **converting a model to a faster runtime
and never verifying the outputs match is how you silently ship a
numerically-different model** — always confirm before trusting a benchmark
number.

### 4.3 Release strategy tradeoffs (how a new model version reaches production)

This is a different axis from "which runtime" — it's about *how you roll
out a new model version safely*.

| Strategy | How it works | Pros | Cons | Typical use |
|---|---|---|---|---|
| **Shadow deployment** | New model runs alongside the old one on live traffic, predictions logged but never acted on | Zero user-facing risk, real production data for comparison | Doesn't tell you actual business-metric impact (no real actions taken); doubles inference compute cost | First step for any high-stakes model (fraud, credit, medical) |
| **Canary release** | New model serves a small % of real traffic (e.g. 5%), ramped up gradually if metrics hold | Real impact measurement, blast radius limited, can auto-rollback on regression | Needs solid real-time monitoring to catch regressions fast; some users get the "wrong" model during the test | Most common pattern at mature ML orgs today |
| **Blue-green deployment** | Two full environments; traffic switches all-at-once from old to new | Instant rollback (just flip back), simple mental model | All-or-nothing — no gradual signal, a bad model hits 100% of traffic before you notice | Good for lower-stakes models or when canary infra isn't available |
| **A/B test** | Traffic split deterministically (often by user ID) between two versions, compared statistically | Rigorous, gives a real causal read on business metrics, not just model metrics | Slower — needs enough traffic/time for statistical power; more of a product-analytics process than a deployment mechanism | When the question is "does this actually move the business metric," not just "is it safe" |
| **Multi-armed bandit** | Traffic allocation to variants adjusts dynamically based on observed performance | More sample-efficient than a fixed A/B split — bad variants get less traffic, faster | More complex to implement and reason about than a static split | High-traffic ranking/recommendation systems where regret matters |

**What's actually used in industry right now, and why**: for anything
risk-sensitive (fraud, credit decisioning, medical, safety-relevant ranking),
the dominant real-world pattern is **shadow deployment first, then canary
with automatic rollback on a monitored metric** — not a straight blue-green
flip. Concretely: run the new model in shadow for some period to confirm no
crashes / latency regressions / wildly different score distributions vs the
old model, then canary at 1% → 5% → 25% → 100%, gated by drift/error metrics
at each step, with an automated rollback if error rate or a business KPI
regresses. Feature-flagging and progressive-delivery tools (e.g. **Flagger**
on Kubernetes, or a service mesh like **Istio**'s traffic splitting) are the
common OSS building blocks for the canary-with-auto-rollback mechanics; the
K8s-native **KServe**/**Seldon Core** serving frameworks build this pattern
in as a first-class "canary rollout" resource rather than something you
script by hand.

For this project's scope (single-box FastAPI service), we implement the
*model* side of this correctly — a versioned `model_version` field on every
prediction response, so a real gateway/mesh in front of it could route
traffic between two running instances (old vs new model) and you'd have the
version tagged on every logged prediction for later comparison. The
traffic-splitting infrastructure itself (Istio/Flagger/KServe) is out of
scope for a laptop demo but is exactly what you'd stand up next.

### 4.4 What's in `Dockerfile` and why

- **Multi-stage build**: dependencies used only for training/tuning (`optuna`,
  `onnxmltools`, `skl2onnx`) never make it into the runtime image — smaller
  image, smaller attack surface, faster cold starts.
- **Only the ONNX model + fitted preprocessor + threshold config are
  copied in** — not the training data, not the raw `.joblib` sklearn model,
  not the notebooks/scripts used to produce it. The serving image is a
  read-only consumer of trained artifacts, never a place training happens.
- **Non-root user** — baseline container hardening; a compromised process
  in the container shouldn't have root inside it.
- **`HEALTHCHECK`** hitting `/health` — lets an orchestrator (Kubernetes
  liveness/readiness probes, or plain Docker) detect and restart a hung
  container automatically.
## 5. Model evaluation — how it actually happened here

Evaluation ran at three levels of granularity, deliberately, because each
catches a different failure mode:

### 5.1 Aggregate metrics (the headline number)
PR-AUC / ROC-AUC / precision / recall at a chosen threshold, on a held-out
test set touched exactly once (see §3.4). This is the number you'd put in a
model card, but it's necessary, not sufficient — it can hide serious
subgroup failures.

### 5.2 Slice-based evaluation (`evaluate.py`) — the number that actually caught something

We recomputed precision/recall broken out by `merchant_category`,
`amount_bucket`, `card_present`, and `country`, and flagged any slice with
≥5 fraud cases and recall below 0.5. **This found real problems the
aggregate metric hid:**

```
merchant_category=grocery:      n_fraud=15   precision=0.667   recall=0.133  ⚠
merchant_category=restaurant:   n_fraud=10   precision=0.667   recall=0.400  ⚠
merchant_category=utilities:    n_fraud=12   precision=1.000   recall=0.333  ⚠
country=US:                     n_fraud=74   precision=0.865   recall=0.432  ⚠
country=NG:                     n_fraud=12   precision=1.000   recall=1.000
country=RU:                     n_fraud=20   precision=0.905   recall=0.950
```

The story this tells: the model is very good at catching the "obvious"
fraud pattern (foreign country + odd hour + high amount) but is missing
most fraud in low-risk-looking categories and in-country (US) transactions
— exactly the fraud that doesn't match the stereotyped pattern, which is
also exactly the fraud a smart fraudster would try to produce. In a real
job, this result would drive next steps: more training examples of
domestic/low-amount fraud, additional features (velocity/frequency features
per customer, device fingerprinting), or a second specialized model for
this slice. **This is the single most interview-relevant point in the whole
project**: a model that looks great in aggregate and is quietly failing on
a subpopulation is one of the most common real-world ML bugs, and "did you
check slices, or just the headline metric" is a very common senior-level
interview question.

### 5.3 Sanity/parity checks before trusting anything downstream
Before believing the ONNX latency win, we checked numerical parity between
native and ONNX outputs (`max |diff| = 0.000000`, see §4.2). Before
believing the drift monitor works, we validated it against synthetic data
with a *known* injected shift (§7) and confirmed it fired on the right
feature and stayed quiet on the unshifted ones. Trusting a number without a
sanity check next to it is a common way teams ship silent regressions.

---

## 6. Optimization strategies for ML models

Two genuinely different kinds of "optimization," worth keeping conceptually
separate (interviewers will sometimes deliberately conflate them to see if
you do too):

### 6.1 Model-quality optimization — hyperparameter tuning
`optimize.py` runs a 30-trial **Optuna** search (Tree-structured Parzen
Estimator sampler, more sample-efficient than grid or random search because
it models which regions of the hyperparameter space look promising and
concentrates trials there) over `n_estimators`, `max_depth`, `learning_rate`,
`subsample`, `colsample_bytree`, `min_child_weight`, `reg_lambda`.

```
Best validation PR-AUC found: 0.7376
Untuned baseline test PR-AUC: 0.7373
Tuned test PR-AUC:            0.7488   (+1.5% relative improvement)
```

A modest but real gain — worth calling out honestly rather than overselling
it: on a well-specified feature set, hyperparameter tuning usually buys you
single-digit-percent gains, not step-changes. The bigger wins in this
project came from feature engineering and imbalance handling, which is
typical — tuning is real but marginal compared to getting the problem
formulation and data right.

Other tuning approaches worth naming: **grid search** (exhaustive,
expensive, fine for small spaces), **random search** (surprisingly
competitive baseline, cheap), **Bayesian optimization** (what Optuna's TPE
approximates), **Hyperband/ASHA** (aggressively kills unpromising trials
early using partial training curves — big wins for deep learning where a
single trial is expensive; less relevant for a model that trains in
seconds like our XGBoost here), **population-based training** (evolves
hyperparameters *during* training, common for RL/deep learning).

### 6.2 Serving-efficiency optimization — the ONNX conversion
Covered in depth in §4.2: model-quality was already "good enough" going in,
so the optimization that mattered *operationally* was inference latency, not
another 1% of PR-AUC. This is the point worth making in an interview: ask
"what's actually the bottleneck — model quality or serving cost" before
picking which kind of optimization to spend time on. For fraud specifically
(runs on every transaction, forever, on a tight latency budget) serving cost
dominates total cost of ownership; for something like an internal analytics
model scored weekly in batch, it wouldn't matter at all.

Other serving-optimization techniques worth knowing beyond ONNX export:
- **Quantization** (float32 → int8) — big wins for deep learning, smaller
  effect for tree ensembles like XGBoost since the compute pattern is
  different (branching, not matrix multiplication)
- **Pruning / distillation** — train a smaller model to mimic a larger one's
  outputs; relevant when the large model's own quality gain is real but its
  serving cost isn't justified
- **Batching** — for GPU-bound deep learning serving, batching requests
  together dramatically improves throughput; less relevant for a CPU tree
  model handling one transaction at a time under a strict latency SLA (there
  batching would *add* latency, the opposite of what you want)
- **Feature-computation caching** — for features that don't change
  request-to-request (e.g. a customer's rolling 30-day spend average),
  precompute and cache in a low-latency store (Redis) rather than
  recomputing per request

---

## 7. Monitoring and debugging in production (`monitor.py`)

### 7.1 What we monitor and why two tests, not one

Every prediction served is logged (`artifacts/prediction_log.jsonl`) with
the input features and the output score. `monitor.py` compares a reference
window (the training-time validation distribution) against a live window
using two complementary statistical tests:

- **PSI (Population Stability Index)** — the industry-standard bucketed
  metric. Thresholds follow common convention: **< 0.1 stable, 0.1–0.25
  investigate, > 0.25 significant drift**. Cheap, interpretable, works for
  both input features and the model's own output-score distribution
  (score drift can indicate a problem even when no single input feature has
  drifted much).
- **KS-test (Kolmogorov-Smirnov)** — a nonparametric test for whether two
  samples come from the same distribution. More sensitive to shape changes
  that PSI's fixed bucketing can smooth over; gives a p-value, so it's a
  genuine statistical test rather than a heuristic threshold.

### 7.2 Validating the monitor actually works, on a known injected shift
To prove the monitor catches real drift (not just print plausible-looking
numbers), we simulated a live traffic window with a **deliberate, known**
shift: transaction amounts inflated ~35%, and 30% of transactions
reassigned to `online_retail` category (simulating, e.g., a genuine
COVID-era shift toward online shopping). Actual output:

```
amount:                    PSI=0.096  (stable)   KS p=0.000 (statistically significant, but PSI says not practically significant)
distance_from_home_km:     PSI=0.012  (stable)
time_since_last_txn_min:   PSI=0.004  (stable)
y_prob (model output):     PSI=0.006  (stable)
merchant_category:         PSI=0.442  (SIGNIFICANT DRIFT)  ← correctly caught the injected shift

⚠ retrain_recommended: True, drifted_features: ['merchant_category']
```

This result is worth understanding, not just quoting: the **KS-test flagged
`amount` as statistically significant (p≈0) even though PSI called it
stable** — with a large sample size, KS will detect even a small, practically
irrelevant shift as "significant" (that's a known property of the test,
not a bug). This is exactly why we run both: PSI's threshold buckets are a
better proxy for "does this matter enough to act on," while KS is more
sensitive to real distributional shape changes; using them together
catches genuine drift (merchant_category) without over-alerting on
sample-size-driven statistical noise (amount).

### 7.3 Debugging workflow when something looks wrong in production
The tools built into this project support the standard debugging sequence:
1. **Is it the model or the data?** — check `drift_report.json` first; if
   input features have drifted, the model degrading is expected, not a bug.
2. **Is it uniform or concentrated?** — rerun `evaluate.py`'s slice
   breakdown on recent labeled data (once ground truth arrives, which for
   fraud is delayed — chargebacks take days/weeks to materialize, a real
   operational constraint worth naming) to see if the regression is
   isolated to a slice.
3. **Is it the serving path, not the model?** — `/metrics` exposes
   `fraud_predict_latency_seconds` and `fraud_predict_errors_total`; a
   latency spike or error-rate jump with no drift signal points at infra
   (a dependency timeout, a resource-starved container), not the model.
4. **Can you reproduce it offline?** — because sanitization and feature code
   are shared modules (not reimplemented), you can pull the exact
   production request, run it through the same `sanitize.py`/`features.py`
   functions offline, and get an identical result to debug against — no
   "well it works when I test it" gap between environments.

### 7.4 What a fuller production monitoring stack adds beyond this
- **Automatic dashboards + alerting** (Prometheus metrics from `serve.py`
  would typically feed **Grafana** for dashboards and **Alertmanager** for
  paging on-call) rather than a script you run manually
- **Label-delay-aware evaluation** — fraud ground truth arrives late
  (chargebacks), so production evaluation pipelines need a delayed-join
  step that revisits predictions once labels land, not just live drift
  proxies
- **Feature-attribution drift** (e.g. via SHAP value distributions, which
  tools like **Evidently AI** or **Arize** compute automatically) — tells
  you not just "the input distribution shifted" but "the shift is actually
  changing which features drive the model's decisions," a stronger signal
  for whether retraining is actually warranted
- **Automated retraining triggers** wired to the drift signal, feeding back
  into the orchestration DAG (Airflow/Dagster) mentioned in §1, rather than
  a human running `python monitor.py` and deciding by hand

---

## 8. What to say in an interview (summary you can actually use)

If asked "walk me through how you'd take a model from idea to production":

1. **Frame the problem and pick a metric before touching data** — for
   imbalanced classification, that's PR-AUC over accuracy, and a
   business-informed operating threshold, not a default 0.5 cutoff.
2. **Sanitize with explicit, domain-informed rules**, never blanket
   fillna/dropna — and share that code between offline and online paths.
3. **Baseline before fancy** — a simple model tells you if the problem is
   even learnable and gives you an interpretable floor.
4. **Split chronologically if there's any time dimension** — random splits
   leak the future into training for anything with temporal structure.
5. **Evaluate in aggregate AND by slice** — the aggregate number can hide a
   subpopulation the model is quietly failing on; that's usually the more
   interesting finding.
6. **Separate model-quality optimization from serving-efficiency
   optimization** — they're different problems with different tools
   (Optuna vs ONNX/quantization), and you should know which one is
   actually your bottleneck before spending time on either.
7. **Choose a release strategy proportional to the stakes** — shadow →
   canary with auto-rollback for anything high-risk; a straight blue-green
   flip is fine for lower-stakes models.
8. **Monitor for drift with more than one statistical test**, and validate
   the monitor itself against a known-shifted sample before trusting it in
   production.
9. **Always sanity-check a benchmark or optimization claim** (numeric
   parity, known-answer tests) before believing the headline number.
