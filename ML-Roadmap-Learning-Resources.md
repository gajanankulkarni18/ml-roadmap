# Learning Resource Map — Companion to the 6-Month ML Roadmap

> Use alongside `ML-Expert-6-Month-Roadmap.md`. Every link below is free to access. Where a topic has one clearly-best resource, that's listed first; I've kept it to 2-4 links per week (one primary "watch/read this end-to-end," plus supporting references) rather than link-dumping — more links isn't more learning at this stage.

---

## Phase 1 — Math Foundations

### Week 1 — Linear Algebra
- **Primary (video):** 3Blue1Brown, *Essence of Linear Algebra* (full playlist, ~3 hrs total) — https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab
- **Primary (text):** *Mathematics for Machine Learning* — free full PDF — https://mml-book.github.io/book/mml-book.pdf (Ch. 2–4 cover linear algebra, analytic geometry, matrix decompositions specifically)
- **Course context (optional, deeper):** MIT OCW 18.06 Linear Algebra, Gilbert Strang — https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/
- **Reference docs:** NumPy linear algebra API — https://numpy.org/doc/stable/reference/routines.linalg.html

### Week 2 — Calculus & Optimization / Autograd
- **Primary (video):** Andrej Karpathy, *The spelled-out intro to neural networks and backpropagation: building micrograd* — https://www.youtube.com/watch?v=VMj-3S1tku0
- **Code to follow along:** micrograd repo — https://github.com/karpathy/micrograd
- **Supplementary (video):** 3Blue1Brown, *Essence of Calculus* — https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr
- **Reference (optimizer theory):** *An overview of gradient descent optimization algorithms* (Sebastian Ruder, free blog, covers SGD/Momentum/Adam derivations) — https://www.ruder.io/optimizing-gradient-descent/

### Week 3 — Probability, Statistics & Information Theory
- **Primary (video series):** StatQuest with Josh Starmer — https://www.youtube.com/c/joshstarmer (search within-channel for: "Bayes' Theorem," "Maximum Likelihood," "Cross Entropy," "KL Divergence," "p-values")
- **Text reference:** *Mathematics for Machine Learning*, Ch. 6 (Probability and Distributions) — same PDF as Week 1, https://mml-book.github.io/book/mml-book.pdf
- **Applied stats tool docs:** SciPy stats module — https://docs.scipy.org/doc/scipy/reference/stats.html ; statsmodels — https://www.statsmodels.org/stable/index.html

---

## Phase 2 — Classical ML Mastery

### Week 4 — Regression, Trees, Ensembles From Scratch
- **Primary (video):** StatQuest playlists on "Decision Trees," "Gradient Boost (Parts 1-4)," "XGBoost (Parts 1-4)" — https://www.youtube.com/c/joshstarmer
- **Reference implementation to compare against:** scikit-learn User Guide — Decision Trees — https://scikit-learn.org/stable/modules/tree.html ; XGBoost docs — https://xgboost.readthedocs.io/en/stable/ ; LightGBM docs — https://lightgbm.readthedocs.io/en/latest/

### Week 5 — Unsupervised Learning & Dimensionality Reduction
- **Primary (video):** StatQuest — "K-means clustering," "Hierarchical Clustering," "t-SNE, Clearly Explained" — https://www.youtube.com/c/joshstarmer
- **UMAP explainer (interactive blog):** *Understanding UMAP* — https://pair-code.github.io/understanding-umap/
- **Tool docs:** scikit-learn clustering guide — https://scikit-learn.org/stable/modules/clustering.html ; UMAP docs — https://umap-learn.readthedocs.io/en/latest/ ; HDBSCAN docs — https://hdbscan.readthedocs.io/en/latest/

### Week 6 — Model Evaluation & Experimentation Rigor
- **Primary (video):** StatQuest — "ROC and AUC," "Cross Validation," "Confidence Intervals" — https://www.youtube.com/c/joshstarmer
- **Causal inference intro:** DoWhy documentation + tutorial notebooks (official, hands-on) — https://www.pywhy.org/dowhy/main/getting_started/index.html
- **Reference paper (off-policy eval, ties to Netflix-style bandits from your interview-prep doc):** *Off-Policy Evaluation for Slate Recommendation* / Covington et al.'s YouTube paper referenced generally — search "counterfactual learning to rank" survey on arXiv for a fuller treatment if you want to go deeper: https://arxiv.org/abs/2104.00214 (Learning to Rank meets counterfactual/off-policy evaluation, one representative survey)

### Week 7 — Feature Engineering & Data-Centric ML
- **Primary (docs + tutorial):** Feast — official Quickstart — https://docs.feast.dev/getting-started/quickstart
- **Data leakage explainer (blog):** *Data Leakage in Machine Learning* — Machine Learning Mastery — https://machinelearningmastery.com/data-leakage-machine-learning/
- **Tool docs:** feature-engine — https://feature-engine.trainindata.com/en/latest/ ; imbalanced-learn (SMOTE etc.) — https://imbalanced-learn.org/stable/

---

## Phase 3 — Deep Learning Foundations & Core Architectures

### Week 8 — Neural Networks From Scratch → PyTorch
- **Primary (video):** continue Karpathy's Zero to Hero — *The spelled-out intro to language modeling: building makemore* — https://www.youtube.com/watch?v=PaCmpygFfXo
- **Full playlist for reference:** https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ
- **PyTorch official:** *Learn the Basics* tutorial series — https://pytorch.org/tutorials/beginner/basics/intro.html

### Week 9 — CNNs & Computer Vision
- **Primary (course):** Stanford CS231n — lecture notes — https://cs231n.github.io/ and course site — https://cs231n.stanford.edu/ ; lecture videos (2017 offering, still the standard reference) — https://www.youtube.com/playlist?list=PLkt2uSq6rBVctENoVBg1TpCC7OQi31AlC
- **Hands-on tutorial:** PyTorch official CIFAR-10 CNN tutorial — https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
- **Transfer learning tutorial:** https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

### Week 10 — RNNs, LSTMs, Attention Bridge
- **Primary (video):** Karpathy — *The Unreasonable Effectiveness of Recurrent Neural Networks* (blog, still the classic intuition-builder) — https://karpathy.github.io/2015/05/21/rnn-effectiveness/
- **LSTM deep-dive (blog, the single best visual explainer):** Christopher Olah, *Understanding LSTM Networks* — https://colah.github.io/posts/2015-08-Understanding-LSTMs/
- **Attention bridge (blog):** *Attention? Attention!* by Lilian Weng — https://lilianweng.github.io/posts/2018-06-24-attention/

### Week 11 — The Transformer Architecture, Deeply
- **Primary (video):** Andrej Karpathy, *Let's build GPT: from scratch, in code, spelled out* — https://www.youtube.com/watch?v=kCc8FmEb1nY
- **Primary (blog, pairs perfectly with the video):** Jay Alammar, *The Illustrated Transformer* — https://jalammar.github.io/illustrated-transformer/
- **Original paper:** Vaswani et al., *Attention Is All You Need* — https://arxiv.org/abs/1706.03762
- **RoPE paper (for your KV-cache/RoPE extension task):** Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding* — https://arxiv.org/abs/2104.09864
- **Companion code:** https://github.com/karpathy/ng-video-lecture

---

## Phase 4 — Modern Architectures & Fine-Tuning Depth

### Week 12 — LLM Variants, MoE, Efficient Attention
- **Primary (blog, MoE intuition):** Hugging Face, *Mixture of Experts Explained* — https://huggingface.co/blog/moe
- **Primary (paper, FlashAttention):** Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness* — https://arxiv.org/abs/2205.14135
- **GQA/MQA paper (used in Llama):** Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models* — https://arxiv.org/abs/2305.13245
- **Encoder vs decoder vs encoder-decoder overview:** Hugging Face LLM Course, Chapter 1 — https://huggingface.co/learn/llm-course/chapter1/1

### Week 13 — Fine-Tuning Techniques Beyond LoRA
- **Primary (paper):** Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models* — https://arxiv.org/abs/2106.09685
- **Primary (paper, QLoRA internals):** Dettmers et al., *QLoRA: Efficient Finetuning of Quantized LLMs* — https://arxiv.org/abs/2305.14314
- **Tool docs:** Hugging Face PEFT — https://huggingface.co/docs/peft/index ; bitsandbytes — https://huggingface.co/docs/bitsandbytes/main/en/index
- **Practical guide (blog):** Hugging Face, *Making LLMs even more accessible with bitsandbytes, 4-bit quantization and QLoRA* — https://huggingface.co/blog/4bit-transformers-bitsandbytes

### Week 14 — RLHF, DPO, and Alignment
- **Primary (blog, best plain-English overview of the full pipeline):** Hugging Face, *Illustrating Reinforcement Learning from Human Feedback (RLHF)* — https://huggingface.co/blog/rlhf
- **Primary (paper, DPO):** Rafailov et al., *Direct Preference Optimization: Your Language Model is Secretly a Reward Model* — https://arxiv.org/abs/2305.18290
- **Tool docs:** Hugging Face TRL (covers RewardTrainer, PPOTrainer, DPOTrainer in one place) — https://huggingface.co/docs/trl/index
- **Supplementary (video):** Hugging Face, *Deep RL Course* — https://huggingface.co/learn/deep-rl-course/unit0/introduction

### Week 15 — Model Compression: Quantization, Pruning, Distillation
- **Primary (paper, distillation):** Hinton et al., *Distilling the Knowledge in a Neural Network* — https://arxiv.org/abs/1503.02531
- **Primary (blog, quantization types explained clearly):** Hugging Face, *A Gentle Introduction to 8-bit Matrix Multiplication* — https://huggingface.co/blog/hf-bitsandbytes-integration
- **Tool docs:** PyTorch Quantization — https://pytorch.org/docs/stable/quantization.html ; Hugging Face Optimum — https://huggingface.co/docs/optimum/index ; ONNX Runtime — https://onnxruntime.ai/docs/

---

## Phase 5 — MLOps, Deployment & Production Systems

### Week 16 — Experiment Tracking & Reproducibility
- **Primary (docs + quickstart):** MLflow — https://mlflow.org/docs/latest/getting-started/index.html
- **Primary (docs + tutorial):** DVC — *Get Started* — https://dvc.org/doc/start
- **Video walkthrough:** MLflow official YouTube intro — search "MLflow Tracking Quickstart" on the official MLflow YouTube channel — https://www.youtube.com/@mlflow

### Week 17 — Pipeline Orchestration
- **Primary (docs + tutorial):** Apache Airflow — *Tutorial* — https://airflow.apache.org/docs/apache-airflow/stable/tutorial/index.html
- **Alternative (gentler, docs):** Prefect — *Quickstart* — https://docs.prefect.io/latest/getting-started/quickstart/
- **Bonus (asset-based alternative):** Dagster — *Tutorial* — https://docs.dagster.io/tutorial

### Week 18 — Serving at Scale
- **Primary (docs + quickstart):** BentoML — https://docs.bentoml.com/en/latest/get-started/quickstart.html
- **Primary (docs):** NVIDIA Triton Inference Server — *Quickstart* — https://github.com/triton-inference-server/server/blob/main/docs/getting_started/quickstart.md
- **Kubernetes-native serving:** KServe — https://kserve.github.io/website/latest/get_started/
- **Load testing tool docs:** Locust — https://docs.locust.io/en/stable/

### Week 19 — Full MLOps Loop: Monitoring, Retraining, Governance
- **Primary (docs + tutorial):** Evidently AI — *Get Started* — https://docs.evidentlyai.com/introduction
- **Primary (docs):** Prometheus — https://prometheus.io/docs/introduction/overview/ ; Grafana — https://grafana.com/docs/grafana/latest/getting-started/
- **Data validation:** Great Expectations — *Quickstart* — https://docs.greatexpectations.io/docs/tutorials/quickstart/
- **Concept reference (blog):** Chip Huyen, *Data Distribution Shifts and Monitoring* — https://huyenchip.com/2022/02/07/data-distribution-shifts-and-monitoring.html

---

## Phase 6 — GenAI Builder Track

### Week 20 — Vector Databases & Retrieval at Scale
- **Primary (blog, ANN intuition):** Pinecone's (vendor blog, but the algorithm explainer is genuinely good and free) *HNSW* explainer — https://www.pinecone.io/learn/series/faiss/hnsw/
- **Tool docs:** FAISS wiki (Meta, open source) — https://github.com/facebookresearch/faiss/wiki ; Qdrant docs — https://qdrant.tech/documentation/ ; Weaviate docs — https://weaviate.io/developers/weaviate
- **Hybrid search reference (blog):** Qdrant, *Hybrid Search Explained* — https://qdrant.tech/articles/hybrid-search/

### Week 21 — Agent Frameworks & Tool-Use Systems
- **Primary (docs + tutorials):** LangGraph — https://langchain-ai.github.io/langgraph/tutorials/introduction/
- **Primary (docs):** CrewAI — https://docs.crewai.com/introduction
- **Primary (docs):** Microsoft AutoGen — https://microsoft.github.io/autogen/stable/
- **Conceptual reference (blog):** Chip Huyen, *Agents* — https://huyenchip.com/2025/01/07/agents.html

### Week 22 — Systematic LLM Evaluation
- **Primary (docs + tutorial):** RAGAS — https://docs.ragas.io/en/stable/getstarted/
- **Primary (docs):** DeepEval — https://deepeval.com/docs/getting-started
- **Primary (docs):** promptfoo — https://www.promptfoo.dev/docs/intro/
- **Concept reference (blog on LLM-as-judge pitfalls):** *Large Language Models are not Fair Evaluators* — https://arxiv.org/abs/2305.17926

### Week 23 — Multimodal & Diffusion Basics
- **Primary (free course):** Hugging Face Diffusion Models Course — https://huggingface.co/learn/diffusion-course/unit0/1
- **Primary (paper, CLIP):** Radford et al., *Learning Transferable Visual Models From Natural Language Supervision* — https://arxiv.org/abs/2103.00020
- **Tool docs:** Hugging Face `diffusers` — https://huggingface.co/docs/diffusers/index ; OpenCLIP repo — https://github.com/mlfoundations/open_clip

---

## Phase 7 — Distributed Training, System Design & Capstone

### Week 24 — Distributed Training Fundamentals
- **Primary (docs + tutorial):** PyTorch, *Distributed Data Parallel* — https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
- **Primary (docs):** PyTorch FSDP — https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html
- **Primary (blog, ZeRO explained clearly):** Microsoft Research, *ZeRO & DeepSpeed: New system optimizations enable training models with over 100 billion parameters* — https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/
- **Original paper:** Rajbhandari et al., *ZeRO: Memory Optimizations Toward Training Trillion Parameter Models* — https://arxiv.org/abs/1910.02054

### Week 25 — ML System Design Practice
- Use your existing `ML-Engineer-Staff-Interview-Guide-MANGOS.md` as the working reference — no new external links needed this week; the point is retrieval practice from what you've already internalized.
- **Optional cross-check (blog):** Chip Huyen, *Machine Learning Systems Design* (free chapter/notes) — https://huyenchip.com/machine-learning-systems-design/toc.html

### Week 26 — Capstone + Portfolio Consolidation
- No new topic-links this week — this is integration/writing. If you want a model for how to document a capstone well, Hugging Face's own model-card guide is a good structural reference: https://huggingface.co/docs/hub/en/model-cards

---

*A note on link rot: docs sites (mlflow.org, airflow.apache.org, etc.) restructure occasionally — if a doc link 404s, the project's GitHub repo (search `<toolname> github`) always has the current docs linked from its README.*
