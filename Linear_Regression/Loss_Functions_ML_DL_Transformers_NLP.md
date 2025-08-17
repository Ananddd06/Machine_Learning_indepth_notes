
# Loss Functions in ML, DL, Transformers, and NLP

A comprehensive categorized list of loss functions used in Machine Learning (ML), Deep Learning (DL), Transformers, and Natural Language Processing (NLP).

---

## 🔹 1. Regression Loss Functions

| Loss Function            | Description                        | Formula                                                | Best Use               |
|--------------------------|------------------------------------|--------------------------------------------------------|------------------------|
| Mean Squared Error (MSE) | Penalizes larger errors more       | (1/n) Σ(y_i - ŷ_i)^2                                   | Standard regression    |
| Root Mean Squared Error  | Square root of MSE                 | √MSE                                                   | Easier interpretation  |
| Mean Absolute Error (MAE)| Penalizes all errors linearly      | (1/n) Σ|y_i - ŷ_i|                                     | Robust to outliers     |
| Huber Loss               | Combines MSE and MAE               | Piecewise defined based on δ                           | Robust regression      |
| Log-Cosh Loss            | Smooth version of MAE              | Σ log(cosh(ŷ - y))                                     | Stable regression      |
| Quantile Loss            | Asymmetric loss                    | max(q(y - ŷ), (q - 1)(y - ŷ))                          | Quantile regression    |

---

## 🔹 2. Classification Loss Functions

### Binary Classification

| Loss Function             | Description                         | Formula                                            | Best Use              |
|---------------------------|-------------------------------------|----------------------------------------------------|-----------------------|
| Binary Cross Entropy      | Error between 0 and 1 predictions   | -[y log(ŷ) + (1 - y) log(1 - ŷ)]                    | Binary classification |
| Hinge Loss                | Used in SVMs                        | max(0, 1 - yŷ)                                     | SVMs                  |

### Multiclass Classification

| Loss Function                 | Description                     | Formula                        | Best Use                         |
|-------------------------------|---------------------------------|--------------------------------|----------------------------------|
| Categorical Cross Entropy     | Multi-class BCE                 | -Σ y_i log(ŷ_i)                | One-hot labels                   |
| Sparse Categorical Cross Entropy| Integer-label version         | Same as above                  | Efficient classification         |
| KL Divergence                 | Distribution divergence         | Σ p(x) log(p(x)/q(x))          | Knowledge distillation           |

---

## 🔹 3. Ranking Loss Functions

| Loss Function         | Description                          | Use Case               |
|-----------------------|--------------------------------------|------------------------|
| Hinge Loss (Ranking)  | Pairwise ranking                     | SVMRank                |
| Triplet Loss          | Embedding similarity                 | FaceNet, Similarity    |
| Contrastive Loss      | Siamese networks                     | Metric learning        |
| Pairwise Ranking Loss | Optimizes correct order              | IR, Recommendations    |
| ListNet, ListMLE      | Listwise ranking loss                | Learning to Rank       |

---

## 🔹 4. Sequence Modeling / NLP Losses

| Loss Function                    | Description                     | Use Case                     |
|----------------------------------|---------------------------------|------------------------------|
| Cross Entropy Loss              | Token-level loss                | Language modeling            |
| Label Smoothing Cross Entropy  | Avoids overconfidence           | Transformers                 |
| CTC Loss                        | Alignment-free sequences        | Speech/OCR                   |
| Sequence-level NLL              | Applied on full sequence        | Sequence-to-sequence models  |
| BLEU Loss (approx.)             | Optimized via RL                | Translation                  |
| ROUGE Loss (approx.)            | Optimized via RL                | Summarization                |

---

## 🔹 5. Transformer-specific Losses

| Loss Function                  | Description                     | Use Case            |
|--------------------------------|---------------------------------|---------------------|
| Masked Language Modeling       | Predict masked tokens           | BERT                |
| Causal Language Modeling       | Predict next token              | GPT                 |
| Next Sentence Prediction       | Binary sentence pair loss       | BERT pretraining    |
| Contrastive Loss               | For alignment tasks             | CLIP, SimCLR        |
| Denoising Autoencoder Loss     | Reconstruct from corruption     | T5                  |
| Permutation Language Modeling  | Predict all token orders        | XLNet               |

---

## 🔹 6. Vision + NLP Multimodal Losses

| Loss Function                 | Description                     | Use Case            |
|-------------------------------|----------------------------------|---------------------|
| Image-Text Contrastive Loss   | Aligns vision/text embeddings    | CLIP, ALIGN         |
| ITC + ITM Loss                | Contrastive + matching loss      | Flamingo, BLIP      |
| VQA Classification Loss       | Predicts answers                 | Visual QA           |
| Masked Region Feature Loss    | Predicts image patches           | ViLT, BEiT          |

---

## 🔹 7. Reinforcement Learning & Approximate Losses

| Loss Function           | Description                        | Use Case         |
|-------------------------|------------------------------------|------------------|
| Policy Gradient Loss    | Maximize expected reward           | RLHF             |
| PPO Loss                | Stable policy training             | OpenAI RLHF      |
| Reward Modeling Loss    | Human preference prediction        | LLM fine-tuning  |
| KL Penalty Loss         | Regularizes model divergence       | LLM tuning       |

---

## 🔹 8. Self-Supervised / Contrastive Learning Losses

| Loss Function       | Description                         | Used In           |
|---------------------|-------------------------------------|-------------------|
| InfoNCE / NT-Xent   | Contrastive embeddings              | SimCLR, MoCo      |
| Barlow Twins Loss   | Redundancy reduction                | Barlow Twins      |
| BYOL Loss           | Augmented view prediction           | BYOL              |
| SimSiam Loss        | Self-supervised similarity          | SimSiam           |
| SwAV Loss           | Cluster assignment                  | SwAV              |
| DINO Loss           | SSL distillation                    | DINO              |

---

## 🔹 9. GAN Loss Functions

| Loss Function         | Description                        | Used In       |
|-----------------------|------------------------------------|---------------|
| Binary Cross Entropy  | Original GAN loss                  | Vanilla GAN   |
| Wasserstein Loss      | Earth Mover Distance               | WGAN          |
| Hinge Loss            | Stability improvement              | HingeGAN      |
| Least Squares Loss    | Gradient vanishing prevention      | LSGAN         |

---

## 🔹 10. Custom / Hybrid Loss Functions

- **Focal Loss** – Handles class imbalance (used in object detection).
- **Tversky Loss** – Variant of Dice for imbalanced segmentation.
- **Combo Loss** – Weighted sum of BCE + Dice (medical imaging).
- **Lovász Loss** – Optimizes IoU directly (image segmentation).
