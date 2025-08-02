# 🧠 Foundation Models in Deep Learning

This document categorizes all major foundational models across:

- 📷 Computer Vision (CV)
- 🧾 Natural Language Processing (NLP)
- 🦙 Large Language Models (LLMs)

---

## 📷 1. Computer Vision Foundation Models

| Model                        | Year | Paper                                                                                                                    | Key Innovation                        |
| ---------------------------- | ---- | ------------------------------------------------------------------------------------------------------------------------ | ------------------------------------- |
| **LeNet-5**                  | 1998 | [LeNet-5](http://yann.lecun.com/exdb/lenet/)                                                                             | Early CNN for digit recognition       |
| **AlexNet**                  | 2012 | [ImageNet Classification](https://papers.nips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf) | First deep CNN to win ImageNet        |
| **ZFNet**                    | 2013 | [Visualizing and Understanding CNNs](https://arxiv.org/abs/1311.2901)                                                    | Improved on AlexNet via visualization |
| **VGGNet**                   | 2014 | [Very Deep CNNs](https://arxiv.org/abs/1409.1556)                                                                        | Deep network with 3×3 convolutions    |
| **GoogLeNet (Inception)**    | 2014 | [Going Deeper with Convolutions](https://arxiv.org/abs/1409.4842)                                                        | Inception modules, efficient design   |
| **ResNet**                   | 2015 | [Deep Residual Learning](https://arxiv.org/abs/1512.03385)                                                               | Skip connections to train deep nets   |
| **DenseNet**                 | 2017 | [Densely Connected CNNs](https://arxiv.org/abs/1608.06993)                                                               | Feature reuse across layers           |
| **EfficientNet**             | 2019 | [EfficientNet](https://arxiv.org/abs/1905.11946)                                                                         | Compound model scaling                |
| **ViT (Vision Transformer)** | 2020 | [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929)                                                        | First Transformer for CV              |
| **Swin Transformer**         | 2021 | [Swin: Hierarchical Vision Transformer](https://arxiv.org/abs/2103.14030)                                                | Local windowed attention              |
| **ConvNeXt**                 | 2022 | [ConvNeXt](https://arxiv.org/abs/2201.03545)                                                                             | CNNs re-imagined in transformer era   |

---

## 🧾 2. NLP Foundation Models (Non-LLM)

| Model           | Year | Paper                                                                                     | Key Innovation                      |
| --------------- | ---- | ----------------------------------------------------------------------------------------- | ----------------------------------- |
| **Word2Vec**    | 2013 | [Efficient Estimation of Word Representations](https://arxiv.org/abs/1301.3781)           | Word embeddings via skip-gram/CBOW  |
| **GloVe**       | 2014 | [GloVe: Global Vectors](https://nlp.stanford.edu/pubs/glove.pdf)                          | Word embeddings from co-occurrence  |
| **ELMo**        | 2018 | [Deep Contextualized Word Representations](https://arxiv.org/abs/1802.05365)              | Contextual word embeddings          |
| **ULMFiT**      | 2018 | [Universal Language Model Fine-tuning](https://arxiv.org/abs/1801.06146)                  | Transfer learning for NLP           |
| **Transformer** | 2017 | [Attention is All You Need](https://arxiv.org/abs/1706.03762)                             | Foundation of all modern NLP models |
| **BERT**        | 2018 | [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805) | Bidirectional masked language model |
| **RoBERTa**     | 2019 | [RoBERTa](https://arxiv.org/abs/1907.11692)                                               | Robust BERT pretraining             |
| **XLNet**       | 2019 | [XLNet](https://arxiv.org/abs/1906.08237)                                                 | Permutation-based language modeling |
| **ALBERT**      | 2019 | [ALBERT](https://arxiv.org/abs/1909.11942)                                                | Parameter-sharing in BERT           |
| **T5**          | 2020 | [T5: Exploring the Limits of Transfer Learning](https://arxiv.org/abs/1910.10683)         | Unified text-to-text model          |

---

## 🦙 3. LLMs (Large Language Models)

| Model          | Year | Paper                                                                                                            | Key Innovation                  |
| -------------- | ---- | ---------------------------------------------------------------------------------------------------------------- | ------------------------------- |
| **GPT**        | 2018 | [Improving Language Understanding by Generative Pre-training](https://openai.com/research/language-unsupervised) | First autoregressive LLM        |
| **GPT-2**      | 2019 | [Language Models are Unsupervised Multitask Learners](https://openai.com/research/better-language-models)        | Scaled-up GPT                   |
| **GPT-3**      | 2020 | [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)                                        | In-context learning             |
| **GPT-4**      | 2023 | [GPT-4 Technical Report](https://openai.com/research/gpt-4)                                                      | Multimodal, high reasoning      |
| **PaLM**       | 2022 | [Pathways Language Model](https://arxiv.org/abs/2204.02311)                                                      | Scaled model (540B) from Google |
| **LaMDA**      | 2022 | [LaMDA: Language Model for Dialogue](https://arxiv.org/abs/2201.08239)                                           | Open-ended dialogue modeling    |
| **Chinchilla** | 2022 | [Chinchilla: Optimal Compute](https://arxiv.org/abs/2203.15556)                                                  | Data vs. parameter tradeoff     |
| **LLaMA**      | 2023 | [LLaMA: Open and Efficient Foundation LM](https://arxiv.org/abs/2302.13971)                                      | Open-weight models from Meta    |
| **Claude**     | 2023 | [Anthropic Claude](https://www.anthropic.com/index/introducing-claude)                                           | Constitutional AI and alignment |
| **Mistral**    | 2023 | [Mistral: Efficient Dense and MoE Models](https://mistral.ai/news/announcing-mistral-7b/)                        | Open LLM alternatives           |
| **Gemini**     | 2023 | [Gemini 1](https://deepmind.google/technologies/gemini/)                                                         | Google’s multimodal model       |
| **Mixtral**    | 2023 | [Mixture of Experts LLM](https://mistral.ai/news/mixtral-of-experts/)                                            | Sparse expert routing           |

---

# 🧠 Annotated Deep Learning Paper Implementations: NLP + LLM Study Guide

This guide is based on the [LabML Annotated Deep Learning Paper Implementations](https://github.com/labmlai/annotated_deep_learning_paper_implementations) repository. It summarizes all the relevant topics you should study to master **NLP** and **Large Language Models (LLMs)** using clean, annotated PyTorch code implementations.

---

## 📚 Topics to Study for NLP and LLMs

### 🔹 Transformer Architecture & Variants

- [ ] Multi‑Head Attention
- [ ] Transformer Building Blocks
- [ ] Transformer‑XL
- [ ] Relative Multi‑Head Attention
- [ ] Rotary Positional Embeddings (RoPE)
- [ ] ALiBi (Attention with Linear Biases)
- [ ] Compressive Transformer
- [ ] Feedback Transformer
- [ ] Switch Transformer
- [ ] Fast Weights Transformer
- [ ] FNet
- [ ] Attention‑Free Transformer

### 🔹 LLM Frameworks & Techniques

- [ ] GPT Architecture Implementation
- [ ] kNN‑LM (Language Modeling through Memorization)
- [ ] Primer EZ
- [ ] LoRA (Low‑Rank Adaptation Modules)
- [ ] GPT‑NeoX Training and Fine-Tuning

### 🔹 Sampling and Inference Methods

- [ ] Greedy Sampling
- [ ] Temperature-Based Sampling
- [ ] Top‑k Sampling
- [ ] Nucleus (Top‑p) Sampling

### 🔹 Training and Scaling Enhancements

- [ ] Zero3 Memory Optimization for Large Model Training
- [ ] DeepNorm for Stabilizing Very Deep Transformers

### 🔹 Normalization Layers in NLP Context

- [ ] Layer Normalization
- [ ] DeepNorm

### 🔹 Optimizers Used in NLP Training

- [ ] Adam
- [ ] AMSGrad
- [ ] AdaBelief
- [ ] Sophia‑G
- [ ] Noam Optimizer

---

## ✅ Essential Topics to Prioritize

### ✅ Must-Learn

- Transformer Block Basics → Multi‑Head Attention → GPT Architecture
- Token Sampling Strategies (Greedy / Top-k / Nucleus)
- LoRA Implementation and Theory
- LayerNorm + DeepNorm
- Adam Optimizer and Variants

### 🔄 Very Valuable

- Transformer Variants like XL, Switch, Feedback
- kNN-LM and Memory-Augmented Models
- Compressive Transformer
- Zero3 Optimization

### 🌱 Nice-to-Know

- Primer‑EZ, FNet, Attention-Free Transformers
- Adaptive Computation Modules like PonderNet

---

## 🔁 Suggested Study Flow

1. **Transformer Basics** → Multi‑Head Attention → GPT
2. **Token Sampling** (Greedy, Top-k, Nucleus, Temperature)
3. **LoRA Fine-Tuning**
4. **Transformer Variants** (XL, Switch, Feedback)
5. **Zero3 & DeepNorm** for Stability + Memory
6. **Optional Extras** (FNet, PonderNet, Memory-Augmented, etc.)

---

## 🌐 Useful Links

- 🔗 [GitHub Repo](https://github.com/labmlai/annotated_deep_learning_paper_implementations)
- 🔗 [Annotated Transformer Index](https://nn.labml.ai/transformers/index.html)
- 🔗 [Sampling Notebook](https://nn.labml.ai/sampling/index.html)

---

> 💡 **Tip:** Walk through the notebooks line by line, match paper equations to code, and re-implement parts using your own datasets to build real understanding.

---

Would you like this as a downloadable `.md` file?

## 📌 Summary of Categories

| Category            | Model Families                                       |
| ------------------- | ---------------------------------------------------- |
| **Computer Vision** | CNNs (AlexNet, VGG, ResNet) → ViTs (ViT, Swin, BEiT) |
| **NLP (Non-LLM)**   | Word2Vec, GloVe, ELMo, BERT, RoBERTa, T5             |
| **LLMs**            | GPT-family, LLaMA, Claude, PaLM, Gemini, Mixtral     |

---
