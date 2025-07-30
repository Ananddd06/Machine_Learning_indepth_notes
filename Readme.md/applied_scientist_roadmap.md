# 🧱 Core Foundations for Applied Scientist & GATE DA

This guide includes foundational topics, standard books, YouTube playlists, GitHub repos, and practical tools to master Mathematics, ML, DL, NLP, and LLMs.

---

## 1. 📐 Mathematics for Machine Learning

### 📘 Standard Books

- **Linear Algebra:**
  - Introduction to Linear Algebra – Gilbert Strang
  - Linear Algebra and Learning from Data – Gilbert Strang
- **Probability & Statistics:**
  - Introduction to Probability – Dimitri Bertsekas
  - All of Statistics – Larry Wasserman
- **Calculus:**
  - Mathematics for Machine Learning – Deisenroth, Faisal, Ong
- **Optimization:**
  - Convex Optimization – Boyd & Vandenberghe
  - Practical Methods of Optimization – R. Fletcher

### 🎥 YouTube Playlists

- [MIT 18.06 – Linear Algebra (Strang)](https://www.youtube.com/playlist?list=PL221E2BBF13BECF6C)
- [MIT 6.041 – Probability](https://www.youtube.com/playlist?list=PLUl4u3cNGP61MdtwGTqZA0MreSaDybji8)
- [3Blue1Brown – Essence of Calculus](https://www.youtube.com/playlist?list=PLZHQObOWTQDNPOjrT6KVlfJuKtYTftqH6)
- [Stanford EE364a – Convex Optimization (Boyd)](https://www.youtube.com/playlist?list=PL3940DD956CDF0622)
- [CMU 10-725 – Optimization for ML (Zico Kolter)](https://www.youtube.com/playlist?list=PLpPXw4zFa0uGdqUknfFaPzGmbq3FtkDx2)

---

## 2. 🤖 Classical Machine Learning

### 📘 Books

- Pattern Recognition and Machine Learning – Bishop
- The Elements of Statistical Learning – Hastie et al.
- Machine Learning: A Probabilistic Perspective – Murphy
- Hands-On ML with Scikit-Learn, Keras, TensorFlow – Géron

### 🎥 Playlists

- [Stanford CS229 – Andrew Ng](https://www.youtube.com/playlist?list=PLA89DCFA6ADACE599)
- [StatQuest – Josh Starmer](https://www.youtube.com/user/joshstarmer/playlists)
- [IIT Madras – Prof. Balaraman Ravindran](https://nptel.ac.in/courses/106106139)

---

## 3. 🧠 Deep Learning

### 📘 Books

- Deep Learning – Ian Goodfellow et al.
- Neural Networks and Deep Learning – Michael Nielsen
- Dive into Deep Learning – d2l.ai

### 🎥 Playlists

- [Deep Learning Specialization – Andrew Ng](https://www.youtube.com/playlist?list=PLkDaE6sCZn6Ec-XTbcX1uRg2_u4xOEky0)
- [MIT 6.S191 – Intro to DL](https://www.youtube.com/playlist?list=PLkDaE6sCZn6F6wUI9tvS_Gw1vaFAx6rd6)
- [Zero to Hero – Andrej Karpathy](https://www.youtube.com/playlist?list=PLpM-Dvs8t0VZr1J1C8jH1d7nHMFYt-7mM)

### 💻 GitHub Implementations

- 🔗 [rasbt/deeplearning-models](https://github.com/rasbt/deeplearning-models) – From scratch implementations of DL architectures
- 🔗 [labmlai/annotated_deep_learning_paper_implementations](https://github.com/labmlai/annotated_deep_learning_paper_implementations) – Annotated paper implementations like Attention, Transformers, GANs, etc.

---

## 4. 🗣️ Natural Language Processing (NLP)

### 📘 Books

- Speech and Language Processing – Jurafsky & Martin ([SLP 3 Draft](https://web.stanford.edu/~jurafsky/slp3/))
- NLP with PyTorch – Delip Rao
- Transformers for NLP – Denis Rothman

### 🎥 Playlists

- [Stanford CS224n – NLP with Deep Learning](https://www.youtube.com/playlist?list=PLoROMvodv4rObpMCir6rNNUlFAn56Js20)
- [Oxford NLP – Phil Blunsom](https://www.youtube.com/playlist?list=PL613dYH5Fjjh9VBndTBDxQQ0cPV_paF0D)
- [HuggingFace NLP Course](https://www.youtube.com/playlist?list=PLo2EIpI_JMQpc_g1t04K9U9LN_JGCsyUO)

---

## 5. 🧠 Large Language Models (LLMs)

### 📘 Books & Papers

- Transformers for NLP – Denis Rothman
- The Annotated Transformer – Harvard NLP
- Scaling Laws for Neural Language Models – OpenAI
- LoRA (Low-Rank Adaptation), RAG (Retrieval-Augmented Generation), GPT papers

### 🎥 Playlists & Courses

- [GPT from Scratch – Karpathy](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- [Stanford CS25 – Transformers United](https://www.youtube.com/playlist?list=PLLHTzKZzVU9eaEyErdV26ikyolxOsz6mq)
- [HuggingFace Course](https://huggingface.co/learn/nlp-course/)
- [Yannic Kilcher’s Paper Reviews](https://www.youtube.com/c/YannicKilcher)

### 💻 GitHub Repos

- 🔗 [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) – GPT-style models from scratch in PyTorch
- 🔗 [lucidrains/transformer implementations](https://github.com/lucidrains) – Modular implementations of transformers and variants
- 🔗 [labmlai Annotated LLMs](https://github.com/labmlai/annotated_deep_learning_paper_implementations)

---

## 🔧 Practical Resources & Tools

| Tool/Repo                                                     | Description                                            |
| ------------------------------------------------------------- | ------------------------------------------------------ |
| [Netron](https://netron.app/)                                 | Visualize model architectures like PyTorch, TensorFlow |
| [Weights & Biases](https://wandb.ai/)                         | Experiment tracking for DL                             |
| [d2l.ai](https://d2l.ai)                                      | Code + theory for DL/NLP in PyTorch/MXNet              |
| [Hugging Face Hub](https://huggingface.co/models)             | Pretrained models and demos                            |
| [Gradient Checkers](https://cs231n.github.io/optimization-2/) | Learn and test your gradients from scratch             |

---

## 🧭 Final Learning Strategy

| Phase               | Focus                          | Purpose                |
| ------------------- | ------------------------------ | ---------------------- |
| 1. Math Foundations | LA, Stats, Optimization        | Core reasoning         |
| 2. Classical ML     | Regression → Trees → SVM       | Breadth & depth        |
| 3. Deep Learning    | MLP → CNN → RNN → Attention    | Power and flexibility  |
| 4. NLP              | Embeddings → Transformers      | Language understanding |
| 5. LLMs             | GPT → LoRA → RAG → Fine-tuning | Production-grade AI    |

---

## 📝 Reminder:

> 🚫 Don’t skip fundamentals.  
> ✅ Build strong math and classical ML foundations **before jumping into** RAG, LoRA, or LangChain.  
> 🧠 Think like a scientist, not just a user of libraries.
