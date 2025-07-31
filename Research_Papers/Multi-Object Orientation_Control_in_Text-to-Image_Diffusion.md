# 🎯 Orientation-Controlled Text-to-Image Diffusion Models

This repository aims to consolidate research efforts on building controllable and orientation-aware multi-object image generation using diffusion models.

---

## 🔬 Core Paper: Compass Control

**Compass Control: Multi-Object Orientation Control in Text-to-Image Diffusion**

- 📄 Paper: [arXiv:2504.06752](https://arxiv.org/abs/2504.06752)
- 🧠 Summary: Introduces compass tokens and a coupled attention localization loss (CALL) to control orientations of multiple objects independently.

---

## 🔍 Related Works

### 1. **Directed Diffusion**

**Title**: Directed Diffusion: Controlling Where to Place Objects in Text-to-Image Generation

- 📄 Paper: [arXiv:2403.11627](https://arxiv.org/abs/2403.11627)
- 🌐 Project: [Website](https://hohonu-vicml.github.io/DirectedDiffusion.Page/)

### 2. **Compositional Synthesis with Attention Map Control (BoxNet)**

**Title**: Compositional Text-to-Image Synthesis with Attention Map Control

- 📄 Paper: [arXiv:2305.13921](https://arxiv.org/abs/2305.13921)

### 3. **Bounded Attention**

**Title**: Bounded Attention: Controlling Leakage in Text-to-Image Generation

- 🌐 Project Page: [ECCV 2024 Bounded Attention](https://omer11a.github.io/bounded-attention/)

### 4. **Masked-Attention Diffusion Guidance (MAG)**

**Title**: MAG: Masked-Attention Diffusion Guidance

- 📄 Paper: [arXiv:2309.13040](https://arxiv.org/abs/2309.13040)
- 🌐 Project: [MAG Website](https://www.cgg.cs.tsukuba.ac.jp/~endo/projects/MAG/)

---

## 🧩 Core Methods & Concepts

- **Compass Tokens**: Learnable vectors that encode directional orientation.
- **Coupled Attention Localization Loss (CALL)**: Forces attention alignment with direction and location.
- **LoRA Fine-Tuning**: Lightweight fine-tuning via low-rank adaptation of model weights.
- **Bounding Box Supervision**: Object-wise positional control via bounding mask constraints.
- **Synthetic Data Generation**: Use of procedural scenes to generate labeled training data with known orientation.
- **Few-Shot Personalization**: Apply orientation control to novel objects with minimal examples.

---

## 📚 Suggested Study Order

1. Read Compass Control: [arxiv.org/abs/2504.06752](https://arxiv.org/abs/2504.06752)
2. Study Directed Diffusion for attention optimization.
3. Understand BoxNet and attention masking in compositional generation.
4. Review Bounded Attention and MAG for prompt disentanglement.
5. Learn LoRA fine-tuning: [LoRA GitHub](https://github.com/microsoft/LoRA)
6. Implement a synthetic dataset with compass labels for testing.

---

## 🚀 To-Do

- [ ] Re-implement Compass Token architecture with orientation encoder
- [ ] Add LoRA adapter to base diffusion model
- [ ] Generate synthetic two-object scenes with angle labels
- [ ] Evaluate orientation accuracy and attention spread

---

## 🧠 Credits

- Compass Control Authors (Meta AI)
- Directed Diffusion (MIT + CMU)
- MAG & Bounded Attention Contributors

---
