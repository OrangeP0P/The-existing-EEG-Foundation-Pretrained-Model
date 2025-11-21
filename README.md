# The-existing-EEG-Foundation-Pretrained-Model

> A curated list of **existing EEG foundation / large pretrained models** – including paper links, short descriptions, and open-source repositories when available.  

---

## Motivation

In the last few years, **EEG foundation models (EEG-FMs)** and large pretrained EEG models have started to appear, inspired by large language and vision models. They usually:

- are trained in a **self-supervised or multi-task** way on **large-scale EEG (or brain-signal) data**,  
- aim to provide **general, reusable representations** across datasets, paradigms, and tasks,  
- are often described in papers as “foundation model”, “large EEG model”, “universal EEG representation”, etc.

This repo tries to **collect existing models that fit this spirit**, with:

- **Model name & year**
- **Paper / arXiv link**
- **GitHub / code link** (if any)
- Very short notes (pretraining data, objective, special design, etc.)

> ⚠️ This is a **curated list, not a perfect census**. Some models may be missing – PRs are highly welcome.

---

## Inclusion criteria

Roughly, a model is listed here if:

1. It has a **peer-reviewed paper or arXiv preprint**, and  
2. It is either
   - explicitly described as an **EEG / brain foundation model**, **large EEG model**, or
   - a **large-scale self-supervised / pretrained EEG model** intended for broad reuse (not just a tiny SSL baseline on one dataset), and  
3. Ideally has **public code or at least announced release**.

We group models into:

- General **EEG foundation models**
- **Multi-modal / brain-signal** foundation models that include EEG
- **EEG-language & instruction-following** models
- **Benchmarks/toolkits**
- Earlier important **self-supervised EEG representation models**

---

## General EEG foundation models (EEG-only)

Models in this section mainly target **scalp EEG** and aim at general-purpose feature learning or broad BCI tasks.

| Model | Year | Paper | Code | Short notes |
|------|------|-------|------|-------------|
| **BENDR** (BErt-like Neurophysiological Data Representation) | 2021 | “BENDR: Using Transformers and a Contrastive Self-Supervised Learning Task to Learn From Massive Amounts of EEG Data” (Frontiers in Human Neuroscience): contentReference[oaicite:0]{index=0} | https://github.com/SPOClab-ca/BENDR :contentReference[oaicite:1]{index=1} | Early large-scale **SSL transformer** for raw EEG. Pretrained contrastively on diverse corpora; shows a **single pretrained model** can transfer to many BCI/sleep tasks. Often regarded as a proto-“EEG foundation model”. |
| **EEGFormer** | 2024 | “EEGFormer: Towards Transferable and Interpretable Large-Scale EEG Foundation Model” (AAAI Clinical Foundation Models Symposium / arXiv:2401.10278) :contentReference[oaicite:2]{index=2} | No official code at time of writing (used as baseline in later works). | **Vector-quantized transformer** foundation model, pretrained on large-scale TUH EEG via masked reconstruction; aims for **transferable & interpretable** EEG tokens. |
| **LaBraM** (Large Brain Model) | 2024 | “Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI” (ICLR 2024): contentReference[oaicite:3]{index=3} | https://github.com/935963004/LaBraM :contentReference[oaicite:4]{index=4} | Unified **EEG foundation model**, ~2,500 hours of EEG across ~20 datasets. Uses **VQ neural tokenizer + masked code prediction** on EEG patches; strong cross-dataset performance (abnormal detection, emotion, gait, etc.). |
| **NeuroGPT** | 2024 | “Neuro-GPT: Towards A Foundation Model for EEG” (ISBI / arXiv:2311.03764): contentReference[oaicite:5]{index=5} | https://github.com/wenhui0206/NeuroGPT: contentReference[oaicite:6]{index=6} | Foundation model with **EEG encoder + GPT decoder**, trained to reconstruct **masked EEG segments** on large TUH-style corpora and fine-tuned on motor imagery. |
| **EEGPT (NeurIPS 2024)** | 2024 | “EEGPT: Pretrained Transformer for Universal and Reliable Representation of EEG Signals” (NeurIPS 2024) :contentReference[oaicite:7]{index=7} | https://github.com/BINE022/EEGPT :contentReference[oaicite:8]{index=8} | ~10M-parameter pretrained transformer with **dual masked self-supervised objectives**; focuses on **robust universal EEG feature extraction** across multiple datasets. |
| **EEGPT (Generalist FM)** | 2024 | “EEGPT: Unleashing the Potential of EEG Generalist Foundation Model by Autoregressive Pre-training” (OpenReview, arXiv:2410.19779) :contentReference[oaicite:9]{index=9} | Official code announced but not fully public at time of writing; an implementation is used in **EEG-FM-Bench**. :contentReference[oaicite:10]{index=10} | First explicitly **generalist EEG foundation model** with up to **1.1B parameters**. Uses **autoregressive next-signal prediction** and **electrode-wise modeling** to support up to 138 electrodes, multi-task transfer with a shared electrode graph. |
| **ALFEE** | 2025 | “ALFEE: Adaptive Large Foundation Model for EEG Representation” (arXiv:2505.06291) :contentReference[oaicite:11]{index=11} | https://github.com/xw1216/ALFEE :contentReference[oaicite:12]{index=12} | **Hybrid transformer** foundation model (~25k hours EEG). Separates **channel encoder** and **temporal encoder**; multi-task pretraining with **task prediction, masked reconstruction (channel & temporal), and forecasting** to improve cross-dataset generalization. |
| **CBraMod** | 2025 | “CBraMod: A Criss-Cross Brain Foundation Model for EEG Decoding” (ICLR 2025) :contentReference[oaicite:13]{index=13} | https://github.com/wjq-learning/CBraMod :contentReference[oaicite:14]{index=14} | **Criss-cross transformer** that models **spatial and temporal dependencies separately** via two attention branches, plus asymmetric positional encoding. Pretrained with **masked patch reconstruction** on a large corpus and evaluated on up to 10 BCI tasks / 12 datasets. |
| **EEGMamba** | 2025 | “EEGMamba: An EEG foundation model with Mamba” (Neural Networks) :contentReference[oaicite:15]{index=15} | https://github.com/wjq-learning/EEGMamba :contentReference[oaicite:16]{index=16} | Foundation model built on **Mamba state-space models**, targeting efficient long-sequence EEG modeling and general EEG decoding. |
| **GEFM** | 2024–2025 | “GEFM: Graph-Enhanced EEG Foundation Model” (arXiv:2411.19507) :contentReference[oaicite:17]{index=17} | Code link not clearly public at time of writing. | Foundation model that combines **GNNs for electrode graphs** with a **masked autoencoder**. Jointly learns temporal dynamics and **inter-channel topology**; shows gains on several downstream tasks. |
| **MIRepNet** | 2025 | “MIRepNet: A Pipeline and Foundation Model for EEG-Based Motor Imagery Classification” (arXiv:2507.20254) :contentReference[oaicite:18]{index=18} | https://github.com/staraink/MIRepNet :contentReference[oaicite:19]{index=19} | **Paradigm-specific EEG FM** explicitly tailored for **motor imagery**. Includes a neurophysiology-informed channel template and preprocessing pipeline; SOTA on 5 MI datasets, even with \<30 trials per class. |
| **Uni-NTFM** | 2025 | “Uni-NTFM: A Unified Foundation Model for EEG Signal Representation Learning” (arXiv:2509.24222) :contentReference[oaicite:20]{index=20} | Code TBA (no public GitHub at time of writing). | Very large (**up to 1.9B params**) **Mixture-of-Experts transformer** with decoupled **time / frequency / raw** streams and **topological electrode embeddings**; pretrained on **28,000+ hours** EEG with dual-domain masked reconstruction. |
| **EEGDM** (borderline FM) | 2025 | “EEGDM: Learning EEG Representation with Latent Diffusion Model” (arXiv:2508.20705) :contentReference[oaicite:21]{index=21} | Code status unclear. | Uses **latent diffusion** for EEG generation as a **self-supervised objective**, turning a diffusion model into a strong EEG encoder with competitive downstream performance. Often treated as a **large SSL EEG representation learner**, very close to FM spirit. |

---

## Multi-modal / brain-signal foundation models including EEG

These models handle **EEG + other neural modalities** (iEEG, MEG, etc.), but are highly relevant to EEG foundation modeling.

| Model | Year | Modalities | Paper | Code / Project | Notes |
|-------|------|------------|-------|----------------|-------|
| **BrainWave / Brant-2** | 2024–2025 | EEG, iEEG and other electrical brain signals | “BrainWave: A Brain Signal Foundation Model for Clinical Applications” (arXiv:2402.10251) :contentReference[oaicite:22]{index=22} | https://github.com/yzz673/Brant-2 (code/weights promised on publication) :contentReference[oaicite:23]{index=23} | Foundation model for **clinical neural recordings**, ~40,000 hours from ~16,000 individuals. Targets **neurological disorder diagnosis** and zero/few-shot transfer across sites and diseases. |
| **BrainBERT** | 2023 | Intracranial field potentials (iEEG) | “BrainBERT: Self-Supervised Representation Learning for Intracranial Recordings” (ICLR 2023) :contentReference[oaicite:24]{index=24} | Implementation used in some labs; public repo situation varies. | **Transformer pretrained on large iEEG recordings** with self-supervision; widely used as a reference for **neural foundation models** though not scalp EEG. |
| **Nested Deep Learning Model (NDL-Brain)** | 2024 | EEG & MEG | “Nested Deep Learning Model Towards A Foundation Model for Brain Signal Data” (arXiv:2410.03191) :contentReference[oaicite:25]{index=25} | (Project page; code situation may evolve) | Uses **nested deep learning architecture** to approach a unified foundation model for **spike detection** and related tasks on EEG/MEG. |
| **Large Brainwave Foundation Models (LBMs)** | 2025 | Survey / analysis across multiple FMs | “Assessing the Capabilities of Large Brainwave Foundation Models” (ICLR 2025 workshop) :contentReference[oaicite:26]{index=26} | — | Focuses on **evaluation & causal analysis** of LBMs (including EEG FMs). Not a single model, but useful context for this repo. |

---

## EEG-language and instruction-following models

Models that explicitly **bridge EEG with language/LLMs** or provide **instruction-like interfaces**.

| Model | Year | Paper | Code | Notes |
|-------|------|-------|------|------|
| **NeuroLM** | 2024–2025 | “NeuroLM: A Universal Multi-task Foundation Model for Bridging the Gap between Language and EEG Signals” (ICLR 2025, arXiv:2409.00101) :contentReference[oaicite:27]{index=27} | https://github.com/935963004/NeuroLM: contentReference[oaicite:28]{index=28} | Learns a **vector-quantized neural tokenizer** for EEG and treats EEG tokens as a “foreign language” for an **LLM**. Supports **multi-task instruction tuning** over EEG + text with up to 1.7B parameters. |
| **EEG-GPT (clinical, abnormal vs normal)** | 2024 | “EEG-GPT: Exploring Capabilities of Large Language Models for EEG Classification and Interpretation” (arXiv:2401.18006) :contentReference[oaicite:29]{index=29} | Official open repo is not clearly standardized; reference implementation & configs are collected in https://github.com/Altaheri/LLMs-in-EEG-Decoding and related projects. :contentReference[oaicite:30]{index=30} | Uses LLMs to **classify and interpret EEG** (normal vs abnormal) in a **few-shot regime**, with step-by-step reasoning and explanations. |
| **Neurosity EEG-GPT** | 2024 | Product-oriented foundational model for Neurosity Crown headset: contentReference[oaicite:31]{index=31} | https://github.com/neurosity/EEG-GPT | A **device-specific foundational model** for a commercial EEG headset, focused on real-time applications. |
| **Large Cognition Model (LCM)** | 2025 | “Large Cognition Model: Towards Pretrained EEG Foundation Model” (arXiv:2502.17464): contentReference[oaicite:32]{index=32} | Evaluated in **EEG-FM-Bench**: https://github.com/xw1216/EEG-FM-Bench: contentReference[oaicite:33]{index=33} | Transformer-based EEG foundation model with **temporal + spectral attention**, trained via large-scale self-supervision and integrated in a fair benchmarking framework. |
| **WaveMind** | 2025 | “WaveMind: Towards a Conversational EEG Foundation Model” (arXiv:2510.00032): contentReference[oaicite:34]{index=34} | https://github.com/ZiyiTsang/WaveMind :contentReference[oaicite:35]{index=35} | Builds a **conversational interface** over EEG via instruction-tuning and synthetic data; code + benchmark and instruction-tuning data are released. |
| **EEG Foundation Models for BCI Learn Diverse Features of Electrophysiology** | 2025 | Preprint (Johns Hopkins APL; EEG FMs for BCI): contentReference[oaicite:36]{index=36} | Authors mention releasing code, but stable repo not clearly discoverable yet. | Studies how **EEG FMs learn diverse electrophysiological features** across BCI paradigms; good reference for understanding what current FMs actually capture. |

---

## Benchmarks and toolkits

| Name | Year | Link | Purpose |
|------|------|------|---------|
| **EEG-FM-Bench** | 2025 | https://github.com/xw1216/EEG-FM-Bench :contentReference[oaicite:37]{index=37} | Standardized evaluation suite for **EEG foundation models**, including baselines like **EEGPT, CBraMod, LaBraM, BENDR, NeuroGPT**, etc. Focuses on **frozen backbone + linear probe**, cross-dataset generalization, and fair comparison. |
| **EEG-Bench** | 2024–2025 | https://github.com/ETH-DISCO/EEG-Bench :contentReference[oaicite:38]{index=38} | Large benchmark for **classical + foundation** models across 25 EEG datasets (clinical + BCI). Provides standardized pipelines and includes FMs such as BENDR, LaBraM, and NeuroGPT. |

These are not “models”, but are extremely useful if you want to **compare or fine-tune** EEG FMs.

---

## Legacy self-supervised EEG representation models (pre-FM era)

Before the term “EEG foundation model” became common, several **large-scale self-supervised EEG models** already existed. They are often used as baselines and are included here for completeness.

| Model | Year | Paper | Code | Why it matters |
|-------|------|-------|------|----------------|
| **BENDR** | 2021 | See above. :contentReference[oaicite:39]{index=39} | https://github.com/SPOClab-ca/BENDR | Widely used **SSL baseline**; demonstrates cross-dataset generalization from massive raw EEG corpora. |
| **EEGMirror** | 2025 | “EEGMirror: Leveraging EEG Data in the Wild via Montage-Agnostic Self-Supervision” (ICCV 2025): contentReference[oaicite:40]{index=40} | (Check paper for code link; often used as SSL baseline) | Montage-agnostic **self-supervision** on uncontrolled EEG data “in the wild”; important precursor for **robust pretraining**. |
| **EEGDM** | 2025 | See above. :contentReference[oaicite:41]{index=41} | — | Diffusion-based SSL; narrows the gap between **generation and representation learning** for EEG. |

> 💡 If you feel some SSL model should be treated as “foundation”, feel free to open an issue / PR with an argument and references.

---

## How to use this repo

This repository is **meant as a directory**, not a library. Typical workflows:

- **Find a starting point** for your project  
  - Generalizable decoding across many BCI tasks → look at **LaBraM, EEGPT, CBraMod, ALFEE, Uni-NTFM**.  
  - MI-specific decoding → **MIRepNet**.  
  - EEG + language / interpretability → **NeuroLM, EEG-GPT, WaveMind**.

- **Benchmark existing FMs** on your own data  
  - Use **EEG-FM-Bench** or **EEG-Bench** as templates for **frozen-backbone + linear probe** or fine-tuning experiments.

- **Research survey / related work section**  
  - The “General EEG foundation models” and “EEG-language models” tables are designed to be **copy-paste-friendly** for literature reviews.

---

## How to contribute

You are **very welcome** to submit PRs or issues for:

- Missing models (especially: **new arXiv preprints** or non-English work)
- Corrections to year/link/dataset/description
- Better categorization (e.g., moving a model from “legacy SSL” to “FM”)
- Adding **parameter counts**, **pretraining hours**, or **downstream benchmarks** columns

When opening a PR, please try to:

1. Put the model in the right section (`EEG`, `Brain-signal`, `EEG-language`, etc.).
2. Add **paper link**, **code link** (if any), and **1–2 sentences** of description.
3. If possible, mention **pretraining data scale** (hours, #datasets) and **main objective** (masked reconstruction, autoregressive, contrastive, diffusion, …).

### 推荐 PR 模板（中文）

```text
Model name: XXX
Section: General EEG FM / EEG-language / Benchmark / Legacy SSL
Paper: arXiv/Conference
Code: Link
Summary (1–2 sentences): Use 1-2 sentences to describe the method
Reason to include: e.g. "explicitly described as EEG foundation model", "large-scale self-supervised EEG representation", "used as baseline in recent FM benchmarks"
