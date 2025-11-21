A curated list of **existing EEG(-related) foundation / pretrained models and benchmarks**, including:

- EEG-only self-supervised / foundation models
- Multimodal / LLM-style models bridging EEG with text / vision
- Benchmarks and toolkits for evaluating EEG foundation models
- Broader brain-signal foundation models and closely related work

> 🍊 Scope: “Foundation / pretrained model” here means models trained on relatively **large, diverse EEG or neural datasets** with the explicit goal of re-use across tasks / datasets, not small task-specific CNNs.

---

## 0. Legend

- **Modality**
  - EEG: scalp EEG
  - iEEG / ECoG: invasive recordings
  - Neuro-signal: EEG + other electrophysiology (ECoG, LFP, etc.)
- **Code**
  - 🔨 Official repo
  - 🔧 Unofficial / third-party or “planned”
  - ⏳ Announced but not released yet

---

## 1. EEG-only foundation / representation models

### 1.1 General-purpose EEG foundation models

| Year | Model | Paper | Code | Modality | Pretraining data (rough) | Highlights |
|------|-------|-------|------|----------|---------------------------|------------|
| 2021 | BENDR | [BENDR: Using Transformers and a Contrastive Self-Supervised Learning Task to Learn From Massive Amounts of EEG Data](https://arxiv.org/abs/2101.12037) | ✅ [SPOClab-ca/BENDR](https://github.com/SPOClab-ca/BENDR) | EEG | ~20k+ hours clinical EEG (Temple U. etc.) | Early **BERT-style** contrastive pretraining on raw EEG; demonstrates cross-dataset transfer (sleep, pathology etc.). |
| 2024 | EEGFormer | [EEGFormer: Towards Transferable and Interpretable Large-Scale EEG Foundation Model](https://arxiv.org/abs/2401.10278) | 🔧 (paper only) | EEG | Large-scale “compound” EEG from multiple datasets | Transformer EEG FM; pretraining on mixed tasks; emphasizes **interpretability** of learned patterns and transfer to anomaly detection, classification. |
| 2024 | LaBraM | [Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI](https://arxiv.org/abs/2405.18765) | ✅ [935963004/LaBraM](https://github.com/935963004/LaBraM) | EEG | ~2,500 h, ~20 datasets | BEiT-style tokenizer + masked prediction on **EEG patches**; strong cross-dataset EEG FM; widely adopted (also in TorchEEG / Braindecode). |
| 2024 | Neuro-GPT | [Neuro-GPT: Towards A Foundation Model for EEG](https://arxiv.org/abs/2311.03764) | ✅ [wenhui0206/NeuroGPT](https://github.com/wenhui0206/NeuroGPT) | EEG | Large mixed EEG corpus (BCI / physionet etc.) | Encoder + GPT-style decoder; masked reconstruction pretraining; early use of **GPT-like decoder** for EEG. |
| 2024 | EEGPT (universal representation) | [EEGPT: Pretrained Transformer for Universal and Reliable Representation of EEG Signals](https://papers.nips.cc/paper_files/paper/2024/file/4540d267eeec4e5dbd9dae9448f0b739-Paper-Conference.pdf) | ✅ [BINE022/EEGPT](https://github.com/BINE022/EEGPT) | EEG | 10k+ hours from multiple datasets | NeurIPS 2024. Large transformer FM with unified preprocessing; strong baselines across many EEG tasks and datasets. |
| 2024 | EEGPT (generalist FM) | [EEGPT: Unleashing the Potential of EEG Generalist Foundation Model by Autoregressive Pre-training](https://arxiv.org/abs/2410.19779) | 🔧 (code promised) | EEG | 37.5M samples, up to 138 electrodes | **Autoregressive** generalist FM; electrode-wise modeling; multi-task transfer with shared electrode graph network. |
| 2024 | GEFM | [GEFM: Graph-Enhanced EEG Foundation Model](https://arxiv.org/abs/2411.19507) | 🔧 (paper only) | EEG | Multi-dataset EEG (hours not explicitly stated) | Combines **GNN + masked autoencoder** to explicitly model inter-channel topology plus temporal dynamics. |
| 2025 | ALFEE | [ALFEE: Adaptive Large Foundation Model for EEG Representation](https://arxiv.org/abs/2505.06291) | ✅ [xw1216/ALFEE](https://github.com/xw1216/ALFEE) | EEG | 25,000+ hours, 6 downstream tasks | Hybrid transformer with **channel encoder + temporal encoder + hybrid decoder**; multi-objective pretraining (masking, forecasting) for robust cross-paradigm generalization. |
| 2025 | CBraMod | [CBraMod: A Criss-Cross Brain Foundation Model for EEG Decoding](https://arxiv.org/abs/2412.07236) | ✅ [wjq-learning/CBraMod](https://github.com/wjq-learning/CBraMod) | EEG | Large multi-dataset EEG corpus (12 public datasets) | **Criss-cross transformer** that separately models spatial / temporal attention; strong SOTA on 10+ BCI tasks. |
| 2025 | EEGMamba | [EEGMamba: An EEG Foundation Model with Mamba](https://doi.org/10.1016/j.neunet.2025.107816) | ✅ [wjq-learning/EEGMamba](https://github.com/wjq-learning/EEGMamba) | EEG | Same / extended corpora as CBraMod | Replaces transformer with **Mamba (state space model)** backbone; more efficient sequence modeling with competitive / better performance. |
| 2025 | MIRepNet | [MIRepNet: A Pipeline and Foundation Model for EEG-Based Motor Imagery Classification](https://arxiv.org/abs/2507.20254) | ✅ [staraink/MIRepNet](https://github.com/staraink/MIRepNet) | EEG | 5 public MI datasets | First **paradigm-specific** FM (motor imagery only); neurophysiology-informed channel template + hybrid supervised/self-supervised pretraining. |
| 2025 | EEGDM | [EEGDM: Learning EEG Representation with Latent Diffusion Model](https://arxiv.org/abs/2508.20705) | 🔧 (paper only) | EEG | Moderate-scale multi-dataset EEG | Uses **latent diffusion** as self-supervised objective: EEG encoder produces latent used to condition diffusion generator; latent also used for downstream tasks. |
| 2025 | Uni-NTFM | [Uni-NTFM: A Unified Foundation Model for EEG Signal Representation Learning](https://arxiv.org/abs/2509.24222) | 🔧 (paper only) | EEG | 28,000+ hours | Neuroscience-inspired **time / freq / raw decoupled encoders**, topological embeddings for 10–20 standards, and Mixture-of-Experts transformer; very large (up to 1.9B params). |
| 2025 | LCM | [Large Cognition Model: Towards Pretrained EEG Foundation Model](https://arxiv.org/abs/2502.17464) | 🔧 (paper only) | EEG | Large-scale EEG, details in paper | Transformer EEG FM with temporal+spectral attention; emphasizes cross-task generalization (cognitive state, disease, neurofeedback). |
| 2025 | EEG-DINO (name varies) | [EEG-DINO: Learning EEG Foundation Models via Hierarchical Self-Distillation](https://www.researchgate.net/publication/395706635_EEG-DINO_Learning_EEG_Foundation_Models_via_Hierarchical_Self-distillation) | 🔧 (paper only) | EEG | Multi-dataset EEG | DINO-style self-distillation adapted to EEG; highlights importance of multiscale spatio-temporal features and multi-task training. |

---

### 1.2 Broader “EEG foundation” / analysis papers

These are not single reusable checkpoints but analyze properties of EEG FMs or propose training pipelines.

| Year | Work | Paper | Code | Notes |
|------|------|-------|------|-------|
| 2025 | EEG FM for BCI diversity | [EEG Foundation Models for BCI Learn Diverse Features of Electrophysiology](https://arxiv.org/abs/2506.01867) | 🔧 (paper; implementation details in text) | Analyzes self-supervised transformer FMs (HuBERT-style) for BCI; shows models capture alpha rhythms, subject identity, etc., beyond classic task labels. |

---

## 2. Multimodal / LLM-style EEG models

### 2.1 EEG + language / instruction-following

| Year | Model | Paper | Code | Modality | Highlights |
|------|-------|-------|------|----------|------------|
| 2024 | EEG-GPT (Kim et al.) | [EEG-GPT: Exploring Capabilities of Large Language Models for EEG Classification and Interpretation](https://arxiv.org/abs/2401.18006) | 🔧 (no official repo) | EEG + LLM tools | Uses an LLM as a **controller** over traditional EEG tools; few-shot abnormal vs normal classification with explicit chain-of-thought interpretations. |
| 2024 | NeuroLM | [NeuroLM: A Universal Multi-task Foundation Model for Bridging the Gap between Language and EEG Signals](https://arxiv.org/abs/2409.00101) | ✅ [935963004/NeuroLM](https://github.com/935963004/NeuroLM) | EEG + text | Treats EEG as a “foreign language” via discrete neural tokens; **LLM backbone** supports instruction-tuned multi-task EEG decoding, report generation, etc. |
| 2025 | WaveMind | [WaveMind: Towards a Conversational EEG Foundation Model Aligned to Textual and Visual Modalities](https://arxiv.org/abs/2510.00032) | 🔧 (paper only) | EEG + text + vision | MLLM-style conversational model; aligns EEG with text / images; introduces **WaveMind-Instruct-338k** for instruction tuning across tasks. |
| 2025 | Neurosity EEG-GPT | (no formal paper; project repo) | ⛏️ [neurosity/EEG-GPT](https://github.com/neurosity/EEG-GPT) | EEG (Neurosity Crown) | Practical project to build a foundation model for Crown device data based on **Neuro-GPT**; currently under active development. |

---

### 2.2 Clinical & cross-modality brain-signal FMs

| Year | Model | Paper | Code | Modality | Highlights |
|------|-------|-------|------|----------|------------|
| 2023 | BrainBERT | [BrainBERT: Self-Supervised Representation Learning for Intracranial Recordings](https://arxiv.org/abs/2302.14367) | 🔧 [lab/BrainBERT2023 (resources page)](https://klab.tch.harvard.edu/resources/BrainBERT2023.html) | iEEG / ECoG | ICLR 2023. Transformer FM for intracranial signals; masked spectrogram modeling; strong cross-subject and cross-task generalization. |
| 2024 | BrainWave / Brant-2 | [BrainWave: A Brain Signal Foundation Model for Clinical Applications](https://arxiv.org/abs/2402.10251) | ✅ [yzz673/Brant-2](https://github.com/yzz673/Brant-2) | EEG + invasive clinical recordings | 40,000+ hours, 16k subjects; large FM across EEG + other electrophysiology for clinical diagnosis / disorder classification. |
| 2024 | NDL-Brain | [Nested Deep Learning Model Towards A Foundation Model for Brain Signal Data](https://arxiv.org/abs/2410.03191) | 🔧 (paper + references) | EEG + MEG | “Nested” architecture that can adapt to varying channel layouts; focuses on spike detection and channel localization (epilepsy). |

---

### 2.3 EEG → vision / generative decoding

| Year | Model | Paper | Code | Modality | Highlights |
|------|-------|-------|------|----------|------------|
| 2025 | EEGMirror | [EEGMirror: Leveraging EEG Data in the Wild via Montage-Agnostic Self-Supervision for EEG to Video Decoding](https://openaccess.thecvf.com/content/ICCV2025/papers/Liu_EEGMirror_Leveraging_EEG_Data_in_the_Wild_via_Montage-Agnostic_Self-Supervision_ICCV_2025_paper.pdf) | 🔧 (paper; possibly code later) | EEG → Video | ICCV 2025. Montage-agnostic pipeline that learns from wild EEG data to decode **high-fidelity video**; uses self-supervised EEG pretraining. |

---

## 3. Benchmarks & toolkits for EEG foundation models

These are essential for fairly comparing EEG FMs.

| Year | Name | Paper | Code | Scope | Notes |
|------|------|-------|------|-------|------|
| 2025 | EEG-FM-Bench | [EEG-FM-Bench: A Comprehensive Benchmark for the Systematic Evaluation of EEG Foundation Models](https://arxiv.org/abs/2508.17742) | ✅ [xw1216/EEG-FM-Bench](https://github.com/xw1216/EEG-FM-Bench) | 14 datasets, 10 paradigms | First comprehensive FM-centric benchmark; defines protocols for **frozen / full fine-tuning / multitask** evaluation across motor imagery, sleep, emotion, seizure, AD, etc. |
| 2025 | EEG-Bench | [EEG-Bench: A Benchmark for EEG Foundation Models in Clinical Applications](https://openreview.net/forum?id=MAudehqShe) | ✅ (link in OpenReview; ETH Zürich group) | Clinical EEG | Focuses on **clinical tasks** (diagnosis, pathology) and compares FMs vs classical baselines; designed for ease-of-use and automated dataset download. |

---

## 4. How to extend this repo

### 4.1 Adding a new model

Please open a PR that adds a new row to the appropriate table with:

- **Year** of first public preprint or publication  
- **Model** name (short, consistent with paper / repo)  
- **Paper**: a Markdown link to arXiv / conference / journal  
- **Code**: GitHub or other public repo  
- **Modality**: `EEG`, `iEEG`, `EEG + text`, `EEG + vision`, etc.  
- **Pretraining data**: rough hours and number of datasets (if reported)  
- **Highlights**: one short sentence on what’s unique (architecture, objective, dataset scale, etc.)

### 4.2 Inclusion criteria

A work is “in scope” if:

1. It pretrains on **multiple EEG/brain-signal datasets** or on a large, diverse single dataset with explicit re-use across tasks; **and/or**  
2. It explicitly positions itself as a **foundation / generalist / universal / large** model for EEG / brain signals; **or**  
3. It defines a **benchmark** specifically aimed at evaluating such models.

Small task-specific CNNs / RNNs trained from scratch on a single dataset are out of scope.

---

## 5. Citation

If this repo helps your research, you can cite it as:

```bibtex
@misc{the_existing_eeg_fm_repo,
  title        = {The-existing-EEG-Foundation-Pretrained-Model},
  author       = {Maintainers of the GitHub repository},
  year         = {2025},
  howpublished = {\url{https://github.com/<your-username>/The-existing-EEG-Foundation-Pretrained-Model}}
}
