# The-existing-EEG-Foundation-Pretrained-Model

A curated list of existing **EEG foundation / large pretrained models**, including paper links, descriptions, and open-source repositories when available.

---

## Motivation

Recent years have seen the emergence of **EEG foundation models (EEG-FMs)** and large pretrained EEG models, inspired by the success of foundation models in NLP and CV. These models are typically:

- trained in a **self-supervised or multi-task** manner on **large-scale EEG datasets**,  
- designed to provide **general-purpose, reusable EEG representations** across datasets and tasks,  
- sometimes integrated with language models or multi-modal architectures.

This repository collects existing EEG foundation models with:

- Model name & year  
- Paper link  
- Code repository  
- Brief technical notes  

---

## Inclusion Criteria

A model is included if:

1. It has a peer-reviewed paper or arXiv preprint, and  
2. It is explicitly positioned as an EEG/brain foundation model **or**  
3. It is a large-scale self-supervised EEG model intended for broad transfer  
4. Code availability is preferred but not required  

Models are grouped into:

- EEG-only foundation models  
- Multi-modal / brain-signal foundation models including EEG  
- EEG-language / instruction models  
- Benchmarks and toolkits  
- Important legacy self-supervised models  

---

## EEG Foundation Models (EEG-only)

| Model | Year | Paper | Code | Notes |
|------|------|-------|------|-------|
| **BENDR** | 2021 | “BENDR: Using Transformers and a Contrastive Self-Supervised Learning Task to Learn From Massive Amounts of EEG Data” | https://github.com/SPOClab-ca/BENDR | Early transformer-based SSL model for EEG; cross-dataset generalization. |
| **EEGFormer** | 2024 | “EEGFormer: Towards Transferable and Interpretable Large-Scale EEG Foundation Model” | N/A | VQ-transformer trained on large-scale EEG via masked reconstruction. |
| **LaBraM** | 2024 | “Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI” | https://github.com/935963004/LaBraM | Large-scale EEG FM with VQ tokenizer + masked code prediction. |
| **NeuroGPT** | 2024 | “Neuro-GPT: Towards A Foundation Model for EEG” | https://github.com/wenhui0206/NeuroGPT | EEG encoder + GPT-style decoder for masked reconstruction. |
| **EEGPT (NeurIPS 2024)** | 2024 | “EEGPT: Pretrained Transformer for Universal and Reliable Representation of EEG Signals” | https://github.com/BINE022/EEGPT | Dual masked SSL objective for robust universal representation. |
| **EEGPT (Generalist FM)** | 2024 | “EEGPT: Unleashing the Potential of EEG Generalist Foundation Model by Autoregressive Pre-training” | Partially available | Up to 1.1B parameters; autoregressive pretraining; electrode-wise modeling. |
| **ALFEE** | 2025 | “ALFEE: Adaptive Large Foundation Model for EEG Representation” | https://github.com/xw1216/ALFEE | Multi-task pretraining with temporal and channel encoders. |
| **CBraMod** | 2025 | “CBraMod: A Criss-Cross Brain Foundation Model for EEG Decoding” | https://github.com/wjq-learning/CBraMod | Criss-cross transformer with spatial/temporal attention branches. |
| **EEGMamba** | 2025 | “EEGMamba: An EEG foundation model with Mamba” | https://github.com/wjq-learning/EEGMamba | State-space model (Mamba) for long-sequence EEG modeling. |
| **GEFM** | 2024–2025 | “GEFM: Graph-Enhanced EEG Foundation Model” | N/A | Integrates GNN electrode topology with transformer MAE. |
| **MIRepNet** | 2025 | “MIRepNet: A Pipeline and Foundation Model for EEG-Based Motor Imagery Classification” | https://github.com/staraink/MIRepNet | MI-specific EEG FM with neuro-inspired preprocessing pipeline. |
| **Uni-NTFM** | 2025 | “Uni-NTFM: A Unified Foundation Model for EEG Signal Representation Learning” | N/A | Up to 1.9B parameters; Mixture-of-Experts with time/frequency/raw streams. |
| **EEGDM** | 2025 | “EEGDM: Learning EEG Representation with Latent Diffusion Model” | N/A | Diffusion-based self-supervised EEG representation learning. |

---

## Multi-modal Brain-Signal Foundation Models (Including EEG)

| Model | Year | Modalities | Paper | Code | Notes |
|-------|------|-----------|--------|-------|-------|
| **BrainWave / Brant-2** | 2024–2025 | EEG, iEEG, others | “BrainWave: A Brain Signal Foundation Model for Clinical Applications” | https://github.com/yzz673/Brant-2 | Large clinical-scale brain-signal FM (~40,000 hours). |
| **BrainBERT** | 2023 | iEEG | “BrainBERT: Self-Supervised Representation Learning for Intracranial Recordings” | Varies | Transformer SSL model for iEEG. |
| **NDL-Brain** | 2024 | EEG, MEG | “Nested Deep Learning Model Towards A Foundation Model for Brain Signal Data” | N/A | Nested architecture across EEG/MEG tasks. |
| **Large Brainwave FMs (Survey)** | 2025 | EEG + others | Review paper | — | Survey and evaluation of brainwave foundation models. |

---

## EEG-Language and Instruction Models

| Model | Year | Paper | Code | Notes |
|-------|------|-------|------|-------|
| **NeuroLM** | 2024–2025 | “NeuroLM: A Universal Multi-task Foundation Model for Bridging Language and EEG Signals” | https://github.com/935963004/NeuroLM | EEG tokenization + LLM instruction tuning. |
| **EEG-GPT** | 2024 | “EEG-GPT: Exploring Capabilities of Large Language Models for EEG Classification and Interpretation” | Various repos | LLM-driven EEG classification and explanation. |
| **Neurosity EEG-GPT** | 2024 | Product-oriented EEG-LM | https://github.com/neurosity/EEG-GPT | Real-time EEG model for commercial headset. |
| **Large Cognition Model (LCM)** | 2025 | “Large Cognition Model: Towards Pretrained EEG Foundation Model” | Integrated in EEG-FM-Bench | FM with temporal + spectral attention. |
| **WaveMind** | 2025 | “WaveMind: Towards a Conversational EEG Foundation Model” | https://github.com/ZiyiTsang/WaveMind | Instruction-following conversational EEG FM. |

---

## Benchmarks & Toolkits

| Name | Year | Link | Purpose |
|------|------|------|---------|
| **EEG-FM-Bench** | 2025 | https://github.com/xw1216/EEG-FM-Bench | Benchmark suite for EEG foundation models. |
| **EEG-Bench** | 2024–2025 | https://github.com/ETH-DISCO/EEG-Bench | Large-scale EEG benchmark covering 25 datasets. |

---

## Legacy Large-Scale EEG Self-Supervised Models

| Model | Year | Paper | Code | Notes |
|-------|------|-------|------|-------|
| **BENDR** | 2021 | (See above) | https://github.com/SPOClab-ca/BENDR | Proto-foundation SSL model. |
| **EEGMirror** | 2025 | “EEGMirror: Leveraging EEG Data in the Wild via Montage-Agnostic Self-Supervision” | N/A | Robust montage-agnostic SSL. |
| **EEGDM** | 2025 | (See above) | — | Diffusion-based SSL for EEG. |

---

## Contributing

Pull requests are welcome for:

- Missing models  
- Additional datasets, details, or parameter counts  
- Corrections and reorganizations  

---

## Citation

```bibtex
@misc{EEGFoundationModelList,
  title  = {The-existing-EEG-Foundation-Pretrained-Model: A curated list of EEG foundation and large pretrained models},
  author = {Community contributors},
  year   = {2025},
  note   = {GitHub repository},
}
