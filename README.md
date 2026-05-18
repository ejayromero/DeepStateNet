# DeepStateNet: A Deep Learning-Based Multimodal EEG Classifier for Motor Imagery Using Microstate Dynamics

> Master Project — [EPFL](https://www.epfl.ch) × [Nanyang Technological University (NTU)](https://www.ntu.edu.sg), Singapore  
> **Author:** Eliser Josan Romero
> **EPFL Supervisor:** Prof. [Silvestro Micera](https://www.epfl.ch/labs/biorob/people/micera/) · **NTU Supervisor:** Prof. [Cuntai Guan](https://dr.ntu.edu.sg/cris/rp/rp00203)  
> **Date:** August 29, 2025

<!-- SUGGESTION: Add a banner image here — e.g., the DeepStateNet architecture figure (Figure 4 from the thesis) -->
<!-- ![DeepStateNet Architecture](assets/figures/deepstatenet_architecture.png) -->

---

## Table of Contents

- [Overview](#overview)
- [Background](#background)
- [Architecture](#architecture)
  - [DeepConvNet (Baseline)](#deepconvnet-baseline)
  - [MicroStateNet](#microstatenet)
  - [DeepStateNet](#deepstatenet)
- [Dataset](#dataset)
- [Microstate Extraction](#microstate-extraction)
- [Results](#results)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [References](#references)
- [Acknowledgements](#acknowledgements)

---

## Overview

This project investigates whether EEG **microstate dynamics** — quasi-stable patterns of whole-brain electrical activity lasting ~30–120 ms — can complement raw EEG signals in a deep learning pipeline for **Motor Imagery (MI) Brain-Computer Interface (BCI)** classification.

Three models are developed and compared:

| Model | Input | Avg. Balanced Accuracy | Avg. F1 Macro |
|---|---|---|---|
| **DeepConvNet** (baseline) | Raw EEG | 73.8% | 68.6% |
| **MicroStateNet** (novel) | Microstate sequences | 52.1–60.0% | 44.1–56.8% |
| **DeepStateNet** (novel) | Raw EEG + Microstates | **84.3–86.2%** | **84.0–85.9%** |

The key finding is that while microstate features alone are insufficient, **fusing them with EEG signals in a multimodal architecture yields a significant improvement** over the EEG-only baseline.

---

## Background

### Brain-Computer Interfaces & Motor Imagery

[Brain-Computer Interfaces (BCIs)](https://doi.org/10.1016/j.cobme.2017.11.004) establish direct communication pathways between the brain and external devices. Among BCI paradigms, **Motor Imagery (MI)** — the mental simulation of motor actions — modulates sensorimotor rhythms (α: 8–12 Hz; β: 18–26 Hz) over the contralateral cortex, enabling device control without physical movement. MI-BCIs are particularly relevant for stroke, ALS, spinal cord injury, and Parkinson's disease rehabilitation.

[EEG](https://doi.org/10.1109/TNSRE.2023.3330500) is the dominant modality for non-invasive BCIs due to its safety, portability, and high temporal resolution.

### EEG Microstates

[EEG microstates](https://doi.org/10.1016/j.neuroimage.2017.11.062) are brief (~30–120 ms), quasi-stable topographic maps of scalp electric potential that reflect coordinated large-scale brain network activity. Often described as the *"atoms of thought"*, they capture whole-brain temporal dynamics at millisecond resolution. Resting-state EEG typically reveals 4–5 canonical microstate classes, each associated with distinct functional networks (auditory, visual, salience, attention).

[Recent work](https://doi.org/10.3390/brainsci13091288) has shown that microstate parameters (duration, occurrence, coverage, transition probabilities) correlate with MI-BCI performance, motivating their use as features in a deep learning classifier.

---

## Architecture

<!-- SUGGESTION: Place the DeepStateNet architecture diagram (Figure 4) here -->
<!-- ![DeepStateNet Architecture](assets/figures/fig4_deepstatenet_architecture.png) -->

### DeepConvNet (Baseline)

Based on [Schirrmeister et al. (2017)](https://doi.org/10.1002/hbm.23730), implemented via [Braindecode](https://braindecode.org) `Deep4Net`. Takes raw EEG `(time × channels)` as input and passes it through four successive convolution–pooling blocks (25→50→100→200 filters), followed by a softmax classifier.

### MicroStateNet

A novel 1D CNN adapted from DeepConvNet for microstate sequences. Key design choices:

- **1D convolutions** (no spatial dimension in microstate sequences)
- **Embedding encoding** of categorical microstate labels (preferred over one-hot encoding; statistically significant improvement in F1, p < 0.01)
- **Multiscale parallel branches** with kernel sizes k=3 (~12 ms), k=11 (~44 ms), k=25 (~100 ms) to capture the full range of microstate durations
- Input concatenates microstate label sequence with normalized Global Field Power (GFP)

<!-- SUGGESTION: Place MicroStateNet architecture diagram (Figure 1) here -->
<!-- ![MicroStateNet Architecture](assets/figures/fig1_microstatenet_architecture.png) -->

<!-- SUGGESTION: Place Multiscale MicroStateNet diagram (Figure 3) here -->
<!-- ![Multiscale MicroStateNet](assets/figures/fig3_multiscale_msn.png) -->

### DeepStateNet

A **multimodal fusion model** combining DeepConvNet (EEG branch) and MicroStateNet (microstate branch) in parallel. Features from both branches are concatenated before a shared classification head (512→256→classes). This late-fusion design preserves spectral and spatial information from raw EEG while incorporating neurologically structured microstate features.

**Final selected configuration:**
- Number of clusters: **K = 12**
- Architecture: **Multiscale** (3 parallel branches)
- Encoding: **Embedding**

---

## Dataset

Data originates from the [CASTNet study](https://doi.org/10.1109/IJCNN60899.2024.10651226) (Ng et al., 2024), collected at CBCR, NTU Singapore (IRB approval: IRB-2018-06-030).

| Property | Value |
|---|---|
| Subjects | 50 healthy adults (31M / 19F, mean age 27.0 ± 5.2 years) |
| Task | Right-hand open / close motor attempt |
| EEG system | BrainAmp ActiCHamp, 61 electrodes, 10-20 system |
| Sampling rate | 1000 Hz |
| Trials per subject | ~358 (balanced: 50% Rest, 25% Open, 25% Close) |
| Preprocessing | 0.3–40 Hz bandpass, CAR, FastICA artifact removal |
| Validation | 5-fold cross-validation (subject-dependent) |
| Metrics | Balanced Accuracy, F1 Macro (imbalanced classes) |

---

## Microstate Extraction

Microstates are extracted using [Modified K-Means clustering](https://doi.org/10.1109/10.391164) (ModKMeans) at Global Field Power (GFP) peaks, implemented in [Pycrostates](https://doi.org/10.21105/joss.04564) (v0.7.0).

**Cluster number selection** balances three metrics across K = 4–50:
- **GEV** (Global Explained Variance) — plateau at K = 12
- **Spatial Correlation** — plateau at K = 11
- **Calinski-Harabasz Index** — knee at K = 17

→ **K = 12** adopted as compromise (GEV ≈ 66.4%, Spatial Correlation ≈ 69.7%).

<!-- SUGGESTION: Place the clustering metrics figure (Figure 5) here -->
<!-- ![Clustering Metrics](assets/figures/fig5_clustering_metrics.png) -->

<!-- SUGGESTION: Place microstate sequence examples (Figure 8) here -->
<!-- ![Microstate Sequences](assets/figures/fig8_microstate_sequences.png) -->

---

## Results

<!-- SUGGESTION: Place the bar plot of model performance (Figure 9) here -->
<!-- ![Model Performance Bar Plot](assets/figures/fig9_model_performance_bar.png) -->

<!-- SUGGESTION: Place the final boxplot comparison (Figure 14) here -->
<!-- ![Final Model Comparison](assets/figures/fig14_final_comparison_boxplot.png) -->

All pairwise comparisons (DCN vs MSN, DCN vs DSN, MSN vs DSN) are statistically significant (p < 0.001, Bonferroni-corrected), with large effect sizes (|Cohen's d| > 1).

### Final Model Comparison (Selected Parameters)

| Comparison | Bal. Acc. Difference | Cohen's d | p-value |
|---|---|---|---|
| DCN vs MSN | +20.3% (DCN higher) | 1.96 | < 0.001 |
| DCN vs DSN | −11.0% (DSN higher) | −1.09 | < 0.001 |
| MSN vs DSN | −31.3% (DSN higher) | −3.87 | < 0.001 |

**Key takeaway:** Microstate features alone are insufficient (MSN < DCN), but their fusion with EEG in DeepStateNet yields a statistically significant, large-effect improvement over the EEG-only baseline.

---

## Repository Structure

```
.
├── README.md
<!-- SUGGESTION: Fill in your actual directory structure here -->
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
```

**Key dependencies:**

| Package | Version | Purpose |
|---|---|---|
| Python | 3.11.9 | — |
| PyTorch | — | Model implementation |
| Braindecode | 0.8.1 | DeepConvNet (Deep4Net) |
| Pycrostates | 0.7.0 | ModKMeans microstate clustering |
| MNE-Python | — | EEG processing & visualization |

<!-- SUGGESTION: Add a requirements.txt or environment.yml to the repo and link it here -->

---

## Usage

<!-- SUGGESTION: Fill in with your actual scripts/notebooks once you share the repo structure -->

```bash
# Example: Extract microstates for all subjects
python src/extract_microstates.py --n_clusters 12

# Example: Train DeepStateNet
python src/train.py --model deepstatenet --clusters 12 --encoding embedding --multiscale

# Example: Evaluate and compare all models
python src/evaluate.py
```

---

## References

- McFarland & Wolpaw (2017). [EEG-based brain–computer interfaces](https://doi.org/10.1016/j.cobme.2017.11.004). *Current Opinion in Biomedical Engineering*.
- Edelman et al. (2025). [Non-Invasive Brain-Computer Interfaces: State of the Art and Trends](https://doi.org/10.1109/RBME.2024.3449790). *IEEE Reviews in Biomedical Engineering*.
- Ng et al. (2024). [CASTNet: Cycle-Consistent Attention-based Network for Decoding Open/Close Hand Movement Attempts using EEG](https://doi.org/10.1109/IJCNN60899.2024.10651226). *IJCNN 2024*.
- Schirrmeister et al. (2017). [Deep learning with convolutional neural networks for EEG decoding and visualization](https://doi.org/10.1002/hbm.23730). *Human Brain Mapping*.
- Lawhern et al. (2018). [EEGNet: A Compact Convolutional Network for EEG-based BCIs](https://doi.org/10.1088/1741-2552/aace8c). *Journal of Neural Engineering*.
- Cui et al. (2023). [Predicting Motor Imagery BCI Performance Based on EEG Microstate Analysis](https://doi.org/10.3390/brainsci13091288). *Brain Sciences*.
- Michel & Koenig (2018). [EEG microstates as a tool for studying temporal dynamics of whole-brain neuronal networks](https://doi.org/10.1016/j.neuroimage.2017.11.062). *NeuroImage*.
- Pascual-Marqui et al. (1995). [Segmentation of brain electrical activity into microstates](https://doi.org/10.1109/10.391164). *IEEE Transactions on Biomedical Engineering*.
- Michel et al. (2024). [Current State of EEG/ERP Microstate Research](https://doi.org/10.1007/s10548-024-01037-3). *Brain Topography*.
- Férat et al. (2022). [Pycrostates: a Python library to study EEG microstates](https://doi.org/10.21105/joss.04564). *Journal of Open Source Software*.

---

## Acknowledgements

This work was conducted at the [Centre for Brain-Computing Research (CBCR)](https://www.ntu.edu.sg/cbcr), Nanyang Technological University, Singapore. Special thanks to Prof. Cuntai Guan, Shuailei, Xiaohao, Prof. Silvestro Micera, Dana, and my family.

> *"Ang hindi marunong lumingon sa pinanggalingan ay hindi makararating sa paroroonan."*  
> ("He who does not look back to where he came from will never reach his destination.") — Filipino proverb

---

<p align="center">
  <img src="https://www.epfl.ch/wp-content/themes/wp-theme-2018/assets/svg/epfl-logo.svg" height="40" alt="EPFL"/>
  &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="https://www.ntu.edu.sg/images/librariesprovider86/ntu-logo/ntu-logo-colour.png" height="40" alt="NTU"/>
</p>
