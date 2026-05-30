# 🌌 Exoplanet Atmospheric Retrieval using Machine Learning

ML-based atmospheric retrieval from exoplanet transmission spectra — benchmarking MLP, SVR, and XGBoost on the Ariel Big Data Challenge dataset and real JWST observations of WASP-39b.

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-Regression-189AB4?style=for-the-badge)](https://xgboost.readthedocs.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)

---

## 📌 Overview

Atmospheric retrieval — inferring what gases and temperatures exist in a planet's atmosphere from its light spectrum — is traditionally a computationally expensive Bayesian problem, often taking hours or days per planet. This project investigates whether supervised machine learning regressors can solve the same inverse problem in a fraction of the time.

Models are trained on the **Ariel Big Data Challenge (ABC)** synthetic dataset and then tested against real **JWST NIRSpec observations of WASP-39b** — one of the best-characterised hot Jupiters and the first exoplanet to have its atmospheric chemistry confirmed by the James Webb Space Telescope.

---

## 🔭 Research Objective

Evaluate and compare classical and neural ML regressors for estimating:

- **Atmospheric gas abundances** — H₂O, CO₂, CO, CH₄, NH₃
- **Planetary equilibrium temperature**

directly from transmission spectra, and assess how preprocessing choices affect model stability and generalisation across synthetic and real observational data.

---

## 📁 Project Structure

```
Exoplanet-Atmospheric-Retrieval/
│
├── 📂 Data Visualization/              # Spectral plots and result visualisations
├── 📂 Dataset Verification/            # Sanity checks on input data
│
├── 📂 Extraction of ABC data/          # Loading and parsing the ABC dataset
├── 📂 Extraction of ABC features/      # Feature engineering on ABC spectra
│
├── 📂 Extraction of WASP-39B data/     # Loading JWST NIRSpec observational data
├── 📂 Preprocessing of WASP-39B data/  # Aligning WASP-39b spectra to model input format
│
├── 📂 Model Training Gases/            # Regressor training for gas abundance targets
├── 📂 Model Training Temperature/      # Regressor training for temperature targets
│
└── 📖 README.md
```

---

## 🗂️ Datasets

### Ariel Big Data Challenge (ABC) Dataset
Synthetic transmission spectra generated from radiative transfer simulations, covering a wide range of planetary atmospheric compositions. Used for all model training and validation. The dataset was released as part of the ESA Ariel mission's ML challenge to accelerate atmospheric retrieval at scale.

### JWST NIRSpec — WASP-39b
Real observational spectra from the James Webb Space Telescope's NIRSpec instrument, targeting WASP-39b — a hot Saturn-mass planet orbiting a sun-like star ~700 light-years away. WASP-39b is the benchmark planet for atmospheric studies; JWST confirmed the presence of CO₂, CO, H₂O, SO₂, and other molecules in its atmosphere in 2022–2023. These observations serve as the out-of-distribution generalisation test for the trained models.

---

## ⚙️ Preprocessing Pipeline

Preprocessing was found to have a **dominant impact** on model performance — in some cases more than the choice of model itself.

| Step | Description |
|---|---|
| **Wavelength Interpolation** | Spectra resampled to a common 52-bin wavelength grid for consistent input dimensionality |
| **Feature Augmentation** | Statistical descriptors (mean, std, gradients) added to the raw spectral bins |
| **Log-Scaling of Targets** | Gas abundances are inherently log-distributed; log-transform stabilises regression |
| **Sample-wise Normalisation** | Applied per-spectrum to remove flux offset variation between observations |

> ⚠️ **Key finding:** Standard (z-score) scaling of targets led to collapsed or unstable models. Log-scaling of gas abundances was essential for meaningful predictions.

---

## 🤖 Models

Three supervised regressors were implemented and benchmarked under identical preprocessing conditions:

| Model | Notes |
|---|---|
| **Multi-Layer Perceptron (MLP)** | Fully connected neural network; sensitive to scaling choices |
| **Support Vector Regression (SVR)** | Kernel-based; effective for small-to-medium data regimes |
| **XGBoost Regression** | Gradient-boosted trees; most stable across both datasets |

Hyperparameter optimisation was performed using **Optuna** where feasible. Each model was trained separately for gas abundance targets and temperature.

---

## 📊 Results

### Gas Abundance Predictions on WASP-39b (XGBoost — best model)

| Gas | XGB (N) Predicted | Published Reference | Source |
|---|---|---|---|
| log H₂O | −5.99 | −4.85 ± 0.38 | Constantinou et al. |
| log CO₂ | −6.35 | −6.59 to −4.16 | Constantinou et al. |
| log CO | −3.99 | −4.25 to −2.58 | Constantinou et al. |
| log CH₄ | −4.76 | < −5.3 | Ahrer et al. |
| log NH₃ | −6.46 | < −6 | Alderson et al. |

### Temperature Prediction on WASP-39b

| Model | Preprocessing | Predicted Temp (K) | Actual Temp (K) |
|---|---|---|---|
| XGBoost | NMM | **958.40 K** | ~1100 K |
| SVR | N | 102.91 K | ~1100 K |
| MLP | N | Collapsed (R² ≈ −0.99) | ~1100 K |

### Impact of Preprocessing on Model Stability

| Preprocessing | XGB | SVR | MLP |
|---|---|---|---|
| N (Z-score, sample-wise) | ✅ Stable | ✅ Stable | ✅ R² ≈ 0.99 |
| NMS (N + stats augmentation) | ✅ Stable | ✅ Stable | ✅ Stable |
| NM (Max normalisation) | ⚠️ Degraded | ⚠️ Degraded | ⚠️ R² 0.6–0.9 |
| NMM (NM + augmentation) | ⚠️ Degraded | ⚠️ Degraded | ⚠️ Degraded |
| S (Global standardisation) | ❌ Collapsed | ❌ Collapsed | ❌ Collapsed |

> **Key finding:** Global standardisation caused complete model collapse across all architectures — compressing the dynamic range of spectral absorption features destroys the signal. Sample-wise normalisation (N/NMS) is essential.

### Summary

- Preprocessing choices dominated model behaviour more than architectural differences
- **XGBoost** was the only model that both performed well on synthetic data *and* generalised to real JWST observations
- MLP achieved the highest R² on synthetic validation (≈ 0.99) but failed to generalise to WASP-39b
- SVR was stable for temperature but unreliable for gas abundances
- XGBoost predictions for all five gases showed qualitative agreement with published JWST retrieval analyses

---

## 🛠️ Tech Stack

- **Python 3.8+**
- **Scikit-learn** — MLP, SVR, preprocessing pipelines
- **XGBoost** — gradient-boosted regression
- **Optuna** — hyperparameter optimisation
- **NumPy & Pandas** — data handling
- **Matplotlib** — visualisation
- **Jupyter Notebook** — experiments and analysis

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Hamshith/Exoplanet-Atmospheric-Retrieval.git
cd Exoplanet-Atmospheric-Retrieval
```

### 2. Install dependencies

```bash
pip install numpy pandas scikit-learn xgboost optuna matplotlib jupyterlab
```

### 3. Run the notebooks

Open any folder in order — a suggested flow:

```
1. Extraction of ABC data        → Load and explore the training dataset
2. Dataset Verification          → Confirm data integrity
3. Extraction of ABC features    → Build feature vectors from spectra
4. Model Training Gases          → Train and evaluate gas abundance models
5. Model Training Temperature    → Train and evaluate temperature model
6. Extraction of WASP-39B data   → Load real JWST observations
7. Preprocessing of WASP-39B data→ Align to model input format
8. Data Visualization            → Inspect predictions vs. published results
```

```bash
jupyter lab
```

---


## 👥 About

Developed by students of the **Department of Information Science & Engineering, BMS College of Engineering (BMSCE), Bengaluru** as part of an academic research project in the intersection of astrophysics and machine learning.

---

## 📄 License

MIT License
