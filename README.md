# EEG-Based Age Estimation

Binary classification of age group (young / older) from resting-state EEG spectral features.

This project develops a machine learning model to estimate chronological age from resting-state electroencephalography (EEG) data.

Brain oscillations exhibit systematic changes across the lifespan, providing a biological signal for age prediction (or better, brain age predction!). We extract spectral features from raw EEG recordings and employ robust cross-validation methods to evaluate predictive performance on a small dataset.

## Dataset

We use publicly available resting-state EEG data from the OpenNeuro repository (OpenNeuro dataset, BIDS-formatted). The dataset comprises 111 partecipant, we dowload 60s with complete spectral recordings. 

e Hatlestad-Hall, C., Rygvold, T. W., & Andersson, S. (2022). BIDS-structured resting-state electroencephalography (EEG) data extracted from an experimental paradigm. *Data in Brief*, 45, 108647. https://doi.org/10.1016/j.dib.2022.108647

## Methodology


Raw EEG signals are processed via Fast Fourier Transform (FFT) to extract spectral features across standard frequency bands. This yields a high-dimensional featurematrix.

For each EDF file:
1. Picks all 64 EEG channels
2. Applies 50 Hz notch filter and 0.5–45 Hz bandpass (Hamming)
3. Resamples to 256 Hz
4. Computes Welch PSD (2-second Hann windows, 50% overlap)
5. Extracts per-channel features for each of the 5 standard bands:

| Feature | Description |
|---|---|
| `{ch}_delta/theta/alpha/beta/gamma_abs` | Absolute band power (µV²) |
| `{ch}_delta/theta/alpha/beta/gamma_rel` | Relative band power (fraction of total) |
| `{ch}_total_abs` | Total broadband power |
| `{ch}_spec_entropy` | Normalised spectral entropy |
| `{ch}_alpha_peak_freq` | Individual alpha peak frequency |
| `{ch}_sef95` | Spectral edge frequency (95%) |

Output: `out/spectral_features.csv` — 90 rows × 900 columns.

With 64 subjects, standard train-test splits lose valuable training data. So Leave-One-Out Cross-Validation (LOO) is use, which uses 63 subjects for training and tests on each held-out subject sequentially.Out-of-fold predictions are aggregated for final evaluation.

To asses final result: accuracy, balanced accuracy (correcting for class imbalance if present), ROC-AUC, per-class precision/recall via classification report, and confusion matrices. ROC curves are generated from aggregated LOO probabilities. Feature importance is assessed from a final model trained on the complete dataset to identify which spectral features drive age predictions.

`model/determine_age.py`

- **Label**: young (age < 35) = 0, older (age ≥ 35) = 1
- **Input**: ses-t1 only (64 subjects, 896 spectral features)
- **Pipeline**: `StandardScaler → SelectKBest(f_classif, k=50) → GradientBoostingClassifier`
- **Evaluation**: Leave-One-Out CV — trains on 63 subjects, tests on 1, repeated 64 times

| Metric | LOO-CV result |
|---|---|
| Accuracy | 0.703 |
| Balanced accuracy | 0.688 |
| ROC-AUC | 0.818 |

---

## How to run

```bash
# activate the venv
.venv\Scripts\activate

# 1. extract features (~5 min)
python model/extract_spectral_features.py

# 2. train and evaluate (~2 min)
python model/determine_age.py
```

---

## Dependencies

`mne`, `scipy`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `joblib`

```bash
pip install mne scikit-learn matplotlib
```

