# EEG-Based Age Estimation


This project develops a machine learning model to estimate chronological age from resting-state electroencephalography (EEG) data. Brain oscillations exhibit systematic changes across the lifespan, providing a biological signal for age prediction. We extract spectral features from raw EEG recordings and employ robust cross-validation methods to evaluate predictive performance on a small dataset.

## Dataset

We use publicly available resting-state EEG data from the OpenNeuro repository (OpenNeuro dataset, BIDS-formatted). The dataset comprises 111 partecipant, we dowload 60s with complete spectral recordings. 

e Hatlestad-Hall, C., Rygvold, T. W., & Andersson, S. (2022). BIDS-structured resting-state electroencephalography (EEG) data extracted from an experimental paradigm. *Data in Brief*, 45, 108647. https://doi.org/10.1016/j.dib.2022.108647

## Methodology


Raw EEG signals are processed via Fast Fourier Transform (FFT) to extract spectral features across standard frequency bands. This yields a high-dimensional feature matrix suitable for machine learning.

We apply a standardized preprocessing pipeline: StandardScaler normalizes the feature space, SelectKBest (f_classif, k=50) reduces dimensionality by selecting the 50 most informative features, and GradientBoostingClassifier performs age prediction. 

With 64 subjects, standard train-test splits lose valuable training data. So Leave-One-Out Cross-Validation (LOO) is use, which uses 63 subjects for training and tests on each held-out subject sequentially.Out-of-fold predictions are aggregated for final evaluation.

To asses final result: accuracy, balanced accuracy (correcting for class imbalance if present), ROC-AUC, per-class precision/recall via classification report, and confusion matrices. ROC curves are generated from aggregated LOO probabilities. Feature importance is assessed from a final model trained on the complete dataset to identify which spectral features drive age predictions.


