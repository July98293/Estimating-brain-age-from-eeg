"""
Resting-state EEG spectral feature extraction.

For every subject/session with a resteyesc EDF file, this script:
  1. Loads the raw EEG (MNE read_raw_edf)
  2. Applies a 50 Hz notch filter and 0.5–45 Hz bandpass
  3. Computes Welch PSD (2-second windows, 50% overlap)
  4. Extracts per-channel spectral features:
       - Absolute and relative band power (delta, theta, alpha, beta, gamma)
       - Spectral entropy
       - Alpha-band peak frequency
       - Spectral edge frequency (SEF95)
  5. Saves one CSV row per (subject, session).

Output: spectral_features.csv in the dataset root.
"""

import os
import warnings
import numpy as np
import pandas as pd
import mne
from scipy.signal import welch
from scipy.integrate import simpson

warnings.filterwarnings("ignore", category=RuntimeWarning)
mne.set_log_level("ERROR")

# ── paths ────────────────────────────────────────────────────────────────────
DATASET_ROOT = r"c:\Users\giuli\Downloads\from eeg to age"
DATA_DIR = os.path.join(DATASET_ROOT, "data")
PARTICIPANTS_TSV = os.path.join(DATASET_ROOT, "participants.tsv")
OUTPUT_CSV = os.path.join(DATASET_ROOT, "spectral_features.csv")

# ── band definitions (Hz) ────────────────────────────────────────────────────
BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

SFREQ_TARGET = 256          # resample to 256 Hz to speed things up
WELCH_WIN_SEC = 2           # Welch window length in seconds
L_FREQ = 0.5
H_FREQ = 45.0
NOTCH_FREQ = 50.0


# ── helpers ──────────────────────────────────────────────────────────────────
def band_power(freqs, psd, fmin, fmax):
    """Absolute power (µV²) in [fmin, fmax] via Simpson integration."""
    mask = (freqs >= fmin) & (freqs <= fmax)
    return simpson(psd[mask], x=freqs[mask])


def spectral_entropy(freqs, psd, fmin=0.5, fmax=45.0):
    """Normalised spectral entropy over [fmin, fmax]."""
    mask = (freqs >= fmin) & (freqs <= fmax)
    p = psd[mask]
    p = p / p.sum()
    p = p[p > 0]
    n = len(p)
    return -np.sum(p * np.log(p)) / np.log(n) if n > 1 else 0.0


def alpha_peak_freq(freqs, psd, fmin=8.0, fmax=13.0):
    """Frequency of maximum power in the alpha band."""
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not mask.any():
        return np.nan
    return freqs[mask][np.argmax(psd[mask])]


def sef95(freqs, psd, fmin=0.5, fmax=45.0):
    """Spectral edge frequency: frequency below which 95% of power lies."""
    mask = (freqs >= fmin) & (freqs <= fmax)
    f = freqs[mask]
    p = psd[mask]
    cumsum = np.cumsum(p)
    threshold = 0.95 * cumsum[-1]
    idx = np.searchsorted(cumsum, threshold)
    return f[min(idx, len(f) - 1)]


def extract_features_from_raw(raw):
    """Return a dict of per-channel spectral features from a preprocessed Raw."""
    n_fft = int(WELCH_WIN_SEC * raw.info["sfreq"])
    n_overlap = n_fft // 2
    data = raw.get_data(units="uV")          # shape: (n_channels, n_times)
    ch_names = raw.ch_names
    sfreq = raw.info["sfreq"]

    features = {}
    for i, ch in enumerate(ch_names):
        freqs, psd = welch(data[i], fs=sfreq, nperseg=n_fft, noverlap=n_overlap,
                           window="hann")

        # band powers
        abs_powers = {band: band_power(freqs, psd, lo, hi)
                      for band, (lo, hi) in BANDS.items()}
        total = sum(abs_powers.values())

        for band, pwr in abs_powers.items():
            features[f"{ch}_{band}_abs"] = pwr
            features[f"{ch}_{band}_rel"] = pwr / total if total > 0 else np.nan

        features[f"{ch}_total_abs"]       = total
        features[f"{ch}_spec_entropy"]    = spectral_entropy(freqs, psd)
        features[f"{ch}_alpha_peak_freq"] = alpha_peak_freq(freqs, psd)
        features[f"{ch}_sef95"]           = sef95(freqs, psd)

    return features


def load_and_preprocess(edf_path):
    """Load EDF, notch + bandpass filter, resample. Returns MNE Raw or None."""
    try:
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        raw.pick("eeg")
        raw.notch_filter(NOTCH_FREQ, verbose=False)
        raw.filter(L_FREQ, H_FREQ, fir_window="hamming", verbose=False)
        raw.resample(SFREQ_TARGET, verbose=False)
        return raw
    except Exception as exc:
        print(f"    ERROR loading {edf_path}: {exc}")
        return None


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    participants = pd.read_csv(PARTICIPANTS_TSV, sep="\t")
    # keep only subjects with a data directory
    available_subs = set(os.listdir(DATA_DIR))
    participants = participants[participants["participant_id"].isin(available_subs)]
    print(f"Subjects with data: {len(participants)}")

    rows = []
    for _, prow in participants.iterrows():
        sub = prow["participant_id"]
        sub_dir = os.path.join(DATA_DIR, sub)

        for ses in ("ses-t1", "ses-t2"):
            eeg_dir = os.path.join(sub_dir, ses, "eeg")
            if not os.path.isdir(eeg_dir):
                continue

            # find the resteyesc EDF
            edf_files = [f for f in os.listdir(eeg_dir)
                         if f.endswith(".edf") and "resteyesc" in f]
            if not edf_files:
                continue

            edf_path = os.path.join(eeg_dir, edf_files[0])
            print(f"  Processing {sub} / {ses} …", end=" ", flush=True)

            raw = load_and_preprocess(edf_path)
            if raw is None:
                continue

            feats = extract_features_from_raw(raw)

            row = {
                "participant_id": sub,
                "session": ses,
                "age": prow.get("age", np.nan),
                "sex": prow.get("sex", np.nan),
            }
            row.update(feats)
            rows.append(row)
            print("done")

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved {len(df)} rows x {len(df.columns)} columns -> {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
