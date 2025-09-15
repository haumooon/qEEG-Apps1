import numpy as np
import pandas as pd
import mne
from scipy.signal import welch   # ✅ use SciPy instead of MNE psd_welch

import matplotlib
matplotlib.use("Agg")  # ✅ safe for Streamlit Cloud (headless)
import matplotlib.pyplot as plt
import io


# -------------------
# PSD computation
# -------------------

def psd_welch(raw, fmin, fmax, n_fft=1024, n_overlap=256, n_per_seg=512, verbose=False):
    """Custom Welch PSD using SciPy, compatible with Python 3.13 & Streamlit Cloud."""
    data = raw.get_data()
    sfreq = raw.info["sfreq"]
    psds = []
    for ch in data:
        freqs, pxx = welch(ch, sfreq, nperseg=n_per_seg, noverlap=n_overlap, nfft=n_fft)
        idx = (freqs >= fmin) & (freqs <= fmax)
        psds.append(pxx[idx])
    return np.array(psds), freqs[idx]


# -------------------
# Utility functions
# -------------------

def normalize_label(ch_name):
    """Normalize EEG channel labels (case-insensitive)."""
    return ch_name.strip().capitalize()


def get(df, ch, band):
    """Get a single value from dataframe with (channel, band)."""
    try:
        return float(df.loc[ch, band])
    except Exception:
        return np.nan


def compute_raw_powers(edf_path):
    """Compute relative band powers from EDF file."""
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    psds, freqs = psd_welch(
        raw, fmin=1, fmax=30, n_fft=1024,
        n_overlap=256, n_per_seg=512, verbose=False
    )
    psds = np.mean(psds, axis=0)
    bands = {"Delta": (1, 4), "Theta": (4, 8), "Alpha": (8, 12), "Beta": (12, 30)}
    data = {}
    for ch_idx, ch in enumerate(raw.ch_names):
        row = {}
        total = np.sum(psds[ch_idx])
        for b, (lo, hi) in bands.items():
            idx = np.logical_and(freqs >= lo, freqs < hi)
            row[f"Rel_{b}"] = np.sum(psds[ch_idx, idx]) / total if total > 0 else np.nan
        data[ch] = row
    return pd.DataFrame(data).T


def compute_clean_powers(edf_path, p2p_thresh=150):
    """Compute relative band powers after simple artifact rejection."""
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    data = raw.get_data()
    kept_epochs = []
    sf = int(raw.info["sfreq"])
    nper = 2 * sf  # 2-second epochs
    for start in range(0, data.shape[1] - nper, nper):
        seg = data[:, start:start+nper]
        if np.max(seg) - np.min(seg) < p2p_thresh:
            kept_epochs.append(seg)
    if not kept_epochs:
        return compute_raw_powers(edf_path), 0
    clean_data = np.concatenate(kept_epochs, axis=1)
    raw._data = clean_data
    return compute_raw_powers(edf_path), len(kept_epochs)


def add_zrel_zabs(df, norms_age):
    """Add Z-scores (relative and absolute) vs Cuban norms for a given age row."""
    dfz = df.copy()
    for band in ["Delta", "Theta", "Alpha", "Beta"]:
        mean = norms_age[f"{band}_mean"].values
        sd = norms_age[f"{band}_sd"].values
        chs = norms_age["Channel"].values
        for i, ch in enumerate(chs):
            try:
                val = df.loc[ch, f"Rel_{band}"]
                dfz.loc[ch, f"Zrel_{band}"] = (val - mean[i]) / sd[i]
            except Exception:
                dfz.loc[ch, f"Zrel_{band}"] = np.nan
    return dfz


def compute_faa(dfz):
    """Compute FAA (frontal alpha asymmetry)."""
    try:
        f3 = dfz.loc["F3", "Rel_Alpha"]
        f4 = dfz.loc["F4", "Rel_Alpha"]
        faa = ((f3 - f4) / (f3 + f4)) * 200
        if faa < -20:
            tag = "Left-dominant"
        elif faa > 20:
            tag = "Right-dominant"
        else:
            tag = "Neutral"
        return faa, tag
    except Exception:
        return np.nan, "Unavailable"


def _topomap_png(df, band, title):
    """Return PNG image of topomap for a given band."""
    from mne.channels import make_standard_montage

    montage = make_standard_montage("standard_1020")
    values = [df.loc[ch, band] if ch in df.index else np.nan for ch in montage.ch_names]
    pos = montage.get_positions()["ch_pos"]

    data = np.array(values)
    fig, ax = plt.subplots()
    im, _ = mne.viz.plot_topomap(
        data, pos, names=montage.ch_names,
        show=False, contours=0, sensors=True, axes=ax
    )
    ax.set_title(title)
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue()
