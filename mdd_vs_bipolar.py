
import numpy as np
import pandas as pd
import mne
from scipy.signal import welch   # ✅ SciPy Welch
from scipy.interpolate import griddata
from matplotlib.patches import Circle
from docx.shared import Inches
from docx import Document

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io


# -------------------
# PSD computation
# -------------------

def psd_welch(raw, fmin, fmax, n_fft=1024, n_overlap=256, n_per_seg=512, verbose=False):
    """Custom Welch PSD using SciPy, Python 3.13 safe."""
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
    return ch_name.strip().capitalize()

def get(df, ch, band):
    try:
        return float(df.loc[ch, band])
    except Exception:
        return np.nan


def compute_raw_powers(edf_path):
    """Compute relative band powers from EDF file."""
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    psds, freqs = psd_welch(raw, 1, 30)

    bands = {"Delta": (1, 4), "Theta": (4, 8), "Alpha": (8, 12), "Beta": (12, 30)}
    data = {}
    for ch_idx, ch in enumerate(raw.ch_names):
        row = {}
        total = np.sum(psds[ch_idx])
        for b, (lo, hi) in bands.items():
            idx = np.logical_and(freqs >= lo, freqs < hi)
            band_power = np.sum(psds[ch_idx, idx]) if total > 0 else np.nan
            row[f"Rel_{b}"] = band_power / total if total > 0 else np.nan
        data[ch] = row
    return pd.DataFrame(data).T


def compute_clean_powers(edf_path, p2p_thresh=150):
    """Compute relative band powers after simple artifact rejection."""
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    data = raw.get_data()
    kept_epochs = []
    sf = int(raw.info["sfreq"])
    nper = 2 * sf

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
    """Add Z-scores (relative power) vs Cuban norms (long format)."""
    dfz = df.copy()
    for band in ["Delta", "Theta", "Alpha", "Beta"]:
        band_rows = norms_age[norms_age["Band"].str.lower() == band.lower()]
        for _, row in band_rows.iterrows():
            ch = row["Channel"]
            if ch in df.index:
                val = df.loc[ch, f"Rel_{band}"]
                mean = row["RelMean"]; sd = row["RelSD"]
                try:
                    dfz.loc[ch, f"Zrel_{band}"] = (val - mean) / sd
                except Exception:
                    dfz.loc[ch, f"Zrel_{band}"] = np.nan
    return dfz


def compute_faa(dfz):
    """Frontal alpha asymmetry."""
    try:
        f3 = dfz.loc["F3", "Rel_Alpha"]
        f4 = dfz.loc["F4", "Rel_Alpha"]
        faa = ((f3 - f4) / (f3 + f4)) * 200
        if faa < -20: tag = "Left-dominant"
        elif faa > 20: tag = "Right-dominant"
        else: tag = "Neutral"
        return faa, tag
    except Exception:
        return np.nan, "Unavailable"


# -------------------
# Custom Topomap
# -------------------

POS = {
    "Fp1": (-0.5, 0.9), "Fp2": (0.5, 0.9),
    "F7": (-0.9, 0.5), "F3": (-0.4, 0.5), "Fz": (0.0, 0.55),
    "F4": (0.4, 0.5), "F8": (0.9, 0.5),
    "T7": (-1.0, 0.0), "C3": (-0.5, 0.0), "Cz": (0.0, 0.0),
    "C4": (0.5, 0.0), "T8": (1.0, 0.0),
    "P7": (-0.9, -0.5), "P3": (-0.4, -0.5), "Pz": (0.0, -0.55),
    "P4": (0.4, -0.5), "P8": (0.9, -0.5),
    "O1": (-0.5, -0.9), "O2": (0.5, -0.9)
}

def _topomap_png(df, key, title):
    """Return PNG of scalp map for given band or Z-score key."""
    xs, ys, zs = [], [], []
    for ch, (x, y) in POS.items():
        if ch not in df.index: continue
        v = df.loc[ch, key] if key in df.columns else np.nan
        if np.isfinite(v):
            xs.append(x); ys.append(y); zs.append(float(v))
    fig = plt.figure(figsize=(4.6, 3.9))
    ax = fig.add_subplot(111, aspect="equal")
    if len(xs) == 0:
        ax.text(0.5,0.5,"No data",ha="center",va="center")
    else:
        xs,ys,zs = np.array(xs),np.array(ys),np.array(zs)
        grid_x,grid_y=np.mgrid[-1:1:220j,-1:1:220j]
        pts=np.vstack([xs,ys]).T
        grid_z=griddata(pts,zs,(grid_x,grid_y),method="cubic")
        r=np.sqrt(grid_x**2+grid_y**2); grid_z[r>1]=np.nan
        if key.startswith("Z"):
            levels=np.linspace(-3,3,13)
            cf=ax.contourf(grid_x,grid_y,grid_z,levels=levels,cmap="RdBu_r",vmin=-3,vmax=3)
        else:
            cf=ax.contourf(grid_x,grid_y,grid_z,levels=12)
        plt.colorbar(cf,ax=ax,shrink=0.75)
    ax.add_patch(Circle((0,0),1.0,fill=False,lw=2))
    ax.scatter([p[0] for p in POS.values()],[p[1] for p in POS.values()],c="k",s=12)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title)
    bio=io.BytesIO(); fig.savefig(bio,format="png",dpi=160,bbox_inches="tight")
    plt.close(fig)
    return bio.getvalue()

def generate_all_topomaps(df):
    maps={}
    for band in ["Rel_Delta","Rel_Theta","Rel_Alpha","Rel_Beta"]:
        if band in df.columns:
            maps[band]=_topomap_png(df,band,f"Topomap: {band}")
    return maps


# -------------------
# Word export helper
# -------------------

def make_two_per_row_section(doc, items):
    table = doc.add_table(rows=0, cols=2)
    row_cells=None
    for i,(title,img_bytes) in enumerate(items):
        if i%2==0: row_cells=table.add_row().cells
        cell=row_cells[i%2]
        cell.add_paragraph(title)
        run=cell.add_paragraph().add_run()
        run.add_picture(io.BytesIO(img_bytes),width=Inches(2.5))
    return doc


# -------------------
# Diagnostic helpers
# -------------------

REGIONS = {
    "Central":["C3","C4","Cz"],
    "Parietal":["P3","P4","P7","P8","Pz"],
    "Occipital":["O1","O2"]
}

def region_alpha_reduced(dfz,chans):
    vals=[dfz.loc[ch,"Zrel_Alpha"] for ch in chans if ch in dfz.index]
    vals=[v for v in vals if np.isfinite(v)]
    if not vals: return False,np.nan,0
    mean=float(np.nanmean(vals))
    n_low=sum(1 for v in vals if v<=-1.0)
    reduced=(mean<=-0.5) or (n_low>=2)
    return reduced,mean,n_low

def bipolar_alpha_signature(dfz):
    cen_red,cen_mean,cen_n=region_alpha_reduced(dfz,REGIONS["Central"])
    par_red,par_mean,par_n=region_alpha_reduced(dfz,REGIONS["Parietal"])
    occ_red,occ_mean,occ_n=region_alpha_reduced(dfz,REGIONS["Occipital"])
    alpha_ok=(cen_red and par_red) or (par_red and occ_red)
    details={
        "Central":{"reduced":cen_red,"mean":cen_mean,"sites_le_-1":cen_n},
        "Parietal":{"reduced":par_red,"mean":par_mean,"sites_le_-1":par_n},
        "Occipital":{"reduced":occ_red,"mean":occ_mean,"sites_le_-1":occ_n},
    }
    return details,alpha_ok

def bipolar_theta_signature(dfz):
    sites={ch:dfz.loc[ch,"Zrel_Theta"] if ch in dfz.index else np.nan
           for ch in ["T8","P8","P4","O2"]}
    hits={ch:(np.isfinite(v) and v>=1.0) for ch,v in sites.items()}
    count=sum(1 for h in hits.values() if h)
    return sites,hits,(count>=2)


# -------------------
# Weighted Scoring
# -------------------

def compute_diagnostic_score(dfz):
    """Return Depression vs Bipolar probability scores."""
    # Rules (example, can adjust thresholds/weights)
    alpha_details, alpha_ok = bipolar_alpha_signature(dfz)
    _, _, theta_ok = bipolar_theta_signature(dfz)

    # Assign weights
    bipolar_points=0; depression_points=0

    if alpha_ok: bipolar_points+=2
    else: depression_points+=1
    if theta_ok: bipolar_points+=2

    total=bipolar_points+depression_points
    if total==0: return {"Depression":0.5,"Bipolar":0.5}

    return {
        "Depression": depression_points/total,
        "Bipolar": bipolar_points/total,
    }
