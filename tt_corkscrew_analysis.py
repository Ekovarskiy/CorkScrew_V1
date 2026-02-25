# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # TT Image Corkscrew Artifact Removal (EDA + Benchmark + Report)
#
# Workflow:
# 1. Load TT matrix from XLSX (expected around `(11999, 180)`).
# 2. Run EDA and spectral stripe-orientation analysis.
# 3. Execute 4 candidate denoising/de-striping methods.
# 4. Score methods and select a best candidate.
# 5. Export artifacts in `outputs/` including:
#    - figures + CSV
#    - markdown report with embedded images
#    - PDF report

# %%
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import io
import textwrap
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages
from scipy import ndimage, signal
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.ndimage import median_filter, gaussian_filter1d


# %%
# -----------------------------
# Configuration
# -----------------------------
ROOT = Path(".")
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_SHAPE = (11999, 180)

FIG_EDA = OUTPUT_DIR / "01_eda_image_hist.png"
FIG_FFT = OUTPUT_DIR / "02_fft_raw.png"
FIG_METHODS = OUTPUT_DIR / "03_method_comparison.png"
FIG_METRICS = OUTPUT_DIR / "04_metrics.png"
CSV_METRICS = OUTPUT_DIR / "method_metrics.csv"
MD_REPORT = OUTPUT_DIR / "report.md"
PDF_REPORT = OUTPUT_DIR / "report.pdf"


# %%
def build_tt_candidates(root: Path) -> list[Path]:
    """Ordered TT candidate list (common names first, then fuzzy TT-like names)."""
    fixed = [
        root / "TT.xlsx",
        root / "tt.xlsx",
        root / "data/TT.xlsx",
        root / "data/tt.xlsx",
    ]
    fuzzy = []
    for p in root.rglob("*.xlsx"):
        stem = p.stem.lower()
        if "tt" in stem and p not in fixed:
            fuzzy.append(p)
    return fixed + sorted(fuzzy)


# %%
def load_tt_data(candidates: Iterable[Path]) -> tuple[np.ndarray, str, str]:
    """Load TT from XLSX; if unavailable, use synthetic demo data."""
    for path in candidates:
        if path.exists():
            df = pd.read_excel(path, header=None)
            arr = df.to_numpy(dtype=float)
            return arr, f"Loaded real TT from: {path}", "real"

    warnings.warn(
        "No TT.xlsx-like file found. Running on synthetic demo data.",
        RuntimeWarning,
    )
    n_depth, n_theta = 2500, 180
    z = np.arange(n_depth)[:, None]
    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)[None, :]

    geology = (
        0.45 * np.sin(2.2 * theta + 0.0017 * z)
        + 0.20 * np.sin(7.5 * theta - 0.0006 * z)
        + 0.18 * np.cos(0.0011 * z)
    )
    corkscrew = 0.38 * np.sin(17.0 * theta + 0.024 * z)
    noise = 0.08 * np.random.default_rng(42).normal(size=(n_depth, n_theta))
    arr = geology + corkscrew + noise
    return arr, "Loaded synthetic demonstration TT data", "synthetic"


TT_CANDIDATES = build_tt_candidates(ROOT)
tt, load_message, data_kind = load_tt_data(TT_CANDIDATES)
print(load_message)
print("TT shape:", tt.shape)
if tt.shape != EXPECTED_SHAPE:
    print(f"Warning: expected shape {EXPECTED_SHAPE}, got {tt.shape}")


# %% [markdown]
# ## EDA

# %%
def nan_safe_stats(img: np.ndarray) -> dict[str, float]:
    finite = img[np.isfinite(img)]
    return {
        "count": float(finite.size),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "nan_ratio": float(np.mean(~np.isfinite(img))),
    }


def normalize_image(img: np.ndarray) -> np.ndarray:
    q1, q99 = np.nanpercentile(img, [1, 99])
    return np.clip((img - q1) / (q99 - q1 + 1e-12), 0, 1)


def spectrum(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    clean = np.nan_to_num(img, nan=np.nanmedian(img))
    f = fftshift(fft2(clean))
    mag = np.log1p(np.abs(f))
    return f, mag


def estimate_stripe_angle_from_fft(img: np.ndarray) -> float:
    """Estimate stripe angle in image domain using weighted FFT orientation."""
    _, mag = spectrum(img)
    h, w = mag.shape
    cy, cx = h // 2, w // 2

    yy, xx = np.indices((h, w))
    dy = yy - cy
    dx = xx - cx
    rr = np.sqrt(dy**2 + dx**2)

    annulus = (rr > min(h, w) * 0.08) & (rr < min(h, w) * 0.45)
    weights = np.where(annulus, mag, 0.0)

    sxx = np.sum(weights * dx * dx)
    syy = np.sum(weights * dy * dy)
    sxy = np.sum(weights * dx * dy)
    angle_freq = 0.5 * np.degrees(np.arctan2(2 * sxy, sxx - syy))

    stripe_angle = (angle_freq + 90.0) % 180.0
    return float(stripe_angle)


stats = nan_safe_stats(tt)
img_norm = normalize_image(tt)
f_raw, mag_raw = spectrum(tt)
stripe_angle = estimate_stripe_angle_from_fft(tt)
print(f"Estimated corkscrew stripe angle: {stripe_angle:.2f}°")

fig, ax = plt.subplots(1, 3, figsize=(18, 5))
ax[0].imshow(tt, aspect="auto", cmap="viridis")
ax[0].set_title("Raw TT")
ax[0].set_xlabel("Angle index")
ax[0].set_ylabel("Depth index")

ax[1].imshow(img_norm, aspect="auto", cmap="viridis")
ax[1].set_title("Contrast-normalized TT")
ax[1].set_xlabel("Angle index")
ax[1].set_ylabel("Depth index")

ax[2].hist(tt[np.isfinite(tt)].ravel(), bins=120, color="steelblue")
ax[2].set_title("Value histogram")
ax[2].set_xlabel("TT value")
ax[2].set_ylabel("Count")

plt.tight_layout()
plt.savefig(FIG_EDA, dpi=170)
plt.close(fig)

fig = plt.figure(figsize=(8, 6))
plt.imshow(mag_raw, cmap="magma", aspect="auto")
plt.title("2D FFT log-magnitude (TT)")
plt.xlabel("freq_theta")
plt.ylabel("freq_depth")
plt.colorbar(label="log(1+|F|)")
plt.tight_layout()
plt.savefig(FIG_FFT, dpi=170)
plt.close(fig)


# %% [markdown]
# ## Candidate methods

# %%
def fft_notch_filter(img: np.ndarray, top_k: int = 8, notch_sigma: float = 3.0) -> np.ndarray:
    f = fftshift(fft2(img))
    mag = np.abs(f)
    h, w = img.shape
    cy, cx = h // 2, w // 2

    yy, xx = np.indices((h, w))
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

    search = mag.copy()
    search[rr < min(h, w) * 0.08] = 0.0

    peak_idx = np.argpartition(search.ravel(), -top_k)[-top_k:]
    peaks = np.column_stack(np.unravel_index(peak_idx, search.shape))

    H = np.ones_like(img, dtype=float)
    for py, px in peaks:
        for qy, qx in [(py, px), (2 * cy - py, 2 * cx - px)]:
            d2 = (yy - qy) ** 2 + (xx - qx) ** 2
            H *= (1 - np.exp(-d2 / (2 * notch_sigma**2)))

    return np.real(ifft2(ifftshift(f * H)))



def directional_median_filter(img: np.ndarray, angle_deg: float, stripe_width: int = 11) -> np.ndarray:
    rot = ndimage.rotate(img, -angle_deg, reshape=False, order=1, mode="reflect")
    filt = median_filter(rot, size=(1, stripe_width), mode="reflect")
    return ndimage.rotate(filt, angle_deg, reshape=False, order=1, mode="reflect")



def estimate_row_shifts(img: np.ndarray, max_shift: int = 25, smooth_sigma: float = 12) -> np.ndarray:
    n_depth, n_theta = img.shape
    shifts = np.zeros(n_depth, dtype=float)
    prev = img[0] - np.mean(img[0])

    lags = np.arange(-n_theta + 1, n_theta)
    valid = (lags >= -max_shift) & (lags <= max_shift)

    for i in range(1, n_depth):
        cur = img[i] - np.mean(img[i])
        corr = signal.correlate(prev, cur, mode="full", method="fft")
        lag = lags[valid][np.argmax(corr[valid])]
        shifts[i] = shifts[i - 1] + lag
        prev = cur

    return gaussian_filter1d(shifts, sigma=smooth_sigma)



def helical_demodulation_filter(img: np.ndarray) -> np.ndarray:
    shifts = estimate_row_shifts(img)

    aligned = np.empty_like(img)
    for i, s in enumerate(shifts):
        aligned[i] = np.roll(img[i], int(np.round(-s)))

    aligned_smooth = gaussian_filter1d(aligned, sigma=2.0, axis=1, mode="wrap")

    restored = np.empty_like(img)
    for i, s in enumerate(shifts):
        restored[i] = np.roll(aligned_smooth[i], int(np.round(s)))
    return restored


@dataclass
class RPCAConfig:
    lam: float | None = None
    mu: float | None = None
    max_iter: int = 100
    tol: float = 1e-5
    depth_stride: int = 4



def _soft_threshold(x: np.ndarray, tau: float) -> np.ndarray:
    return np.sign(x) * np.maximum(np.abs(x) - tau, 0.0)



def robust_pca_lowrank(img: np.ndarray, cfg: RPCAConfig = RPCAConfig()) -> np.ndarray:
    ds = max(1, int(cfg.depth_stride))
    M_full = img.astype(float)
    M = M_full[::ds]
    m, n = M.shape

    lam = cfg.lam if cfg.lam is not None else 1.0 / np.sqrt(max(m, n))
    mu = cfg.mu if cfg.mu is not None else (m * n) / (4.0 * np.sum(np.abs(M)) + 1e-12)

    L = np.zeros_like(M)
    S = np.zeros_like(M)
    Y = np.zeros_like(M)
    normM = np.linalg.norm(M, ord="fro") + 1e-12

    for _ in range(cfg.max_iter):
        U, s, Vt = np.linalg.svd(M - S + (1 / mu) * Y, full_matrices=False)
        s_sh = np.maximum(s - 1 / mu, 0)
        rank = np.sum(s_sh > 0)

        if rank > 0:
            L = (U[:, :rank] * s_sh[:rank]) @ Vt[:rank]
        else:
            L = np.zeros_like(M)

        S = _soft_threshold(M - L + (1 / mu) * Y, lam / mu)
        residual = M - L - S
        Y = Y + mu * residual

        if np.linalg.norm(residual, ord="fro") / normM < cfg.tol:
            break

    idx_ds = np.arange(0, M_full.shape[0], ds)
    idx_full = np.arange(M_full.shape[0])

    L_full = np.empty_like(M_full)
    for j in range(M_full.shape[1]):
        L_full[:, j] = np.interp(idx_full, idx_ds, L[:, j])

    return L_full


# %% [markdown]
# ## Benchmark and selection

# %%
def stripe_energy_metric(img: np.ndarray, stripe_angle_deg: float, bandwidth_deg: float = 12.0) -> float:
    _, mag = spectrum(img)
    h, w = mag.shape
    cy, cx = h // 2, w // 2

    yy, xx = np.indices((h, w))
    dy = yy - cy
    dx = xx - cx
    rr = np.sqrt(dy**2 + dx**2)

    ang = (np.degrees(np.arctan2(dy, dx)) + 180.0) % 180.0
    stripe_freq_angle = (stripe_angle_deg - 90.0) % 180.0
    d_ang = np.minimum(np.abs(ang - stripe_freq_angle), 180.0 - np.abs(ang - stripe_freq_angle))

    mask = (rr > min(h, w) * 0.08) & (d_ang <= bandwidth_deg)
    return float(np.mean(mag[mask]))



def preservation_metric(raw: np.ndarray, filt: np.ndarray) -> float:
    gx0, gz0 = np.gradient(raw, axis=1), np.gradient(raw, axis=0)
    gxf, gzf = np.gradient(filt, axis=1), np.gradient(filt, axis=0)

    v0 = np.concatenate([gx0.ravel(), gz0.ravel()]).astype(float)
    vf = np.concatenate([gxf.ravel(), gzf.ravel()]).astype(float)

    v0 -= v0.mean()
    vf -= vf.mean()
    denom = np.linalg.norm(v0) * np.linalg.norm(vf) + 1e-12
    return float(np.dot(v0, vf) / denom)


results: dict[str, np.ndarray] = {"raw": tt}
results["fft_notch"] = fft_notch_filter(tt, top_k=8, notch_sigma=3.0)
results["directional_median"] = directional_median_filter(tt, stripe_angle, stripe_width=11)
results["helical_demod"] = helical_demodulation_filter(tt)
results["rpca_lowrank"] = robust_pca_lowrank(tt, RPCAConfig(max_iter=100, tol=1e-5, depth_stride=4))

metrics = []
base_stripe = stripe_energy_metric(results["raw"], stripe_angle)

for name, img in results.items():
    if name == "raw":
        continue
    stripe = stripe_energy_metric(img, stripe_angle)
    stripe_reduction = (base_stripe - stripe) / (base_stripe + 1e-12)
    preserve = preservation_metric(results["raw"], img)
    score = 0.65 * stripe_reduction + 0.35 * preserve
    metrics.append((name, stripe, stripe_reduction, preserve, score))

metrics_df = pd.DataFrame(
    metrics,
    columns=["method", "stripe_energy", "stripe_reduction", "preservation", "score"],
).sort_values("score", ascending=False)

best_method = str(metrics_df.iloc[0]["method"])
print("Selected best method:", best_method)

fig, axes = plt.subplots(2, 3, figsize=(16, 10), sharex=True, sharey=True)
plot_order = ["raw", "fft_notch", "directional_median", "helical_demod", "rpca_lowrank"]
for ax, name in zip(axes.ravel(), plot_order):
    ax.imshow(results[name], aspect="auto", cmap="viridis")
    ax.set_title(name)
    ax.set_xlabel("Angle index")
    ax.set_ylabel("Depth index")

axes.ravel()[-1].axis("off")
plt.tight_layout()
plt.savefig(FIG_METHODS, dpi=170)
plt.close(fig)

fig, ax1 = plt.subplots(figsize=(10, 5))
ax1.bar(metrics_df["method"], metrics_df["stripe_reduction"], alpha=0.75)
ax1.set_ylabel("Stripe reduction (higher better)")
ax1.tick_params(axis="x", rotation=20)

ax2 = ax1.twinx()
ax2.plot(metrics_df["method"], metrics_df["preservation"], "o-r")
ax2.set_ylabel("Structure preservation (higher better)")

plt.title("Method benchmark")
plt.tight_layout()
plt.savefig(FIG_METRICS, dpi=170)
plt.close(fig)

metrics_df.to_csv(CSV_METRICS, index=False)


# %% [markdown]
# ## Report writing (Markdown + PDF)

# %%
def dataframe_to_markdown(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False, floatfmt=".5f")
    except Exception:
        # fallback if tabulate is not available
        return df.to_csv(index=False)



def write_markdown_report(
    out_path: Path,
    data_kind: str,
    load_message: str,
    shape: tuple[int, int],
    stripe_angle: float,
    stats: dict[str, float],
    metrics_df: pd.DataFrame,
    best_method: str,
) -> None:
    table_md = dataframe_to_markdown(metrics_df)

    text = f"""\
# TT Corkscrew Artifact Removal Report

## 1) Data summary
- Source: **{data_kind}**
- Loader note: {load_message}
- Shape used: `{shape}`
- Expected nominal shape: `{EXPECTED_SHAPE}`
- NaN ratio: `{stats['nan_ratio']:.6f}`
- Value range: `{stats['min']:.5f}` to `{stats['max']:.5f}`
- Mean ± std: `{stats['mean']:.5f}` ± `{stats['std']:.5f}`

## 2) EDA
- Estimated dominant corkscrew stripe angle (from FFT): **{stripe_angle:.2f}°**

![EDA overview](01_eda_image_hist.png)

![Raw FFT spectrum](02_fft_raw.png)

## 3) Methods evaluated
1. FFT notch suppression
2. Directional median filtering
3. Helical demodulation (row-shift alignment)
4. Robust PCA low-rank reconstruction (depth-downsampled)

## 4) Quantitative benchmark
Scoring formula:

`score = 0.65 * stripe_reduction + 0.35 * preservation`

{table_md}

![Method visual comparison](03_method_comparison.png)

![Metric comparison](04_metrics.png)

## 5) Final selection
Selected best method: **{best_method}**

## 6) Notes
- Results depend on interval/geology; validate best method on several intervals.
- After TT validation, reuse this benchmark on AMP for cross-channel consistency.
"""
    out_path.write_text(textwrap.dedent(text), encoding="utf-8")



def _render_wrapped_text_page(pdf: PdfPages, title: str, body: str, fontsize: int = 10) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))  # A4 portrait in inches
    fig.patch.set_facecolor("white")
    plt.axis("off")

    wrapped_lines = []
    for line in body.splitlines():
        wrapped = textwrap.wrap(line, width=100) or [""]
        wrapped_lines.extend(wrapped)

    text = f"{title}\n\n" + "\n".join(wrapped_lines)
    plt.text(0.05, 0.97, text, va="top", ha="left", fontsize=fontsize, family="monospace")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)



def _add_image_page(pdf: PdfPages, image_path: Path, title: str) -> None:
    if not image_path.exists():
        return
    img = mpimg.imread(image_path)
    fig = plt.figure(figsize=(8.27, 11.69))
    ax = fig.add_subplot(111)
    ax.imshow(img)
    ax.set_title(title)
    ax.axis("off")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)



def markdown_to_pdf(markdown_path: Path, pdf_path: Path) -> None:
    """
    Create a readable PDF report without external dependencies:
    - page 1..n: markdown text (wrapped)
    - next pages: embedded figure images
    """
    md_text = markdown_path.read_text(encoding="utf-8")

    # split markdown text into chunks for multiple pages
    lines = md_text.splitlines()
    chunk_size = 70
    chunks = ["\n".join(lines[i : i + chunk_size]) for i in range(0, len(lines), chunk_size)]

    with PdfPages(pdf_path) as pdf:
        for idx, chunk in enumerate(chunks, start=1):
            _render_wrapped_text_page(pdf, title=f"TT Report (Text) - page {idx}", body=chunk, fontsize=9)

        _add_image_page(pdf, FIG_EDA, "EDA overview")
        _add_image_page(pdf, FIG_FFT, "FFT spectrum")
        _add_image_page(pdf, FIG_METHODS, "Method visual comparison")
        _add_image_page(pdf, FIG_METRICS, "Benchmark metrics")


write_markdown_report(
    out_path=MD_REPORT,
    data_kind=data_kind,
    load_message=load_message,
    shape=tt.shape,
    stripe_angle=stripe_angle,
    stats=stats,
    metrics_df=metrics_df,
    best_method=best_method,
)

markdown_to_pdf(MD_REPORT, PDF_REPORT)

print(f"Artifacts written to: {OUTPUT_DIR.resolve()}")
print(f"- {FIG_EDA.name}")
print(f"- {FIG_FFT.name}")
print(f"- {FIG_METHODS.name}")
print(f"- {FIG_METRICS.name}")
print(f"- {CSV_METRICS.name}")
print(f"- {MD_REPORT.name}")
print(f"- {PDF_REPORT.name}")
