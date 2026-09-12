"""
ForgeLens-X — Milestone 8: 2D Fourier Transform (FFT) Frequency Forensics
=========================================================================
Analyzes the 2D Discrete Fourier Transform (DFT) log-magnitude spectrum to expose
periodic interpolation artifacts, digital re-sampling, and synthetic text insertion.
Authentic print substrates exhibit smooth decaying radial power spectra (1/f^alpha),
while digital character tampering introduces periodic high-frequency harmonic spikes.
"""

from typing import Any, Dict, Optional, Tuple
import cv2
import numpy as np


def analyze_fft_spectrum(
    image: np.ndarray,
    roi_bbox: Optional[Tuple[int, int, int, int]] = None,
) -> Dict[str, Any]:
    """
    Computes 2D Fast Fourier Transform spectral energy and kurtosis metrics.

    Args:
        image: BGR or Grayscale numpy image array.
        roi_bbox: Optional (x1, y1, x2, y2) region of interest to inspect.

    Returns:
        Dict containing:
          - spectral_kurtosis: Tail heaviness of high-frequency energy.
          - high_freq_energy_ratio: Proportion of power in upper frequency octaves.
          - periodic_peak_count: Number of anomalous harmonic peaks in spectrum.
          - is_anomalous: Boolean indicating detected digital resampling.
          - anomaly_score: Normalized fraud probability [0.0 - 1.0].
    """
    if image is None or image.size == 0:
        return {
            "spectral_kurtosis": 0.0,
            "high_freq_energy_ratio": 0.0,
            "periodic_peak_count": 0,
            "is_anomalous": False,
            "anomaly_score": 0.0,
        }

    # Extract ROI if specified
    if roi_bbox is not None:
        x1, y1, x2, y2 = [int(v) for v in roi_bbox]
        h, w = image.shape[:2]
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(x1 + 1, min(x2, w))
        y2 = max(y1 + 1, min(y2, h))
        crop = image[y1:y2, x1:x2]
    else:
        crop = image

    # Convert to grayscale float32
    if len(crop.shape) == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop.copy()

    h, w = gray.shape
    if h < 16 or w < 16:
        return {
            "spectral_kurtosis": 0.0,
            "high_freq_energy_ratio": 0.0,
            "periodic_peak_count": 0,
            "is_anomalous": False,
            "anomaly_score": 0.0,
        }

    # 1. Compute 2D Fast Fourier Transform (centered)
    dft = np.fft.fft2(gray.astype(np.float32))
    dft_shift = np.fft.fftshift(dft)
    magnitude = np.abs(dft_shift)
    log_magnitude = np.log1p(magnitude)

    # 2. Partition into low-frequency core and high-frequency annulus
    cy, cx = h // 2, w // 2
    y_coords, x_coords = np.ogrid[:h, :w]
    radii = np.sqrt((x_coords - cx) ** 2 + (y_coords - cy) ** 2)
    max_radius = np.sqrt(cx**2 + cy**2)

    # High-frequency band: outer 50% of frequency space
    high_freq_mask = radii > (0.45 * max_radius)
    low_freq_mask = radii <= (0.45 * max_radius)

    total_energy = np.sum(magnitude) + 1e-7
    high_freq_energy = np.sum(magnitude[high_freq_mask])
    high_freq_ratio = float(high_freq_energy / total_energy)

    # 3. Compute Spectral Kurtosis (measures spikiness vs Gaussian decay)
    hf_vals = log_magnitude[high_freq_mask]
    if len(hf_vals) > 0:
        mean_hf = np.mean(hf_vals)
        std_hf = np.std(hf_vals) + 1e-7
        kurtosis = float(np.mean(((hf_vals - mean_hf) / std_hf) ** 4) - 3.0)
    else:
        kurtosis = 0.0

    # 4. Detect Periodic Harmonic Peak Spikes (evidence of bilinear/bicubic resampling)
    # Threshold at mean + 3.5 * std of high-frequency log spectrum
    thresh = np.mean(hf_vals) + 3.2 * np.std(hf_vals)
    peak_count = int(np.sum(hf_vals > thresh))

    # Anomaly scoring
    # Authentic printed documents have smooth decay (kurtosis in [-1.0, 2.5], few outlier peaks).
    # Resampled forged characters produce sharp spikes in the outer annulus.
    is_anomalous = bool(kurtosis > 4.5 or (peak_count > 120 and high_freq_ratio > 0.35))
    anomaly_score = float(np.clip((max(0.0, kurtosis) / 8.0) * 0.5 + (min(200, peak_count) / 200.0) * 0.5, 0.0, 1.0))

    return {
        "spectral_kurtosis": round(kurtosis, 3),
        "high_freq_energy_ratio": round(high_freq_ratio, 4),
        "periodic_peak_count": peak_count,
        "is_anomalous": is_anomalous,
        "anomaly_score": round(anomaly_score, 3),
    }


def generate_fft_spectrum_image(image: np.ndarray) -> np.ndarray:
    """
    Renders the centered log-magnitude 2D Fourier spectrum as a color visualizer.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    dft = np.fft.fft2(gray.astype(np.float32))
    dft_shift = np.fft.fftshift(dft)
    log_mag = np.log1p(np.abs(dft_shift))

    # Normalize to 0-255 uint8
    norm = cv2.normalize(log_mag, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    heatmap = cv2.applyColorMap(norm, cv2.COLORMAP_VIRIDIS)
    return heatmap
