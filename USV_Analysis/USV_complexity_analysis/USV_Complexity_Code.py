
from __future__ import annotations

import csv
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


@dataclass
class SyllableInterval:
    start_s: float
    end_s: float


@dataclass
class SyllableFeatures:
    interval: SyllableInterval
    fm_index: float
    contour_vector: np.ndarray


@dataclass
class CallComponents:
    cintra: float
    cinter: float
    ctemporal: float


@dataclass
class CallComplexity:
    file_name: str
    cintra: float
    cinter: float
    ctemporal: float
    ctotal: float


def read_wav_mono(path: str | Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        n_frames = wf.getnframes()
        audio_bytes = wf.readframes(n_frames)

    if sampwidth == 2:
        audio = np.frombuffer(audio_bytes, dtype="<i2").astype(np.float64) / 32768.0
    elif sampwidth == 1:
        audio = (np.frombuffer(audio_bytes, dtype=np.uint8).astype(np.float64) - 128.0) / 128.0
    elif sampwidth == 4:
        audio = np.frombuffer(audio_bytes, dtype="<i4").astype(np.float64) / 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth}")

    if n_channels > 1:
        audio = audio.reshape(-1, n_channels).mean(axis=1)
    return audio, int(sr)


def extract_dominant_frequency_contour(
    signal: np.ndarray,
    sr: int,
    freq_min_hz: float = 30_000.0,
    freq_max_hz: float = 110_000.0,
    window_ms: float = 1.0,
    overlap: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nperseg = max(32, int(round(sr * window_ms * 1e-3)))
    noverlap = int(round(nperseg * overlap))
    hop = max(1, nperseg - noverlap)
    nfft = 1 << int(np.ceil(np.log2(nperseg)))
    window = np.hamming(nperseg)

    if signal.size < nperseg:
        return np.array([]), np.array([]), np.array([])

    starts = np.arange(0, signal.size - nperseg + 1, hop, dtype=int)
    times = (starts + nperseg * 0.5) / float(sr)
    freqs = np.fft.rfftfreq(nfft, d=1.0 / sr)
    spec = np.empty((freqs.size, starts.size), dtype=np.float64)

    for idx, start in enumerate(starts):
        frame = signal[start : start + nperseg] * window
        mag = np.abs(np.fft.rfft(frame, n=nfft))
        spec[:, idx] = mag

    band_mask = (freqs >= freq_min_hz) & (freqs <= freq_max_hz)
    if band_mask.sum() < 2 or spec.shape[1] < 2:
        return np.array([]), np.array([]), np.array([])

    band_freqs = freqs[band_mask]
    band_mag = spec[band_mask, :]

    peak_idx = np.argmax(band_mag, axis=0)
    track_hz = band_freqs[peak_idx]
    peak_mag = band_mag[peak_idx, np.arange(band_mag.shape[1])]
    median_mag = np.median(band_mag, axis=0) + 1e-12
    confidence = peak_mag / median_mag
    return times, track_hz, confidence


def trim_contour_edges(
    times_s: np.ndarray,
    track_hz: np.ndarray,
    confidence: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if track_hz.size <= 2:
        return times_s, track_hz, confidence

    diff = np.diff(track_hz, prepend=track_hz[0])
    trim_start = 0
    trim_end = track_hz.size

    if track_hz.size > 20:
        if np.mean(np.abs(diff[:20])) > 1000.0:
            trim_start = min(20, trim_end)
        if trim_end - trim_start > 20 and np.mean(np.abs(diff[-20:])) > 1000.0:
            trim_end = max(trim_start, trim_end - 20)
    elif track_hz.size > 10:
        if np.mean(np.abs(diff[:10])) > 2000.0:
            trim_start = min(10, trim_end)
        if trim_end - trim_start > 10 and np.mean(np.abs(diff[-10:])) > 2000.0:
            trim_end = max(trim_start, trim_end - 10)

    times_s = times_s[trim_start:trim_end]
    track_hz = track_hz[trim_start:trim_end]
    confidence = confidence[trim_start:trim_end]

    if times_s.size > 0:
        times_s = times_s - times_s[0]
    return times_s, track_hz, confidence


def clean_contour(
    times_s: np.ndarray,
    track_hz: np.ndarray,
    confidence: np.ndarray,
    freq_min_hz: float = 25_000.0,
    freq_max_hz: float = 110_000.0,
) -> tuple[np.ndarray, np.ndarray]:
    if track_hz.size == 0:
        return np.array([]), np.array([])

    positive_conf = confidence[confidence > 0]
    if positive_conf.size == 0:
        return np.array([]), np.array([])

    score_floor = max(2.5, 0.55 * float(np.median(positive_conf)))
    valid = (
        (track_hz >= freq_min_hz)
        & (track_hz <= freq_max_hz)
        & (confidence >= score_floor)
    )

    times_s = times_s[valid]
    track_hz = track_hz[valid]
    if times_s.size < 2:
        return np.array([]), np.array([])

    keep = np.concatenate([[True], np.diff(times_s) > 0])
    return times_s[keep], track_hz[keep]


def resample_series(
    times_s: np.ndarray,
    values: np.ndarray,
    n_points: int,
) -> np.ndarray:
    if times_s.size < 2 or values.size < 2:
        return np.array([])

    duration = times_s[-1] - times_s[0]
    if duration <= 0:
        return np.array([])

    t_norm = (times_s - times_s[0]) / duration
    grid = np.linspace(0.0, 1.0, n_points)
    return np.interp(grid, t_norm, values)


def compute_fm_index(
    times_s: np.ndarray,
    track_hz: np.ndarray,
    n_points: int = 101,
) -> float:
    resampled = resample_series(times_s, track_hz, n_points=n_points)
    if resampled.size < 2 or np.any(resampled <= 0):
        return float("nan")

    mean_freq = float(np.mean(resampled))
    if mean_freq <= 0:
        return float("nan")

    log_norm = np.log(resampled / mean_freq)
    return float(np.std(log_norm, ddof=0))


def build_contour_vector(
    times_s: np.ndarray,
    track_hz: np.ndarray,
    n_points: int = 20,
) -> np.ndarray:
    if times_s.size < 2 or track_hz.size < 2:
        return np.array([])

    mean_freq = float(np.mean(track_hz))
    sd_freq = float(np.std(track_hz, ddof=0))
    if sd_freq <= 1e-9:
        standardized = np.zeros_like(track_hz)
    else:
        standardized = (track_hz - mean_freq) / sd_freq

    return resample_series(times_s, standardized, n_points=n_points)


def pairwise_mean_euclidean(vectors: Sequence[np.ndarray]) -> float:
    vectors = [v for v in vectors if v.size > 0]
    n = len(vectors)
    if n < 2:
        return 0.0

    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt(np.mean((vectors[i] - vectors[j]) ** 2))
            dists.append(float(d))
    return float(np.mean(dists))


def compute_isi_cv(intervals: Sequence[SyllableInterval]) -> float:
    intervals = sorted(intervals, key=lambda x: x.start_s)
    if len(intervals) < 3:
        return 0.0

    isi = []
    for i in range(len(intervals) - 1):
        gap = intervals[i + 1].start_s - intervals[i].end_s
        if gap >= 0:
            isi.append(gap)

    if len(isi) < 2:
        return 0.0

    isi = np.asarray(isi, dtype=float)
    mean_isi = float(np.mean(isi))
    if mean_isi <= 0:
        return 0.0
    return float(np.std(isi, ddof=0) / mean_isi)


def compute_call_components(
    wav_path: str | Path,
    syllables: Sequence[SyllableInterval],
    fm_points: int = 101,
    contour_points: int = 20,
) -> tuple[CallComponents, list[SyllableFeatures]]:
    audio, sr = read_wav_mono(wav_path)
    syllable_features: list[SyllableFeatures] = []

    for interval in syllables:
        start_idx = max(0, int(np.floor(interval.start_s * sr)))
        end_idx = min(audio.size - 1, int(np.floor(interval.end_s * sr)))
        if end_idx <= start_idx:
            continue

        seg = audio[start_idx : end_idx + 1]
        times, track, conf = extract_dominant_frequency_contour(seg, sr)
        times, track, conf = trim_contour_edges(times, track, conf)
        times, track = clean_contour(times, track, conf)

        if track.size < 2:
            continue

        fm_index = compute_fm_index(times, track, n_points=fm_points)
        contour_vec = build_contour_vector(times, track, n_points=contour_points)
        if not np.isfinite(fm_index) or contour_vec.size == 0:
            continue

        syllable_features.append(
            SyllableFeatures(
                interval=interval,
                fm_index=float(fm_index),
                contour_vector=contour_vec,
            )
        )

    if not syllable_features:
        return CallComponents(0.0, 0.0, 0.0), []

    cintra = float(np.mean([s.fm_index for s in syllable_features]))
    cinter = pairwise_mean_euclidean([s.contour_vector for s in syllable_features])
    ctemporal = compute_isi_cv([s.interval for s in syllable_features])

    return CallComponents(cintra=cintra, cinter=cinter, ctemporal=ctemporal), syllable_features


def minmax_normalize(values: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(values), dtype=float)
    if values.size == 0:
        return values
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if vmax <= vmin:
        return np.zeros_like(values)
    return (values - vmin) / (vmax - vmin)


def compute_dataset_ctotal(
    call_components: Sequence[tuple[str, CallComponents]],
    weights: tuple[float, float, float] | None = None,
) -> list[CallComplexity]:
    cintra = minmax_normalize([comp.cintra for _, comp in call_components])
    cinter = minmax_normalize([comp.cinter for _, comp in call_components])
    ctemporal = minmax_normalize([comp.ctemporal for _, comp in call_components])

    if weights is None:
        w1, w2, w3 = 1.0, 1.0, 1.0
    else:
        w1, w2, w3 = weights

    results: list[CallComplexity] = []
    for i, (file_name, comp) in enumerate(call_components):
        ctotal = w1 * cintra[i] + w2 * cinter[i] + w3 * ctemporal[i]
        results.append(
            CallComplexity(
                file_name=file_name,
                cintra=comp.cintra,
                cinter=comp.cinter,
                ctemporal=comp.ctemporal,
                ctotal=float(ctotal),
            )
        )
    return results


def load_intervals_from_csv(csv_path: str | Path) -> dict[str, list[SyllableInterval]]:
    call_map: dict[str, list[SyllableInterval]] = {}
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["FileName"]
            call_map.setdefault(name, []).append(
                SyllableInterval(
                    start_s=float(row["StartSeconds"]),
                    end_s=float(row["EndSeconds"]),
                )
            )
    for name in call_map:
        call_map[name] = sorted(call_map[name], key=lambda x: x.start_s)
    return call_map


def main() -> None:
    seg_csv = Path(r"D:\pyl\复杂度\example9\complexity_segmented_syllables.csv")
    wav_dir = Path(r"D:\pyl\复杂度\example9")
    out_csv = Path(r"D:\pyl\复杂度\example9\supplementary_complexity_results.csv")

    call_map = load_intervals_from_csv(seg_csv)
    components = []
    for wav_name in sorted(call_map.keys()):
        wav_path = wav_dir / wav_name
        comp, _ = compute_call_components(wav_path, call_map[wav_name])
        components.append((wav_name, comp))

    results = compute_dataset_ctotal(components)

    with open(out_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["FileName", "Cintra", "Cinter", "Ctemporal", "Ctotal"])
        for row in results:
            writer.writerow(
                [
                    row.file_name,
                    f"{row.cintra:.6f}",
                    f"{row.cinter:.6f}",
                    f"{row.ctemporal:.6f}",
                    f"{row.ctotal:.6f}",
                ]
            )

    print(out_csv)


if __name__ == "__main__":
    main()
