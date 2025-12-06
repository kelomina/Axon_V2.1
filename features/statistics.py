import numpy as np
from config.config import BYTE_HISTOGRAM_BINS, STAT_CHUNK_COUNT

def extract_statistical_features(byte_sequence, pe_features):
    byte_array = np.array(byte_sequence, dtype=np.uint8)
    features = []
    features.extend([
        float(np.mean(byte_array)),
        float(np.std(byte_array)),
        float(np.min(byte_array)),
        float(np.max(byte_array)),
        float(np.median(byte_array)),
        float(np.percentile(byte_array, 25)),
        float(np.percentile(byte_array, 75)),
    ])
    features.extend([
        int(np.sum(byte_array == 0)),
        int(np.sum(byte_array == 0xFF)),
        int(np.sum(byte_array == 0x90)),
        int(np.sum((byte_array >= 32) & (byte_array <= 126))),
    ])
    hist, _ = np.histogram(byte_array, bins=BYTE_HISTOGRAM_BINS, range=(0, 255), density=True)
    hist = hist[hist > 0]
    entropy = -np.sum(hist * np.log2(hist)) if len(hist) > 0 else 0.0
    features.append(float(entropy))
    length = len(byte_array)
    if length >= 3:
        one_third = length // 3
        segments = [
            byte_array[:one_third],
            byte_array[one_third:2 * one_third],
            byte_array[2 * one_third:],
        ]
    else:
        segments = [byte_array, byte_array, byte_array]
    for seg in segments:
        if len(seg) == 0:
            seg_mean = 0.0
            seg_std = 0.0
            seg_entropy = 0.0
        else:
            seg_mean = float(np.mean(seg))
            seg_std = float(np.std(seg))
            seg_hist, _ = np.histogram(seg, bins=BYTE_HISTOGRAM_BINS, range=(0, 255), density=True)
            seg_hist = seg_hist[seg_hist > 0]
            seg_entropy = -np.sum(seg_hist * np.log2(seg_hist)) if len(seg_hist) > 0 else 0.0
        features.extend([seg_mean, seg_std, seg_entropy])
    chunk_size = max(1, len(byte_array) // STAT_CHUNK_COUNT)
    chunk_means = []
    chunk_stds = []
    for i in range(STAT_CHUNK_COUNT):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size if i < 9 else len(byte_array)
        chunk = byte_array[start_idx:end_idx]
        if len(chunk) > 0:
            chunk_means.append(float(np.mean(chunk)))
            chunk_stds.append(float(np.std(chunk)))
        else:
            chunk_means.append(0.0)
            chunk_stds.append(0.0)
    features.extend(chunk_means)
    features.extend(chunk_stds)
    chunk_means = np.array(chunk_means, dtype=np.float32)
    chunk_stds = np.array(chunk_stds, dtype=np.float32)
    if len(chunk_means) > 1:
        mean_diffs = np.diff(chunk_means)
        std_diffs = np.diff(chunk_stds)
        features.extend([
            float(np.mean(np.abs(mean_diffs))),
            float(np.std(mean_diffs)),
            float(np.max(mean_diffs)),
            float(np.min(mean_diffs)),
        ])
        features.extend([
            float(np.mean(np.abs(std_diffs))),
            float(np.std(std_diffs)),
            float(np.max(std_diffs)),
            float(np.min(std_diffs)),
        ])
    else:
        features.extend([0.0, 0.0, 0.0, 0.0])
        features.extend([0.0, 0.0, 0.0, 0.0])
    features.extend(pe_features.tolist())
    return np.array(features, dtype=np.float32)