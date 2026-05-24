import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from itertools import groupby
from scipy.interpolate import pchip_interpolate
from scipy.signal import medfilt
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

filename = r"data.h5" 
FPS = 60             
GAP_LIMIT = 5           
SPIKE_THRESHOLD = 50    
PC_CLEAN_STD = 3.0      

def format_time(frame, fps):
    total_seconds = frame / fps
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def preprocess_pipeline(sig):
    if np.all(np.isnan(sig)): return sig
    temp_sig = sig.copy()
    n = len(temp_sig)

    for i in range(1, n - 1):
        p, c, n_val = temp_sig[i-1], temp_sig[i], temp_sig[i+1]
        if not np.isnan([p, c, n_val]).any():
            if abs(c - (p + n_val)/2) > SPIKE_THRESHOLD and abs(p - n_val) < SPIKE_THRESHOLD:
                temp_sig[i] = np.nan

    mask = np.isnan(temp_sig)
    if np.any(mask) and not np.all(mask):
        idx = 0
        for is_nan, group in groupby(mask):
            g_list = list(group)
            length = len(g_list)
            if is_nan and length <= GAP_LIMIT:
                start, end = idx, idx + length
                if start > 0 and end < n:
                    v_idx = np.where(~mask)[0]
                    v_val = temp_sig[~mask]
                    temp_sig[start:end] = pchip_interpolate(v_idx, v_val, np.arange(start, end))
            idx += length

    res = np.full_like(temp_sig, np.nan)
    new_mask = ~np.isnan(temp_sig)
    idx = 0
    for is_valid, group in groupby(new_mask):
        g_list = list(group)
        grp_len = len(g_list)
        s, e = idx, idx + grp_len
        if is_valid:
            seg = temp_sig[s:e]
            if grp_len >= 3:
                res[s:e] = gaussian_filter1d(medfilt(seg, 3), sigma=1)
            else:
                res[s:e] = seg
        idx += grp_len
    return res

def clean_pc_jumps(pc_array, threshold_std=PC_CLEAN_STD):
    cleaned = pc_array.copy()
    for m in range(cleaned.shape[0]):
        for i in range(cleaned.shape[1]):
            sig = cleaned[m, i, :].copy() 
            if np.all(np.isnan(sig)): continue
            diff = np.diff(sig, prepend=sig[0])
            std_val = np.nanstd(diff)
            bad_mask = np.abs(diff) > (threshold_std * std_val)
            sig[bad_mask] = np.nan
            
            mask_nan = np.isnan(sig)
            if np.any(mask_nan) and not np.all(mask_nan):
                inds = np.arange(len(sig))
                sig[mask_nan] = np.interp(inds[mask_nan], inds[~mask_nan], sig[~mask_nan])
            cleaned[m, i, :] = medfilt(sig, kernel_size=5)
    return cleaned

def get_wavelet_feature_stack(pcs, fs=30.0, n_freqs=25):
    num_mice, num_pcs, n_frames = pcs.shape
    widths = np.geomspace(1, fs/2, n_freqs)
    wavelet_stack = np.full((num_mice, num_pcs * n_freqs, n_frames), np.nan)
    
    def manual_ricker(points, a):
        t = np.arange(0, points) - (points - 1.0) / 2.0
        const = 2 / (np.sqrt(3 * a) * (np.pi**0.25))
        term = (t / a)**2
        return const * (1 - term) * np.exp(-term / 2)

    for m in range(num_mice):
        features_m = []
        for i in range(num_pcs):
            sig = pcs[m, i, :]
            if np.all(np.isnan(sig)):
                features_m.append(np.full((n_freqs, n_frames), np.nan))
                continue
            sig_filled = np.nan_to_num(sig, nan=np.nanmean(sig))
            output = np.zeros((len(widths), len(sig_filled)))
            for ind, width in enumerate(widths):
                points = int(min(10 * width, len(sig_filled)))
                if points % 2 == 0: points += 1 
                wavelet_kernel = manual_ricker(points, width)
                res = np.convolve(sig_filled, wavelet_kernel, mode='same')
                output[ind, :] = np.abs(res)**2
            features_m.append(output)
        wavelet_stack[m] = np.vstack(features_m)
    return wavelet_stack

def build_final_feature_matrix(wavelet_feats, vel_feats, acc_feats, sq_feats):
    m, d, f = wavelet_feats.shape
    wavelet_flat = wavelet_feats.transpose(0, 2, 1).reshape(-1, d)
    vel_flat = vel_feats.transpose(0, 2, 1).reshape(-1, vel_feats.shape[1])
    acc_flat = acc_feats.transpose(0, 2, 1).reshape(-1, acc_feats.shape[1])
    sq_flat = sq_feats.transpose(0, 2, 1).reshape(-1, sq_feats.shape[1])
    combined = np.hstack([wavelet_flat, vel_flat, acc_flat, sq_flat])
    return combined

try:
    with h5py.File(filename, "r") as f:
        raw_tracks = f["tracks"][:]
        node_names = [name.decode() for name in f["node_names"][:]]
except FileNotFoundError:
    raise FileNotFoundError(f"Cannot find the specified H5 file: {filename}. Please check the path.")

num_mice, _, num_nodes, num_frames = raw_tracks.shape
scaled_tracks = np.full_like(raw_tracks, np.nan)

for m_idx in range(num_mice):
    m_data = np.full((2, num_nodes, num_frames), np.nan)
    for xy in range(2):
        for node in range(num_nodes):
            m_data[xy, node, :] = preprocess_pipeline(raw_tracks[m_idx, xy, node, :])
    
    dists = np.sqrt(np.sum((m_data[:, 0, :] - m_data[:, 3, :])**2, axis=0))
    valid_dists = dists[~np.isnan(dists)]
    if len(valid_dists) > 0:
        scale = np.percentile(valid_dists, 97.5)
        scaled_tracks[m_idx] = m_data / scale

node_pairs = list(itertools.combinations(range(num_nodes), 2))
dist_features = np.full((num_mice, 6, num_frames), np.nan)

for m_idx in range(num_mice):
    for p_idx, (i1, i2) in enumerate(node_pairs):
        p1, p2 = scaled_tracks[m_idx, :, i1, :], scaled_tracks[m_idx, :, i2, :]
        dist_features[m_idx, p_idx, :] = np.sqrt(np.sum((p1 - p2)**2, axis=0))

velocity_features = np.full((num_mice, num_nodes, num_frames), np.nan)
accel_features = np.full((num_mice, num_nodes, num_frames), np.nan)
speed_sq_features = np.full((num_mice, num_nodes, num_frames), np.nan)

for m in range(num_mice):
    for n in range(num_nodes):
        pos = scaled_tracks[m, :, n, :] 
        vel_xy = np.diff(pos, axis=1, prepend=pos[:, :1]) * FPS
        speed = np.sqrt(np.sum(vel_xy**2, axis=0))
        accel = np.diff(speed, prepend=speed[0]) * FPS
        
        velocity_features[m, n, :] = speed
        accel_features[m, n, :] = np.abs(accel)
        speed_sq_features[m, n, :] = speed**2

flat_features = dist_features.transpose(0, 2, 1).reshape(-1, 6)
valid_pca_mask = ~np.isnan(flat_features).any(axis=1)
train_data = flat_features[valid_pca_mask]

scaler_pca = StandardScaler()
train_data_std = scaler_pca.fit_transform(train_data)

pca = PCA()
pca.fit(train_data_std)
cum_var = np.cumsum(pca.explained_variance_ratio_)
n_components_95 = np.argmax(cum_var >= 0.95) + 1

pcs_flat = np.full((flat_features.shape[0], n_components_95), np.nan)
pcs_flat[valid_pca_mask] = pca.transform(train_data_std)[:, :n_components_95]
posture_pcs = pcs_flat.reshape(num_mice, num_frames, n_components_95).transpose(0, 2, 1)

posture_pcs_final = clean_pc_jumps(posture_pcs)
wavelet_features = get_wavelet_feature_stack(posture_pcs_final, fs=FPS, n_freqs=25)

final_data_matrix = build_final_feature_matrix(
    wavelet_features, velocity_features, accel_features, speed_sq_features
)

num_features = final_data_matrix.shape[1]
features_3d = final_data_matrix.reshape(num_mice, num_frames, num_features)

nan_count_per_frame = np.isnan(features_3d).sum(axis=(0, 2))
total_features_per_frame = num_mice * num_features
tolerance_ratio = 0.30
valid_frame_mask = nan_count_per_frame <= int(total_features_per_frame * tolerance_ratio)

features_3d_cleaned = np.zeros_like(features_3d)
for m in range(num_mice):
    for f_idx in range(num_features):
        sig = features_3d[m, :, f_idx]
        if np.isnan(sig).all():
            features_3d_cleaned[m, :, f_idx] = np.nan
            continue
        sig_series = pd.Series(sig)
        sig_filtered = sig_series.rolling(window=5, center=True, min_periods=1).median().values
        upper_limit = np.nanpercentile(sig_filtered, 99.9)
        features_3d_cleaned[m, :, f_idx] = np.clip(sig_filtered, None, upper_limit)

features_flat_cleaned = features_3d_cleaned.reshape(-1, num_features)
scaler_clean = StandardScaler()
valid_data_std = scaler_clean.fit_transform(features_flat_cleaned)
reshaped_data = valid_data_std.reshape(num_mice, num_frames, num_features)

ind_msq = np.nanmean(reshaped_data**2, axis=2) 
ind_energy_clean = np.sqrt(ind_msq * num_features) 

group_msq = np.nanmean(ind_energy_clean**2, axis=0)
group_movescore_clean = np.sqrt(group_msq * num_mice)
group_movescore_clean[~valid_frame_mask] = np.nan

full_len = num_frames
frames_axis = np.arange(full_len)

plt.figure(figsize=(20, 6)) 
plt.plot(frames_axis, group_movescore_clean, color='#1a5276', lw=1.0, alpha=0.9)
plt.fill_between(frames_axis, group_movescore_clean, color='#5dade2', alpha=0.3)

m_val = np.nanmean(group_movescore_clean)
s_val = np.nanstd(group_movescore_clean)
threshold = m_val + 2 * s_val
plt.axhline(y=threshold, color='#e74c3c', linestyle='--', lw=1.5)

plt.title(f"Full Session Group MoveScore", fontsize=16, fontweight='bold')
plt.ylabel("Integrated Energy (L2 Norm)", fontsize=14)
plt.xlabel("Time (HH:MM:SS) / Frame Number", fontsize=14)
plt.grid(axis='y', linestyle=':', alpha=0.5)

tick_pos = np.linspace(0, full_len-1, 10).astype(int)
tick_labels = [f"{format_time(f, FPS)}\n({f})" for f in tick_pos]
plt.xticks(tick_pos, tick_labels)
plt.xlim(0, full_len)

plt.tight_layout()
save_img_path = "Group_MoveScore_Timeline.png"
plt.savefig(save_img_path, dpi=300, bbox_inches='tight')

export_dict = {
    'frame': np.arange(num_frames),
    'time_str': [format_time(f, FPS) for f in range(num_frames)],
    'group_movescore': group_movescore_clean
}

export_df = pd.DataFrame(export_dict)
export_csv_path = "Group_MoveScore_Results.csv"
export_df.to_csv(export_csv_path, index=False)