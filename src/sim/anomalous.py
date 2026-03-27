"""Feature extraction and classifier for anomalous diffusion mechanism identification."""

from __future__ import annotations

import numpy as np
import scipy.stats
import networkx as nx
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from ..graphs.carpet import build_carpet
from .ctrw import ctrw_positions
from .fbm import fbm_positions


LABELS = ("Fractal", "CTRW", "fBm")


def fractal_positions(
    G: nx.Graph,
    node_pos: dict,
    n_walkers: int,
    n_steps: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Run walkers on G using padded-adjacency vectorised walk.

    Parameters
    ----------
    G : nx.Graph
    node_pos : dict mapping node -> array([x, y]) in [0,1]^2
    n_walkers : int
    n_steps : int
    rng : np.random.Generator

    Returns
    -------
    positions : ndarray of shape (n_walkers, n_steps+1, 2)
    """
    nodes = list(G.nodes())
    n = len(nodes)
    idx_of = {v: i for i, v in enumerate(nodes)}

    # Padded adjacency (n x max_degree)
    degrees = np.array([G.degree(v) for v in nodes], dtype=np.int32)
    max_deg = int(degrees.max())
    adj = np.zeros((n, max_deg), dtype=np.int32)
    for i, v in enumerate(nodes):
        nbs = [idx_of[nb] for nb in G.neighbors(v)]
        adj[i, : len(nbs)] = nbs

    # Physical XY positions as array (n, 2)
    xy = np.array([node_pos[v] for v in nodes], dtype=np.float64)

    # Random starting positions
    start_idx = rng.integers(0, n, size=n_walkers, dtype=np.int32)
    cur = start_idx.copy()

    positions = np.zeros((n_walkers, n_steps + 1, 2), dtype=np.float64)
    positions[:, 0, :] = xy[cur]

    for t in range(1, n_steps + 1):
        deg_cur = degrees[cur]
        col = (rng.random(n_walkers) * deg_cur).astype(np.int32)
        cur = adj[cur, col]
        positions[:, t, :] = xy[cur]

    return positions


def compute_trajectory_features(positions: np.ndarray) -> dict:
    """
    Extract physics-inspired features from an ensemble of 2-D trajectories.

    Parameters
    ----------
    positions : ndarray of shape (n_walkers, n_steps+1, 2)

    Returns
    -------
    features : dict with keys msd_slope, nongaussian, ergodicity, vacf_lag1, kurtosis
    """
    n_walkers, T_plus1, _ = positions.shape
    n_steps = T_plus1 - 1

    # Displacements from origin
    disp = positions - positions[:, :1, :]  # (n_walkers, T+1, 2)
    r2 = np.sum(disp ** 2, axis=-1)         # (n_walkers, T+1)

    # Ensemble MSD at each time
    msd = r2.mean(axis=0)                   # (T+1,)
    t_arr = np.arange(T_plus1, dtype=np.float64)

    # --- MSD log-log slope ---
    t_start = max(5, n_steps // 20)
    t_end = n_steps // 2
    mask = (t_arr[1:] >= t_start) & (t_arr[1:] <= t_end) & (msd[1:] > 1e-12)
    if mask.sum() >= 3:
        lx = np.log(t_arr[1:][mask])
        ly = np.log(msd[1:][mask])
        msd_slope = float(np.polyfit(lx, ly, 1)[0])
    else:
        msd_slope = float("nan")

    # --- Non-Gaussian parameter at t = n_steps // 2 ---
    t_ng = n_steps // 2
    r2_ng = r2[:, t_ng]
    r4_ng = np.sum(disp[:, t_ng, :] ** 2, axis=-1) ** 2
    mean_r2 = float(r2_ng.mean())
    mean_r4 = float(r4_ng.mean())
    if mean_r2 > 1e-12:
        nongaussian = float(mean_r4 / (2.0 * mean_r2 ** 2) - 1.0)
    else:
        nongaussian = 0.0

    # --- Ergodicity: time-avg MSD / ensemble MSD at lag = n_steps // 4 ---
    lag = n_steps // 4
    if lag < 1:
        lag = 1
    # Ensemble MSD at lag
    ens_msd_lag = float(r2[:, lag].mean())

    # Time-averaged MSD per walker at lag
    # time_avg_msd_w = mean over t of |pos[t+lag] - pos[t]|^2
    n_windows = n_steps - lag
    if n_windows > 0:
        diff = positions[:, lag:, :] - positions[:, :n_steps - lag + 1, :]
        # diff shape: (n_walkers, n_windows+1, 2) but we want exactly n_windows points
        diff = positions[:, lag:lag + n_windows, :] - positions[:, :n_windows, :]
        time_avg_msds = np.mean(np.sum(diff ** 2, axis=-1), axis=1)  # (n_walkers,)
        ergodicity_raw = float(time_avg_msds.mean()) / (ens_msd_lag + 1e-12)
        ergodicity = float(np.clip(ergodicity_raw, 0.0, 3.0))
    else:
        ergodicity = 1.0

    # --- VACF at lag 1 ---
    v = positions[:, 1:, :] - positions[:, :-1, :]   # (n_walkers, n_steps, 2)
    vv0 = np.mean(np.sum(v ** 2, axis=-1))            # <v(t)·v(t)>
    if n_steps > 1:
        vv1 = np.mean(np.sum(v[:, :-1, :] * v[:, 1:, :], axis=-1))
    else:
        vv1 = 0.0
    vacf_lag1 = float(vv1 / (vv0 + 1e-12))

    # --- Kurtosis of |displacement| at t = n_steps // 4 ---
    t_k = n_steps // 4
    if t_k < 1:
        t_k = 1
    r_at_tk = np.sqrt(r2[:, t_k])
    kurtosis = float(scipy.stats.kurtosis(r_at_tk, fisher=True, nan_policy="omit"))

    return {
        "msd_slope": msd_slope,
        "nongaussian": nongaussian,
        "ergodicity": ergodicity,
        "vacf_lag1": vacf_lag1,
        "kurtosis": kurtosis,
    }


def generate_anomalous_dataset(
    n_per_class: int = 40,
    n_walkers: int = 200,
    n_steps: int = 300,
    alpha_ctrw: float = 0.70,
    H_fbm: float = 0.35,
    noise_scale: float = 0.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Generate labeled dataset of trajectory-feature vectors for three anomalous
    diffusion mechanisms: Fractal (label 0), CTRW (label 1), fBm (label 2).

    Returns
    -------
    X : ndarray of shape (n_per_class * 3, 5)
    y : ndarray of shape (n_per_class * 3,) — integer labels 0/1/2
    feature_names : list of 5 strings
    """
    rng = np.random.default_rng(seed)
    feature_names = ["msd_slope", "nongaussian", "ergodicity", "vacf_lag1", "kurtosis"]

    rows: list[np.ndarray] = []
    labels: list[int] = []

    # --- Fractal (label 0): Sierpinski carpet depth 3 ---
    G, node_pos, _ = build_carpet(3)

    for _ in range(n_per_class):
        pos = fractal_positions(G, node_pos, n_walkers, n_steps, rng)
        feats = compute_trajectory_features(pos)
        row = np.array([feats[k] for k in feature_names], dtype=np.float64)
        rows.append(row)
        labels.append(0)

    # --- CTRW (label 1) ---
    n_jumps = 15 * n_steps

    for _ in range(n_per_class):
        t_eval_inner = np.arange(1, n_steps + 1, dtype=np.float64)
        pos_ctrw = ctrw_positions(n_walkers, n_jumps, alpha_ctrw, t_eval_inner, rng)
        # Prepend zeros for t=0
        zeros = np.zeros((n_walkers, 1, 2), dtype=np.float64)
        pos = np.concatenate([zeros, pos_ctrw], axis=1)  # (n_walkers, n_steps+1, 2)
        feats = compute_trajectory_features(pos)
        row = np.array([feats[k] for k in feature_names], dtype=np.float64)
        rows.append(row)
        labels.append(1)

    # --- fBm (label 2) ---
    for _ in range(n_per_class):
        pos = fbm_positions(n_walkers, n_steps, H_fbm, rng)
        feats = compute_trajectory_features(pos)
        row = np.array([feats[k] for k in feature_names], dtype=np.float64)
        rows.append(row)
        labels.append(2)

    X = np.vstack(rows)
    y = np.array(labels, dtype=np.int64)

    # Add per-feature Gaussian noise scaled to each feature's dataset std.
    # noise_scale=0 → clean data; noise_scale=0.5 → realistic measurement noise.
    if noise_scale > 0.0:
        feature_stds = X.std(axis=0)
        feature_stds = np.where(feature_stds < 1e-8, 1.0, feature_stds)
        X = X + rng.normal(0.0, noise_scale * feature_stds, size=X.shape)

    return X, y, feature_names


def train_anomalous_classifier(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
) -> dict:
    """
    Train a RandomForest classifier on trajectory features.

    Returns dict with keys:
        confusion_matrix, classification_report, feature_importances,
        accuracy_1feat, accuracy_all
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
    report = classification_report(
        y_test, y_pred, labels=[0, 1, 2],
        target_names=list(LABELS), zero_division=0,
    )
    importances = dict(zip(feature_names, clf.feature_importances_))

    accuracy_all = float(np.mean(y_pred == y_test))

    # Single-feature accuracy
    accuracy_1feat: dict[str, float] = {}
    for fi, fname in enumerate(feature_names):
        Xtr1 = X_train[:, fi:fi + 1]
        Xte1 = X_test[:, fi:fi + 1]
        clf1 = RandomForestClassifier(n_estimators=100, random_state=42)
        clf1.fit(Xtr1, y_train)
        accuracy_1feat[fname] = float(np.mean(clf1.predict(Xte1) == y_test))

    return {
        "confusion_matrix": cm,
        "classification_report": report,
        "feature_importances": importances,
        "accuracy_1feat": accuracy_1feat,
        "accuracy_all": accuracy_all,
        "y_test": y_test,
        "y_pred": y_pred,
    }
