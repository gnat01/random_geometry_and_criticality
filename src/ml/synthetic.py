"""
Chapter C: synthetic physics-style dataset + sklearn classifier.

Each row = one random graph instance + short random-walk statistics → label = geometry family.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ..graphs.carpet import build_carpet
from ..graphs.percolation import build_percolation
from ..graphs.vicsek import build_vicsek
from ..graphs.common import mean_degree
from ..sim.observables import estimate_msd_rms_series
from ..sim.walks import stationary_distribution_degree


LABELS = ("carpet", "vicsek", "percolation")


@dataclass
class SyntheticConfig:
    n_samples: int = 200
    seed: int = 42
    depth_min: int = 2
    depth_max: int = 4
    perc_size_min: int = 12
    perc_size_max: int = 28
    p_open_min: float = 0.45
    p_open_max: float = 0.65
    n_walkers: int = 80
    n_steps: int = 400
    # Add realistic "measurement noise" to the extracted physics-style features.
    # This prevents the inverse problem from becoming perfectly separable.
    feature_noise: float = 0.8


def _pos_walk_square(nodes):
    return {n: np.array([float(n[0]), float(n[1])], dtype=float) for n in nodes}


def _loglog_slope(steps: np.ndarray, y: np.ndarray) -> float:
    mask = (steps > 15) & (y > 1e-12)
    if mask.sum() < 8:
        return float("nan")
    ls = np.log(steps[mask])
    ly = np.log(y[mask])
    m, _ = np.polyfit(ls, ly, 1)
    return float(m)


def _sample_graph(
    cfg: SyntheticConfig, rng: np.random.Generator, forced_label: int | None = None
):
    """Return (G, pos_walk, label_index, meta dict)."""
    k = int(forced_label) if forced_label is not None else int(rng.integers(0, 3))
    if k == 0:
        d = int(rng.integers(cfg.depth_min, cfg.depth_max + 1))
        G, _, _ = build_carpet(d)
        meta = {"family": "carpet", "depth": d, "p_open": np.nan}
    elif k == 1:
        d = int(rng.integers(cfg.depth_min, cfg.depth_max + 1))
        G, _, _ = build_vicsek(d)
        meta = {"family": "vicsek", "depth": d, "p_open": np.nan}
    else:
        sz = int(rng.integers(cfg.perc_size_min, cfg.perc_size_max + 1))
        p = float(rng.uniform(cfg.p_open_min, cfg.p_open_max))
        G, _, _ = build_percolation(sz, p, rng=rng)
        meta = {"family": "percolation", "depth": np.nan, "p_open": p, "grid_size": sz}

    nodes = list(G.nodes())
    if len(nodes) < 4:
        return None

    pw = _pos_walk_square(nodes)
    start = nodes[rng.integers(len(nodes))]
    # Emulate finite experimental budget: not every sample uses the exact same
    # number of walkers/steps.
    n_walkers_eff = max(10, int(rng.uniform(0.7, 1.0) * cfg.n_walkers))
    n_steps_eff = max(50, int(rng.uniform(0.7, 1.0) * cfg.n_steps))
    st, rms, msd = estimate_msd_rms_series(
        G, pw, start, n_walkers_eff, n_steps_eff, rng
    )
    pi = stationary_distribution_degree(G)
    ent = -sum(
        pv * np.log(pv + 1e-18) for pv in pi.values() if pv > 0
    )

    feats = {
        "log_n_nodes": np.log(len(nodes)),
        "mean_degree": mean_degree(G),
        "rms_loglog_slope": _loglog_slope(st, rms),
        "msd_loglog_slope": _loglog_slope(st, msd),
        "pi_entropy": ent,
    }

    # Add Gaussian noise to make feature extraction less idealized.
    # Keep the noise heteroscedastic across feature types.
    s = float(cfg.feature_noise)
    feats["log_n_nodes"] = feats["log_n_nodes"] + rng.normal(0.0, 0.06 * s)
    feats["mean_degree"] = feats["mean_degree"] + rng.normal(0.0, 0.12 * s)
    feats["rms_loglog_slope"] = feats["rms_loglog_slope"] + rng.normal(0.0, 0.25 * s)
    feats["msd_loglog_slope"] = feats["msd_loglog_slope"] + rng.normal(0.0, 0.25 * s)
    feats["pi_entropy"] = feats["pi_entropy"] + rng.normal(0.0, 0.12 * s)

    row = np.array(
        [
            feats["log_n_nodes"],
            feats["mean_degree"],
            feats["rms_loglog_slope"],
            feats["msd_loglog_slope"],
            feats["pi_entropy"],
        ],
        dtype=float,
    )
    return row, k, feats


def generate_labeled_batch(cfg: SyntheticConfig):
    """Build X (n,5), y (n,), feature_names."""
    rng = np.random.default_rng(cfg.seed)
    rows = []
    labels = []
    attempts = 0
    max_attempts = cfg.n_samples * 20
    n_classes = len(LABELS)
    per_class_target = cfg.n_samples // n_classes
    remainder = cfg.n_samples % n_classes
    target_counts = [per_class_target] * n_classes
    for i in range(remainder):
        target_counts[i] += 1

    class_counts = [0] * n_classes
    while len(rows) < cfg.n_samples and attempts < max_attempts:
        pending = [i for i in range(n_classes) if class_counts[i] < target_counts[i]]
        if not pending:
            break
        attempts += 1
        forced_k = int(pending[rng.integers(len(pending))])
        out = _sample_graph(cfg, rng, forced_label=forced_k)
        if out is None:
            continue
        row, k, _ = out
        if np.any(np.isnan(row[:5])):
            continue
        rows.append(row)
        labels.append(int(k))
        class_counts[k] += 1

    if not rows:
        raise RuntimeError("Failed to generate any valid synthetic samples.")

    X = np.vstack(rows)
    y = np.asarray(labels, dtype=int)
    feature_names = [
        "log_n_nodes",
        "mean_degree",
        "rms_loglog_slope",
        "msd_loglog_slope",
        "pi_entropy",
    ]
    return X, y, feature_names


def train_classifier(X, y, feature_names, test_size: float = 0.25, seed: int = 42):
    """Train RandomForest with scaling; return dict with metrics and model artifacts."""
    counts = Counter(y)
    strat = y if len(counts) >= 2 and min(counts.values()) >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=strat
    )
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        random_state=seed,
        class_weight="balanced_subsample",
    )
    clf.fit(Xtr, y_train)
    pred = clf.predict(Xte)
    report = classification_report(
        y_test,
        pred,
        labels=[0, 1, 2],
        target_names=list(LABELS),
        zero_division=0,
    )
    cm = confusion_matrix(y_test, pred, labels=[0, 1, 2])
    importances = dict(zip(feature_names, clf.feature_importances_))
    return {
        "classifier": clf,
        "scaler": scaler,
        "confusion_matrix": cm,
        "classification_report": report,
        "feature_importances": importances,
        "X_test": Xte,
        "y_test": y_test,
        "y_pred": pred,
    }
