import os
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from training.data_loader import load_dataset
from training.train_lightgbm import train_lightgbm_model
from training.evaluate import evaluate_model
from config.config import PROCESSED_DATA_DIR, METADATA_FILE, FEATURES_PKL_PATH, DEFAULT_TEST_SIZE, DEFAULT_RANDOM_STATE, FEATURE_GATING_TOP_K, FEATURE_GATING_REPORT_PATH, PE_FEATURE_VECTOR_DIM, DEFAULT_NUM_BOOST_ROUND, FEATURE_GATING_K_START, FEATURE_GATING_K_STEP

def load_features(use_existing_features: bool):
    if use_existing_features and os.path.exists(FEATURES_PKL_PATH):
        df = pd.read_pickle(FEATURES_PKL_PATH)
        files = df['filename'].tolist()
        y = df['label'].values
        X = df.drop(['filename', 'label'], axis=1).values.astype(np.float32)
        return X, y, files
    X, y, files = load_dataset(PROCESSED_DATA_DIR, METADATA_FILE)
    return X, y, files

def split_sets(X, y, files):
    if len(X) > 10:
        X_temp, X_test, y_temp, y_test, files_temp, files_test = train_test_split(
            X, y, files, test_size=DEFAULT_TEST_SIZE, random_state=DEFAULT_RANDOM_STATE, stratify=y if len(np.unique(y)) > 1 else None
        )
        if len(X_temp) > 5:
            X_train, X_val, y_train, y_val, files_train, files_val = train_test_split(
                X_temp, y_temp, files_temp, test_size=DEFAULT_TEST_SIZE, random_state=DEFAULT_RANDOM_STATE, stratify=y_temp if len(np.unique(y_temp)) > 1 else None
            )
        else:
            X_train, X_val = X_temp, X_temp
            y_train, y_val = y_temp, y_temp
            X_test, y_test = X_temp, y_temp
            files_train, files_val, files_test = files_temp, files_temp, files_temp
    else:
        X_train, X_val, X_test = X, X, X
        y_train, y_val, y_test = y, y, y
        files_train, files_val, files_test = files, files, files
    return X_train, y_train, files_train, X_val, y_val, files_val, X_test, y_test, files_test

def mask_top_k(importances: np.ndarray, k: int, n_features: int) -> np.ndarray:
    k = max(1, min(k, n_features))
    idx = np.argsort(importances)[::-1][:k]
    mask = np.zeros(n_features, dtype=bool)
    mask[idx] = True
    return mask

def mask_random(k: int, n_features: int, seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    k = max(1, min(k, n_features))
    idx = rng.choice(n_features, size=k, replace=False)
    mask = np.zeros(n_features, dtype=bool)
    mask[idx] = True
    return mask

def mask_pre_pe_only(n_features: int, pe_dim: int) -> np.ndarray:
    pre_len = max(0, n_features - pe_dim)
    mask = np.zeros(n_features, dtype=bool)
    mask[:pre_len] = True
    return mask

def mask_pe_only(n_features: int, pe_dim: int) -> np.ndarray:
    pre_len = max(0, n_features - pe_dim)
    mask = np.zeros(n_features, dtype=bool)
    mask[pre_len:] = True
    return mask

def apply_mask(X: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return X[:, mask]

def get_k_values(n_features: int, k_start: int, k_step: int) -> list:
    values = []
    start = max(1, min(k_start, n_features))
    step = max(1, k_step)
    cur = start
    while cur <= n_features:
        values.append(cur)
        cur += step
    if values[-1] != n_features:
        values.append(n_features)
    return values

def run_experiments(use_existing_features: bool, k_start: int, k_step: int, num_boost_round: int):
    X, y, files = load_features(use_existing_features)
    X_train, y_train, files_train, X_val, y_val, files_val, X_test, y_test, files_test = split_sets(X, y, files)
    base_model = train_lightgbm_model(X_train, y_train, X_val, y_val, iteration=1, num_boost_round=num_boost_round)
    base_acc, _ = evaluate_model(base_model, X_test, y_test, files_test)
    importances = base_model.feature_importance(importance_type='gain')
    n_features = X_train.shape[1]
    k_values = get_k_values(n_features, k_start, k_step)
    series = []
    for k in k_values:
        mk = mask_top_k(importances, k, n_features)
        X_train_k = apply_mask(X_train, mk)
        X_val_k = apply_mask(X_val, mk)
        X_test_k = apply_mask(X_test, mk)
        model_k = train_lightgbm_model(X_train_k, y_train, X_val_k, y_val, iteration=1, num_boost_round=num_boost_round)
        acc_k, _ = evaluate_model(model_k, X_test_k, y_test, files_test)
        series.append({'k': int(k), 'accuracy': float(acc_k)})
    result = {
        'n_features': int(n_features),
        'k_start': int(k_values[0] if k_values else 0),
        'k_step': int(k_step),
        'k_values': [int(v) for v in k_values],
        'accuracy_baseline': float(base_acc),
        'series_topk': series
    }
    os.makedirs(os.path.dirname(FEATURE_GATING_REPORT_PATH), exist_ok=True)
    with open(FEATURE_GATING_REPORT_PATH, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))

def main():
    parser = argparse.ArgumentParser(description='Feature gating cross experiments')
    parser.add_argument('--use-existing-features', action='store_true')
    parser.add_argument('--k-start', type=int, default=FEATURE_GATING_K_START)
    parser.add_argument('--k-step', type=int, default=FEATURE_GATING_K_STEP)
    parser.add_argument('--num-boost-round', type=int, default=DEFAULT_NUM_BOOST_ROUND)
    args = parser.parse_args()
    run_experiments(args.use_existing_features, args.k_start, args.k_step, args.num_boost_round)

if __name__ == '__main__':
    main()
