import os
import json
import argparse
import numpy as np
import pandas as pd
import lightgbm as lgb
import multiprocessing
from sklearn.model_selection import train_test_split
from training.data_loader import load_dataset
from training.evaluate import evaluate_model
from config.config import PROCESSED_DATA_DIR, METADATA_FILE, FEATURES_PKL_PATH, DEFAULT_TEST_SIZE, DEFAULT_RANDOM_STATE, DEFAULT_NUM_BOOST_ROUND, WARMUP_ROUNDS, WARMUP_START_LR, LIGHTGBM_FEATURE_FRACTION, LIGHTGBM_BAGGING_FRACTION, LIGHTGBM_BAGGING_FREQ, LIGHTGBM_MIN_GAIN_TO_SPLIT, LIGHTGBM_MIN_DATA_IN_LEAF, LIGHTGBM_NUM_THREADS_MAX, LEARNING_RATE_SWEEP_NO_WARMUP, LEARNING_RATE_SWEEP_WARMUP_TARGETS, COMPLEXITY_SWEEP_NUM_LEAVES, COMPLEXITY_SWEEP_REPORT_PATH, MODEL_EVAL_FIG_DIR, DEFAULT_EARLY_STOPPING_ROUNDS

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

def parse_csv_ints(s: str):
    if not s:
        return None
    parts = [p.strip() for p in s.split(',') if p.strip()]
    return [int(p) for p in parts] if parts else None

def parse_csv_floats(s: str):
    if not s:
        return None
    parts = [p.strip() for p in s.split(',') if p.strip()]
    return [float(p) for p in parts] if parts else None

def make_warmup_callback(warmup_rounds: int, start_lr: float, target_lr: float):
    def callback(env):
        if env.iteration < warmup_rounds:
            lr = start_lr + (target_lr - start_lr) * (env.iteration / warmup_rounds)
            env.model.params['learning_rate'] = lr
    return callback

def train_no_warmup(X_train, y_train, X_val, y_val, lr, leaves, num_boost_round: int):
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': int(leaves),
        'learning_rate': float(lr),
        'feature_fraction': LIGHTGBM_FEATURE_FRACTION,
        'bagging_fraction': LIGHTGBM_BAGGING_FRACTION,
        'bagging_freq': LIGHTGBM_BAGGING_FREQ,
        'min_gain_to_split': LIGHTGBM_MIN_GAIN_TO_SPLIT,
        'min_data_in_leaf': LIGHTGBM_MIN_DATA_IN_LEAF,
        'verbose': -1,
        'num_threads': min(multiprocessing.cpu_count(), LIGHTGBM_NUM_THREADS_MAX)
    }
    callbacks = [lgb.early_stopping(DEFAULT_EARLY_STOPPING_ROUNDS), lgb.log_evaluation(50)]
    model = lgb.train(params, train_data, valid_sets=[val_data], valid_names=['validation'], num_boost_round=num_boost_round, callbacks=callbacks)
    return model

def train_with_warmup(X_train, y_train, X_val, y_val, target_lr, leaves, num_boost_round: int):
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': int(leaves),
        'learning_rate': float(target_lr),
        'feature_fraction': LIGHTGBM_FEATURE_FRACTION,
        'bagging_fraction': LIGHTGBM_BAGGING_FRACTION,
        'bagging_freq': LIGHTGBM_BAGGING_FREQ,
        'min_gain_to_split': LIGHTGBM_MIN_GAIN_TO_SPLIT,
        'min_data_in_leaf': LIGHTGBM_MIN_DATA_IN_LEAF,
        'verbose': -1,
        'num_threads': min(multiprocessing.cpu_count(), LIGHTGBM_NUM_THREADS_MAX)
    }
    callbacks = [lgb.early_stopping(DEFAULT_EARLY_STOPPING_ROUNDS), lgb.log_evaluation(50), make_warmup_callback(WARMUP_ROUNDS, WARMUP_START_LR, target_lr)]
    model = lgb.train(params, train_data, valid_sets=[val_data], valid_names=['validation'], num_boost_round=num_boost_round, callbacks=callbacks)
    return model

def build_grid(leaves_list, lr_list):
    grid = []
    for nl in leaves_list:
        for lr in lr_list:
            grid.append({'num_leaves': int(nl), 'learning_rate': float(lr)})
    return grid

def run_grid(use_existing_features: bool, num_boost_round: int, leaves_override=None, lr_no_override=None, lr_warm_override=None):
    X, y, files = load_features(use_existing_features)
    X_train, y_train, files_train, X_val, y_val, files_val, X_test, y_test, files_test = split_sets(X, y, files)
    leaves_list = leaves_override if leaves_override else COMPLEXITY_SWEEP_NUM_LEAVES
    lr_no_list = lr_no_override if lr_no_override else LEARNING_RATE_SWEEP_NO_WARMUP
    lr_warm_list = lr_warm_override if lr_warm_override else LEARNING_RATE_SWEEP_WARMUP_TARGETS
    grid_no = build_grid(leaves_list, lr_no_list)
    series_no = []
    for g in grid_no:
        model = train_no_warmup(X_train, y_train, X_val, y_val, g['learning_rate'], g['num_leaves'], num_boost_round)
        acc, _ = evaluate_model(model, X_test, y_test, files_test)
        series_no.append({'num_leaves': g['num_leaves'], 'learning_rate': g['learning_rate'], 'accuracy': float(acc)})
    grid_warm = build_grid(leaves_list, lr_warm_list)
    series_warm = []
    for g in grid_warm:
        model = train_with_warmup(X_train, y_train, X_val, y_val, g['learning_rate'], g['num_leaves'], num_boost_round)
        acc, _ = evaluate_model(model, X_test, y_test, files_test)
        series_warm.append({'num_leaves': g['num_leaves'], 'target_lr': g['learning_rate'], 'accuracy': float(acc)})
    best_no = max(series_no, key=lambda x: x['accuracy']) if series_no else {'num_leaves': 0, 'learning_rate': 0.0, 'accuracy': 0.0}
    best_warm = max(series_warm, key=lambda x: x['accuracy']) if series_warm else {'num_leaves': 0, 'target_lr': 0.0, 'accuracy': 0.0}
    result = {
        'num_boost_round': int(num_boost_round),
        'warmup_rounds': int(WARMUP_ROUNDS),
        'warmup_start_lr': float(WARMUP_START_LR),
        'leaves': [int(nl) for nl in leaves_list],
        'lr_no_warmup': [float(lr) for lr in lr_no_list],
        'lr_warmup_targets': [float(lr) for lr in lr_warm_list],
        'series_no_warmup': series_no,
        'series_warmup': series_warm,
        'best_no_warmup': best_no,
        'best_warmup': best_warm
    }
    os.makedirs(MODEL_EVAL_FIG_DIR, exist_ok=True)
    with open(COMPLEXITY_SWEEP_REPORT_PATH, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))

def main():
    parser = argparse.ArgumentParser(description='Model complexity x learning rate grid experiment')
    parser.add_argument('--use-existing-features', action='store_true')
    parser.add_argument('--num-boost-round', type=int, default=DEFAULT_NUM_BOOST_ROUND)
    parser.add_argument('--leaves', type=str, help='Comma-separated num_leaves list, e.g. 31,64,96')
    parser.add_argument('--lr-no', type=str, help='Comma-separated learning rates (no warmup), e.g. 0.03,0.05')
    parser.add_argument('--lr-warm', type=str, help='Comma-separated target learning rates (warmup), e.g. 0.03,0.05')
    args = parser.parse_args()
    leaves_override = parse_csv_ints(args.leaves) if args.leaves else None
    lr_no_override = parse_csv_floats(args.lr_no) if args.lr_no else None
    lr_warm_override = parse_csv_floats(args.lr_warm) if args.lr_warm else None
    run_grid(args.use_existing_features, args.num_boost_round, leaves_override, lr_no_override, lr_warm_override)

if __name__ == '__main__':
    main()
