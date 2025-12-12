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
from config.config import PROCESSED_DATA_DIR, METADATA_FILE, FEATURES_PKL_PATH, DEFAULT_TEST_SIZE, DEFAULT_RANDOM_STATE, DEFAULT_NUM_BOOST_ROUND, WARMUP_ROUNDS, WARMUP_START_LR, LIGHTGBM_FEATURE_FRACTION, LIGHTGBM_BAGGING_FRACTION, LIGHTGBM_BAGGING_FREQ, LIGHTGBM_MIN_GAIN_TO_SPLIT, LIGHTGBM_MIN_DATA_IN_LEAF, LIGHTGBM_NUM_THREADS_MAX, LEARNING_RATE_SWEEP_NO_WARMUP, LEARNING_RATE_SWEEP_WARMUP_TARGETS, LEARNING_RATE_SWEEP_REPORT_PATH, MODEL_EVAL_FIG_DIR, DEFAULT_EARLY_STOPPING_ROUNDS

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

def train_no_warmup(X_train, y_train, X_val, y_val, lr, num_boost_round: int):
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 36,
        'learning_rate': lr,
        'feature_fraction': LIGHTGBM_FEATURE_FRACTION,
        'bagging_fraction': LIGHTGBM_BAGGING_FRACTION,
        'bagging_freq': LIGHTGBM_BAGGING_FREQ,
        'min_gain_to_split': LIGHTGBM_MIN_GAIN_TO_SPLIT,
        'min_data_in_leaf': LIGHTGBM_MIN_DATA_IN_LEAF,
        'verbose': -1,
        'num_threads': min(multiprocessing.cpu_count(), LIGHTGBM_NUM_THREADS_MAX)
    }
    callbacks = [lgb.early_stopping(DEFAULT_EARLY_STOPPING_ROUNDS), lgb.log_evaluation(50)]
    model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        valid_names=['validation'],
        num_boost_round=num_boost_round,
        callbacks=callbacks
    )
    return model

def simulate_warmup_sequence(warmup_rounds: int, start_lr: float, target_lr: float, total_rounds: int) -> list:
    seq = []
    for i in range(total_rounds):
        if i < warmup_rounds:
            lr = start_lr + (target_lr - start_lr) * (i / warmup_rounds)
        else:
            lr = target_lr
        seq.append(lr)
    return seq

def make_warmup_callback(warmup_rounds: int, start_lr: float, target_lr: float):
    def callback(env):
        if env.iteration < warmup_rounds:
            lr = start_lr + (target_lr - start_lr) * (env.iteration / warmup_rounds)
            env.model.params['learning_rate'] = lr
    return callback

def train_with_warmup(X_train, y_train, X_val, y_val, target_lr, num_boost_round: int):
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 36,
        'learning_rate': target_lr,
        'feature_fraction': LIGHTGBM_FEATURE_FRACTION,
        'bagging_fraction': LIGHTGBM_BAGGING_FRACTION,
        'bagging_freq': LIGHTGBM_BAGGING_FREQ,
        'min_gain_to_split': LIGHTGBM_MIN_GAIN_TO_SPLIT,
        'min_data_in_leaf': LIGHTGBM_MIN_DATA_IN_LEAF,
        'verbose': -1,
        'num_threads': min(multiprocessing.cpu_count(), LIGHTGBM_NUM_THREADS_MAX)
    }
    callbacks = [lgb.early_stopping(DEFAULT_EARLY_STOPPING_ROUNDS), lgb.log_evaluation(50), make_warmup_callback(WARMUP_ROUNDS, WARMUP_START_LR, target_lr)]
    model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        valid_names=['validation'],
        num_boost_round=num_boost_round,
        callbacks=callbacks
    )
    return model

def run_sweep(use_existing_features: bool, num_boost_round: int):
    X, y, files = load_features(use_existing_features)
    X_train, y_train, files_train, X_val, y_val, files_val, X_test, y_test, files_test = split_sets(X, y, files)
    series_no = []
    for lr in LEARNING_RATE_SWEEP_NO_WARMUP:
        model = train_no_warmup(X_train, y_train, X_val, y_val, lr, num_boost_round)
        acc, _ = evaluate_model(model, X_test, y_test, files_test)
        series_no.append({'lr': float(lr), 'accuracy': float(acc)})
    series_warm = []
    for tlr in LEARNING_RATE_SWEEP_WARMUP_TARGETS:
        model = train_with_warmup(X_train, y_train, X_val, y_val, tlr, num_boost_round)
        acc, _ = evaluate_model(model, X_test, y_test, files_test)
        series_warm.append({'target_lr': float(tlr), 'accuracy': float(acc)})
    best_no = max(series_no, key=lambda x: x['accuracy']) if series_no else {'lr': 0.0, 'accuracy': 0.0}
    best_warm = max(series_warm, key=lambda x: x['accuracy']) if series_warm else {'target_lr': 0.0, 'accuracy': 0.0}
    result = {
        'num_boost_round': int(num_boost_round),
        'warmup_rounds': int(WARMUP_ROUNDS),
        'warmup_start_lr': float(WARMUP_START_LR),
        'series_no_warmup': series_no,
        'series_warmup': series_warm,
        'best_no_warmup': best_no,
        'best_warmup': best_warm
    }
    os.makedirs(MODEL_EVAL_FIG_DIR, exist_ok=True)
    with open(LEARNING_RATE_SWEEP_REPORT_PATH, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))

def main():
    parser = argparse.ArgumentParser(description='Learning rate sweep for warmup and no-warmup')
    parser.add_argument('--use-existing-features', action='store_true')
    parser.add_argument('--num-boost-round', type=int, default=DEFAULT_NUM_BOOST_ROUND)
    args = parser.parse_args()
    run_sweep(args.use_existing_features, args.num_boost_round)

if __name__ == '__main__':
    main()
