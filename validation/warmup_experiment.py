import os
import json
import argparse
import numpy as np
import pandas as pd
import lightgbm as lgb
import multiprocessing
from sklearn.model_selection import train_test_split
from training.data_loader import load_dataset
from training.train_lightgbm import train_lightgbm_model
from training.evaluate import evaluate_model
from config.config import PROCESSED_DATA_DIR, METADATA_FILE, FEATURES_PKL_PATH, DEFAULT_TEST_SIZE, DEFAULT_RANDOM_STATE, DEFAULT_NUM_BOOST_ROUND, LIGHTGBM_FEATURE_FRACTION, LIGHTGBM_BAGGING_FRACTION, LIGHTGBM_BAGGING_FREQ, LIGHTGBM_MIN_GAIN_TO_SPLIT, LIGHTGBM_MIN_DATA_IN_LEAF, LIGHTGBM_NUM_THREADS_MAX, MODEL_EVAL_FIG_DIR, DEFAULT_EARLY_STOPPING_ROUNDS

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

def train_without_warmup(X_train, y_train, X_val, y_val, num_boost_round: int):
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    learning_rate = max(0.05 / (1.0 + 1 * 0.1), 0.01)
    num_leaves = min(31 + 1 * 5, 128)
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': num_leaves,
        'learning_rate': learning_rate,
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
    return model, learning_rate, num_leaves

def run_experiment(use_existing_features: bool, num_boost_round: int):
    X, y, files = load_features(use_existing_features)
    X_train, y_train, files_train, X_val, y_val, files_val, X_test, y_test, files_test = split_sets(X, y, files)
    model_warm = train_lightgbm_model(X_train, y_train, X_val, y_val, iteration=1, num_boost_round=num_boost_round)
    acc_warm, _ = evaluate_model(model_warm, X_test, y_test, files_test)
    model_no, lr, nl = train_without_warmup(X_train, y_train, X_val, y_val, num_boost_round)
    acc_no, _ = evaluate_model(model_no, X_test, y_test, files_test)
    result = {
        'accuracy_warmup': float(acc_warm),
        'accuracy_no_warmup': float(acc_no),
        'learning_rate': float(lr),
        'num_leaves': int(nl)
    }
    os.makedirs(MODEL_EVAL_FIG_DIR, exist_ok=True)
    report_path = os.path.join(MODEL_EVAL_FIG_DIR, 'warmup_experiment.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))

def main():
    parser = argparse.ArgumentParser(description='Warmup vs no-warmup cross experiment')
    parser.add_argument('--use-existing-features', action='store_true')
    parser.add_argument('--num-boost-round', type=int, default=DEFAULT_NUM_BOOST_ROUND)
    args = parser.parse_args()
    run_experiment(args.use_existing_features, args.num_boost_round)

if __name__ == '__main__':
    main()
