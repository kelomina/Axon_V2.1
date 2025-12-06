import numpy as np
import lightgbm as lgb
import multiprocessing
from config.config import LIGHTGBM_FEATURE_FRACTION, LIGHTGBM_BAGGING_FRACTION, LIGHTGBM_BAGGING_FREQ, LIGHTGBM_MIN_GAIN_TO_SPLIT, LIGHTGBM_MIN_DATA_IN_LEAF, LIGHTGBM_NUM_THREADS_MAX, DEFAULT_LIGHTGBM_NUM_LEAVES, DEFAULT_LIGHTGBM_LEARNING_RATE

def incremental_train_lightgbm_model(existing_model, X_train, y_train, X_val, y_val, false_positive_files=None, files_train=None, num_boost_round=100, early_stopping_rounds=50):
    print("[*] Performing incremental reinforcement training...")
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    if false_positive_files is not None and files_train is not None:
        print(f"[*] Detected {len(false_positive_files)} false positive samples, increasing their training weights")
        weights = np.ones(len(X_train), dtype=np.float32)
        false_positive_count = 0
        for i, file in enumerate(files_train):
            if file in false_positive_files:
                weights[i] = 10.0
                false_positive_count += 1
        print(f"[+] Identified {false_positive_count} false positive samples, adjusted weights")
        train_data = lgb.Dataset(X_train, label=y_train, weight=weights)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    params = existing_model.params if existing_model else {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': DEFAULT_LIGHTGBM_NUM_LEAVES,
        'learning_rate': DEFAULT_LIGHTGBM_LEARNING_RATE,
        'feature_fraction': LIGHTGBM_FEATURE_FRACTION,
        'bagging_fraction': LIGHTGBM_BAGGING_FRACTION,
        'bagging_freq': LIGHTGBM_BAGGING_FREQ,
        'min_gain_to_split': LIGHTGBM_MIN_GAIN_TO_SPLIT,
        'min_data_in_leaf': LIGHTGBM_MIN_DATA_IN_LEAF,
        'verbose': -1,
        'num_threads': min(multiprocessing.cpu_count(), LIGHTGBM_NUM_THREADS_MAX)
    }
    if existing_model:
        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            valid_names=['validation'],
            num_boost_round=num_boost_round,
            init_model=existing_model,
            callbacks=[lgb.early_stopping(early_stopping_rounds), lgb.log_evaluation(10)]
        )
    else:
        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            valid_names=['validation'],
            num_boost_round=num_boost_round,
            callbacks=[lgb.early_stopping(early_stopping_rounds), lgb.log_evaluation(10)]
        )
    print("[+] Incremental reinforcement training completed")
    return model