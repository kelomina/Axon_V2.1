import numpy as np
import lightgbm as lgb
import multiprocessing
from config.config import WARMUP_ROUNDS, WARMUP_START_LR, LIGHTGBM_FEATURE_FRACTION, LIGHTGBM_BAGGING_FRACTION, LIGHTGBM_BAGGING_FREQ, LIGHTGBM_MIN_GAIN_TO_SPLIT, LIGHTGBM_MIN_DATA_IN_LEAF, LIGHTGBM_NUM_THREADS_MAX, FP_WEIGHT_BASE, FP_WEIGHT_GROWTH_PER_ITER, FP_WEIGHT_MAX

def warmup_scheduler(warmup_rounds=WARMUP_ROUNDS, start_lr=WARMUP_START_LR, target_lr=0.05):
    def callback(env):
        if env.iteration < warmup_rounds:
            lr = start_lr + (target_lr - start_lr) * (env.iteration / warmup_rounds)
            env.model.params['learning_rate'] = lr
            if env.iteration % 20 == 0:
                 print(f"[*] Warmup: Iteration {env.iteration}, LR: {lr:.6f}")
    return callback

def train_lightgbm_model(X_train, y_train, X_val, y_val, false_positive_files=None, files_train=None, iteration=1, num_boost_round=5000, init_model=None):
    print(f"[*] Training LightGBM model (Round {iteration})...")
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    if false_positive_files is not None and files_train is not None:
        print(f"[*] Detected {len(false_positive_files)} false positive samples, increasing their training weights")
        weights = np.ones(len(X_train), dtype=np.float32)
        false_positive_count = 0
        weight_factor = min(FP_WEIGHT_BASE + iteration * FP_WEIGHT_GROWTH_PER_ITER, FP_WEIGHT_MAX)
        print(f"[*] Current false positive weight factor: {weight_factor}")
        for i, file in enumerate(files_train):
            if file in false_positive_files:
                weights[i] = weight_factor
                false_positive_count += 1
        print(f"[+] Identified {false_positive_count} false positive samples, adjusted weights")
        train_data = lgb.Dataset(X_train, label=y_train, weight=weights)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    learning_rate = max(0.05 / (1.0 + iteration * 0.1), 0.01)
    num_leaves = min(31 + iteration * 5, 128)
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
    print(f"[*] Current training parameters - Learning rate: {learning_rate:.4f}, Number of leaves: {num_leaves}")
    callbacks = [lgb.early_stopping(50), lgb.log_evaluation(50)]
    if iteration == 1 and init_model is None:
        print("[*] Applying Cold Start mechanism (Warm-up)...")
        callbacks.append(warmup_scheduler(warmup_rounds=WARMUP_ROUNDS, start_lr=WARMUP_START_LR, target_lr=learning_rate))
    model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        valid_names=['validation'],
        num_boost_round=num_boost_round,
        init_model=init_model,
        callbacks=callbacks
    )
    return model
