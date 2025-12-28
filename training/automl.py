import os
import json
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, accuracy_score
from config.config import (
    FEATURES_PKL_PATH, PROCESSED_DATA_DIR, METADATA_FILE, DEFAULT_MAX_FILE_SIZE,
    DEFAULT_RANDOM_STATE, DEFAULT_NUM_BOOST_ROUND,
    DEFAULT_LIGHTGBM_NUM_LEAVES, DEFAULT_LIGHTGBM_LEARNING_RATE,
    LIGHTGBM_FEATURE_FRACTION, LIGHTGBM_BAGGING_FRACTION, LIGHTGBM_BAGGING_FREQ,
    LIGHTGBM_MIN_GAIN_TO_SPLIT, LIGHTGBM_MIN_DATA_IN_LEAF,
    AUTOML_RESULTS_PATH, AUTOML_TRIALS_DEFAULT, AUTOML_CV_FOLDS_DEFAULT, AUTOML_METHOD_DEFAULT,
    AUTOML_METRIC_DEFAULT, AUTOML_ADDITIONAL_METRICS,
    AUTOML_LGBM_NUM_LEAVES_MIN, AUTOML_LGBM_NUM_LEAVES_MAX,
    AUTOML_LGBM_LEARNING_RATE_MIN, AUTOML_LGBM_LEARNING_RATE_MAX,
    AUTOML_LGBM_FEATURE_FRACTION_MIN, AUTOML_LGBM_FEATURE_FRACTION_MAX,
    AUTOML_LGBM_BAGGING_FRACTION_MIN, AUTOML_LGBM_BAGGING_FRACTION_MAX,
    AUTOML_LGBM_MIN_DATA_IN_LEAF_MIN, AUTOML_LGBM_MIN_DATA_IN_LEAF_MAX,
    AUTOML_LGBM_MIN_GAIN_TO_SPLIT_MIN, AUTOML_LGBM_MIN_GAIN_TO_SPLIT_MAX,
    AUTOML_LGBM_BAGGING_FREQ_MIN, AUTOML_LGBM_BAGGING_FREQ_MAX,
    AUTOML_LGBM_SCALE_POS_WEIGHT_MIN, AUTOML_LGBM_SCALE_POS_WEIGHT_MAX
)
from training.data_loader import load_dataset

def _load_data(use_existing_features=False, max_file_size=DEFAULT_MAX_FILE_SIZE, fast_dev_run=False):
    import pandas as pd
    X = None
    y = None
    if use_existing_features and os.path.exists(FEATURES_PKL_PATH):
        df = pd.read_pickle(FEATURES_PKL_PATH)
        feature_cols = [c for c in df.columns if c.startswith('feature_')]
        try:
            feature_cols = sorted(feature_cols, key=lambda c: int(c.split('_')[1]))
        except Exception:
            pass
        X = df[feature_cols]
        y = df['label'].astype(int)
    else:
        X_np, y_np, _ = load_dataset(PROCESSED_DATA_DIR, METADATA_FILE, max_file_size, fast_dev_run=fast_dev_run)
        feature_cols = [f'feature_{i}' for i in range(X_np.shape[1])]
        X = pd.DataFrame(X_np, columns=feature_cols)
        y = pd.Series(y_np)
    return X, y

def _make_baseline_model():
    return lgb.LGBMClassifier(
        objective='binary',
        n_estimators=DEFAULT_NUM_BOOST_ROUND,
        num_leaves=DEFAULT_LIGHTGBM_NUM_LEAVES,
        learning_rate=DEFAULT_LIGHTGBM_LEARNING_RATE,
        feature_fraction=LIGHTGBM_FEATURE_FRACTION,
        bagging_fraction=LIGHTGBM_BAGGING_FRACTION,
        bagging_freq=LIGHTGBM_BAGGING_FREQ,
        min_gain_to_split=LIGHTGBM_MIN_GAIN_TO_SPLIT,
        min_data_in_leaf=LIGHTGBM_MIN_DATA_IN_LEAF,
        subsample=LIGHTGBM_BAGGING_FRACTION,
        subsample_freq=LIGHTGBM_BAGGING_FREQ,
        verbosity=-1,
        random_state=DEFAULT_RANDOM_STATE
    )

def _cv_score(model, X, y, cv_folds, metric):
    if len(np.unique(y)) < 2:
        return 0.0
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=DEFAULT_RANDOM_STATE)
    # 支持更多指标
    valid_metrics = {
        'roc_auc': 'roc_auc',
        'accuracy': 'accuracy',
        'f1': 'f1',
        'precision': 'precision',
        'recall': 'recall'
    }
    scoring = valid_metrics.get(metric, 'roc_auc')
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    return float(np.mean(scores))

def _optuna_tune_lgbm(X, y, cv_folds, trials, metric):
    import optuna
    def objective(trial):
        params = {
            'objective': 'binary',
            'n_estimators': DEFAULT_NUM_BOOST_ROUND,
            'num_leaves': trial.suggest_int('num_leaves', AUTOML_LGBM_NUM_LEAVES_MIN, AUTOML_LGBM_NUM_LEAVES_MAX),
            'learning_rate': trial.suggest_float('learning_rate', AUTOML_LGBM_LEARNING_RATE_MIN, AUTOML_LGBM_LEARNING_RATE_MAX, log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', AUTOML_LGBM_FEATURE_FRACTION_MIN, AUTOML_LGBM_FEATURE_FRACTION_MAX),
            'bagging_fraction': trial.suggest_float('bagging_fraction', AUTOML_LGBM_BAGGING_FRACTION_MIN, AUTOML_LGBM_BAGGING_FRACTION_MAX),
            'bagging_freq': trial.suggest_int('bagging_freq', AUTOML_LGBM_BAGGING_FREQ_MIN, AUTOML_LGBM_BAGGING_FREQ_MAX),
            'min_gain_to_split': trial.suggest_float('min_gain_to_split', AUTOML_LGBM_MIN_GAIN_TO_SPLIT_MIN, AUTOML_LGBM_MIN_GAIN_TO_SPLIT_MAX),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', AUTOML_LGBM_MIN_DATA_IN_LEAF_MIN, AUTOML_LGBM_MIN_DATA_IN_LEAF_MAX),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', AUTOML_LGBM_SCALE_POS_WEIGHT_MIN, AUTOML_LGBM_SCALE_POS_WEIGHT_MAX),
            'verbosity': -1,
            'random_state': DEFAULT_RANDOM_STATE
        }
        model = lgb.LGBMClassifier(**params)
        score = _cv_score(model, X, y, cv_folds, metric)
        return score
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=trials)
    best_params = study.best_params
    best_model = lgb.LGBMClassifier(
        objective='binary',
        n_estimators=DEFAULT_NUM_BOOST_ROUND,
        verbosity=-1,
        random_state=DEFAULT_RANDOM_STATE,
        **best_params
    )
    best_score = _cv_score(best_model, X, y, cv_folds, metric)
    return best_score, best_params

def _hyperopt_tune_lgbm(X, y, cv_folds, trials, metric):
    from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
    space = {
        'num_leaves': hp.quniform('num_leaves', AUTOML_LGBM_NUM_LEAVES_MIN, AUTOML_LGBM_NUM_LEAVES_MAX, 1),
        'learning_rate': hp.loguniform('learning_rate', np.log(AUTOML_LGBM_LEARNING_RATE_MIN), np.log(AUTOML_LGBM_LEARNING_RATE_MAX)),
        'feature_fraction': hp.uniform('feature_fraction', AUTOML_LGBM_FEATURE_FRACTION_MIN, AUTOML_LGBM_FEATURE_FRACTION_MAX),
        'bagging_fraction': hp.uniform('bagging_fraction', AUTOML_LGBM_BAGGING_FRACTION_MIN, AUTOML_LGBM_BAGGING_FRACTION_MAX),
        'bagging_freq': hp.quniform('bagging_freq', AUTOML_LGBM_BAGGING_FREQ_MIN, AUTOML_LGBM_BAGGING_FREQ_MAX, 1),
        'min_gain_to_split': hp.uniform('min_gain_to_split', AUTOML_LGBM_MIN_GAIN_TO_SPLIT_MIN, AUTOML_LGBM_MIN_GAIN_TO_SPLIT_MAX),
        'min_data_in_leaf': hp.quniform('min_data_in_leaf', AUTOML_LGBM_MIN_DATA_IN_LEAF_MIN, AUTOML_LGBM_MIN_DATA_IN_LEAF_MAX, 1),
        'scale_pos_weight': hp.uniform('scale_pos_weight', AUTOML_LGBM_SCALE_POS_WEIGHT_MIN, AUTOML_LGBM_SCALE_POS_WEIGHT_MAX)
    }
    def objective(params):
        params_cast = {
            'objective': 'binary',
            'n_estimators': DEFAULT_NUM_BOOST_ROUND,
            'num_leaves': int(params['num_leaves']),
            'learning_rate': float(params['learning_rate']),
            'feature_fraction': float(params['feature_fraction']),
            'bagging_fraction': float(params['bagging_fraction']),
            'bagging_freq': int(params['bagging_freq']),
            'min_gain_to_split': float(params['min_gain_to_split']),
            'min_data_in_leaf': int(params['min_data_in_leaf']),
            'scale_pos_weight': float(params['scale_pos_weight']),
            'verbosity': -1,
            'random_state': DEFAULT_RANDOM_STATE
        }
        model = lgb.LGBMClassifier(**params_cast)
        score = _cv_score(model, X, y, cv_folds, metric)
        loss = -score
        return {'loss': loss, 'status': STATUS_OK}
    trials_obj = Trials()
    best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=trials, trials=trials_obj)
    best_params = {
        'num_leaves': int(best['num_leaves']),
        'learning_rate': float(best['learning_rate']),
        'feature_fraction': float(best['feature_fraction']),
        'bagging_fraction': float(best['bagging_fraction']),
        'bagging_freq': int(best['bagging_freq']),
        'min_gain_to_split': float(best['min_gain_to_split']),
        'min_data_in_leaf': int(best['min_data_in_leaf']),
        'scale_pos_weight': float(best['scale_pos_weight'])
    }
    best_model = lgb.LGBMClassifier(
        objective='binary',
        n_estimators=DEFAULT_NUM_BOOST_ROUND,
        verbosity=-1,
        random_state=DEFAULT_RANDOM_STATE,
        **best_params
    )
    best_score = _cv_score(best_model, X, y, cv_folds, metric)
    return best_score, best_params

def run_cross_test(method=AUTOML_METHOD_DEFAULT, trials=AUTOML_TRIALS_DEFAULT, cv_folds=AUTOML_CV_FOLDS_DEFAULT, metric=AUTOML_METRIC_DEFAULT, use_existing_features=True, max_file_size=DEFAULT_MAX_FILE_SIZE, fast_dev_run=False):
    X, y = _load_data(use_existing_features=use_existing_features, max_file_size=max_file_size, fast_dev_run=fast_dev_run)
    baseline_model = _make_baseline_model()
    baseline_score = _cv_score(baseline_model, X, y, cv_folds, metric)
    
    # 计算基线的额外指标
    baseline_additional = {}
    for m in AUTOML_ADDITIONAL_METRICS:
        baseline_additional[m] = _cv_score(baseline_model, X, y, cv_folds, m)

    tuned_score = baseline_score
    best_params = {}
    if method == 'optuna':
        tuned_score, best_params = _optuna_tune_lgbm(X, y, cv_folds, trials, metric)
    elif method == 'hyperopt':
        tuned_score, best_params = _hyperopt_tune_lgbm(X, y, cv_folds, trials, metric)
    
    # 计算调优后的额外指标
    best_model = lgb.LGBMClassifier(
        objective='binary',
        n_estimators=DEFAULT_NUM_BOOST_ROUND,
        verbosity=-1,
        random_state=DEFAULT_RANDOM_STATE,
        **best_params
    )
    tuned_additional = {}
    for m in AUTOML_ADDITIONAL_METRICS:
        tuned_additional[m] = _cv_score(best_model, X, y, cv_folds, m)

    os.makedirs(os.path.dirname(AUTOML_RESULTS_PATH), exist_ok=True)
    result = {
        'method': method,
        'metric': metric,
        'cv_folds': cv_folds,
        'trials': trials,
        'baseline_score': baseline_score,
        'baseline_additional_metrics': baseline_additional,
        'tuned_score': tuned_score,
        'tuned_additional_metrics': tuned_additional,
        'best_params': best_params
    }
    with open(AUTOML_RESULTS_PATH, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result

def main(args):
    method = getattr(args, 'method', AUTOML_METHOD_DEFAULT)
    trials = getattr(args, 'trials', AUTOML_TRIALS_DEFAULT)
    cv = getattr(args, 'cv', AUTOML_CV_FOLDS_DEFAULT)
    metric = getattr(args, 'metric', AUTOML_METRIC_DEFAULT)
    use_existing = getattr(args, 'use_existing_features', True)
    fast_dev_run = getattr(args, 'fast_dev_run', False)
    max_file_size = getattr(args, 'max_file_size', DEFAULT_MAX_FILE_SIZE)
    return run_cross_test(method=method, trials=trials, cv_folds=cv, metric=metric, use_existing_features=use_existing, max_file_size=max_file_size, fast_dev_run=fast_dev_run)
