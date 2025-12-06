import os
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from training.data_loader import load_dataset, extract_features_from_raw_files, load_incremental_dataset
from training.feature_io import save_features_to_pickle
from training.train_lightgbm import train_lightgbm_model
from training.evaluate import evaluate_model
from training.model_io import save_model, load_existing_model
from training.incremental import incremental_train_lightgbm_model
from config.config import PROCESSED_DATA_DIR, METADATA_FILE, SAVED_MODEL_DIR, MODEL_PATH, FEATURES_PKL_PATH, DEFAULT_MAX_FILE_SIZE, DEFAULT_NUM_BOOST_ROUND, DEFAULT_INCREMENTAL_ROUNDS, DEFAULT_INCREMENTAL_EARLY_STOPPING, DEFAULT_MAX_FINETUNE_ITERATIONS, HELP_MAX_FILE_SIZE, HELP_FAST_DEV_RUN, HELP_SAVE_FEATURES, HELP_FINETUNE_ON_FALSE_POSITIVES, HELP_INCREMENTAL_TRAINING, HELP_INCREMENTAL_DATA_DIR, HELP_INCREMENTAL_RAW_DATA_DIR, HELP_FILE_EXTENSIONS, HELP_LABEL_INFERENCE, HELP_NUM_BOOST_ROUND, HELP_INCREMENTAL_ROUNDS, HELP_INCREMENTAL_EARLY_STOPPING, HELP_MAX_FINETUNE_ITERATIONS, HELP_USE_EXISTING_FEATURES


def main(args):
    os.makedirs(SAVED_MODEL_DIR, exist_ok=True)

    if args.use_existing_features and os.path.exists(FEATURES_PKL_PATH):

        print("[*] Loading existing feature file...")
        try:
            df = pd.read_pickle(FEATURES_PKL_PATH)
            files = df['filename'].tolist()
            y = df['label'].values
            X = df.drop(['filename', 'label'], axis=1).values

            print(f"[+] Successfully loaded feature file, total {len(files)} samples, feature dimension: {X.shape[1]}")
        except Exception as e:

            print(f"[!] Failed to load feature file: {e}")

            print("[-] Exiting training")
            return
    else:
        if args.incremental_training and args.incremental_data_dir:
            if args.incremental_raw_data_dir:

                print("[*] Extracting features from raw files...")
                output_features_dir = args.incremental_data_dir
                file_names, labels = extract_features_from_raw_files(
                    args.incremental_raw_data_dir,
                    output_features_dir,
                    args.max_file_size,
                    args.file_extensions,
                    args.label_inference
                )

                if not file_names:

                    print("[!] Failed to extract features from raw files, exiting training")
                    return

            X, y, files = load_incremental_dataset(args.incremental_data_dir, args.max_file_size)
            if X is None:

                print("[!] Failed to load incremental data, exiting training")
                return
        else:
            X, y, files = load_dataset(PROCESSED_DATA_DIR, METADATA_FILE, args.max_file_size, args.fast_dev_run)

        save_features_to_pickle(X, y, files, FEATURES_PKL_PATH)

    if len(X) > 10:
        X_temp, X_test, y_temp, y_test, files_temp, files_test = train_test_split(
            X, y, files, test_size=0.1, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
        )
        if len(X_temp) > 5:
            X_train, X_val, y_train, y_val, files_train, files_val = train_test_split(
                X_temp, y_temp, files_temp, test_size=0.1, random_state=42, stratify=y_temp if len(np.unique(y_temp)) > 1 else None
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

    print(f"[*] Dataset split completed:")
    print(f"    Training set: {len(X_train)} samples")
    print(f"    Validation set: {len(X_val)} samples")
    print(f"    Test set: {len(X_test)} samples")
    print(f"    Class distribution - Train: Benign={np.sum(y_train==0)}, Malicious={np.sum(y_train==1)}")
    print(f"    Class distribution - Val: Benign={np.sum(y_val==0)}, Malicious={np.sum(y_val==1)}")
    print(f"    Class distribution - Test: Benign={np.sum(y_test==0)}, Malicious={np.sum(y_test==1)}")

    existing_model = None
    if args.incremental_training:
        existing_model = load_existing_model(MODEL_PATH)

    model = None

    if args.incremental_training and existing_model:

        print("\n[*] Performing incremental training...")
        model = incremental_train_lightgbm_model(
            existing_model, X_train, y_train, X_val, y_val,
            num_boost_round=args.incremental_rounds,
            early_stopping_rounds=args.incremental_early_stopping
        )
    else:
        model = train_lightgbm_model(X_train, y_train, X_val, y_val, iteration=1, num_boost_round=args.num_boost_round)

    max_finetune_iterations = args.max_finetune_iterations
    finetune_iteration = 0
    false_positives = []

    while finetune_iteration < max_finetune_iterations:
        if args.finetune_on_false_positives:
            finetune_iteration += 1

            print(f"\n[*] Performing round {finetune_iteration} reinforcement training...")
            # Gradient Restart / Incremental: Pass the previous model as init_model
            model = train_lightgbm_model(X_train, y_train, X_val, y_val, 
                                       iteration=finetune_iteration+1, 
                                       num_boost_round=args.num_boost_round,
                                       init_model=model)

            if finetune_iteration >= max_finetune_iterations:

                print("[*] Reached maximum reinforcement training rounds")
                break
        else:

            print("[*] Reinforcement training not enabled, skipping reinforcement training phase")
            break

    print("\n[*] Reinforcement training completed, performing final evaluation...")
    if len(X_test) > 0:
        test_accuracy, false_positives = evaluate_model(model, X_test, y_test, files_test)

        if false_positives and args.finetune_on_false_positives:

            print(f"\n[*] Detected {len(false_positives)} false positive samples, performing targeted reinforcement training...")

            targeted_iteration = 0
            max_targeted_iterations = 5
            previous_fp_count = len(false_positives)

            while len(false_positives) > 0 and targeted_iteration < max_targeted_iterations:
                targeted_iteration += 1

                print(f"\n[*] Performing round {targeted_iteration} targeted reinforcement training...")
                # Gradient Restart / Incremental: Pass the previous model as init_model
                model = train_lightgbm_model(X_train, y_train, X_val, y_val,
                                           false_positives, files_train,
                                           finetune_iteration + targeted_iteration,
                                           num_boost_round=args.num_boost_round,
                                           init_model=model)

                print(f"\n[*] Evaluating after round {targeted_iteration} targeted reinforcement training...")
                test_accuracy, false_positives = evaluate_model(model, X_test, y_test, files_test)

                if len(false_positives) >= previous_fp_count:

                    print("[*] Targeted reinforcement training failed to reduce false positives, stopping training")
                    break
                previous_fp_count = len(false_positives)

            if len(false_positives) == 0:

                print("[*] Successfully eliminated all false positive samples")
            else:

                print(f"[*] Targeted reinforcement training completed, remaining {len(false_positives)} false positive samples")
        elif false_positives:

            print(f"\n[*] Detected {len(false_positives)} false positive samples, but reinforcement training is not enabled")

            print("    To enable reinforcement training, use the --finetune-on-false-positives parameter")
    else:

        print("[*] Test set is empty, skipping model evaluation")

    save_model(model, MODEL_PATH)

    print("\n[*] Top 20 important features:")
    feature_importance = model.feature_importance(importance_type='gain')
    feature_names = [f'feature_{i}' for i in range(len(feature_importance))]
    importance_pairs = list(zip(feature_names, feature_importance))
    importance_pairs.sort(key=lambda x: x[1], reverse=True)

    for i, (name, importance) in enumerate(importance_pairs[:20]):

        print(f"    {i+1:2d}. {name}: {importance:.2f}")

    print(f"\n[+] LightGBM pre-training completed! Model saved to: {MODEL_PATH}")

    print(f"[+] Extracted features saved to: {FEATURES_PKL_PATH}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="LightGBM-based malware detection pre-training script")
    parser.add_argument('--max-file-size', type=int, default=DEFAULT_MAX_FILE_SIZE, help=HELP_MAX_FILE_SIZE)
    parser.add_argument('--fast-dev-run', action='store_true', help=HELP_FAST_DEV_RUN)
    parser.add_argument('--save-features', action='store_true', help=HELP_SAVE_FEATURES)
    parser.add_argument('--finetune-on-false-positives', action='store_true', help=HELP_FINETUNE_ON_FALSE_POSITIVES)
    parser.add_argument('--incremental-training', action='store_true', help=HELP_INCREMENTAL_TRAINING)
    parser.add_argument('--incremental-data-dir', type=str, help=HELP_INCREMENTAL_DATA_DIR)
    parser.add_argument('--incremental-raw-data-dir', type=str, help=HELP_INCREMENTAL_RAW_DATA_DIR)
    parser.add_argument('--file-extensions', type=str, nargs='+', help=HELP_FILE_EXTENSIONS)
    parser.add_argument('--label-inference', type=str, default='filename', choices=['filename', 'directory'], help=HELP_LABEL_INFERENCE)
    parser.add_argument('--num-boost-round', type=int, default=DEFAULT_NUM_BOOST_ROUND, help=HELP_NUM_BOOST_ROUND)
    parser.add_argument('--incremental-rounds', type=int, default=DEFAULT_INCREMENTAL_ROUNDS, help=HELP_INCREMENTAL_ROUNDS)
    parser.add_argument('--incremental-early-stopping', type=int, default=DEFAULT_INCREMENTAL_EARLY_STOPPING, help=HELP_INCREMENTAL_EARLY_STOPPING)
    parser.add_argument('--max-finetune-iterations', type=int, default=DEFAULT_MAX_FINETUNE_ITERATIONS, help=HELP_MAX_FINETUNE_ITERATIONS)
    parser.add_argument('--use-existing-features', action='store_true', help=HELP_USE_EXISTING_FEATURES)

    args = parser.parse_args()

    if args.incremental_training and not args.incremental_data_dir:

        print("[!] --incremental-data-dir parameter must be specified when enabling incremental training")
        exit(1)

    if args.incremental_raw_data_dir and not args.incremental_data_dir:

        print("[!] --incremental-data-dir parameter must be specified when specifying --incremental-raw-data-dir")
        exit(1)

    main(args)