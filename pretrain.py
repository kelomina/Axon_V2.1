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
        from config.config import DEFAULT_TEST_SIZE, DEFAULT_VAL_SIZE, DEFAULT_RANDOM_STATE
        X_temp, X_test, y_temp, y_test, files_temp, files_test = train_test_split(
            X, y, files, test_size=DEFAULT_TEST_SIZE, random_state=DEFAULT_RANDOM_STATE, stratify=y if len(np.unique(y)) > 1 else None
        )
        if len(X_temp) > 5:
            X_train, X_val, y_train, y_val, files_train, files_val = train_test_split(
                X_temp, y_temp, files_temp, test_size=DEFAULT_VAL_SIZE, random_state=DEFAULT_RANDOM_STATE, stratify=y_temp if len(np.unique(y_temp)) > 1 else None
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
        model = train_lightgbm_model(X_train, y_train, X_val, y_val, iteration=1, num_boost_round=args.num_boost_round, params_override=getattr(args, 'override_params', None))

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
                                       init_model=model,
                                       params_override=getattr(args, 'override_params', None))

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
                                           init_model=model,
                                           params_override=getattr(args, 'override_params', None))

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
        try:
            y_pred_proba = model.predict(X_test, num_iteration=model.best_iteration)
            thresholds = np.arange(0.90, 0.99, 0.01)
            from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
            print("\n[*] Threshold sensitivity (0.90–0.98):")
            for t in thresholds:
                y_pred_t = (y_pred_proba > t).astype(int)
                cm = confusion_matrix(y_test, y_pred_t)
                if cm.shape == (2, 2):
                    tn, fp, fn, tp = cm.ravel()
                else:
                    tn = fp = fn = tp = 0
                acc = accuracy_score(y_test, y_pred_t)
                pre = precision_score(y_test, y_pred_t, zero_division=0)
                rec = recall_score(y_test, y_pred_t, zero_division=0)
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
                tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                fp_count = int(fp)
                print(f"    t={t:.2f} acc={acc:.4f} pre={pre:.4f} rec={rec:.4f} FPR={fpr:.4f} TPR={tpr:.4f} FP={fp_count}")
        except Exception:
            pass
    else:

        print("[*] Test set is empty, skipping model evaluation")

    save_model(model, MODEL_PATH)

    from config.config import EVAL_TOP_FEATURE_COUNT
    print(f"\n[*] Top {EVAL_TOP_FEATURE_COUNT} important features:")
    feature_importance = model.feature_importance(importance_type='gain')
    indices_sorted = np.argsort(feature_importance)[::-1]
    def get_feature_semantics(index):
        n_stat = 49
        if index < n_stat:
            if index == 0:
                return '字节均值'
            elif index == 1:
                return '字节标准差'
            elif index == 2:
                return '字节最小值'
            elif index == 3:
                return '字节最大值'
            elif index == 4:
                return '字节中位数'
            elif index == 5:
                return '字节25分位'
            elif index == 6:
                return '字节75分位'
            elif index == 7:
                return '零字节计数'
            elif index == 8:
                return '0xFF字节计数'
            elif index == 9:
                return '0x90字节计数'
            elif index == 10:
                return '可打印字节计数'
            elif index == 11:
                return '全局熵'
            elif 12 <= index <= 20:
                pos = (index - 12) // 3
                mod = (index - 12) % 3
                seg = ['前段','中段','后段'][pos]
                name = ['均值','标准差','熵'][mod]
                return seg + name
            elif 21 <= index <= 30:
                return f'分块均值_{index-21}'
            elif 31 <= index <= 40:
                return f'分块标准差_{index-31}'
            elif 41 <= index <= 44:
                return ['分块均值差绝对均值','分块均值差标准差','分块均值差最大值','分块均值差最小值'][index-41]
            elif 45 <= index <= 48:
                return ['分块标准差差绝对均值','分块标准差差标准差','分块标准差差最大值','分块标准差差最小值'][index-45]
            else:
                return '统计特征'
        j = index - n_stat
        if j < 256:
            if j < 128:
                return '轻量哈希位:导入DLL'
            elif j < 224:
                return '轻量哈希位:导入API'
            else:
                return '轻量哈希位:节名'
        k = j - 256
        order = [
            'size','log_size','sections_count','symbols_count','imports_count','exports_count',
            'unique_imports','unique_dlls','unique_apis','section_names_count','section_total_size',
            'section_total_vsize','avg_section_size','avg_section_vsize','section_entropy_avg','section_entropy_min','section_entropy_max','section_entropy_std','packed_sections_ratio','subsystem','dll_characteristics',
            'code_section_ratio','data_section_ratio','code_vsize_ratio','data_vsize_ratio',
            'has_nx_compat','has_aslr','has_seh','has_guard_cf','has_resources','has_debug_info',
            'has_tls','has_relocs','has_exceptions','dll_name_avg_length','dll_name_max_length',
            'dll_name_min_length','section_name_avg_length','section_name_max_length','section_name_min_length',
            'export_name_avg_length','export_name_max_length','export_name_min_length','max_section_size',
            'min_section_size','long_sections_count','short_sections_count','section_size_std','section_size_cv',
            'executable_writable_sections','file_entropy_avg','file_entropy_min','file_entropy_max','file_entropy_range',
            'zero_byte_ratio','printable_byte_ratio','trailing_data_size','trailing_data_ratio','imported_system_dlls_count',
            'exports_density','has_large_trailing_data','pe_header_size','header_size_ratio','file_entropy_std',
            'file_entropy_q25','file_entropy_q75','file_entropy_median','high_entropy_ratio','low_entropy_ratio',
            'entropy_change_rate','entropy_change_std','executable_sections_count','writable_sections_count',
            'readable_sections_count','executable_sections_ratio','writable_sections_ratio','readable_sections_ratio',
            'executable_code_density','non_standard_executable_sections_count','rwx_sections_count','rwx_sections_ratio',
            'special_char_ratio','long_sections_ratio','short_sections_ratio','has_.text_section','has_.data_section','has_.rdata_section','has_.reloc_section','has_.rsrc_section',
            'has_signature','signature_size','signature_has_signing_time','version_info_present','company_name_len','product_name_len','file_version_len','original_filename_len',
            'has_upx_section','has_mpress_section','has_aspack_section','has_themida_section','api_network_ratio','api_process_ratio','api_filesystem_ratio','api_registry_ratio','overlay_entropy','overlay_high_entropy_flag','packer_keyword_hits_count','packer_keyword_hits_ratio','timestamp','timestamp_year'
        ]
        if k < len(order):
            m = order[k]
            return m
        return 'PE特征'
    for rank, idx in enumerate(indices_sorted[:EVAL_TOP_FEATURE_COUNT], 1):
        semantics = get_feature_semantics(idx)
        print(f"    {rank:2d}. feature_{idx}: {feature_importance[idx]:.2f} ({semantics})")

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
