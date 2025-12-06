import os
import argparse
from config.config import (
    MODEL_PATH, FAMILY_CLASSIFIER_PATH, FEATURES_PKL_PATH, PROCESSED_DATA_DIR, METADATA_FILE,
    BENIGN_SAMPLES_DIR, MALICIOUS_SAMPLES_DIR,
    DEFAULT_MAX_FILE_SIZE, DEFAULT_NUM_BOOST_ROUND, DEFAULT_INCREMENTAL_ROUNDS,
    DEFAULT_INCREMENTAL_EARLY_STOPPING, DEFAULT_MAX_FINETUNE_ITERATIONS,
    DEFAULT_MIN_CLUSTER_SIZE, DEFAULT_MIN_SAMPLES, DEFAULT_MIN_FAMILY_SIZE,
    DEFAULT_SERVE_PORT, SCAN_CACHE_PATH, SCAN_OUTPUT_DIR, HDBSCAN_SAVE_DIR,
    HELP_MAX_FILE_SIZE, HELP_FAST_DEV_RUN, HELP_SAVE_FEATURES,
    HELP_FINETUNE_ON_FALSE_POSITIVES, HELP_INCREMENTAL_TRAINING,
    HELP_INCREMENTAL_DATA_DIR, HELP_INCREMENTAL_RAW_DATA_DIR, HELP_FILE_EXTENSIONS,
    HELP_LABEL_INFERENCE, HELP_NUM_BOOST_ROUND, HELP_INCREMENTAL_ROUNDS,
    HELP_INCREMENTAL_EARLY_STOPPING, HELP_MAX_FINETUNE_ITERATIONS,
    HELP_USE_EXISTING_FEATURES, HELP_DATA_DIR, HELP_FEATURES_PATH, HELP_SAVE_DIR,
    HELP_MIN_CLUSTER_SIZE, HELP_MIN_SAMPLES, HELP_MIN_FAMILY_SIZE, HELP_PLOT_PCA,
    HELP_EXPLAIN_DISCREPANCY, HELP_TREAT_NOISE_AS_FAMILY, HELP_LIGHTGBM_MODEL_PATH,
    HELP_FAMILY_CLASSIFIER_PATH, HELP_CACHE_FILE, HELP_FILE_PATH, HELP_DIR_PATH,
    HELP_RECURSIVE, HELP_OUTPUT_PATH, HELP_PORT
)
from utils.logging_utils import get_logger

def main():
    logger = get_logger('kolo')
    parser = argparse.ArgumentParser(prog='KoloVirusDetector', description='KoloVirusDetector 项目入口')
    subs = parser.add_subparsers(dest='command', required=True)

    sp_pretrain = subs.add_parser('pretrain', help='预训练LightGBM模型')
    sp_pretrain.add_argument('--max-file-size', type=int, default=DEFAULT_MAX_FILE_SIZE, help=HELP_MAX_FILE_SIZE)
    sp_pretrain.add_argument('--fast-dev-run', action='store_true', help=HELP_FAST_DEV_RUN)
    sp_pretrain.add_argument('--save-features', action='store_true', help=HELP_SAVE_FEATURES)
    sp_pretrain.add_argument('--finetune-on-false-positives', action='store_true', help=HELP_FINETUNE_ON_FALSE_POSITIVES)
    sp_pretrain.add_argument('--incremental-training', action='store_true', help=HELP_INCREMENTAL_TRAINING)
    sp_pretrain.add_argument('--incremental-data-dir', type=str, help=HELP_INCREMENTAL_DATA_DIR)
    sp_pretrain.add_argument('--incremental-raw-data-dir', type=str, help=HELP_INCREMENTAL_RAW_DATA_DIR)
    sp_pretrain.add_argument('--file-extensions', type=str, nargs='+', help=HELP_FILE_EXTENSIONS)
    sp_pretrain.add_argument('--label-inference', type=str, default='filename', choices=['filename', 'directory'], help=HELP_LABEL_INFERENCE)
    sp_pretrain.add_argument('--num-boost-round', type=int, default=DEFAULT_NUM_BOOST_ROUND, help=HELP_NUM_BOOST_ROUND)
    sp_pretrain.add_argument('--incremental-rounds', type=int, default=DEFAULT_INCREMENTAL_ROUNDS, help=HELP_INCREMENTAL_ROUNDS)
    sp_pretrain.add_argument('--incremental-early-stopping', type=int, default=DEFAULT_INCREMENTAL_EARLY_STOPPING, help=HELP_INCREMENTAL_EARLY_STOPPING)
    sp_pretrain.add_argument('--max-finetune-iterations', type=int, default=DEFAULT_MAX_FINETUNE_ITERATIONS, help=HELP_MAX_FINETUNE_ITERATIONS)
    sp_pretrain.add_argument('--use-existing-features', action='store_true', help=HELP_USE_EXISTING_FEATURES)

    sp_finetune = subs.add_parser('finetune', help='HDBSCAN 家族发现与分类器训练')
    sp_finetune.add_argument('--data-dir', type=str, default=PROCESSED_DATA_DIR, help=HELP_DATA_DIR)
    sp_finetune.add_argument('--features-path', type=str, default=FEATURES_PKL_PATH, help=HELP_FEATURES_PATH)
    sp_finetune.add_argument('--save-dir', type=str, default=HDBSCAN_SAVE_DIR, help=HELP_SAVE_DIR)
    sp_finetune.add_argument('--max-file-size', type=int, default=DEFAULT_MAX_FILE_SIZE, help=HELP_MAX_FILE_SIZE)
    sp_finetune.add_argument('--min-cluster-size', type=int, default=DEFAULT_MIN_CLUSTER_SIZE, help=HELP_MIN_CLUSTER_SIZE)
    sp_finetune.add_argument('--min-samples', type=int, default=DEFAULT_MIN_SAMPLES, help=HELP_MIN_SAMPLES)
    sp_finetune.add_argument('--min-family-size', type=int, default=DEFAULT_MIN_FAMILY_SIZE, help=HELP_MIN_FAMILY_SIZE)
    sp_finetune.add_argument('--plot-pca', action='store_true', help=HELP_PLOT_PCA)
    sp_finetune.add_argument('--explain-discrepancy', action='store_true', help=HELP_EXPLAIN_DISCREPANCY)
    sp_finetune.add_argument('--treat-noise-as-family', action='store_true', help=HELP_TREAT_NOISE_AS_FAMILY)

    sp_scan = subs.add_parser('scan', help='单次扫描或目录扫描')
    sp_scan.add_argument('--lightgbm-model-path', type=str, default=MODEL_PATH, help=HELP_LIGHTGBM_MODEL_PATH)
    sp_scan.add_argument('--family-classifier-path', type=str, default=FAMILY_CLASSIFIER_PATH, help=HELP_FAMILY_CLASSIFIER_PATH)
    sp_scan.add_argument('--cache-file', type=str, default=SCAN_CACHE_PATH, help=HELP_CACHE_FILE)
    sp_scan.add_argument('--file-path', type=str, help=HELP_FILE_PATH)
    sp_scan.add_argument('--dir-path', type=str, help=HELP_DIR_PATH)
    sp_scan.add_argument('--recursive', action='store_true', help=HELP_RECURSIVE)
    sp_scan.add_argument('--output-path', type=str, default=SCAN_OUTPUT_DIR, help=HELP_OUTPUT_PATH)
    sp_scan.add_argument('--max-file-size', type=int, default=DEFAULT_MAX_FILE_SIZE, help=HELP_MAX_FILE_SIZE)

    sp_extract = subs.add_parser('extract', help='从默认样本目录提取并生成处理数据')
    sp_extract.add_argument('--output-dir', type=str, default=PROCESSED_DATA_DIR, help=HELP_SAVE_DIR)
    sp_extract.add_argument('--file-extensions', type=str, nargs='+', help=HELP_FILE_EXTENSIONS)
    sp_extract.add_argument('--label-inference', type=str, default='directory', choices=['filename', 'directory'], help=HELP_LABEL_INFERENCE)
    sp_extract.add_argument('--max-file-size', type=int, default=DEFAULT_MAX_FILE_SIZE, help=HELP_MAX_FILE_SIZE)

    sp_serve = subs.add_parser('serve', help='启动FastAPI扫描服务')
    sp_serve.add_argument('--port', type=int, default=DEFAULT_SERVE_PORT, help=HELP_PORT)

    args = parser.parse_args()

    if args.command == 'pretrain':
        import pretrain
        try:
            pretrain.main(args)
        except Exception as e:
            logger.error(f'预训练失败: {e}')
            raise
    elif args.command == 'finetune':
        import finetune
        try:
            finetune.main(args)
        except Exception as e:
            logger.error(f'微调失败: {e}')
            raise
    elif args.command == 'scan':
        import scanner
        try:
            scanner_instance = scanner.MalwareScanner(
                lightgbm_model_path=args.lightgbm_model_path,
                family_classifier_path=args.family_classifier_path,
                max_file_size=args.max_file_size,
                cache_file=args.cache_file,
                enable_cache=True,
            )
            results = []
            if args.file_path:
                result = scanner_instance.scan_file(args.file_path)
                if result is not None:
                    results.append(result)
            elif args.dir_path:
                results = scanner_instance.scan_directory(args.dir_path, args.recursive)
            else:
                logger.error('请指定 --file-path 或 --dir-path')
                return
            scanner_instance.save_results(results, args.output_path)
        except Exception as e:
            logger.error(f'扫描失败: {e}')
            raise
    elif args.command == 'extract':
        from training.data_loader import extract_features_from_raw_files
        try:
            sources = [BENIGN_SAMPLES_DIR, MALICIOUS_SAMPLES_DIR]
            all_files = []
            all_labels = []
            for src in sources:
                file_names, labels = extract_features_from_raw_files(
                    src, args.output_dir, args.max_file_size, args.file_extensions, args.label_inference
                )
                all_files.extend(file_names)
                all_labels.extend(labels)
            if all_files:
                import json
                os.makedirs(os.path.dirname(METADATA_FILE), exist_ok=True)
                mapping = {fn: ('benign' if lab == 0 else 'malicious') for fn, lab in zip(all_files, all_labels)}
                with open(METADATA_FILE, 'w', encoding='utf-8') as f:
                    json.dump(mapping, f, ensure_ascii=False, indent=2)
                logger.info(f'已生成元数据: {METADATA_FILE}，样本数: {len(all_files)}')
        except Exception as e:
            logger.error(f'提取失败: {e}')
            raise
    elif args.command == 'serve':
        import uvicorn
        import scanner_service
        try:
            uvicorn.run(scanner_service.app, host='0.0.0.0', port=args.port, reload=False)
        except Exception as e:
            logger.error(f'服务启动失败: {e}')
            raise

if __name__ == '__main__':
    main()