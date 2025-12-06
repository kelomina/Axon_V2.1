import os

# 路径参数：统一管理数据与模型的存储位置
# BASE_DIR：配置文件所在目录；用途：定位项目根；推荐值：自动计算
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# PROJECT_ROOT：项目根目录；用途：拼接各模块路径；推荐值：BASE_DIR 的上级目录
PROJECT_ROOT = os.path.dirname(BASE_DIR)
# PROCESSED_DATA_DIR：预处理输出目录；用途：存放 .npz 与 metadata.json；推荐值：data/processed_lightgbm
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed_lightgbm')
# METADATA_FILE：数据元信息文件；用途：提供文件名与标签映射；推荐值：PROCESSED_DATA_DIR/metadata.json
METADATA_FILE = os.path.join(PROCESSED_DATA_DIR, 'metadata.json')
# SAVED_MODEL_DIR：模型保存目录；用途：保存训练生成的模型文件；推荐值：saved_models
SAVED_MODEL_DIR = os.path.join(PROJECT_ROOT, 'saved_models')
# MODEL_PATH：LightGBM 模型文件路径；用途：扫描与评估加载模型；推荐值：saved_models/lightgbm_model.txt
MODEL_PATH = os.path.join(SAVED_MODEL_DIR, 'lightgbm_model.txt')
# FEATURES_PKL_PATH：特征持久化文件；用途：跳过重复特征提取；推荐值：项目根/extracted_features.pkl
FEATURES_PKL_PATH = os.path.join(PROJECT_ROOT, 'extracted_features.pkl')
# FAMILY_CLASSIFIER_PATH：家族分类器模型路径；用途：家族预测；推荐值：hdbscan_cluster_results/family_classifier.pkl
FAMILY_CLASSIFIER_PATH = os.path.join(PROJECT_ROOT, 'hdbscan_cluster_results', 'family_classifier.pkl')
# BENIGN_SAMPLES_DIR：良性样本目录；用途：训练/评估数据来源；推荐值：benign_samples
BENIGN_SAMPLES_DIR = os.path.join(PROJECT_ROOT, 'benign_samples')
# MALICIOUS_SAMPLES_DIR：恶意样本目录；用途：训练/评估数据来源；推荐值：malicious_samples
MALICIOUS_SAMPLES_DIR = os.path.join(PROJECT_ROOT, 'malicious_samples')


# 训练参数：控制特征长度与 LightGBM 超参数
# DEFAULT_MAX_FILE_SIZE：每个文件字节序列长度；用途：截断/填充字节序列；推荐值：64KB-256KB（训练与存储需一致）
DEFAULT_MAX_FILE_SIZE = 64 * 1024
# DEFAULT_NUM_BOOST_ROUND：总迭代轮数；用途：控制训练步数与拟合能力；推荐值：2000-5000
DEFAULT_NUM_BOOST_ROUND = 5000
# DEFAULT_INCREMENTAL_ROUNDS：增量训练轮数；用途：在已有模型基础上追加训练；推荐值：50-200
DEFAULT_INCREMENTAL_ROUNDS = 200
# DEFAULT_INCREMENTAL_EARLY_STOPPING：增量早停轮数；用途：验证集长期不提升则停止；推荐值：30-100
DEFAULT_INCREMENTAL_EARLY_STOPPING = 100
# DEFAULT_MAX_FINETUNE_ITERATIONS：强化训练最大迭代次数；用途：循环微调以降低误报；推荐值：5-15
DEFAULT_MAX_FINETUNE_ITERATIONS = 15
# STAT_CHUNK_COUNT：统计分块数量；用途：分段计算均值/方差/熵；推荐值：10（5-20）
STAT_CHUNK_COUNT = 10
# BYTE_HISTOGRAM_BINS：字节直方图分箱；用途：构建熵等统计；推荐值：256（64-256）
BYTE_HISTOGRAM_BINS = 256
# ENTROPY_BLOCK_SIZE：熵块大小；用途：计算分块熵；推荐值：1024（512-4096）
ENTROPY_BLOCK_SIZE = 2048
# ENTROPY_SAMPLE_SIZE：熵采样大小；用途：快速估计全局熵；推荐值：10240（8192-16384）
ENTROPY_SAMPLE_SIZE = 10240
# LIGHTWEIGHT_FEATURE_DIM：轻量特征维度；用途：前 256 维统计特征长度；推荐值：256
LIGHTWEIGHT_FEATURE_DIM = 256
# LIGHTWEIGHT_FEATURE_SCALE：轻量特征缩放系数；用途：融合时权重调整；推荐值：1.5（1.0-2.0）
LIGHTWEIGHT_FEATURE_SCALE = 1.5
# PE_FEATURE_VECTOR_DIM：综合特征向量总维度；用途：模型输入维度；推荐值：1000
PE_FEATURE_VECTOR_DIM = 1000
# SIZE_NORM_MAX：文件大小归一化上限；用途：避免尺度过大；推荐值：100MB
SIZE_NORM_MAX = 128 * 1024 * 1024
# TIMESTAMP_MAX/TIMESTAMP_YEAR_*：时间戳归一化参数；用途：规范时间特征；推荐值：MAX=2147483647，范围 1970-2038
TIMESTAMP_MAX = 2147483647
TIMESTAMP_YEAR_BASE = 1970
TIMESTAMP_YEAR_MAX = 2038
# LIGHTGBM_FEATURE_FRACTION：特征采样比例；用途：提升泛化、降过拟合；推荐值：0.7-0.9
LIGHTGBM_FEATURE_FRACTION = 0.9
# LIGHTGBM_BAGGING_FRACTION：样本采样比例；用途：随机采样增强稳健性；推荐值：0.7-0.9
LIGHTGBM_BAGGING_FRACTION = 0.9
# LIGHTGBM_BAGGING_FREQ：Bagging 频率；用途：每 N 轮进行一次样本采样；推荐值：5（3-10）
LIGHTGBM_BAGGING_FREQ = 10
# LIGHTGBM_MIN_GAIN_TO_SPLIT：最小分裂增益；用途：控制树的复杂度；推荐值：0.01（0.0-0.1）
LIGHTGBM_MIN_GAIN_TO_SPLIT = 0.01
# LIGHTGBM_MIN_DATA_IN_LEAF：叶子最小样本数；用途：避免过拟合；推荐值：20（10-50）
LIGHTGBM_MIN_DATA_IN_LEAF = 30
# LIGHTGBM_NUM_THREADS_MAX：最大线程数；用途：并行训练；推荐值：8（按 CPU 调整）
LIGHTGBM_NUM_THREADS_MAX = 16
# DEFAULT_LIGHTGBM_NUM_LEAVES/LEARNING_RATE：默认叶子数与学习率；用途：基础复杂度与步长；推荐值：31/0.05
DEFAULT_LIGHTGBM_NUM_LEAVES = 31
DEFAULT_LIGHTGBM_LEARNING_RATE = 0.05


# 帮助文本（训练 CLI）：用途：命令行参数说明文字；推荐值：按需维护
HELP_MAX_FILE_SIZE = 'Maximum file size in bytes to process'
HELP_FAST_DEV_RUN = 'Use a small portion of data for quick development testing'
HELP_SAVE_FEATURES = 'Save extracted features to file'
HELP_FINETUNE_ON_FALSE_POSITIVES = 'Perform reinforcement training when false positive samples are detected'
HELP_INCREMENTAL_TRAINING = 'Enable incremental training (continue training based on existing model)'
HELP_INCREMENTAL_DATA_DIR = 'Incremental training data directory (.npz files)'
HELP_INCREMENTAL_RAW_DATA_DIR = 'Incremental training raw data directory (for feature extraction)'
HELP_FILE_EXTENSIONS = 'File extensions to process, e.g. .exe .dll'
HELP_LABEL_INFERENCE = 'Label inference method: filename (based on file name) or directory (based on directory name)'
HELP_NUM_BOOST_ROUND = 'Number of boosting rounds for training'
HELP_INCREMENTAL_ROUNDS = 'Number of rounds for incremental training'
HELP_INCREMENTAL_EARLY_STOPPING = 'Early stopping rounds for incremental training'
HELP_MAX_FINETUNE_ITERATIONS = 'Maximum reinforcement training iterations'
HELP_USE_EXISTING_FEATURES = 'Use existing extracted_features.pkl file, skip feature extraction'


# 聚类与服务参数：控制 HDBSCAN 与服务端行为
# DEFAULT_MIN_CLUSTER_SIZE：最小簇大小；用途：过滤小簇噪声；推荐值：50（30-100）
DEFAULT_MIN_CLUSTER_SIZE = 30
# DEFAULT_MIN_SAMPLES：核心点最小样本数；用途：影响簇密度判定；推荐值：10（5-20）
DEFAULT_MIN_SAMPLES = 5
# DEFAULT_MIN_FAMILY_SIZE：家族保留阈值；用途：过小家族视为噪声；推荐值：20（10-50）
DEFAULT_MIN_FAMILY_SIZE = 10
# DEFAULT_SERVE_PORT：服务端口；用途：FastAPI 监听端口；推荐值：8000
DEFAULT_SERVE_PORT = 8000
# SCAN_CACHE_PATH：扫描缓存路径；用途：避免重复计算；推荐值：项目根/scan_cache.json
SCAN_CACHE_PATH = os.path.join(PROJECT_ROOT, 'scan_cache.json')
# SCAN_OUTPUT_DIR：扫描结果输出目录；用途：保存 JSON/CSV 结果；推荐值：scan_results
SCAN_OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'scan_results')
# HDBSCAN_SAVE_DIR：聚类结果目录；用途：保存标签与可视化；推荐值：hdbscan_cluster_results
HDBSCAN_SAVE_DIR = os.path.join(PROJECT_ROOT, 'hdbscan_cluster_results')


# 帮助文本（聚类/服务 CLI）：用途：命令行参数说明文字；推荐值：按需维护
HELP_DATA_DIR = 'Directory of processed dataset (.npz & metadata)'
HELP_FEATURES_PATH = 'Path to extracted features pickle'
HELP_SAVE_DIR = 'Directory to save HDBSCAN results'
HELP_MIN_CLUSTER_SIZE = 'Minimum cluster size for HDBSCAN'
HELP_MIN_SAMPLES = 'Minimum samples for HDBSCAN core points'
HELP_MIN_FAMILY_SIZE = 'Minimum family size to keep'
HELP_PLOT_PCA = 'Plot PCA for clusters'
HELP_EXPLAIN_DISCREPANCY = 'Explain discrepancies between cluster and ground truth'
HELP_TREAT_NOISE_AS_FAMILY = 'Treat HDBSCAN noise as a separate family'
HELP_LIGHTGBM_MODEL_PATH = 'Path to LightGBM model file'
HELP_FAMILY_CLASSIFIER_PATH = 'Path to family classifier pickle'
HELP_CACHE_FILE = 'Path to scan cache file'
HELP_FILE_PATH = 'Single file path to scan'
HELP_DIR_PATH = 'Directory path to scan'
HELP_RECURSIVE = 'Recursively scan directories'
HELP_OUTPUT_PATH = 'Directory to save scan results'
HELP_PORT = 'Port for FastAPI service'


# 可视化输出路径：统一保存训练与聚类图表
# MODEL_EVAL_FIG_DIR：图表输出目录；用途：集中管理报告文件；推荐值：reports
MODEL_EVAL_FIG_DIR = os.path.join(PROJECT_ROOT, 'reports')
# MODEL_EVAL_FIG_PATH：模型评估图路径；用途：保存准确率/混淆矩阵等；推荐值：reports/model_evaluation.png
MODEL_EVAL_FIG_PATH = os.path.join(MODEL_EVAL_FIG_DIR, 'model_evaluation.png')
# HDBSCAN_CLUSTER_FIG_PATH：聚类可视化路径；用途：保存聚类热图等；推荐值：reports/hdbscan_clustering_visualization.png
HDBSCAN_CLUSTER_FIG_PATH = os.path.join(MODEL_EVAL_FIG_DIR, 'hdbscan_clustering_visualization.png')
# HDBSCAN_PCA_FIG_PATH：聚类 PCA 图路径；用途：PCA 降维可视化；推荐值：reports/hdbscan_clustering_visualization_pca.png
HDBSCAN_PCA_FIG_PATH = os.path.join(MODEL_EVAL_FIG_DIR, 'hdbscan_clustering_visualization_pca.png')
# 环境变量键：用于在运行时覆盖默认配置
# ENV_LIGHTGBM_MODEL_PATH：覆盖二分类模型路径；用途：部署时灵活配置；推荐值：SCANNER_LIGHTGBM_MODEL_PATH
ENV_LIGHTGBM_MODEL_PATH = 'SCANNER_LIGHTGBM_MODEL_PATH'
# ENV_FAMILY_CLASSIFIER_PATH：覆盖家族分类器路径；用途：部署时灵活配置；推荐值：SCANNER_FAMILY_CLASSIFIER_PATH
ENV_FAMILY_CLASSIFIER_PATH = 'SCANNER_FAMILY_CLASSIFIER_PATH'
# ENV_CACHE_PATH：覆盖扫描缓存路径；用途：持久化缓存位置；推荐值：SCANNER_CACHE_PATH
ENV_CACHE_PATH = 'SCANNER_CACHE_PATH'
# ENV_MAX_FILE_SIZE：覆盖最大字节序列长度；用途：动态调整扫描性能；推荐值：SCANNER_MAX_FILE_SIZE
ENV_MAX_FILE_SIZE = 'SCANNER_MAX_FILE_SIZE'
# ENV_SERVICE_PORT：覆盖服务端口；用途：部署时端口选择；推荐值：SCANNER_SERVICE_PORT
ENV_SERVICE_PORT = 'SCANNER_SERVICE_PORT'
# ENV_ALLOWED_SCAN_ROOT：限制扫描根路径；用途：安全访问控制；推荐值：SCANNER_ALLOWED_SCAN_ROOT
ENV_ALLOWED_SCAN_ROOT = 'SCANNER_ALLOWED_SCAN_ROOT'



# 评估与训练细节参数：控制可视化与学习率策略
# SCAN_PREDICTION_THRESHOLD：恶意概率阈值；用途：二分类输出转标签；推荐值：0.85-0.95（生产 0.90）
SCAN_PREDICTION_THRESHOLD = 0.95
# VIS_SAMPLE_SIZE：可视化采样数；用途：绘图子样本大小；推荐值：5000-20000（默认 10000）
VIS_SAMPLE_SIZE = 20000
# VIS_TSNE_PERPLEXITY：t-SNE 困惑度；用途：嵌入稳定性与结构；推荐值：30（10-50）
VIS_TSNE_PERPLEXITY = 30
# PCA_DIMENSION_FOR_CLUSTERING：聚类前 PCA 维度；用途：降噪与提速；推荐值：50（30-100）
PCA_DIMENSION_FOR_CLUSTERING = 50
# EVAL_HIST_BINS：评估直方图分箱；用途：概率分布可视化精度；推荐值：50（30-100）
EVAL_HIST_BINS = 100
EVAL_PREDICTION_THRESHOLD = 0.5
EVAL_TOP_FEATURE_COUNT = 20
EVAL_FONT_FAMILY = ['SimHei', 'Microsoft YaHei']
DEFAULT_TEST_SIZE = 0.1
DEFAULT_RANDOM_STATE = 42
COMMON_SECTIONS = ['.text', '.data', '.rdata', '.reloc', '.rsrc']
PACKER_SECTION_KEYWORDS = ['upx', 'mpress', 'aspack', 'themida']
SYSTEM_DLLS = {'kernel32','user32','gdi32','advapi32','shell32','ole32','comctl32'}
ENTROPY_HIGH_THRESHOLD = 0.8
ENTROPY_LOW_THRESHOLD = 0.2
LARGE_TRAILING_DATA_SIZE = 1024 * 1024
# FAMILY_THRESHOLD_PERCENTILE：家族阈值百分位；用途：确定家族置信阈；推荐值：95（90-99）
FAMILY_THRESHOLD_PERCENTILE = 90
# FAMILY_THRESHOLD_MULTIPLIER：家族阈值放大倍数；用途：放宽/收紧家族判定；推荐值：1.2（1.0-1.5）
FAMILY_THRESHOLD_MULTIPLIER = 1.0
# WARMUP_ROUNDS：学习率暖启动轮数；用途：前期小步长稳定训练；推荐值：100（50-200）
WARMUP_ROUNDS = 200
# WARMUP_START_LR：暖启动起始学习率；用途：初始学习率；推荐值：0.001（0.0005-0.005）
WARMUP_START_LR = 0.001
# WARMUP_TARGET_LR：暖启动目标学习率；用途：结束时学习率；推荐值：0.05（0.01-0.10）
WARMUP_TARGET_LR = 0.1
