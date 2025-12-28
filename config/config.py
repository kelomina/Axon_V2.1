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
# BENIGN_WHITELIST_PENDING_DIR：待白名单样本目录；用途：存放待审核的良性样本；推荐值：benign_samples/待加入白名单
BENIGN_WHITELIST_PENDING_DIR = os.path.join(BENIGN_SAMPLES_DIR, '待加入白名单')
# MALICIOUS_SAMPLES_DIR：恶意样本目录；用途：训练/评估数据来源；推荐值：malicious_samples
MALICIOUS_SAMPLES_DIR = os.path.join(PROJECT_ROOT, 'malicious_samples')


# 训练参数：控制特征长度与 LightGBM 超参数
# DEFAULT_MAX_FILE_SIZE：每个文件字节序列长度；用途：截断/填充字节序列；推荐值：64KB-256KB（训练与存储需一致）
DEFAULT_MAX_FILE_SIZE = 64 * 1024
# DEFAULT_NUM_BOOST_ROUND：总迭代轮数；用途：控制训练步数与拟合能力；推荐值：2000-5000
DEFAULT_NUM_BOOST_ROUND = 5000
# DEFAULT_INCREMENTAL_ROUNDS：增量训练轮数；用途：在已有模型基础上追加训练；推荐值：50-200
DEFAULT_INCREMENTAL_ROUNDS = 200
# DEFAULT_EARLY_STOPPING_ROUNDS：训练早停轮数；用途：防止过拟合；推荐值：100-300
DEFAULT_EARLY_STOPPING_ROUNDS = 200
# DEFAULT_INCREMENTAL_EARLY_STOPPING：增量训练早停轮数；用途：增量阶段防止过拟合；推荐值：100-300
DEFAULT_INCREMENTAL_EARLY_STOPPING = 200
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
# PE_FEATURE_VECTOR_DIM：综合特征向量总维度；用途：模型输入维度；推荐值：1500
PE_FEATURE_VECTOR_DIM = 1500
# SIZE_NORM_MAX：文件大小归一化上限；用途：避免尺度过大；推荐值：100MB
SIZE_NORM_MAX = 128 * 1024 * 1024
# TIMESTAMP_MAX/TIMESTAMP_YEAR_*：时间戳归一化参数；用途：规范时间特征；推荐值：MAX=2147483647，范围 1970-2038
TIMESTAMP_MAX = 2147483647
TIMESTAMP_YEAR_BASE = 1970
TIMESTAMP_YEAR_MAX = 2038
# LIGHTGBM_FEATURE_FRACTION：特征采样比例；用途：提升泛化、降过拟合；推荐值：0.7-0.9
LIGHTGBM_FEATURE_FRACTION = 0.6734990233925464
# LIGHTGBM_BAGGING_FRACTION：样本采样比例；用途：随机采样增强稳健性；推荐值：0.7-0.9
LIGHTGBM_BAGGING_FRACTION = 0.8951208064215719
# LIGHTGBM_BAGGING_FREQ：Bagging 频率；用途：每 N 轮进行一次样本采样；推荐值：5（3-10）
LIGHTGBM_BAGGING_FREQ = 13
# LIGHTGBM_MIN_GAIN_TO_SPLIT：最小分裂增益；用途：控制树的复杂度；推荐值：0.01（0.0-0.1）
LIGHTGBM_MIN_GAIN_TO_SPLIT = 0.002330096559042368
# LIGHTGBM_MIN_DATA_IN_LEAF：叶子最小样本数；用途：避免过拟合；推荐值：20（10-50）
LIGHTGBM_MIN_DATA_IN_LEAF = 71
# LIGHTGBM_NUM_THREADS_MAX：最大线程数；用途：并行训练；推荐值：8（按 CPU 调整）
LIGHTGBM_NUM_THREADS_MAX = 16
# DEFAULT_LIGHTGBM_NUM_LEAVES/LEARNING_RATE：默认叶子数与学习率；用途：基础复杂度与步长；推荐值：30/0.07
DEFAULT_LIGHTGBM_NUM_LEAVES = 281
DEFAULT_LIGHTGBM_LEARNING_RATE = 0.0054273608259950085


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
HELP_AUTOML_METHOD = 'AutoML method: optuna or hyperopt'
HELP_AUTOML_TRIALS = 'AutoML tuning trials count'
HELP_AUTOML_CV = 'Cross-validation folds for AutoML comparison'
HELP_AUTOML_METRIC = 'Evaluation metric: roc_auc or accuracy'
HELP_AUTOML_FAST_DEV_RUN = 'Use small subset for AutoML quick run'
HELP_SKIP_TUNING = 'Skip AutoML hyperparameter tuning phase'


# 聚类与服务参数：控制 HDBSCAN 与服务端行为
# DEFAULT_MIN_CLUSTER_SIZE：最小簇大小；用途：过滤小簇噪声；推荐值：2（每一个样本独立成家时设为 2）
DEFAULT_MIN_CLUSTER_SIZE = 2
# DEFAULT_MIN_SAMPLES：核心点最小样本数；用途：影响簇密度判定；推荐值：1（每一个样本独立成家时设为 1）
DEFAULT_MIN_SAMPLES = 1
# DEFAULT_MIN_FAMILY_SIZE：家族保留阈值；用途：过小家族视为噪声；推荐值：1（每一个样本独立成家时设为 1）
DEFAULT_MIN_FAMILY_SIZE = 1
# DEFAULT_TREAT_NOISE_AS_FAMILY：将噪声视为独立家族；用途：开启后每个噪声点将分配独立家族 ID；推荐值：True
DEFAULT_TREAT_NOISE_AS_FAMILY = True
# SCAN_CACHE_PATH：扫描缓存路径；用途：避免重复计算；推荐值：项目根/scan_cache.json
SCAN_CACHE_PATH = os.path.join(PROJECT_ROOT, 'scan_cache.json')
# SCAN_OUTPUT_DIR：扫描结果输出目录；用途：保存 JSON/CSV 结果；推荐值：scan_results
SCAN_OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'scan_results')
# PE_DIM_SUMMARY_DATASET：数据集 PE 维度摘要；用途：记录数据集特征分布；推荐值：scan_results/pe_dim_summary_dataset.json
PE_DIM_SUMMARY_DATASET = os.path.join(SCAN_OUTPUT_DIR, 'pe_dim_summary_dataset.json')
# PE_DIM_SUMMARY_INCREMENTAL：增量数据 PE 维度摘要；用途：记录增量特征分布；推荐值：scan_results/pe_dim_summary_incremental.json
PE_DIM_SUMMARY_INCREMENTAL = os.path.join(SCAN_OUTPUT_DIR, 'pe_dim_summary_incremental.json')
# PE_DIM_SUMMARY_RAW：原始数据 PE 维度摘要；用途：记录原始特征分布；推荐值：scan_results/pe_dim_summary_raw.json
PE_DIM_SUMMARY_RAW = os.path.join(SCAN_OUTPUT_DIR, 'pe_dim_summary_raw.json')
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


# 可视化输出路径：统一保存训练与聚类图表
# MODEL_EVAL_FIG_DIR：图表输出目录；用途：集中管理报告文件；推荐值：reports
MODEL_EVAL_FIG_DIR = os.path.join(PROJECT_ROOT, 'reports')
# MODEL_EVAL_FIG_PATH：模型评估图路径；用途：保存准确率/混淆矩阵等；推荐值：reports/model_evaluation.png
MODEL_EVAL_FIG_PATH = os.path.join(MODEL_EVAL_FIG_DIR, 'model_evaluation.png')
# MODEL_EVAL_AUC_PATH：ROC-AUC 曲线路径；用途：评估分类器性能；推荐值：reports/model_auc_curve.png
MODEL_EVAL_AUC_PATH = os.path.join(MODEL_EVAL_FIG_DIR, 'model_auc_curve.png')
# ROUTING_EVAL_REPORT_PATH：路由评估报告路径；用途：保存路由系统评估文本；推荐值：reports/routing_evaluation_report.txt
ROUTING_EVAL_REPORT_PATH = os.path.join(MODEL_EVAL_FIG_DIR, 'routing_evaluation_report.txt')
# ROUTING_CONFUSION_MATRIX_PATH：路由混淆矩阵路径；用途：保存路由系统混淆矩阵图；推荐值：reports/routing_confusion_matrix.png
ROUTING_CONFUSION_MATRIX_PATH = os.path.join(MODEL_EVAL_FIG_DIR, 'routing_confusion_matrix.png')
# ROUTING_ROC_AUC_PATH：路由 ROC-AUC 曲线路径；用途：评估路由门控性能；推荐值：reports/routing_roc_auc.png
ROUTING_ROC_AUC_PATH = os.path.join(MODEL_EVAL_FIG_DIR, 'routing_roc_auc.png')
# AUTOML_RESULTS_PATH：AutoML 结果路径；用途：保存调优实验结果；推荐值：reports/automl_comparison.json
AUTOML_RESULTS_PATH = os.path.join(MODEL_EVAL_FIG_DIR, 'automl_comparison.json')
# DETECTED_MALICIOUS_PATHS_REPORT_PATH：恶意路径报告路径；用途：记录扫描发现的威胁；推荐值：reports/detected_malicious_paths.txt
DETECTED_MALICIOUS_PATHS_REPORT_PATH = os.path.join(MODEL_EVAL_FIG_DIR, 'detected_malicious_paths.txt')
# SCAN_PRINT_ONLY_MALICIOUS：仅打印恶意样本；用途：简化扫描输出；推荐值：True
SCAN_PRINT_ONLY_MALICIOUS = True
# SERVICE_CONCURRENCY_LIMIT：服务并发限制；用途：控制服务端最大连接数；推荐值：256
SERVICE_CONCURRENCY_LIMIT = 256
# SERVICE_PRINT_MALICIOUS_PATHS：服务打印恶意路径；用途：服务端实时输出威胁；推荐值：False
SERVICE_PRINT_MALICIOUS_PATHS = False
# SERVICE_EXIT_COMMAND：服务退出指令；用途：远程关闭服务的口令；推荐值：'exit'
SERVICE_EXIT_COMMAND = 'exit'
# SERVICE_ADMIN_TOKEN：服务管理令牌；用途：身份验证；推荐值：自定义字符串
SERVICE_ADMIN_TOKEN = ''
# SERVICE_CONTROL_LOCALHOSTS：允许控制的本地地址；用途：安全访问控制；推荐值：['127.0.0.1', '::1']
SERVICE_CONTROL_LOCALHOSTS = ['127.0.0.1', '::1']
# SERVICE_MAX_BATCH_SIZE：服务最大批处理大小；用途：限制单次扫描请求的文件数；推荐值：64
SERVICE_MAX_BATCH_SIZE = 64
# SERVICE_IPC_HOST：IPC 服务主机；用途：进程间通信监听地址；推荐值：'127.0.0.1'
SERVICE_IPC_HOST = '127.0.0.1'
# SERVICE_IPC_PORT：IPC 服务端口；用途：进程间通信监听端口；推荐值：8765
SERVICE_IPC_PORT = 8765
# SERVICE_IPC_MAX_MESSAGE_BYTES：IPC 最大消息字节数；用途：限制通信包大小；推荐值：1MB
SERVICE_IPC_MAX_MESSAGE_BYTES = 1024 * 1024
# SERVICE_IPC_READ_TIMEOUT_SEC：IPC 读取超时；用途：防止读取阻塞；推荐值：5
SERVICE_IPC_READ_TIMEOUT_SEC = 5
# SERVICE_IPC_WRITE_TIMEOUT_SEC：IPC 写入超时；用途：防止写入阻塞；推荐值：5
SERVICE_IPC_WRITE_TIMEOUT_SEC = 5
# SERVICE_IPC_REQUEST_TIMEOUT_SEC：IPC 请求总超时；用途：控制单次扫描最长时间；推荐值：120
SERVICE_IPC_REQUEST_TIMEOUT_SEC = 120
# SERVICE_IPC_MAX_REQUESTS_PER_CONNECTION：IPC 单连接最大请求数；用途：防止连接长久占用；推荐值：128
SERVICE_IPC_MAX_REQUESTS_PER_CONNECTION = 128

# 路由系统训练参数
# PACKED_SECTIONS_RATIO_THRESHOLD：加壳节比例阈值；用途：判定是否加壳；推荐值：0.4
PACKED_SECTIONS_RATIO_THRESHOLD = 0.4
# PACKER_KEYWORD_HITS_THRESHOLD：加壳关键词命中阈值；用途：判定是否加壳；推荐值：0
PACKER_KEYWORD_HITS_THRESHOLD = 0

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
# ENV_ALLOWED_SCAN_ROOT：限制扫描根路径；用途：安全访问控制；推荐值：SCANNER_ALLOWED_SCAN_ROOT
ENV_ALLOWED_SCAN_ROOT = 'SCANNER_ALLOWED_SCAN_ROOT'
# ENV_SERVICE_ADMIN_TOKEN：环境变量-服务管理令牌；用途：运行时覆盖服务令牌
ENV_SERVICE_ADMIN_TOKEN = 'SCANNER_SERVICE_ADMIN_TOKEN'
# ENV_SERVICE_EXIT_COMMAND：环境变量-服务退出指令；用途：运行时覆盖退出指令
ENV_SERVICE_EXIT_COMMAND = 'SCANNER_SERVICE_EXIT_COMMAND'
# ENV_SERVICE_IPC_HOST：环境变量-IPC 主机；用途：运行时覆盖 IPC 监听地址
ENV_SERVICE_IPC_HOST = 'SCANNER_SERVICE_IPC_HOST'
# ENV_SERVICE_IPC_PORT：环境变量-IPC 端口；用途：运行时覆盖 IPC 监听端口
ENV_SERVICE_IPC_PORT = 'SCANNER_SERVICE_IPC_PORT'
# ENV_SERVICE_IPC_MAX_MESSAGE_BYTES：环境变量-IPC 最大消息；用途：运行时覆盖最大消息大小
ENV_SERVICE_IPC_MAX_MESSAGE_BYTES = 'SCANNER_SERVICE_IPC_MAX_MESSAGE_BYTES'
# ENV_SERVICE_IPC_READ_TIMEOUT_SEC：环境变量-IPC 读取超时；用途：运行时覆盖读取超时
ENV_SERVICE_IPC_READ_TIMEOUT_SEC = 'SCANNER_SERVICE_IPC_READ_TIMEOUT_SEC'
# ENV_SERVICE_IPC_WRITE_TIMEOUT_SEC：环境变量-IPC 写入超时；用途：运行时覆盖写入超时
ENV_SERVICE_IPC_WRITE_TIMEOUT_SEC = 'SCANNER_SERVICE_IPC_WRITE_TIMEOUT_SEC'
# ENV_SERVICE_IPC_REQUEST_TIMEOUT_SEC：环境变量-IPC 请求超时；用途：运行时覆盖请求总超时
ENV_SERVICE_IPC_REQUEST_TIMEOUT_SEC = 'SCANNER_SERVICE_IPC_REQUEST_TIMEOUT_SEC'
# ENV_SERVICE_IPC_MAX_REQUESTS_PER_CONNECTION：环境变量-IPC 最大请求数；用途：运行时覆盖单连接最大请求
ENV_SERVICE_IPC_MAX_REQUESTS_PER_CONNECTION = 'SCANNER_SERVICE_IPC_MAX_REQUESTS_PER_CONNECTION'

# COLLECT_SOURCE_ROOT：采集源码根目录；用途：样本自动化采集的起始路径；推荐值：'C:\\'
COLLECT_SOURCE_ROOT = os.getenv(ENV_ALLOWED_SCAN_ROOT) or 'C:\\'
# COLLECT_ALLOWED_EXTENSIONS：采集允许的后缀；用途：过滤采集的文件类型；推荐值：['.exe', '.dll']
COLLECT_ALLOWED_EXTENSIONS = ['.exe', '.dll']
# COLLECT_MAX_FILE_SIZE：采集最大文件大小；用途：限制采集样本的大小；推荐值：SIZE_NORM_MAX
COLLECT_MAX_FILE_SIZE = SIZE_NORM_MAX



# 评估与训练细节参数：控制可视化与学习率策略
# PREDICTION_THRESHOLD：恶意预测阈值；用途：判定样本为恶意的概率下限；推荐值：0.95-0.99
PREDICTION_THRESHOLD = 0.98
# VIS_SAMPLE_SIZE：可视化采样数；用途：绘图子样本大小；推荐值：5000-20000（默认 10000）
VIS_SAMPLE_SIZE = 20000
# VIS_TSNE_PERPLEXITY：t-SNE 困惑度；用途：嵌入稳定性与结构；推荐值：30（10-50）
VIS_TSNE_PERPLEXITY = 30
# PCA_DIMENSION_FOR_CLUSTERING：聚类前 PCA 维度；用途：降噪与提速；推荐值：50（30-100）
PCA_DIMENSION_FOR_CLUSTERING = 50
# EVAL_HIST_BINS：评估直方图分箱；用途：概率分布可视化精度；推荐值：50（30-100）
EVAL_HIST_BINS = 100
# EVAL_TOP_FEATURE_COUNT：Top 特征数量；用途：训练后输出前 N 个重要特征；推荐值：20（10-50）
EVAL_TOP_FEATURE_COUNT = 50
# EVAL_FONT_FAMILY：评估图中文字体；用途：确保中文标签正常显示；推荐值：['SimHei','Microsoft YaHei']
EVAL_FONT_FAMILY = ['SimHei', 'Microsoft YaHei']
# DEFAULT_TEST_SIZE：测试集比例；用途：数据集划分的测试集占比；推荐值：0.2
DEFAULT_TEST_SIZE = 0.2
# DEFAULT_VAL_SIZE：验证集比例（占训练+验证集的比例）；用途：配合 DEFAULT_TEST_SIZE 实现 6:2:2 划分；推荐值：0.25
DEFAULT_VAL_SIZE = 0.25
# DEFAULT_RANDOM_STATE：随机种子；用途：保证训练/可视化结果可复现；推荐值：42
DEFAULT_RANDOM_STATE = 42
# COMMON_SECTIONS：常见节名列表；用途：节存在布尔特征与结构判断；推荐值：['.text','.data','.rdata','.reloc','.rsrc']
COMMON_SECTIONS = ['.text', '.data', '.rdata', '.reloc', '.rsrc']
# PACKER_SECTION_KEYWORDS：打包器关键词；用途：基于节名识别常见打包器；推荐值：按需扩展
PACKER_SECTION_KEYWORDS = ['upx', 'mpress', 'aspack', 'themida', 'petite', 'pecompact', 'fsg']
# SYSTEM_DLLS：系统 DLL 集合；用途：统计导入系统 DLL 的数量/占比；推荐值：常见基础系统库集合
SYSTEM_DLLS = {'kernel32', 'user32', 'gdi32', 'advapi32', 'shell32', 'ole32', 'comctl32'}
# ENTROPY_HIGH_THRESHOLD：高熵阈值；用途：计算高熵块占比；推荐值：0.8（0.7-0.9）
ENTROPY_HIGH_THRESHOLD = 0.8
# ENTROPY_LOW_THRESHOLD：低熵阈值；用途：计算低熵块占比；推荐值：0.2（0.1-0.3）
ENTROPY_LOW_THRESHOLD = 0.2
# LARGE_TRAILING_DATA_SIZE：大尾部数据大小阈值（字节）；用途：识别异常附加数据；推荐值：1MB（512KB-4MB）
LARGE_TRAILING_DATA_SIZE = 1024 * 1024
# SECTION_ENTROPY_MIN_SIZE：节熵计算的最小字节数；用途：避免小样本噪声
SECTION_ENTROPY_MIN_SIZE = 256
# OVERLAY_ENTROPY_MIN_SIZE：叠加区熵计算的最小字节数；用途：稳定估计
OVERLAY_ENTROPY_MIN_SIZE = 1024
# FP_WEIGHT_BASE：误报权重基数；用途：强化训练中错误分类良性样本的惩罚倍数；推荐值：5.0
FP_WEIGHT_BASE = 5.0
# FP_WEIGHT_GROWTH_PER_ITER：误报权重增长步长；用途：随迭代次数增加惩罚力度；推荐值：3.0
FP_WEIGHT_GROWTH_PER_ITER = 3.0
# FP_WEIGHT_MAX：最大误报权重；用途：限制惩罚上限；推荐值：100.0
FP_WEIGHT_MAX = 100.0
# FAMILY_THRESHOLD_PERCENTILE：家族阈值百分位；用途：确定家族置信阈；推荐值：95（90-99）
FAMILY_THRESHOLD_PERCENTILE = 90
# FAMILY_THRESHOLD_MULTIPLIER：家族阈值放大倍数；用途：放宽/收紧家族判定；推荐值：1.2（1.0-1.5）
FAMILY_THRESHOLD_MULTIPLIER = 1.0
# WARMUP_ROUNDS：学习率暖启动轮数；用途：前期小步长稳定训练；推荐值：100（50-200）
WARMUP_ROUNDS = 200
# WARMUP_START_LR：暖启动起始学习率；用途：初始学习率；推荐值：0.001（0.0005-0.005）
WARMUP_START_LR = 0.001
# WARMUP_TARGET_LR：暖启动目标学习率；用途：结束时学习率
WARMUP_TARGET_LR = 0.07

# API 分类关键词：用于特征提取中的语义分析
# API_CATEGORY_NETWORK：网络相关 API；用途：识别网络通信行为
API_CATEGORY_NETWORK = ['ws2_32', 'wininet', 'winhttp', 'internet', 'socket', 'connect', 'send', 'recv', 'http', 'url']
# API_CATEGORY_PROCESS：进程相关 API；用途：识别进程注入与控制行为
API_CATEGORY_PROCESS = ['createprocess', 'openprocess', 'terminateprocess', 'getprocaddress', 'loadlibrary', 'virtualallocex', 'writeprocessmemory']
# API_CATEGORY_FILESYSTEM：文件系统相关 API；用途：识别文件读写与遍历行为
API_CATEGORY_FILESYSTEM = ['createfile', 'readfile', 'writefile', 'deletefile', 'movefile', 'copyfile', 'findfirstfile', 'findnextfile', 'setfileattributes', 'getfileattributes', 'getfilesize']
# API_CATEGORY_REGISTRY：注册表相关 API；用途：识别持久化与配置修改行为
API_CATEGORY_REGISTRY = ['regopenkey', 'regsetvalue', 'regcreatekey', 'regdeletekey', 'regqueryvalue', 'regenumkey', 'regclosekey']

# 路由门控与专家模型配置
# GATING_ENABLED：启用路由门控；用途：开启混合专家模型 (MoE) 架构；推荐值：True
GATING_ENABLED = True
# GATING_MODE：路由模式；用途：'rule' (基于规则) 或 'model' (基于神经网络)；推荐值：'rule'
GATING_MODE = 'rule'
# GATING_MODEL_PATH：门控模型路径；用途：加载训练好的门控神经网络；推荐值：saved_models/gating_model.pth
GATING_MODEL_PATH = os.path.join(SAVED_MODEL_DIR, 'gating_model.pth')
# GATING_INPUT_DIM corresponds to the total feature dimension (Statistical + PE features)
# Statistical features: 49 (based on current logic with STAT_CHUNK_COUNT=10)
# PE features: 1500 (PE_FEATURE_VECTOR_DIM)
# Total: 1549
# GATING_INPUT_DIM：门控输入维度；用途：统计特征 + PE 特征总和；推荐值：1549
GATING_INPUT_DIM = 1549 
# GATING_HIDDEN_DIM：门控隐藏层维度；用途：控制门控网络复杂度；推荐值：256
GATING_HIDDEN_DIM = 256
# GATING_OUTPUT_DIM：门控输出维度；用途：0 为普通样本，1 为加壳样本；推荐值：2
GATING_OUTPUT_DIM = 2  # 0: Normal, 1: Packed
# GATING_THRESHOLD：门控判定阈值；用途：加壳判定的概率阈值；推荐值：0.5
GATING_THRESHOLD = 0.5
# GATING_LEARNING_RATE：门控训练学习率；用途：优化门控网络的步长；推荐值：0.001
GATING_LEARNING_RATE = 0.001
# GATING_EPOCHS：门控训练轮数；用途：训练门控网络的最大轮数；推荐值：20
GATING_EPOCHS = 20
# GATING_BATCH_SIZE：门控训练批大小；用途：梯度下降的样本块大小；推荐值：64
GATING_BATCH_SIZE = 64
# GATE_HIGH_ENTROPY_RATIO：高熵占比门限；用途：规则路由判定加壳的依据；推荐值：0.8
GATE_HIGH_ENTROPY_RATIO = 0.8
# GATE_PACKED_SECTIONS_RATIO：加壳节占比门限；用途：规则路由判定加壳的依据；推荐值：0.3
GATE_PACKED_SECTIONS_RATIO = 0.3
# GATE_PACKER_RATIO：打包器特征门限；用途：规则路由判定加壳的依据；推荐值：0.1
GATE_PACKER_RATIO = 0.1

# EXPERT_NORMAL_MODEL_PATH：普通样本专家模型；用途：处理未加壳样本的 LightGBM 模型
EXPERT_NORMAL_MODEL_PATH = os.path.join(SAVED_MODEL_DIR, 'lightgbm_model_normal.txt')
# EXPERT_PACKED_MODEL_PATH：加壳样本专家模型；用途：处理加壳样本的 LightGBM 模型
EXPERT_PACKED_MODEL_PATH = os.path.join(SAVED_MODEL_DIR, 'lightgbm_model_packed.txt')

# FEATURE_GATING_TOP_K：特征选择 K 值；用途：特征重要性实验中的保留数量；推荐值：1150
FEATURE_GATING_TOP_K = 1150
# FEATURE_GATING_REPORT_PATH：特征选择报告路径；用途：保存特征消融实验结果
FEATURE_GATING_REPORT_PATH = os.path.join(MODEL_EVAL_FIG_DIR, 'feature_gating_experiment.json')
# FEATURE_GATING_K_START：特征实验起始 K 值；用途：实验步进的起点
FEATURE_GATING_K_START = 50
# FEATURE_GATING_K_STEP：特征实验步进值；用途：每次增加的特征数量
FEATURE_GATING_K_STEP = 50

# AutoML 配置
# AUTOML_ENABLED：启用 AutoML；用途：开启自动超参数优化
AUTOML_ENABLED = True
# AUTOML_METHOD_DEFAULT：默认调优方法；用途：'optuna' 或 'hyperopt'
AUTOML_METHOD_DEFAULT = 'optuna'
# AUTOML_TRIALS_DEFAULT：默认试验次数；用途：超参数搜索的迭代次数
AUTOML_TRIALS_DEFAULT = 50
# AUTOML_CV_FOLDS_DEFAULT：默认交叉验证折数；用途：评估参数时的 CV 次数
AUTOML_CV_FOLDS_DEFAULT = 5
# AUTOML_METRIC_DEFAULT：默认评估指标；用途：优化目标 (roc_auc/accuracy/f1/precision/recall)
AUTOML_METRIC_DEFAULT = 'f1'
# AUTOML_ADDITIONAL_METRICS：额外监控指标；用途：在调优报告中显示的辅助指标
AUTOML_ADDITIONAL_METRICS = ['precision', 'recall', 'f1']
# AUTOML_TIMEOUT：调优超时时间；用途：限制搜索的最长时间（秒）
AUTOML_TIMEOUT = None
# AUTOML_LGBM_SCALE_POS_WEIGHT_MIN/MAX：类别权重搜索范围；用途：平衡正负样本比例
AUTOML_LGBM_SCALE_POS_WEIGHT_MIN = 0.3
AUTOML_LGBM_SCALE_POS_WEIGHT_MAX = 1.0
# AUTOML_LGBM_NUM_LEAVES_MIN/MAX：叶子数搜索范围
AUTOML_LGBM_NUM_LEAVES_MIN = 16
AUTOML_LGBM_NUM_LEAVES_MAX = 512
# AUTOML_LGBM_LEARNING_RATE_MIN/MAX：学习率搜索范围
AUTOML_LGBM_LEARNING_RATE_MIN = 0.005
AUTOML_LGBM_LEARNING_RATE_MAX = 0.2
# AUTOML_LGBM_FEATURE_FRACTION_MIN/MAX：特征采样比例搜索范围
AUTOML_LGBM_FEATURE_FRACTION_MIN = 0.6
AUTOML_LGBM_FEATURE_FRACTION_MAX = 1.0
# AUTOML_LGBM_BAGGING_FRACTION_MIN/MAX：样本采样比例搜索范围
AUTOML_LGBM_BAGGING_FRACTION_MIN = 0.6
AUTOML_LGBM_BAGGING_FRACTION_MAX = 1.0
# AUTOML_LGBM_MIN_DATA_IN_LEAF_MIN/MAX：叶子最小样本数搜索范围
AUTOML_LGBM_MIN_DATA_IN_LEAF_MIN = 10
AUTOML_LGBM_MIN_DATA_IN_LEAF_MAX = 100
# AUTOML_LGBM_MIN_GAIN_TO_SPLIT_MIN/MAX：最小分裂增益搜索范围
AUTOML_LGBM_MIN_GAIN_TO_SPLIT_MIN = 0.0
AUTOML_LGBM_MIN_GAIN_TO_SPLIT_MAX = 0.2
# AUTOML_LGBM_BAGGING_FREQ_MIN/MAX：Bagging 频率搜索范围
AUTOML_LGBM_BAGGING_FREQ_MIN = 1
AUTOML_LGBM_BAGGING_FREQ_MAX = 20
