# 神枢 - Axon V2

基于机器学习的恶意软件检测与家族分类系统，核心由 LightGBM 二分类模型与基于 HDBSCAN 的家族识别组成，可通过命令行与 IPC 扫描服务进行扫描与集成。

## 项目概述

- 主要功能
  - 恶意软件检测（二分类）：判断文件是否为恶意样本
  - 家族识别（聚类 + 分类器）：对恶意样本进行家族归属或标记为未知家族
  - 本地扫描与目录批量扫描，支持缓存与结果导出（JSON/CSV）
  - 提供 IPC 接口的扫描服务，便于系统集成

- 新增特性：路由门控与专家模型系统
  - 引入动态路由机制（Gating Model），基于样本特征（如加壳情况）自动将请求分发至最适合的专家模型。
  - 包含两个专家模型：
    - Normal Expert：专注于处理未加壳的常规恶意样本。
    - Packed Expert：专注于处理加壳或高熵的恶意样本。
  - 提升了对加壳样本的检测能力，同时保持对常规样本的高精度。

- 技术栈与核心依赖
  - Python、NumPy、Pandas、scikit-learn、LightGBM（运行时门控为规则/NumPy，无需 PyTorch）
  - fast-hdbscan（多核优化的 HDBSCAN 实现，用于家族聚类；更适合低维欧氏空间输入）
- pefile（PE 结构读取）、matplotlib / seaborn（可视化）、tqdm（进度条）

## 环境要求

- 操作系统：Windows 10/11 或 Linux（x86_64）
- Python：建议 3.10 及以上
- 必备软件：`pip`、可选 `virtualenv`/`venv`
- 依赖：参见 `requirements.txt`
  - 家族聚类默认使用 `fast-hdbscan`；如遇聚类阶段异常退出，可在 `config/config.py` 调整 `FAST_HDBSCAN_PCA_DIMENSION`（建议 10-20）与 `HDBSCAN_FLOAT32_FOR_CLUSTERING` 以降低内存与稳定运行

## 安装指南

- 创建并激活虚拟环境（Windows PowerShell）
  ```powershell
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1
  ```
- 安装依赖
  ```bash
  pip install -r requirements.txt
  ```
- 可选：设置环境变量（仅当前会话）
  ```powershell
  $env:SCANNER_ALLOWED_SCAN_ROOT="E:\\SampleRoot"
  $env:SCANNER_SERVICE_IPC_PORT="8765"
  $env:SCANNER_MAX_FILE_SIZE="65536"
  ```
- 目录准备
  - 良性样本：`benign_samples/`
  - 恶意样本：`malicious_samples/`
  - 处理输出：`data/processed_lightgbm/`
  - 模型：`saved_models/lightgbm_model.txt`
  - 家族分类器：`hdbscan_cluster_results/family_classifier.pkl`

## 使用说明

- 提取原始样本并生成处理数据
  ```bash
  python main.py extract --output-dir data/processed_lightgbm --label-inference directory
  ```

- 预训练二分类模型（可保存/复用特征）
  ```bash
  python main.py pretrain --save-features
  ```

- 一键训练集成（特征→预训练→路由训练→评估→聚类→保存）
  ```bash
  python main.py train-all
  ```
  - 自动执行特征提取或复用 `extracted_features.pkl`
  - 训练 LightGBM 基础模型并保存至 `saved_models/lightgbm_model.txt`
  - 内置 AutoML 超参调优并应用于本次训练（默认使用 `optuna`，结果保存至 `reports/automl_comparison.json`）
  - 训练路由系统（门控模型与两套专家模型），保存至 `saved_models/`
  - 生成评估可视化，包括混淆矩阵与 ROC AUC 曲线：
    - `reports/model_evaluation.png`
    - `reports/model_auc_curve.png`
    - `reports/routing_confusion_matrix.png`
    - `reports/routing_roc_auc.png`
  - 进行家族聚类并训练分类器，保存至 `hdbscan_cluster_results/family_classifier.pkl`

- 家族聚类与分类器训练（生成 `family_classifier.pkl` 与可视化图）
  ```bash
  python main.py finetune --plot-pca --min-cluster-size 2 --min-samples 1 --min-family-size 1 --treat-noise-as-family
  ```

- 扫描文件或目录并导出结果
  ```bash
  python main.py scan --file-path E:\\path\\to\\sample.exe
  python main.py scan --dir-path E:\\path\\to\\dump_dir --recursive
  ```
  - 扫描过程中实时在控制台输出被识别为恶意样本的文件路径
  - 同时在 `reports/detected_malicious_paths.txt` 保存所有被识别为恶意的文件路径

- 启动扫描服务（IPC）
  ```bash
  python main.py serve
  ```
  - 监听地址与端口由 `SCANNER_SERVICE_IPC_HOST`（默认 `127.0.0.1`）与 `SCANNER_SERVICE_IPC_PORT`（默认 `8765`）控制

- 自动调优与交叉测试（AutoML）
  ```bash
  python main.py auto-tune --method optuna --trials 50 --cv 5 --metric f1 --use-existing-features
  python main.py auto-tune --method hyperopt --trials 50 --cv 5 --metric precision --use-existing-features
  ```
  - 输出结果保存至 `reports/automl_comparison.json`
  - 支持参数：`--method`、`--trials`、`--cv`、`--metric`（支持 `f1`、`precision`、`recall`、`roc_auc`、`accuracy`）、`--fast-dev-run`、`--max-file-size`
  - 增强特性：
    - 支持自动优化类别权重 `scale_pos_weight` 以平衡样本。
    - 调优过程中自动监控并记录多个辅助指标（Precision, Recall, F1）。

- IPC 接口（本机进程间通信，TCP + JSON）
  - 启用方式
    - 监听地址与端口：`SCANNER_SERVICE_IPC_HOST`（默认 `127.0.0.1`）、`SCANNER_SERVICE_IPC_PORT`（默认 `8765`）
  - 传输与帧格式
    - 传输层：TCP（通常用于同机回环地址）
    - 帧格式：`4字节大端长度` + `UTF-8 JSON`
  - 请求格式（JSON）
    - `version`：协议版本，当前为 `1`
    - `id`：请求ID（字符串，可选）
    - `type`：消息类型（字符串）
    - `payload`：消息体（对象）
    - `timeout_ms`：单次请求处理超时（可选，毫秒；不能大于服务端默认上限）
  - 响应格式（JSON）
    - `version`：协议版本
    - `id`：回显请求ID
    - `ok`：是否成功
    - 成功时：`payload` 为返回数据
    - 失败时：`error` 为 `{code, message, details?}`
  - 消息类型
    - `health`
      - 请求：`{"type":"health","payload":{}}`
      - 响应：`{"ok":true,"payload":{"status":"ok"}}`
    - `scan_file`
      - 请求：`{"type":"scan_file","payload":{"file_path":"C:\\sample.exe"}}`
      - 响应：返回扫描结果，包含 `virus_family`
    - `scan_batch`
      - 请求：`{"type":"scan_batch","payload":{"file_paths":["C:\\a.exe","C:\\b.exe"]}}`
      - 响应：返回结果数组，包含 `virus_family`
    - `control`
      - 请求：`{"type":"control","payload":{"command":"exit","token":"..."}}`
      - 行为：触发服务优雅退出
  - Python 调用示例
  ```python
  import json, socket, struct

  def ipc_call(host, port, msg):
      data = json.dumps(msg, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
      with socket.create_connection((host, port), timeout=3) as s:
          s.sendall(struct.pack(">I", len(data)) + data)
          header = s.recv(4)
          size = struct.unpack(">I", header)[0]
          buf = b""
          while len(buf) < size:
              buf += s.recv(size - len(buf))
      return json.loads(buf.decode("utf-8"))

  resp = ipc_call("127.0.0.1", 8765, {"version": 1, "id": "1", "type": "scan_file", "payload": {"file_path": r"C:\Windows\System32\notepad.exe"}})
  print(resp)
  ```
  - 性能指标与限制
    - 单次消息体大小上限：`config.SERVICE_IPC_MAX_MESSAGE_BYTES`（默认 1MB），超过将返回错误并断开连接
    - 批量扫描数量上限：`config.SERVICE_MAX_BATCH_SIZE`
    - 并发限制：`config.SERVICE_CONCURRENCY_LIMIT`
    - 默认超时：读取/写入超时由 `config.SERVICE_IPC_READ_TIMEOUT_SEC` / `config.SERVICE_IPC_WRITE_TIMEOUT_SEC` 控制；处理超时由 `config.SERVICE_IPC_REQUEST_TIMEOUT_SEC` 控制
    - 单连接最大请求数：`config.SERVICE_IPC_MAX_REQUESTS_PER_CONNECTION`（默认 128），超过将自动断开
    - 仅建议绑定到回环地址（默认 `127.0.0.1`），避免暴露本机文件扫描能力到外网

## 路由门控系统训练

新的路由门控系统需要单独的训练流程来生成 Gating Model 和两个 Expert Models。

- 运行路由系统训练脚本：
  ```bash
  python main.py train-routing
  # 或者复用已提取的特征（默认行为）
  python main.py train-routing --use-existing-features
  # 或者强制提取并保存特征
  python main.py train-routing --save-features
  # 一键执行完整流程（特征提取 -> 超参调优 -> 路由训练 -> 家族聚类）
  python main.py train-all
  # 一键执行流程并跳过超参调优（直接使用 config.py 中的参数）
  python main.py train-all --skip-tuning
  ```
  该脚本会自动：
  1. 加载或提取特征（支持 `--use-existing-features` 和 `--save-features` 参数）。
  2. 根据启发式规则（加壳特征）生成路由标签。
  3. 应用规则门控（运行时无需 PyTorch，基于高熵/打包器特征）。
  4. 分割数据集并分别训练 Normal Expert 和 Packed Expert (LightGBM)。
  5. 保存所有模型至 `saved_models/` 目录。

- 配置：
  可以在 `config/config.py` 中调整 `GATING_*` 相关参数与规则阈值；当前默认使用 `GATING_MODE='rule'`，门控采用启发式规则并跳过门控模型训练。

### 门控机制验证

- 用途：基于高熵比例、打包器特征等信号，对样本进行路由判定（packed/normal），用于验证门控策略有效性
- 运行
  ```bash
  python -m validation.gating_validator --file-path E:\\path\\to\\sample.exe
  python -m validation.gating_validator --dir-path E:\\path\\to\\dump_dir --recursive
  ```
- 可调参数（`config/config.py`）
  - `GATE_HIGH_ENTROPY_RATIO`、`GATE_PACKED_SECTIONS_RATIO`、`GATE_PACKER_RATIO`
  - `GATING_MODE`、`GATING_ENABLED`
  - 专家模型路径：`EXPERT_NORMAL_MODEL_PATH`、`EXPERT_PACKED_MODEL_PATH`

### 特征门控交叉实验

- 用途：在训练阶段对不重要的特征进行屏蔽，进行对照实验，评估对准确率与误报的影响
- 运行
  ```bash
  python -m validation.feature_gating_experiment --use-existing-features --num-boost-round 1000
  ```
- 实验内容
  - 基线：使用全部特征训练与评估（不使用门控）
  - Top-K 门控序列：自动从 Top-50 到全特征，每次步进 20，逐一训练与评估
- 输出
  - 结果保存 `reports/feature_gating_experiment.json`
- 可调参数（`config/config.py`）
  - `FEATURE_GATING_K_START`（默认 50）
  - `FEATURE_GATING_K_STEP`（默认 20）
  - `FEATURE_GATING_REPORT_PATH`

### 学习率暖启动

- 默认启用学习率暖启动策略
- 参数（`config/config.py`）
  - `WARMUP_ROUNDS`（默认 200）
  - `WARMUP_START_LR`（默认 0.001）
  - `WARMUP_TARGET_LR`（默认 0.07）

- 常用参数（CLI，对应 `main.py:28-74`）
  - `--max-file-size`：最大读取字节数（`config/config.py:27-31`）
  - `--file-extensions`、`--label-inference`（`main.py:34-37,66-71`）
  - `--num-boost-round`、`--incremental-rounds`、`--incremental-early-stopping`（`main.py:38-41`）
  - `--min-cluster-size`、`--min-samples`、`--min-family-size`（`main.py:49-51`）
  - `--treat-noise-as-family`：将聚类噪声点视为独立家族（当 `min-family-size` 为 1 时，每个噪声点分配唯一家族 ID）

- 环境变量覆盖（服务与扫描）
  - `SCANNER_LIGHTGBM_MODEL_PATH`、`SCANNER_FAMILY_CLASSIFIER_PATH`（`config/config.py:142-151`）
  - `SCANNER_CACHE_PATH`、`SCANNER_MAX_FILE_SIZE`、`SCANNER_ALLOWED_SCAN_ROOT`（`config/config.py:146-153`）
  - `SCANNER_SERVICE_ADMIN_TOKEN`、`SCANNER_SERVICE_EXIT_COMMAND`（`config/config.py`）
  - `SCANNER_SERVICE_IPC_HOST`、`SCANNER_SERVICE_IPC_PORT` 等（`config/config.py`）
  - 服务端并发与打印行为：`SERVICE_CONCURRENCY_LIMIT`、`SERVICE_PRINT_MALICIOUS_PATHS`（`config/config.py:150-151`）

## 开发指南

- 代码结构与职责
  - `main.py`：统一 CLI 入口与子命令调度（`main.py:23-74`）
  - 特征提取：`features/extractor_in_memory.py`、`features/statistics.py`（`statistics.py:4`）
  - 数据集加载：`data/dataset.py`（`data/dataset.py:4-49`）
  - 训练与评估：`training/train_lightgbm.py`（`training/train_lightgbm.py:15`）、`training/evaluate.py`（`training/evaluate.py:8`）
  - 增量训练：`training/incremental.py`（`training/incremental.py:6`）
  - 家族分类器：`models/family_classifier.py`（`models/family_classifier.py:7-69`）
  - 扫描器与服务：`scanner.py`（`scanner.py:47-78,184-253`）、`scanner_service.py`（`scanner_service.py:97-134`）
  - 配置集中：`config/config.py`（所有自定义参数与帮助文本）

- 贡献规范
  - 新增库前先评估必要性与维护成本；如确需引入，请在 `requirements.txt` 与本 README 同步说明
  - 所有可调参数统一写入 `config/config.py`，避免散落各处
  - 保持模块化与现有代码风格；提交前请自测与同步文档

- 测试方法（unittest）
  - 测试代码放置于 `tests/`，文件名以 `test_*.py` 命名
  - 运行：
    ```bash
    python -m unittest discover -s tests -p "test_*.py"
    ```
  - 建议覆盖：
    - 特征提取边界（空文件、非 PE、超长截断）
    - 检测阈值与误报处理（`config.PREDICTION_THRESHOLD`、增量训练权重）
    - 服务端路径校验与上传扫描（`scanner_service.py:97-134`）

- 数据集划分 (Train/Val/Test)
  - 系统采用 **6:2:2** 的比例进行数据集划分。
  - 划分逻辑通过 `DEFAULT_TEST_SIZE` (0.2) 和 `DEFAULT_VAL_SIZE` (0.25) 共同控制：
    1. 首先从全量数据中划分出 20% 作为独立测试集。
    2. 从剩余的 80% 数据中划分出 25%（即总量的 20%）作为验证集。
    3. 剩余 60% 数据用于模型训练。
  - 相关参数均在 `config/config.py` 中集中管理。

## 部署说明

- 准备工件
  - `saved_models/lightgbm_model.txt` 或同名 `lightgbm_model.txt.gz`
  - `hdbscan_cluster_results/family_classifier.pkl`

- 环境变量与安全
  - 设置 `SCANNER_ALLOWED_SCAN_ROOT` 限制可扫描目录，避免越权访问（`scanner.py:29-38`）
  - 根据需要调整 `SCANNER_MAX_FILE_SIZE` 以平衡性能与准确性

- 启动服务
  ```bash
  python main.py serve
  ```

- 性能优化建议
  - 训练阶段：调整 `LIGHTGBM_NUM_THREADS_MAX`、`DEFAULT_LIGHTGBM_NUM_LEAVES`、`DEFAULT_LIGHTGBM_LEARNING_RATE`、`DEFAULT_EARLY_STOPPING_ROUNDS`（默认 200）以及 `DEFAULT_INCREMENTAL_EARLY_STOPPING`。当前默认：`num_leaves=281`、`learning_rate=0.0054273608259950085`（`config/config.py`，`training/train_lightgbm.py`）
  - 推理阶段：合理设置 `DEFAULT_MAX_FILE_SIZE`，使用缓存文件 `scan_cache.json`
  - 可视化与聚类：降维到 `PCA_DIMENSION_FOR_CLUSTERING`，控制采样量 `VIS_SAMPLE_SIZE`

## 维护信息

- 已知问题与解决方案
  - 未安装 `fast-hdbscan` 时，家族聚类直接退出（请安装依赖）`finetune.py:19-23`
  - Windows 下聚类阶段可能出现无 Python 报错的异常退出（例如退出码 -1073741571）；可在 `config/config.py` 将 `FAMILY_CLUSTERING_BACKEND` 设为 `minibatch_kmeans`，或保持 `auto` 由程序自动切换
  - 模型/分类器路径不存在导致启动失败（请检查路径）`scanner_service.py:59-65`
  - 非 PE 文件会被跳过（属预期行为）`scanner.py:211-213`
  - 特征维度不一致会自动填充/截断并记录汇总（由 `PE_DIM_SUMMARY_DATASET`、`PE_DIM_SUMMARY_INCREMENTAL`、`PE_DIM_SUMMARY_RAW` 指定的 JSON 文件输出统计）`training/data_loader.py:61-73,99-113`
  - 训练与评估统一使用带列名的特征 DataFrame，列名规范为 `feature_{i}` 且顺序一致，避免出现 sklearn 提示 “X does not have valid feature names”，相关处理见 `training/automl.py:25-36,54-60`
  - 扫描完成后立即释放 PE 解析句柄与上传临时文件，避免 Windows 上文件锁定问题（`features/extractor_in_memory.py:117-684`、`scanner_service.py:179-205`）

- 故障排查指南
  - 确认依赖与 Python 版本；运行 `pip list` 检查 `fast-hdbscan`、`lightgbm`
  - 检查模型与分类器文件是否存在于预期路径
  - 使用 `--max-file-size` 与 `SCANNER_ALLOWED_SCAN_ROOT` 缩小问题范围
  - 查看 `reports/`、`scan_results/` 与日志输出定位问题

- 联系方式
  - 通过仓库 Issue 反馈问题与建议

## 许可证

本项目采用 Apache-2.0 许可证，详见 `LICENSE`。
