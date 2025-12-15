# 神枢 - Axon V2

基于机器学习的恶意软件检测与家族分类系统，核心由 LightGBM 二分类模型与基于 HDBSCAN 的家族识别组成，可通过命令行与 FastAPI 服务进行扫描与集成。

## 项目概述

- 主要功能
  - 恶意软件检测（二分类）：判断文件是否为恶意样本
  - 家族识别（聚类 + 分类器）：对恶意样本进行家族归属或标记为未知家族
  - 本地扫描与目录批量扫描，支持缓存与结果导出（JSON/CSV）
  - 提供 HTTP 接口的扫描服务，便于系统集成

- 新增特性：路由门控与专家模型系统
  - 引入动态路由机制（Gating Model），基于样本特征（如加壳情况）自动将请求分发至最适合的专家模型。
  - 包含两个专家模型：
    - Normal Expert：专注于处理未加壳的常规恶意样本。
    - Packed Expert：专注于处理加壳或高熵的恶意样本。
  - 提升了对加壳样本的检测能力，同时保持对常规样本的高精度。

- 技术栈与核心依赖
  - Python、NumPy、Pandas、scikit-learn、LightGBM、PyTorch (新增)
  - FastAPI + Uvicorn（服务端）
  - fast-hdbscan（多核优化的 HDBSCAN 实现，用于家族聚类，finetune 必备，`finetune.py:21`）
  - pefile（PE 结构读取）、matplotlib / seaborn（可视化）、tqdm（进度条）

## 环境要求

- 操作系统：Windows 10/11 或 Linux（x86_64）
- Python：建议 3.10 及以上
- 必备软件：`pip`、可选 `virtualenv`/`venv`
- 依赖：参见 `requirements.txt`
  - 需确保安装 `fast-hdbscan`，否则家族聚类流程会退出（`finetune.py:21`）

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
  $env:SCANNER_SERVICE_PORT="8000"
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
  python main.py finetune --plot-pca --min-cluster-size 30 --min-family-size 10
  ```

- 扫描文件或目录并导出结果
  ```bash
  python main.py scan --file-path E:\\path\\to\\sample.exe
  python main.py scan --dir-path E:\\path\\to\\dump_dir --recursive
  ```
  - 扫描过程中实时在控制台输出被识别为恶意样本的文件路径
  - 同时在 `reports/detected_malicious_paths.txt` 保存所有被识别为恶意的文件路径

- 启动扫描服务（FastAPI）
  ```bash
  python main.py serve --port 8000
  ```

- 自动调优与交叉测试（AutoML）
  ```bash
  python main.py auto-tune --method optuna --trials 50 --cv 5 --metric roc_auc --use-existing-features
  python main.py auto-tune --method hyperopt --trials 50 --cv 5 --metric accuracy --use-existing-features
  ```
  - 输出结果保存至 `reports/automl_comparison.json`
  - 支持参数：`--method`、`--trials`、`--cv`、`--metric`、`--fast-dev-run`、`--max-file-size`

- HTTP 接口
  - `POST /scan/file`：`{"file_path": "C:\\sample.exe"}`
  - `POST /scan/upload`：上传文件进行扫描（multipart/form-data）
  - `GET /health`：服务健康检查
  - 示例
  ```bash
  curl -X POST "http://127.0.0.1:8000/scan/file" -H "Content-Type: application/json" -d '{"file_path":"C:\\Windows\\System32\\notepad.exe"}'
  ```

## 路由门控系统训练

新的路由门控系统需要单独的训练流程来生成 Gating Model 和两个 Expert Models。

- 运行路由系统训练脚本：
  ```bash
  python main.py train-routing
  # 或者复用已提取的特征（默认行为）
  python main.py train-routing --use-existing-features
  # 或者强制提取并保存特征
  python main.py train-routing --save-features
  ```
  该脚本会自动：
  1. 加载或提取特征（支持 `--use-existing-features` 和 `--save-features` 参数）。
  2. 根据启发式规则（加壳特征）生成路由标签。
  3. 训练 Gating Model（PyTorch MLP/Transformer）。
  4. 分割数据集并分别训练 Normal Expert 和 Packed Expert (LightGBM)。
  5. 保存所有模型至 `saved_models/` 目录。

- 配置：
  可以在 `config/config.py` 中调整 `GATING_*` 相关参数，如网络结构、阈值等。

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

- 环境变量覆盖（服务与扫描）
  - `SCANNER_LIGHTGBM_MODEL_PATH`、`SCANNER_FAMILY_CLASSIFIER_PATH`（`config/config.py:142-151`）
  - `SCANNER_CACHE_PATH`、`SCANNER_MAX_FILE_SIZE`、`SCANNER_SERVICE_PORT`、`SCANNER_ALLOWED_SCAN_ROOT`（`config/config.py:146-153`）

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

## 部署说明

- 准备工件
  - `saved_models/lightgbm_model.txt` 或同名 `lightgbm_model.txt.gz`
  - `hdbscan_cluster_results/family_classifier.pkl`

- 环境变量与安全
  - 设置 `SCANNER_ALLOWED_SCAN_ROOT` 限制可扫描目录，避免越权访问（`scanner.py:29-38`）
  - 根据需要调整 `SCANNER_MAX_FILE_SIZE` 以平衡性能与准确性

- 启动服务
  ```bash
  python main.py serve --port 8000
  ```

- 性能优化建议
- 训练阶段：调整 `LIGHTGBM_NUM_THREADS_MAX`、`num_leaves`、`learning_rate`、`DEFAULT_EARLY_STOPPING_ROUNDS`（默认 200）。当前默认：`num_leaves=281`、`learning_rate=0.0054273608259950085`（`config/config.py`，`training/train_lightgbm.py`）
  - 推理阶段：合理设置 `DEFAULT_MAX_FILE_SIZE`，使用缓存文件 `scan_cache.json`
  - 可视化与聚类：降维到 `PCA_DIMENSION_FOR_CLUSTERING`，控制采样量 `VIS_SAMPLE_SIZE`

## 维护信息

- 已知问题与解决方案
  - 未安装 `fast-hdbscan` 时，家族聚类直接退出（请安装依赖）`finetune.py:19-23`
  - 模型/分类器路径不存在导致启动失败（请检查路径）`scanner_service.py:59-65`
  - 非 PE 文件会被跳过（属预期行为）`scanner.py:211-213`
  - 特征维度不一致会自动填充/截断并记录汇总（数据/增量/扫描路径均有统计文件输出）`training/data_loader.py:61-73,99-113`
  - 训练与评估统一使用带列名的特征 DataFrame，列名规范为 `feature_{i}` 且顺序一致，避免出现 sklearn 提示 “X does not have valid feature names”，相关处理见 `training/automl.py:25-36,54-60`

- 故障排查指南
  - 确认依赖与 Python 版本；运行 `pip list` 检查 `fast-hdbscan`、`lightgbm`
  - 检查模型与分类器文件是否存在于预期路径
  - 使用 `--max-file-size` 与 `SCANNER_ALLOWED_SCAN_ROOT` 缩小问题范围
  - 查看 `reports/`、`scan_results/` 与日志输出定位问题

- 联系方式
  - 通过仓库 Issue 反馈问题与建议

## 许可证

本项目采用 Apache-2.0 许可证，详见 `LICENSE`。
