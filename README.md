# 神枢 - Axon V2

基于机器学习的恶意软件检测和分类系统，使用LightGBM和深度学习技术实现恶意软件的检测和家族分类。

## 项目概述

神枢 - Axon V2 是一个基于机器学习的恶意软件检测系统，能够：

1. 检测文件是否为恶意软件（二分类）
2. 对恶意软件进行家族分类（多分类）

该系统使用多种技术组合来实现高精度的恶意软件检测和分类，包括：
- 基于字节序列的统计特征提取
- PE文件结构分析
- LightGBM机器学习模型
- HDBSCAN聚类算法进行恶意软件家族发现

## 目录结构

```
KoloVirusDetector_ML_V2-main/
├── config/                    # 配置模块，集中默认值与帮助文本
├── data/                      # 数据相关模块与处理输出
│   └── dataset.py            # 数据集装载与规范化
├── features/                  # 通用特征模块
│   └── statistics.py         # 统计特征提取
├── training/                  # 训练与评估流水线
│   ├── data_loader.py
│   ├── train_lightgbm.py
│   ├── evaluate.py
│   ├── feature_io.py
│   ├── incremental.py
│   └── model_io.py
├── models/                    # 家族分类器等模型
│   └── family_classifier.py
├── utils/                     # 工具模块
│   └── logging_utils.py
├── saved_models/              # 模型保存目录
├── hdbscan_cluster_results/   # 聚类结果与分类器
├── reports/                   # 评估与聚类可视化输出
├── feature_extractor_enhanced.py
├── main.py                    # 项目入口
├── pretrain.py                # 预训练入口（调用 training/*）
├── finetune.py                # 聚类与家族训练入口
├── scanner.py                 # 扫描器
├── scanner_service.py         # FastAPI 服务
└── extracted_features.pkl     # 特征文件（可选）
```

## 功能模块

- `features/statistics.py`：统计特征提取
- `data/dataset.py`：数据集加载与规范化
- `training/*`：数据装载、训练、评估、特征IO、增量训练、模型读写
- `models/family_classifier.py`：家族分类器
- `scanner.py`：文件与目录扫描
- `scanner_service.py`：FastAPI 扫描服务
- `config/config.py`：集中配置与默认值

## 安装依赖

```bash
pip install numpy pandas scikit-learn lightgbm matplotlib seaborn tqdm pefile torch fast-hdbscan
```

## 使用方法

### 1. 提取数据集&预训练二分类模型
```bat
python main.py extract&python main.py pretrain
```

### 2. 家族聚类与分类器训练
```bat
python main.py finetune --plot-pca
```

### 3. 扫描文件或目录
```bat
python main.py scan --file-path e:\path\to\sample.exe
python main.py scan --dir-path e:\path\to\dump_dir --recursive
```

### 4. 启动扫描服务
```bat
python main.py serve --port 8000
```

默认路径与参数集中在 `config/config.py`，也可通过环境变量或命令行覆盖。

常用端点：
- `POST /scan/file`：接受 JSON `{ "file_path": "C:\\sample.exe" }`
- `POST /scan/upload`：上传文件内容（multipart/form-data）进行扫描
- `GET /health`：服务健康检查

通过以下环境变量可以覆盖默认配置：

| 变量名 | 作用 |
| --- | --- |
| `SCANNER_LIGHTGBM_MODEL_PATH` | LightGBM 模型文件路径 |
| `SCANNER_FAMILY_CLASSIFIER_PATH` | 家族分类器模型路径 |
| `SCANNER_CACHE_PATH` | 扫描缓存文件路径 |
| `SCANNER_MAX_FILE_SIZE` | 最大读取文件大小（字节） |
| `SCANNER_SERVICE_PORT` | 扫描服务端口 |
| `SCANNER_ALLOWED_SCAN_ROOT` | 限制允许扫描的根目录（路径前缀约束） |

## 模型性能

评估图保存在 `reports/model_evaluation.png`，包括：
- 准确率指标
- 混淆矩阵
- 预测概率分布图

## 数据集

项目使用以下数据集进行训练和测试：
- 正常软件样本：`benign_samples/` 目录
- 恶意软件样本：`malicious_samples/` 目录

处理后的数据存储在 `data/processed_lightgbm/` 目录中。

## 聚类结果

聚类结果保存在 `hdbscan_cluster_results/`，可视化图表保存在 `reports/`：
- 聚类标签与统计
- `reports/hdbscan_clustering_visualization.png`
- `reports/hdbscan_clustering_visualization_pca.png`
- 家族名称映射与分类器

## 项目特点

1. **高准确性**：结合多种特征和先进的机器学习算法
2. **高效性**：优化的特征提取和预测过程
3. **可扩展性**：模块化设计，易于添加新功能
4. **可视化**：提供丰富的结果可视化功能

## 贡献

欢迎提交Issue和Pull Request来改进项目。

## 许可证

本项目采用Apache-2.0许可证，详情请见LICENSE文件。





# Shenshu - Axon V2

A machine learning-based malware detection and classification system that leverages LightGBM and deep learning techniques to detect malware and classify it into families.

## Project Overview

**Shenshu - Axon V2** is a machine learning-powered malware detection system capable of:

1. Detecting whether a file is malicious (binary classification)  
2. Classifying malware into specific families (multi-class classification)

The system combines multiple advanced techniques to achieve high-precision malware detection and classification, including:
- Statistical feature extraction from byte sequences
- PE file structure analysis
- LightGBM machine learning models
- HDBSCAN clustering algorithm for malware family discovery

## Directory Structure

```
KoloVirusDetectorML/
├── benign_samples/           # Benign software samples
├── malicious_samples/        # Malware samples
├── data/                     # Processed datasets
├── saved_models/             # Saved model files
├── hdbscan_cluster_results/  # HDBSCAN clustering results
├── feature_extractor_enhanced.py  # Feature extraction module
├── pretrain.py               # Pre-training module
├── finetune.py               # Fine-tuning module
├── scanner.py                # Scanner module
├── extracted_features.pkl    # Extracted features file
└── model_evaluation.png      # Model evaluation result chart
```

## Functional Modules

### 1. Feature Extraction (feature_extractor_enhanced.py)

This module extracts features from executable files, including:
- Statistical features from byte sequences
- File entropy calculation
- PE file structural information
- File attribute metadata

### 2. Pre-training (pretrain.py)

Implements binary malware detection:
- Trains a binary classification model using LightGBM
- Distinguishes between benign and malicious software
- Provides model evaluation and validation capabilities

### 3. Fine-tuning (finetune.py)

Implements malware family classification:
- Uses the HDBSCAN clustering algorithm to discover malware families
- Performs fine-grained classification of malware samples
- Visualizes clustering results

### 4. Scanner (scanner.py)

Provides a command-line interface for scanning files:
- Single file scanning
- Batch scanning of directories
- Outputs detailed scan reports

## Installation Dependencies

```bash
pip install numpy pandas scikit-learn lightgbm matplotlib seaborn tqdm pefile torch fast-hdbscan
```

## Usage Instructions

### 1. Train the Binary Classification Model

```bash
python pretrain.py --data_dir data/processed_lightgbm --metadata_file data/metadata.json
```

### 2. Perform Malware Family Clustering

```bash
python finetune.py --features_file extracted_features.pkl --output_dir hdbscan_cluster_results
```

### 3. Scan Files

```bash
# Scan a single file
python scanner.py --model saved_models/lightgbm_model.txt --file /path/to/file.exe

# Scan an entire directory
python scanner.py --model saved_models/lightgbm_model.txt --dir /path/to/directory
```

## Model Performance

Refer to `model_evaluation.png` for model evaluation results, which include:
- Accuracy metrics
- Confusion matrix
- ROC curve

## Dataset

The project uses the following datasets for training and testing:
- Benign software samples: located in `benign_samples/`
- Malware samples: located in `malicious_samples/`

Processed data is stored in the `data/processed_lightgbm/` directory.

## Clustering Results

HDBSCAN clustering results are saved in the `hdbscan_cluster_results/` directory, including:
- Cluster label files
- Clustering visualization charts
- Malware family name mappings

## Key Features

1. **High Accuracy**: Combines multiple features and advanced machine learning algorithms  
2. **Efficiency**: Optimized feature extraction and prediction pipeline  
3. **Scalability**: Modular design that supports easy extension and integration of new features  
4. **Visualization**: Rich visualization tools for analysis and reporting

## Contributions

We welcome contributions via Issues and Pull Requests to improve the project.

## License

This project is licensed under the Apache-2.0 License. See the LICENSE file for details.
