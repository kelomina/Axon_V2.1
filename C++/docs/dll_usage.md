# kvd.dll 使用说明（EXE 调用 DLL）

本项目对外提供一个 C ABI 的 DLL：`kvd.dll`，用于扫描文件并输出 JSON 结果。

你可以用两种方式在 EXE 中调用它：

- **方式 A：静态链接（隐式加载）**：编译期链接 `kvd.lib`（import library），运行时由系统自动加载 `kvd.dll`。
- **方式 B：静态调用（显式加载）**：EXE 运行时 `LoadLibrary/GetProcAddress` 调用 `kvd.dll`，不需要 `kvd.lib`。

“静态链接 DLL”在 Windows 语境下一般指方式 A（import lib），它不是把 DLL 代码打进 EXE，而是让 EXE 运行时自动加载 DLL。

---

## 分发文件清单

最小分发包含：

- `kvd.dll`
- `kvd.lib`（仅方式 A 需要；方式 B 不需要）
- 头文件：`C++/include/kvd/api.h`
- 运行时依赖 DLL（随构建输出一起拷贝）：`LIEF.dll`、`lib_lightgbm.dll`、`fmt.dll`、`spdlog.dll` 等
- 模型文件（默认路径）：`saved_models/lightgbm_model.txt`（可选 `lightgbm_model_normal.txt`/`lightgbm_model_packed.txt`）
- 家族分类器（可选）：`hdbscan_cluster_results/family_classifier.json`

---

## API 概览

头文件位置：[api.h](file:///e:/Project/python/KoloVirusDetector_ML_V2-main/C%2B%2B/include/kvd/api.h)

### 数据结构：kvd_config

```c
typedef struct kvd_config {
  const char* model_path;
  const char* model_normal_path;
  const char* model_packed_path;
  const char* family_classifier_json_path;
  const char* allowed_scan_root;
  unsigned int max_file_size;
  float prediction_threshold;
} kvd_config;
```

- `model_path`：主 LightGBM 模型路径（留空则走默认/环境变量）。
- `model_normal_path` / `model_packed_path`：路由专家模型（可选；两者都存在才会启用路由）。
- `family_classifier_json_path`：家族分类 JSON（可选；仅在判定为恶意时尝试输出 family）。
- `allowed_scan_root`：允许扫描的根目录（可选；不在该目录下的路径会拒绝）。
- `max_file_size`：最多读取字节数（0 表示走默认/环境变量/内置默认）。
- `prediction_threshold`：恶意阈值 (0..1]（0 表示走默认/环境变量/内置默认）。

环境变量支持（优先级低于 `kvd_config` 显式传参）：

- `SCANNER_LIGHTGBM_MODEL_PATH`
- `SCANNER_LIGHTGBM_MODEL_NORMAL_PATH`
- `SCANNER_LIGHTGBM_MODEL_PACKED_PATH`
- `SCANNER_FAMILY_CLASSIFIER_PATH`
- `SCANNER_ALLOWED_SCAN_ROOT`
- `SCANNER_MAX_FILE_SIZE`
- `SCANNER_PREDICTION_THRESHOLD`

### 函数：kvd_create

```c
kvd_handle* kvd_create(const kvd_config* config);
```

- 用途：创建扫描器句柄（加载模型、可选加载家族分类器）。
- 参数：
  - `config`：不能为空；可以传零初始化的 `kvd_config cfg = {0};`
- 返回：
  - 成功：非空 `kvd_handle*`
  - 失败：返回 `NULL`

### 函数：kvd_destroy

```c
void kvd_destroy(kvd_handle* handle);
```

- 用途：销毁句柄，释放模型与相关资源。
- 约束：允许传 `NULL`（等同于 no-op）是常见做法，但当前实现直接 `delete`，建议调用前自行判空。

### 函数：kvd_scan_path

```c
int kvd_scan_path(kvd_handle* handle, const char* path, char** out_json, size_t* out_len);
```

- 用途：扫描一个文件路径，输出 JSON 字符串。
- 参数：
  - `handle`：由 `kvd_create` 创建
  - `path`：待扫描文件路径
  - `out_json/out_len`：输出 JSON 与长度（字节数）
- 返回值：
  - `0`：成功（即使扫描内部失败，也会尽量返回 JSON，内部错误在 JSON 的 `error` 字段中）
  - `<0`：调用层错误（例如参数非法、内存分配失败等）
- 内存释放：
  - 成功时 `*out_json` 由 DLL 内 `malloc` 分配，必须用 `kvd_free(*out_json)` 释放（不要用 `free/delete`，避免 CRT 不一致）。

典型 JSON 字段：

- `is_malware`：bool
- `confidence`：float
- `error`：string（可选；例如 `invalid_path_or_read_failed` / `pe_features_failed` / `predict_failed`）
- `malware_family`：object（可选，且仅恶意时尝试给出）

### 函数：kvd_scan_bytes

```c
int kvd_scan_bytes(kvd_handle* handle, const unsigned char* bytes, size_t len, char** out_json, size_t* out_len);
```

- 用途：扫描内存缓冲区（当前实现会返回 JSON，`error` 为 `scan_bytes_not_implemented`）。
- 返回值/释放规则：同 `kvd_scan_path`。

### 函数：kvd_free

```c
void kvd_free(char* p);
```

- 用途：释放 `kvd_scan_path/kvd_scan_bytes` 返回的 `out_json`。
- 约束：必须用它释放，避免跨 CRT 释放导致崩溃。

### 函数：kvd_validate_models

```c
int kvd_validate_models(const kvd_config* config, char** out_error, size_t* out_len);
```

- 用途：校验模型文件与家族分类器文件是否缺失或损坏。
- 参数：
  - `config`：不能为空；可以传零初始化的 `kvd_config cfg = {0};`
  - `out_error/out_len`：输出错误码字符串与长度；不需要时可同时传 `NULL`
- 返回值：
  - `0`：校验通过
  - `<0`：校验失败，错误含义见下表
- 内存释放：
  - `out_error` 由 DLL 内 `malloc` 分配，必须用 `kvd_free(*out_error)` 释放

---

## 错误对照表（kvd_validate_models）

| 返回值 | out_error 字符串 | 说明 |
| --- | --- | --- |
| 0 | ok | 校验通过 |
| -1 | n/a | 参数非法（config 为空或 out_error/out_len 仅给其一） |
| -10 | model_main_missing | 主模型文件缺失 |
| -11 | model_main_invalid | 主模型文件损坏或无法加载 |
| -12 | model_route_incomplete | 路由模型不完整（只提供 normal 或 packed 其中之一） |
| -13 | model_normal_missing | normal 模型文件缺失 |
| -14 | model_normal_invalid | normal 模型文件损坏或无法加载 |
| -15 | model_packed_missing | packed 模型文件缺失 |
| -16 | model_packed_invalid | packed 模型文件损坏或无法加载 |
| -17 | family_classifier_missing | 家族分类器文件缺失 |
| -18 | family_classifier_invalid | 家族分类器文件损坏或无法加载 |
| -100 | n/a | 输出错误信息内存分配失败 |

---

## 方式 A：静态链接（import lib，隐式加载）

### 1) 工程依赖

- 编译期：
  - 头文件：`include/kvd/api.h`
  - 库：`kvd.lib`
- 运行期（与 exe 同目录或 PATH 可找到）：
  - `kvd.dll` + 依赖 DLL

### 2) 最小示例

```cpp
#include "kvd/api.h"
#include <iostream>

int main(int argc, char** argv) {
  if (argc < 2) return 2;

  kvd_config cfg{};
  kvd_handle* h = kvd_create(&cfg);
  if (!h) return 1;

  char* out_json = nullptr;
  size_t out_len = 0;
  int rc = kvd_scan_path(h, argv[1], &out_json, &out_len);
  if (rc == 0 && out_json) {
    std::cout.write(out_json, (std::streamsize)out_len);
    std::cout << "\n";
    kvd_free(out_json);
  }

  kvd_destroy(h);
  return rc == 0 ? 0 : 1;
}
```

---

## 方式 B：静态调用（LoadLibrary/GetProcAddress，显式加载）

特点：EXE 不需要链接 `kvd.lib`，只需要头文件定义 `kvd_config` 等结构体，然后在运行时解析函数指针。

参考实现：`C++/examples/kvd_scan_loader.cpp`。

注意：出于稳定性考虑，示例程序默认**不调用** `FreeLibrary(kvd.dll)`，让进程退出时由系统统一卸载（部分第三方依赖在手动卸载场景下可能触发崩溃）。

### 1) 最小示例（核心片段）

```cpp
HMODULE mod = LoadLibraryA("kvd.dll");
auto kvd_create_p = (kvd_handle* (KVD_CALL*)(const kvd_config*))GetProcAddress(mod, "kvd_create");
auto kvd_scan_path_p = (int (KVD_CALL*)(kvd_handle*, const char*, char**, size_t*))GetProcAddress(mod, "kvd_scan_path");
auto kvd_free_p = (void (KVD_CALL*)(char*))GetProcAddress(mod, "kvd_free");
auto kvd_destroy_p = (void (KVD_CALL*)(kvd_handle*))GetProcAddress(mod, "kvd_destroy");
```

### 2) 为什么需要 `.def`

为了让 `GetProcAddress(mod, \"kvd_create\")` 的名字稳定一致，Windows 下我们用 `kvd.def` 明确导出名，避免编译器/平台差异导致的符号装饰问题。

---

## 示例程序

- 链接模式示例：`kvd_scan`（见 `C++/examples/scan_path.cpp`）\n
- 显式加载示例：`kvd_scan_loader`（见 `C++/examples/kvd_scan_loader.cpp`）
