#include "config.h"

#include <cstdlib>
#include <filesystem>

namespace kvd {

static constexpr const char* ENV_LIGHTGBM_MODEL_PATH = "SCANNER_LIGHTGBM_MODEL_PATH";
static constexpr const char* ENV_LIGHTGBM_MODEL_NORMAL_PATH = "SCANNER_LIGHTGBM_MODEL_NORMAL_PATH";
static constexpr const char* ENV_LIGHTGBM_MODEL_PACKED_PATH = "SCANNER_LIGHTGBM_MODEL_PACKED_PATH";
static constexpr const char* ENV_FAMILY_CLASSIFIER_PATH = "SCANNER_FAMILY_CLASSIFIER_PATH";
static constexpr const char* ENV_ALLOWED_SCAN_ROOT = "SCANNER_ALLOWED_SCAN_ROOT";
static constexpr const char* ENV_MAX_FILE_SIZE = "SCANNER_MAX_FILE_SIZE";
static constexpr const char* ENV_PREDICTION_THRESHOLD = "SCANNER_PREDICTION_THRESHOLD";

std::optional<std::string> getenv_string(const char* name) {
  const char* v = std::getenv(name);
  if (!v || !*v) {
    return std::nullopt;
  }
  return std::string(v);
}

static bool path_exists(const std::string& path) {
  std::error_code ec;
  return !path.empty() && std::filesystem::exists(std::filesystem::path(path), ec) && !ec;
}

static std::optional<std::size_t> parse_size_t(const std::string& s) {
  if (s.empty()) return std::nullopt;
  char* end = nullptr;
  unsigned long long v = std::strtoull(s.c_str(), &end, 10);
  if (!end || *end != '\0') return std::nullopt;
  return static_cast<std::size_t>(v);
}

static std::optional<float> parse_float(const std::string& s) {
  if (s.empty()) return std::nullopt;
  char* end = nullptr;
  float v = std::strtof(s.c_str(), &end);
  if (!end || *end != '\0') return std::nullopt;
  return v;
}

Config config_from_api(
    const char* model_path,
    const char* model_normal_path,
    const char* model_packed_path,
    const char* family_classifier_json_path,
    const char* allowed_scan_root,
    unsigned int max_file_size,
    float prediction_threshold) {
  Config cfg;
  if (model_path && *model_path) cfg.model_path = model_path;
  if (model_normal_path) cfg.model_normal_path = model_normal_path;
  if (model_packed_path) cfg.model_packed_path = model_packed_path;
  if (family_classifier_json_path && *family_classifier_json_path) cfg.family_classifier_json_path = family_classifier_json_path;
  if (allowed_scan_root && *allowed_scan_root) cfg.allowed_scan_root = std::string(allowed_scan_root);
  if (max_file_size > 0) cfg.max_file_size = static_cast<std::size_t>(max_file_size);
  if (prediction_threshold > 0.0f && prediction_threshold <= 1.0f) cfg.prediction_threshold = prediction_threshold;

  if (cfg.model_path.empty()) {
    auto env_model = getenv_string(ENV_LIGHTGBM_MODEL_PATH);
    if (env_model) cfg.model_path = *env_model;
  }

  if (cfg.model_normal_path.empty()) {
    auto env_model_normal = getenv_string(ENV_LIGHTGBM_MODEL_NORMAL_PATH);
    if (env_model_normal) cfg.model_normal_path = *env_model_normal;
  }

  if (cfg.model_packed_path.empty()) {
    auto env_model_packed = getenv_string(ENV_LIGHTGBM_MODEL_PACKED_PATH);
    if (env_model_packed) cfg.model_packed_path = *env_model_packed;
  }

  if (cfg.family_classifier_json_path.empty()) {
    auto env_fc = getenv_string(ENV_FAMILY_CLASSIFIER_PATH);
    if (env_fc) cfg.family_classifier_json_path = *env_fc;
  }

  if (!cfg.allowed_scan_root) {
    auto env_root = getenv_string(ENV_ALLOWED_SCAN_ROOT);
    if (env_root) cfg.allowed_scan_root = *env_root;
  }

  if (max_file_size == 0) {
    auto env_mfs = getenv_string(ENV_MAX_FILE_SIZE);
    if (env_mfs) {
      auto v = parse_size_t(*env_mfs);
      if (v && *v > 0) cfg.max_file_size = *v;
    }
  }

  if (!(prediction_threshold > 0.0f && prediction_threshold <= 1.0f)) {
    auto env_th = getenv_string(ENV_PREDICTION_THRESHOLD);
    if (env_th) {
      auto v = parse_float(*env_th);
      if (v && *v > 0.0f && *v <= 1.0f) cfg.prediction_threshold = *v;
    }
  }

  if (cfg.model_path.empty()) {
    std::string default_model = (std::filesystem::path("saved_models") / "lightgbm_model.txt").string();
    if (path_exists(default_model)) {
      cfg.model_path = default_model;
    }
  }

  if (cfg.model_normal_path.empty()) {
    std::string default_model_normal = (std::filesystem::path("saved_models") / "lightgbm_model_normal.txt").string();
    if (path_exists(default_model_normal)) {
      cfg.model_normal_path = default_model_normal;
    }
  }

  if (cfg.model_packed_path.empty()) {
    std::string default_model_packed = (std::filesystem::path("saved_models") / "lightgbm_model_packed.txt").string();
    if (path_exists(default_model_packed)) {
      cfg.model_packed_path = default_model_packed;
    }
  }

  if (cfg.family_classifier_json_path.empty()) {
    std::string default_fc = (std::filesystem::path("hdbscan_cluster_results") / "family_classifier.json").string();
    if (path_exists(default_fc)) {
      cfg.family_classifier_json_path = default_fc;
    }
  }

  return cfg;
}

}
