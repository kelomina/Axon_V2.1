#include "kvd/api.h"

#include "config.h"
#include "malware_scanner.h"

#include <cstdlib>
#include <filesystem>
#include <new>

#include <nlohmann/json.hpp>

struct kvd_handle {
  kvd::MalwareScanner scanner;
};

static nlohmann::json to_json(const kvd::ScanResult& r) {
  nlohmann::json j;
  j["is_malware"] = r.is_malware;
  j["confidence"] = r.confidence;
  if (!r.error.empty()) {
    j["error"] = r.error;
  }
  if (r.family) {
    nlohmann::json f;
    f["family_name"] = r.family->family_name;
    f["cluster_id"] = r.family->cluster_id;
    f["is_new_family"] = r.family->is_new_family;
    j["malware_family"] = f;
  }
  return j;
}

kvd_handle* kvd_create(const kvd_config* config) {
  if (!config) {
    return nullptr;
  }
  kvd::Config cfg = kvd::config_from_api(
      config->model_path,
      config->model_normal_path,
      config->model_packed_path,
      config->family_classifier_json_path,
      config->allowed_scan_root,
      config->max_file_size,
      config->prediction_threshold);

  auto scanner_opt = kvd::MalwareScanner::create(cfg);
  if (!scanner_opt) {
    return nullptr;
  }
  kvd_handle* h = new (std::nothrow) kvd_handle{std::move(*scanner_opt)};
  return h;
}

void kvd_destroy(kvd_handle* handle) {
  delete handle;
}

static int write_json_out(const nlohmann::json& j, char** out_json, size_t* out_len) {
  if (!out_json || !out_len) {
    return -1;
  }
  std::string s = j.dump();
  char* buf = static_cast<char*>(std::malloc(s.size() + 1));
  if (!buf) {
    return -2;
  }
  std::memcpy(buf, s.data(), s.size());
  buf[s.size()] = '\0';
  *out_json = buf;
  *out_len = s.size();
  return 0;
}

static int write_string_out(const std::string& s, char** out_json, size_t* out_len) {
  if (!out_json || !out_len) {
    return -1;
  }
  char* buf = static_cast<char*>(std::malloc(s.size() + 1));
  if (!buf) {
    return -2;
  }
  std::memcpy(buf, s.data(), s.size());
  buf[s.size()] = '\0';
  *out_json = buf;
  *out_len = s.size();
  return 0;
}

static bool path_exists(const std::string& path) {
  std::error_code ec;
  return !path.empty() && std::filesystem::exists(std::filesystem::path(path), ec) && !ec;
}

int kvd_scan_path(kvd_handle* handle, const char* path, char** out_json, size_t* out_len) {
  if (!handle || !path) {
    return -1;
  }
  kvd::ScanResult r = handle->scanner.scan_path(path);
  return write_json_out(to_json(r), out_json, out_len);
}

int kvd_scan_bytes(kvd_handle* handle, const unsigned char* bytes, size_t len, char** out_json, size_t* out_len) {
  if (!handle || !bytes) {
    return -1;
  }
  std::vector<std::uint8_t> v(bytes, bytes + len);
  kvd::ScanResult r = handle->scanner.scan_bytes(v);
  return write_json_out(to_json(r), out_json, out_len);
}

void kvd_free(char* p) {
  std::free(p);
}

int kvd_validate_models(const kvd_config* config, char** out_error, size_t* out_len) {
  if (!config) {
    return KVD_MODEL_ERR_INVALID_ARGUMENT;
  }
  if ((out_error && !out_len) || (!out_error && out_len)) {
    return KVD_MODEL_ERR_INVALID_ARGUMENT;
  }

  kvd::Config cfg = kvd::config_from_api(
      config->model_path,
      config->model_normal_path,
      config->model_packed_path,
      config->family_classifier_json_path,
      config->allowed_scan_root,
      config->max_file_size,
      config->prediction_threshold);

  auto write_error = [&](const std::string& code) -> int {
    if (!out_error && !out_len) {
      return 0;
    }
    int rc = write_string_out(code, out_error, out_len);
    return rc == 0 ? 0 : KVD_MODEL_ERR_OOM;
  };

  if (cfg.model_path.empty()) {
    int rc = write_error("model_main_missing");
    return rc == 0 ? KVD_MODEL_ERR_MAIN_MISSING : rc;
  }
  if (!path_exists(cfg.model_path)) {
    int rc = write_error("model_main_missing");
    return rc == 0 ? KVD_MODEL_ERR_MAIN_MISSING : rc;
  }
  if (!kvd::LightGbmModel::load_from_file(cfg.model_path)) {
    int rc = write_error("model_main_invalid");
    return rc == 0 ? KVD_MODEL_ERR_MAIN_INVALID : rc;
  }

  bool has_normal = !cfg.model_normal_path.empty();
  bool has_packed = !cfg.model_packed_path.empty();
  if (has_normal != has_packed) {
    int rc = write_error("model_route_incomplete");
    return rc == 0 ? KVD_MODEL_ERR_ROUTE_INCOMPLETE : rc;
  }

  if (has_normal) {
    if (!path_exists(cfg.model_normal_path)) {
      int rc = write_error("model_normal_missing");
      return rc == 0 ? KVD_MODEL_ERR_NORMAL_MISSING : rc;
    }
    if (!kvd::LightGbmModel::load_from_file(cfg.model_normal_path)) {
      int rc = write_error("model_normal_invalid");
      return rc == 0 ? KVD_MODEL_ERR_NORMAL_INVALID : rc;
    }
  }

  if (has_packed) {
    if (!path_exists(cfg.model_packed_path)) {
      int rc = write_error("model_packed_missing");
      return rc == 0 ? KVD_MODEL_ERR_PACKED_MISSING : rc;
    }
    if (!kvd::LightGbmModel::load_from_file(cfg.model_packed_path)) {
      int rc = write_error("model_packed_invalid");
      return rc == 0 ? KVD_MODEL_ERR_PACKED_INVALID : rc;
    }
  }

  if (!cfg.family_classifier_json_path.empty()) {
    if (!path_exists(cfg.family_classifier_json_path)) {
      int rc = write_error("family_classifier_missing");
      return rc == 0 ? KVD_MODEL_ERR_FAMILY_MISSING : rc;
    }
    if (!kvd::FamilyClassifier::load_from_json(cfg.family_classifier_json_path)) {
      int rc = write_error("family_classifier_invalid");
      return rc == 0 ? KVD_MODEL_ERR_FAMILY_INVALID : rc;
    }
  }

  int rc = write_error("ok");
  return rc == 0 ? KVD_MODEL_OK : rc;
}
