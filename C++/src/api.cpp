#include "kvd/api.h"

#include "config.h"
#include "malware_scanner.h"

#include <cstdlib>
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
