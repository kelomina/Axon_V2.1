#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace kvd {

struct Config {
  std::string model_path;
  std::string model_normal_path;
  std::string model_packed_path;
  std::string family_classifier_json_path;
  std::optional<std::string> allowed_scan_root;
  std::size_t max_file_size = 64 * 1024;
  float prediction_threshold = 0.98f;
};

struct ScanResultFamily {
  std::string family_name;
  int cluster_id = -1;
  bool is_new_family = false;
};

struct ScanResult {
  bool is_malware = false;
  float confidence = 0.0f;
  std::optional<ScanResultFamily> family;
  std::string error;
};

}
