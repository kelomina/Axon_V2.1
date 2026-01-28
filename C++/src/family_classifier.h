#pragma once

#include "kvd_internal.h"

#include <optional>
#include <string>
#include <vector>

namespace kvd {

class FamilyClassifier {
 public:
  static std::optional<FamilyClassifier> load_from_json(const std::string& path);

  std::optional<ScanResultFamily> predict(const std::vector<float>& features) const;

  bool ok() const;

 private:
  std::vector<int> cluster_ids_;
  std::vector<std::vector<float>> centroids_;
  std::vector<float> thresholds_;
  std::vector<std::string> family_names_;
  std::vector<float> scaler_mean_;
  std::vector<float> scaler_scale_;
};

}
