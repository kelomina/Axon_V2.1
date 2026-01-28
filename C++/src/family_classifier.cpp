#include "family_classifier.h"

#include <cmath>
#include <fstream>

#include <nlohmann/json.hpp>

namespace kvd {

static float l2_distance_sq(const std::vector<float>& a, const std::vector<float>& b) {
  if (a.size() != b.size()) {
    return std::numeric_limits<float>::infinity();
  }
  float s = 0.0f;
  for (std::size_t i = 0; i < a.size(); ++i) {
    float d = a[i] - b[i];
    s += d * d;
  }
  return s;
}

static std::vector<float> apply_scaler(const std::vector<float>& x, const std::vector<float>& mean, const std::vector<float>& scale) {
  if (x.size() != mean.size() || x.size() != scale.size()) {
    return {};
  }
  std::vector<float> y;
  y.resize(x.size());
  for (std::size_t i = 0; i < x.size(); ++i) {
    float denom = scale[i];
    if (denom == 0.0f) denom = 1.0f;
    y[i] = (x[i] - mean[i]) / denom;
  }
  return y;
}

std::optional<FamilyClassifier> FamilyClassifier::load_from_json(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    return std::nullopt;
  }
  nlohmann::json j;
  try {
    f >> j;
  } catch (...) {
    return std::nullopt;
  }

  FamilyClassifier fc;
  try {
    if (j.contains("cluster_ids")) {
      fc.cluster_ids_ = j.at("cluster_ids").get<std::vector<int>>();
    }
    fc.centroids_ = j.at("centroids").get<std::vector<std::vector<float>>>();
    fc.thresholds_ = j.at("thresholds").get<std::vector<float>>();
    fc.family_names_ = j.at("family_names").get<std::vector<std::string>>();
    fc.scaler_mean_ = j.at("scaler_mean").get<std::vector<float>>();
    fc.scaler_scale_ = j.at("scaler_scale").get<std::vector<float>>();
  } catch (...) {
    return std::nullopt;
  }
  if (fc.centroids_.empty() || fc.centroids_.size() != fc.thresholds_.size() || fc.centroids_.size() != fc.family_names_.size()) {
    return std::nullopt;
  }
  if (!fc.cluster_ids_.empty() && fc.cluster_ids_.size() != fc.centroids_.size()) {
    return std::nullopt;
  }
  if (fc.cluster_ids_.empty()) {
    fc.cluster_ids_.resize(fc.centroids_.size());
    for (std::size_t i = 0; i < fc.cluster_ids_.size(); ++i) {
      fc.cluster_ids_[i] = static_cast<int>(i);
    }
  }
  return fc;
}

std::optional<ScanResultFamily> FamilyClassifier::predict(const std::vector<float>& features) const {
  if (centroids_.empty() || thresholds_.empty() || family_names_.empty()) {
    return std::nullopt;
  }
  std::vector<float> x = features;
  if (!scaler_mean_.empty() && scaler_mean_.size() == x.size()) {
    x = apply_scaler(x, scaler_mean_, scaler_scale_);
    if (x.empty()) {
      return std::nullopt;
    }
  }

  std::size_t best_i = 0;
  float best_d = std::numeric_limits<float>::infinity();
  for (std::size_t i = 0; i < centroids_.size(); ++i) {
    float d = l2_distance_sq(x, centroids_[i]);
    if (d < best_d) {
      best_d = d;
      best_i = i;
    }
  }
  float dist = std::sqrt(best_d);
  float thr = thresholds_[best_i];
  ScanResultFamily r;
  r.cluster_id = cluster_ids_.empty() ? static_cast<int>(best_i) : cluster_ids_[best_i];
  r.family_name = family_names_[best_i];
  r.is_new_family = dist > thr;
  return r;
}

bool FamilyClassifier::ok() const {
  return !centroids_.empty();
}

}
