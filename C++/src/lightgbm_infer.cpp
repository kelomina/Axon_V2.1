#include "lightgbm_infer.h"

#include <LightGBM/c_api.h>

namespace kvd {

LightGbmModel::LightGbmModel(LightGbmModel&& other) noexcept {
  handle_ = other.handle_;
  num_iterations_ = other.num_iterations_;
  other.handle_ = nullptr;
  other.num_iterations_ = 0;
}

LightGbmModel& LightGbmModel::operator=(LightGbmModel&& other) noexcept {
  if (this == &other) return *this;
  if (handle_) {
    LGBM_BoosterFree(handle_);
  }
  handle_ = other.handle_;
  num_iterations_ = other.num_iterations_;
  other.handle_ = nullptr;
  other.num_iterations_ = 0;
  return *this;
}

LightGbmModel::~LightGbmModel() {
  if (handle_) {
    LGBM_BoosterFree(handle_);
    handle_ = nullptr;
  }
}

std::optional<LightGbmModel> LightGbmModel::load_from_file(const std::string& path) {
  void* booster = nullptr;
  int num_iterations = 0;
  if (LGBM_BoosterCreateFromModelfile(path.c_str(), &num_iterations, &booster) != 0) {
    return std::nullopt;
  }
  LightGbmModel m;
  m.handle_ = booster;
  m.num_iterations_ = num_iterations;
  return m;
}

std::optional<float> LightGbmModel::predict_one(const std::vector<float>& features) const {
  if (!handle_) {
    return std::nullopt;
  }
  if (features.empty()) {
    return std::nullopt;
  }
  double out = 0.0;
  int64_t out_len = 0;
  if (LGBM_BoosterPredictForMat(
          handle_,
          features.data(),
          C_API_DTYPE_FLOAT32,
          1,
          static_cast<int>(features.size()),
          1,
          C_API_PREDICT_NORMAL,
          0,
          -1,
          "",
          &out_len,
          &out) != 0) {
    return std::nullopt;
  }
  if (out_len < 1) {
    return std::nullopt;
  }
  return static_cast<float>(out);
}

bool LightGbmModel::ok() const {
  return handle_ != nullptr;
}

}
