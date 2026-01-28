#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

namespace kvd {

class LightGbmModel {
 public:
  LightGbmModel() = default;
  LightGbmModel(const LightGbmModel&) = delete;
  LightGbmModel& operator=(const LightGbmModel&) = delete;
  LightGbmModel(LightGbmModel&&) noexcept;
  LightGbmModel& operator=(LightGbmModel&&) noexcept;
  ~LightGbmModel();

  static std::optional<LightGbmModel> load_from_file(const std::string& path);

  std::optional<float> predict_one(const std::vector<float>& features) const;

  bool ok() const;

 private:
  void* handle_ = nullptr;
  int num_iterations_ = 0;
};

}
