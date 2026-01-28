#include "features_file_attributes.h"

#include "util_filesystem.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <numeric>
#include <vector>

namespace kvd {

static double entropy_u8(const std::vector<std::uint8_t>& data, std::size_t start, std::size_t end) {
  std::size_t n = end > start ? (end - start) : 0;
  if (n == 0) return 0.0;
  std::array<std::uint32_t, 256> counts{};
  counts.fill(0);
  for (std::size_t i = start; i < end; ++i) {
    counts[data[i]]++;
  }
  double inv = 1.0 / static_cast<double>(n);
  double s = 0.0;
  for (std::size_t i = 0; i < 256; ++i) {
    if (counts[i] == 0) continue;
    double p = static_cast<double>(counts[i]) * inv;
    s += p * std::log2(p);
  }
  return (-s) / 8.0;
}

static double std_f64(const std::vector<double>& v) {
  if (v.empty()) return 0.0;
  double m = 0.0;
  for (double x : v) m += x;
  m /= static_cast<double>(v.size());
  double acc = 0.0;
  for (double x : v) {
    double d = x - m;
    acc += d * d;
  }
  return std::sqrt(acc / static_cast<double>(v.size()));
}

static double percentile_f64(std::vector<double> v, double q) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  double pos = (static_cast<double>(v.size()) - 1.0) * q;
  std::size_t lo = static_cast<std::size_t>(std::floor(pos));
  std::size_t hi = static_cast<std::size_t>(std::ceil(pos));
  double w = pos - static_cast<double>(lo);
  double a = v[lo];
  double b = v[hi];
  return a + (b - a) * w;
}

std::unordered_map<std::string, float> extract_file_attributes(
    const std::string& path,
    const std::optional<std::string>& allowed_root) {
  std::unordered_map<std::string, float> features;

  auto valid = validate_path(path, allowed_root);
  if (!valid) {
    return features;
  }

  std::error_code ec;
  std::uintmax_t file_size = std::filesystem::file_size(std::filesystem::path(*valid), ec);
  if (ec) {
    return features;
  }

  features["size"] = static_cast<float>(file_size);
  features["log_size"] = static_cast<float>(std::log(static_cast<double>(file_size) + 1.0));

  static constexpr std::size_t ENTROPY_SAMPLE_SIZE = 10240;
  static constexpr std::size_t ENTROPY_BLOCK_SIZE = 2048;

  std::vector<std::uint8_t> sample;
  if (!read_file_bytes_seek(*valid, 0, ENTROPY_SAMPLE_SIZE, sample)) {
    sample.clear();
  }

  double overall_entropy = entropy_u8(sample, 0, sample.size());
  std::vector<double> block_entropies;
  if (!sample.empty()) {
    for (std::size_t start = 0; start < sample.size(); start += ENTROPY_BLOCK_SIZE) {
      std::size_t end = std::min(sample.size(), start + ENTROPY_BLOCK_SIZE);
      if (end > start) {
        block_entropies.push_back(entropy_u8(sample, start, end));
      }
    }
  }

  double min_entropy = overall_entropy;
  double max_entropy = overall_entropy;
  double entropy_std = 0.0;
  if (!block_entropies.empty()) {
    auto mm = std::minmax_element(block_entropies.begin(), block_entropies.end());
    min_entropy = *mm.first;
    max_entropy = *mm.second;
    entropy_std = std_f64(block_entropies);
  }

  features["file_entropy_avg"] = static_cast<float>(overall_entropy);
  features["file_entropy_min"] = static_cast<float>(min_entropy);
  features["file_entropy_max"] = static_cast<float>(max_entropy);
  features["file_entropy_range"] = static_cast<float>(max_entropy - min_entropy);
  features["file_entropy_std"] = static_cast<float>(entropy_std);

  if (!block_entropies.empty()) {
    features["file_entropy_q25"] = static_cast<float>(percentile_f64(block_entropies, 0.25));
    features["file_entropy_q75"] = static_cast<float>(percentile_f64(block_entropies, 0.75));
    features["file_entropy_median"] = static_cast<float>(percentile_f64(block_entropies, 0.5));

    int high = 0;
    int low = 0;
    for (double e : block_entropies) {
      if (e > 0.8) high++;
      if (e < 0.2) low++;
    }
    features["high_entropy_ratio"] = static_cast<float>(static_cast<double>(high) / static_cast<double>(block_entropies.size()));
    features["low_entropy_ratio"] = static_cast<float>(static_cast<double>(low) / static_cast<double>(block_entropies.size()));

    if (block_entropies.size() > 1) {
      std::vector<double> diffs;
      diffs.reserve(block_entropies.size() - 1);
      for (std::size_t i = 1; i < block_entropies.size(); ++i) {
        diffs.push_back(block_entropies[i] - block_entropies[i - 1]);
      }
      double mean_abs = 0.0;
      for (double d : diffs) mean_abs += std::abs(d);
      mean_abs /= static_cast<double>(diffs.size());
      features["entropy_change_rate"] = static_cast<float>(mean_abs);
      features["entropy_change_std"] = static_cast<float>(std_f64(diffs));
    } else {
      features["entropy_change_rate"] = 0.0f;
      features["entropy_change_std"] = 0.0f;
    }
  } else {
    features["file_entropy_q25"] = 0.0f;
    features["file_entropy_q75"] = 0.0f;
    features["file_entropy_median"] = 0.0f;
    features["high_entropy_ratio"] = 0.0f;
    features["low_entropy_ratio"] = 0.0f;
    features["entropy_change_rate"] = 0.0f;
    features["entropy_change_std"] = 0.0f;
  }

  if (!sample.empty()) {
    int zero = 0;
    int printable = 0;
    for (std::uint8_t b : sample) {
      if (b == 0) zero++;
      if (b >= 32 && b <= 126) printable++;
    }
    features["zero_byte_ratio"] = static_cast<float>(static_cast<double>(zero) / static_cast<double>(sample.size()));
    features["printable_byte_ratio"] = static_cast<float>(static_cast<double>(printable) / static_cast<double>(sample.size()));
  } else {
    features["zero_byte_ratio"] = 0.0f;
    features["printable_byte_ratio"] = 0.0f;
  }

  return features;
}

}
