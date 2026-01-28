#include "features_statistics.h"

#include <array>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace kvd {

static double mean_u8(const std::vector<std::uint8_t>& v, std::size_t n) {
  if (n == 0) return 0.0;
  double s = 0.0;
  for (std::size_t i = 0; i < n; ++i) s += static_cast<double>(v[i]);
  return s / static_cast<double>(n);
}

static double std_u8(const std::vector<std::uint8_t>& v, std::size_t n, double mean) {
  if (n == 0) return 0.0;
  double acc = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    double d = static_cast<double>(v[i]) - mean;
    acc += d * d;
  }
  return std::sqrt(acc / static_cast<double>(n));
}

static std::uint8_t min_u8(const std::vector<std::uint8_t>& v, std::size_t n) {
  if (n == 0) return 0;
  std::uint8_t m = v[0];
  for (std::size_t i = 1; i < n; ++i) m = std::min(m, v[i]);
  return m;
}

static std::uint8_t max_u8(const std::vector<std::uint8_t>& v, std::size_t n) {
  if (n == 0) return 0;
  std::uint8_t m = v[0];
  for (std::size_t i = 1; i < n; ++i) m = std::max(m, v[i]);
  return m;
}

static double entropy_u8(const std::vector<std::uint8_t>& v, std::size_t start, std::size_t end) {
  std::size_t n = end > start ? (end - start) : 0;
  if (n == 0) return 0.0;
  std::array<std::uint32_t, 256> counts{};
  counts.fill(0);
  for (std::size_t i = start; i < end; ++i) {
    counts[v[i]]++;
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

static float percentile_u8(std::vector<std::uint8_t> tmp, double q) {
  std::size_t n = tmp.size();
  if (n == 0) return 0.0f;
  std::sort(tmp.begin(), tmp.end());
  double pos = (static_cast<double>(n) - 1.0) * q;
  std::size_t lo = static_cast<std::size_t>(std::floor(pos));
  std::size_t hi = static_cast<std::size_t>(std::ceil(pos));
  double w = pos - static_cast<double>(lo);
  double a = static_cast<double>(tmp[lo]);
  double b = static_cast<double>(tmp[hi]);
  return static_cast<float>(a + (b - a) * w);
}

static float std_f32(const std::vector<float>& v) {
  if (v.empty()) return 0.0f;
  double m = 0.0;
  for (float x : v) m += static_cast<double>(x);
  m /= static_cast<double>(v.size());
  double acc = 0.0;
  for (float x : v) {
    double d = static_cast<double>(x) - m;
    acc += d * d;
  }
  return static_cast<float>(std::sqrt(acc / static_cast<double>(v.size())));
}

std::vector<float> extract_statistical_features(const std::vector<std::uint8_t>& padded_sequence, std::size_t orig_length) {
  std::size_t n = orig_length;
  if (n > padded_sequence.size()) n = padded_sequence.size();

  std::vector<float> features;
  features.reserve(49);

  double mean_val = mean_u8(padded_sequence, n);
  double std_val = std_u8(padded_sequence, n, mean_val);
  std::uint8_t min_val = min_u8(padded_sequence, n);
  std::uint8_t max_val = max_u8(padded_sequence, n);

  float median_val = 0.0f;
  float q25 = 0.0f;
  float q75 = 0.0f;
  if (n > 0) {
    std::vector<std::uint8_t> tmp(padded_sequence.begin(), padded_sequence.begin() + static_cast<std::ptrdiff_t>(n));
    median_val = percentile_u8(tmp, 0.5);
    q25 = percentile_u8(tmp, 0.25);
    q75 = percentile_u8(tmp, 0.75);
  }

  features.push_back(static_cast<float>(mean_val));
  features.push_back(static_cast<float>(std_val));
  features.push_back(static_cast<float>(min_val));
  features.push_back(static_cast<float>(max_val));
  features.push_back(median_val);
  features.push_back(q25);
  features.push_back(q75);

  int count_0 = 0;
  int count_255 = 0;
  int count_90 = 0;
  int count_printable = 0;
  for (std::size_t i = 0; i < n; ++i) {
    std::uint8_t b = padded_sequence[i];
    if (b == 0) count_0++;
    if (b == 255) count_255++;
    if (b == 0x90) count_90++;
    if (b >= 32 && b <= 126) count_printable++;
  }
  features.push_back(static_cast<float>(count_0));
  features.push_back(static_cast<float>(count_255));
  features.push_back(static_cast<float>(count_90));
  features.push_back(static_cast<float>(count_printable));

  features.push_back(static_cast<float>(entropy_u8(padded_sequence, 0, n)));

  std::size_t one_third = 0;
  if (n >= 3) one_third = n / 3;

  for (int seg_i = 0; seg_i < 3; ++seg_i) {
    std::size_t start = 0;
    std::size_t end = n;
    if (n >= 3) {
      if (seg_i == 0) {
        start = 0;
        end = one_third;
      } else if (seg_i == 1) {
        start = one_third;
        end = 2 * one_third;
      } else {
        start = 2 * one_third;
        end = n;
      }
    }

    std::size_t seg_len = end > start ? (end - start) : 0;
    if (seg_len == 0) {
      features.push_back(0.0f);
      features.push_back(0.0f);
      features.push_back(0.0f);
    } else {
      double m = 0.0;
      for (std::size_t i = start; i < end; ++i) m += static_cast<double>(padded_sequence[i]);
      m /= static_cast<double>(seg_len);
      double sd = 0.0;
      for (std::size_t i = start; i < end; ++i) {
        double d = static_cast<double>(padded_sequence[i]) - m;
        sd += d * d;
      }
      sd = std::sqrt(sd / static_cast<double>(seg_len));
      features.push_back(static_cast<float>(m));
      features.push_back(static_cast<float>(sd));
      features.push_back(static_cast<float>(entropy_u8(padded_sequence, start, end)));
    }
  }

  static constexpr std::size_t STAT_CHUNK_COUNT = 10;
  std::size_t chunk_size = n / STAT_CHUNK_COUNT;
  if (chunk_size < 1) chunk_size = 1;

  std::vector<float> chunk_means;
  std::vector<float> chunk_stds;
  chunk_means.reserve(STAT_CHUNK_COUNT);
  chunk_stds.reserve(STAT_CHUNK_COUNT);

  for (std::size_t i = 0; i < STAT_CHUNK_COUNT; ++i) {
    std::size_t start = i * chunk_size;
    std::size_t end = (i < STAT_CHUNK_COUNT - 1) ? (start + chunk_size) : n;
    if (start > n) start = n;
    if (end > n) end = n;
    std::size_t len = end > start ? (end - start) : 0;
    if (len == 0) {
      chunk_means.push_back(0.0f);
      chunk_stds.push_back(0.0f);
    } else {
      double m = 0.0;
      for (std::size_t j = start; j < end; ++j) m += static_cast<double>(padded_sequence[j]);
      m /= static_cast<double>(len);
      double sd = 0.0;
      for (std::size_t j = start; j < end; ++j) {
        double d = static_cast<double>(padded_sequence[j]) - m;
        sd += d * d;
      }
      sd = std::sqrt(sd / static_cast<double>(len));
      chunk_means.push_back(static_cast<float>(m));
      chunk_stds.push_back(static_cast<float>(sd));
    }
  }

  for (float x : chunk_means) features.push_back(x);
  for (float x : chunk_stds) features.push_back(x);

  if (chunk_means.size() > 1) {
    std::vector<float> mean_diffs;
    std::vector<float> std_diffs;
    mean_diffs.reserve(chunk_means.size() - 1);
    std_diffs.reserve(chunk_stds.size() - 1);
    for (std::size_t i = 1; i < chunk_means.size(); ++i) {
      mean_diffs.push_back(chunk_means[i] - chunk_means[i - 1]);
      std_diffs.push_back(chunk_stds[i] - chunk_stds[i - 1]);
    }

    double mean_abs = 0.0;
    for (float x : mean_diffs) mean_abs += std::abs(static_cast<double>(x));
    mean_abs /= static_cast<double>(mean_diffs.size());

    double std_abs = 0.0;
    for (float x : std_diffs) std_abs += std::abs(static_cast<double>(x));
    std_abs /= static_cast<double>(std_diffs.size());

    auto mm1 = std::minmax_element(mean_diffs.begin(), mean_diffs.end());
    auto mm2 = std::minmax_element(std_diffs.begin(), std_diffs.end());

    features.push_back(static_cast<float>(mean_abs));
    features.push_back(std_f32(mean_diffs));
    features.push_back(*mm1.second);
    features.push_back(*mm1.first);

    features.push_back(static_cast<float>(std_abs));
    features.push_back(std_f32(std_diffs));
    features.push_back(*mm2.second);
    features.push_back(*mm2.first);
  } else {
    for (int i = 0; i < 8; ++i) features.push_back(0.0f);
  }

  return features;
}

}
