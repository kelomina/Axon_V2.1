#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace kvd {

std::vector<float> extract_statistical_features(const std::vector<std::uint8_t>& padded_sequence, std::size_t orig_length);

}
