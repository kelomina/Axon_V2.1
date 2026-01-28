#pragma once

#include <optional>
#include <string>
#include <unordered_map>

namespace kvd {

std::unordered_map<std::string, float> extract_file_attributes(
    const std::string& path,
    const std::optional<std::string>& allowed_root);

}
