#pragma once

#include "kvd_internal.h"

#include <optional>
#include <string>

namespace kvd {

Config config_from_api(
    const char* model_path,
    const char* model_normal_path,
    const char* model_packed_path,
    const char* family_classifier_json_path,
    const char* allowed_scan_root,
    unsigned int max_file_size,
    float prediction_threshold);

std::optional<std::string> getenv_string(const char* name);

}
