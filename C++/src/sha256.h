#pragma once

#include <cstdint>
#include <string>

namespace kvd {

std::uint8_t sha256_last_byte_utf8(const std::string& s);

}
