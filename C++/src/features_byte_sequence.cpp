#include "features.h"

#include "util_filesystem.h"

namespace kvd {

std::optional<ByteSequenceResult> extract_byte_sequence_from_path(
    const std::string& path,
    std::size_t max_file_size,
    const std::optional<std::string>& allowed_root) {
  auto valid = validate_path(path, allowed_root);
  if (!valid) {
    return std::nullopt;
  }
  std::size_t offset = 8;
  if (max_file_size <= offset) {
    return std::nullopt;
  }
  std::vector<std::uint8_t> raw;
  if (!read_file_bytes_seek(*valid, offset, max_file_size - offset, raw)) {
    return std::nullopt;
  }
  ByteSequenceResult r;
  r.original_length = raw.size();
  r.padded_sequence.assign(max_file_size, 0);
  if (!raw.empty()) {
    std::size_t copy_n = raw.size();
    if (copy_n > max_file_size) copy_n = max_file_size;
    std::copy(raw.begin(), raw.begin() + static_cast<std::ptrdiff_t>(copy_n), r.padded_sequence.begin());
  }
  return r;
}

std::optional<ByteSequenceResult> extract_byte_sequence_from_bytes(
    const std::uint8_t* bytes,
    std::size_t len,
    std::size_t max_file_size) {
  if (!bytes) {
    return std::nullopt;
  }
  std::size_t offset = 8;
  if (max_file_size <= offset) {
    return std::nullopt;
  }
  if (len <= offset) {
    ByteSequenceResult r;
    r.original_length = 0;
    r.padded_sequence.assign(max_file_size, 0);
    return r;
  }
  std::size_t avail = len - offset;
  std::size_t want = max_file_size - offset;
  std::size_t take = avail < want ? avail : want;

  ByteSequenceResult r;
  r.original_length = take;
  r.padded_sequence.assign(max_file_size, 0);
  if (take > 0) {
    std::copy(bytes + offset, bytes + offset + take, r.padded_sequence.begin());
  }
  return r;
}

}
