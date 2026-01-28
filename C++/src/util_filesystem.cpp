#include "util_filesystem.h"

#include <algorithm>
#include <cctype>
#include <cwctype>
#include <filesystem>
#include <fstream>

namespace kvd {

static bool contains_nul(const std::string& s) {
  return std::find(s.begin(), s.end(), '\0') != s.end();
}

static std::wstring to_lower(std::wstring s) {
  std::transform(s.begin(), s.end(), s.begin(), [](wchar_t c) {
    return static_cast<wchar_t>(std::towlower(c));
  });
  return s;
}

std::optional<std::string> validate_path(const std::string& path, const std::optional<std::string>& allowed_root) {
  if (path.empty() || contains_nul(path)) {
    return std::nullopt;
  }
  std::error_code ec;
  std::filesystem::path p = std::filesystem::absolute(std::filesystem::path(path), ec);
  if (ec) {
    return std::nullopt;
  }
  p = std::filesystem::weakly_canonical(p, ec);
  if (ec) {
    return std::nullopt;
  }
  if (!std::filesystem::exists(p, ec) || ec) {
    return std::nullopt;
  }
  if (allowed_root) {
    std::filesystem::path root = std::filesystem::absolute(std::filesystem::path(*allowed_root), ec);
    if (ec) {
      return std::nullopt;
    }
    root = std::filesystem::weakly_canonical(root, ec);
    if (ec) {
      return std::nullopt;
    }

    std::wstring abs_p = p.wstring();
    std::wstring base = root.wstring();
    abs_p = to_lower(std::move(abs_p));
    base = to_lower(std::move(base));

    if (abs_p == base) {
    } else {
      if (!base.empty() && base.back() != L'\\' && base.back() != L'/') {
        base.push_back(L'\\');
      }
      if (abs_p.size() < base.size()) {
        return std::nullopt;
      }
      if (abs_p.compare(0, base.size(), base) != 0) {
        return std::nullopt;
      }
    }
  }
  auto u8 = p.u8string();
  return std::string(reinterpret_cast<const char*>(u8.data()), u8.size());
}

bool read_file_bytes(const std::string& path, std::vector<std::uint8_t>& out) {
  out.clear();
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    return false;
  }
  f.seekg(0, std::ios::end);
  std::streamoff size = f.tellg();
  if (size < 0) {
    return false;
  }
  f.seekg(0, std::ios::beg);
  out.resize(static_cast<std::size_t>(size));
  if (!out.empty()) {
    f.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(out.size()));
    if (!f) {
      out.clear();
      return false;
    }
  }
  return true;
}

bool read_file_bytes_seek(const std::string& path, std::size_t offset, std::size_t max_count, std::vector<std::uint8_t>& out) {
  out.clear();
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    return false;
  }
  f.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
  if (!f) {
    return false;
  }
  out.resize(max_count);
  f.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(max_count));
  std::streamsize read_count = f.gcount();
  if (read_count < 0) {
    out.clear();
    return false;
  }
  out.resize(static_cast<std::size_t>(read_count));
  return true;
}

}
