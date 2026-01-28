#include "features.h"

#include "features_file_attributes.h"
#include "sha256.h"
#include "util_filesystem.h"

#include <LIEF/PE.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <iterator>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace kvd {

static constexpr std::size_t PE_FEATURE_VECTOR_DIM = 1500;
static constexpr std::size_t LIGHTWEIGHT_FEATURE_DIM = 256;
static constexpr float LIGHTWEIGHT_FEATURE_SCALE = 1.5f;
static constexpr float PE_FEATURE_SCALE = 0.8f;

static constexpr float SIZE_NORM_MAX = 134217728.0f;
static constexpr float TIMESTAMP_MAX = 2147483647.0f;
static constexpr float TIMESTAMP_YEAR_BASE = 1970.0f;
static constexpr float TIMESTAMP_YEAR_MAX = 2038.0f;
static constexpr float ENTROPY_HIGH_THRESHOLD = 0.8f;
static constexpr float ENTROPY_LOW_THRESHOLD = 0.2f;
static constexpr std::size_t SECTION_ENTROPY_MIN_SIZE = 256;
static constexpr std::size_t OVERLAY_ENTROPY_MIN_SIZE = 512;
static constexpr std::size_t LARGE_OVERLAY_THRESHOLD = 1048576;

static constexpr std::array<const char*, 18> SYSTEM_DLLS = {
    "kernel32",   "user32",    "gdi32",     "advapi32", "ole32",   "oleaut32",
    "shell32",    "comdlg32",  "ws2_32",    "wininet",  "crypt32", "ntdll",
    "msvcrt",     "rpcrt4",    "shlwapi",   "comctl32", "version", "wintrust"};

static constexpr std::array<const char*, 9> COMMON_SECTIONS = {
    ".text", ".data", ".rdata", ".rsrc", ".reloc", ".tls", ".pdata", ".idata", ".edata"};

static constexpr std::array<const char*, 12> PACKER_SECTION_KEYWORDS = {
    "upx", "aspack", "fsg", "mpress", "petite", "pec2to", "nspack", "telock", "themida", "vmp", "vmprotect", "enigma"};

static constexpr std::array<const char*, 12> NETWORK_API_KEYWORDS = {
    "internet", "http", "url", "socket", "connect", "send", "recv", "wsastartup", "wininet", "ws2_32", "dns", "wget"};
static constexpr std::array<const char*, 13> PROCESS_API_KEYWORDS = {
    "createprocess", "openprocess", "terminateprocess", "writeprocessmemory", "readprocessmemory", "virtualallocex", "createremotethread",
    "getprocaddress", "loadlibrary", "winexec", "shellexecute", "rtlcreateuserthread", "ntcreateprocess"};
static constexpr std::array<const char*, 12> FILESYSTEM_API_KEYWORDS = {
    "createfile", "readfile", "writefile", "deletefile", "movefile", "copyfile", "setfileattributes", "getfileattributes",
    "findfirstfile", "findnextfile", "createdirectory", "removedirectory"};
static constexpr std::array<const char*, 12> REGISTRY_API_KEYWORDS = {
    "regopenkey", "regcreatekey", "regsetvalue", "regqueryvalue", "regdeletevalue", "regdeletekey", "regclosekey", "regenumkey",
    "regenumvalue", "regconnectregistry", "regloadkey", "regsavekey"};

static std::string lower_ascii(std::string s) {
  for (char& c : s) {
    if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
  }
  return s;
}

static std::size_t range_count(const auto& r) {
  return static_cast<std::size_t>(std::distance(r.begin(), r.end()));
}

static bool contains_any(const std::string& haystack, const auto& keywords) {
  for (const char* k : keywords) {
    if (!k) continue;
    if (haystack.find(k) != std::string::npos) return true;
  }
  return false;
}

static double entropy_bytes(const std::vector<std::uint8_t>& data) {
  if (data.empty()) return 0.0;
  std::array<std::uint32_t, 256> counts{};
  counts.fill(0);
  for (std::uint8_t b : data) counts[b]++;
  double inv = 1.0 / static_cast<double>(data.size());
  double s = 0.0;
  for (std::size_t i = 0; i < 256; ++i) {
    if (counts[i] == 0) continue;
    double p = static_cast<double>(counts[i]) * inv;
    s += p * std::log2(p);
  }
  return (-s) / 8.0;
}

static float l2_norm(const std::vector<float>& v) {
  double s = 0.0;
  for (float x : v) s += static_cast<double>(x) * static_cast<double>(x);
  return static_cast<float>(std::sqrt(s));
}

static void l2_normalize(std::vector<float>& v) {
  float n = l2_norm(v);
  if (n <= 0.0f || !std::isfinite(n)) return;
  for (float& x : v) x /= n;
}

static std::unordered_map<std::string, float> extract_lightweight_features(const LIEF::PE::Binary& bin) {
  std::unordered_map<std::string, float> out;
  std::vector<float> v(LIGHTWEIGHT_FEATURE_DIM, 0.0f);

  std::vector<std::string> dlls;
  std::vector<std::string> apis;
  if (bin.has_imports()) {
    for (const LIEF::PE::Import& imp : bin.imports()) {
      std::string dll = lower_ascii(imp.name());
      if (!dll.empty()) dlls.push_back(dll);
      for (const LIEF::PE::ImportEntry& e : imp.entries()) {
        if (!e.name().empty()) {
          apis.push_back(e.name());
        }
      }
    }
  }

  for (const std::string& dll : dlls) {
    std::uint8_t last = sha256_last_byte_utf8(dll);
    std::size_t idx = static_cast<std::size_t>(last % 128);
    v[idx] = 1.0f;
  }
  for (const std::string& api : apis) {
    std::uint8_t last = sha256_last_byte_utf8(api);
    std::size_t idx = 128 + static_cast<std::size_t>(last % 128);
    v[idx] = 1.0f;
  }

  for (const LIEF::PE::Section& s : bin.sections()) {
    std::string name = s.name();
    if (name.empty()) continue;
    std::uint8_t last = sha256_last_byte_utf8(name);
    std::size_t idx = 224 + static_cast<std::size_t>(last % 32);
    v[idx] = 1.0f;
  }

  l2_normalize(v);

  for (std::size_t i = 0; i < v.size(); ++i) {
    out[std::string("lw_") + std::to_string(i)] = v[i];
  }
  return out;
}

static std::unordered_map<std::string, float> extract_enhanced_pe_features(const std::string& valid_path, const LIEF::PE::Binary& bin) {
  std::unordered_map<std::string, float> features;

  std::error_code ec;
  std::uintmax_t file_size = std::filesystem::file_size(std::filesystem::path(valid_path), ec);
  if (ec) file_size = 0;

  features["sections_count"] = static_cast<float>(range_count(bin.sections()));
  features["symbols_count"] = 0.0f;

  std::vector<std::pair<std::string, std::string>> imports;
  std::vector<std::string> api_names;
  std::vector<std::string> dll_names;
  std::size_t import_ordinal_only_count = 0;
  std::size_t total_import_functions = 0;

  if (bin.has_imports()) {
    features["imports_count"] = static_cast<float>(range_count(bin.imports()));
    for (const LIEF::PE::Import& entry : bin.imports()) {
      std::string dll_name = lower_ascii(entry.name());
      if (!dll_name.empty()) dll_names.push_back(dll_name);
      for (const LIEF::PE::ImportEntry& imp : entry.entries()) {
        if (!imp.name().empty()) {
          imports.emplace_back(dll_name, imp.name());
          api_names.push_back(imp.name());
          total_import_functions += 1;
        } else if (imp.is_ordinal()) {
          import_ordinal_only_count += 1;
          total_import_functions += 1;
        }
      }
    }
  } else {
    features["imports_count"] = 0.0f;
  }

  if (!dll_names.empty() || !api_names.empty()) {
    std::unordered_set<std::string> uniq_imp;
    uniq_imp.reserve(imports.size());
    for (const auto& p : imports) {
      uniq_imp.insert(p.first + "|" + p.second);
    }
    std::unordered_set<std::string> uniq_dll(dll_names.begin(), dll_names.end());
    std::unordered_set<std::string> uniq_api(api_names.begin(), api_names.end());

    features["unique_imports"] = static_cast<float>(uniq_imp.size());
    features["unique_dlls"] = static_cast<float>(uniq_dll.size());
    features["unique_apis"] = static_cast<float>(uniq_api.size());

    features["import_ordinal_only_count"] = static_cast<float>(import_ordinal_only_count);
    features["import_ordinal_only_ratio"] =
        static_cast<float>(static_cast<double>(import_ordinal_only_count) / static_cast<double>(total_import_functions + 1));
    features["avg_imports_per_dll"] =
        static_cast<float>(static_cast<double>(total_import_functions) / static_cast<double>(uniq_dll.size() + 1));

    features["imports_density"] = static_cast<float>(static_cast<double>(imports.size()) / static_cast<double>(file_size + 1));
    features["imports_name_ratio"] = static_cast<float>(static_cast<double>(api_names.size()) / static_cast<double>(total_import_functions + 1));

    if (!dll_names.empty()) {
      std::vector<std::size_t> lens;
      lens.reserve(dll_names.size());
      for (const auto& n : dll_names) {
        if (!n.empty()) lens.push_back(n.size());
      }
      if (!lens.empty()) {
        double avg = std::accumulate(lens.begin(), lens.end(), 0.0) / static_cast<double>(lens.size());
        auto mm = std::minmax_element(lens.begin(), lens.end());
        features["dll_name_avg_length"] = static_cast<float>(avg);
        features["dll_name_max_length"] = static_cast<float>(*mm.second);
        features["dll_name_min_length"] = static_cast<float>(*mm.first);
      }
    }

    std::unordered_set<std::string> imported_system;
    for (const auto& d : dll_names) {
      std::string base = d;
      auto pos = base.find('.');
      if (pos != std::string::npos) base = base.substr(0, pos);
      base = lower_ascii(base);
      for (const char* s : SYSTEM_DLLS) {
        if (base == s) imported_system.insert(base);
      }
    }
    features["imported_system_dlls_count"] = static_cast<float>(imported_system.size());
    features["imported_system_dlls_ratio"] =
        static_cast<float>(static_cast<double>(imported_system.size()) / static_cast<double>(dll_names.size() + 1));

    auto entropy_from_counts = [](const std::unordered_map<std::string, int>& counts) -> float {
      int total = 0;
      for (const auto& kv : counts) total += kv.second;
      if (total <= 0) return 0.0f;
      double s = 0.0;
      for (const auto& kv : counts) {
        double p = static_cast<double>(kv.second) / static_cast<double>(total);
        if (p > 0.0) s += p * std::log2(p);
      }
      return static_cast<float>(-s / 8.0);
    };

    std::unordered_map<std::string, int> dll_counts;
    for (const auto& d : dll_names) dll_counts[d] += 1;
    std::unordered_map<std::string, int> api_counts;
    for (const auto& a : api_names) api_counts[a] += 1;
    features["dll_imports_entropy"] = entropy_from_counts(dll_counts);
    features["api_imports_entropy"] = entropy_from_counts(api_counts);

    int syscall_count = 0;
    for (const auto& a : api_names) {
      std::string al = lower_ascii(a);
      if (al.rfind("nt", 0) == 0 || al.rfind("zw", 0) == 0) syscall_count++;
    }
    features["syscall_api_ratio"] = static_cast<float>(static_cast<double>(syscall_count) / static_cast<double>(api_names.size() + 1));

    int network_calls = 0;
    int process_calls = 0;
    int filesystem_calls = 0;
    int registry_calls = 0;
    for (const auto& p : imports) {
      std::string dll = lower_ascii(p.first);
      std::string api = lower_ascii(p.second);
      if (contains_any(dll, NETWORK_API_KEYWORDS) || contains_any(api, NETWORK_API_KEYWORDS)) network_calls++;
      if (contains_any(dll, PROCESS_API_KEYWORDS) || contains_any(api, PROCESS_API_KEYWORDS)) process_calls++;
      if (contains_any(dll, FILESYSTEM_API_KEYWORDS) || contains_any(api, FILESYSTEM_API_KEYWORDS)) filesystem_calls++;
      if (contains_any(dll, REGISTRY_API_KEYWORDS) || contains_any(api, REGISTRY_API_KEYWORDS)) registry_calls++;
    }
    double denom = static_cast<double>(total_import_functions + 1);
    features["api_network_ratio"] = static_cast<float>(static_cast<double>(network_calls) / denom);
    features["api_process_ratio"] = static_cast<float>(static_cast<double>(process_calls) / denom);
    features["api_filesystem_ratio"] = static_cast<float>(static_cast<double>(filesystem_calls) / denom);
    features["api_registry_ratio"] = static_cast<float>(static_cast<double>(registry_calls) / denom);
  } else {
    features["unique_imports"] = 0.0f;
    features["unique_dlls"] = 0.0f;
    features["unique_apis"] = 0.0f;
    features["dll_name_avg_length"] = 0.0f;
    features["dll_name_max_length"] = 0.0f;
    features["dll_name_min_length"] = 0.0f;
    features["imported_system_dlls_count"] = 0.0f;
    features["dll_imports_entropy"] = 0.0f;
    features["api_imports_entropy"] = 0.0f;
    features["imported_system_dlls_ratio"] = 0.0f;
    features["syscall_api_ratio"] = 0.0f;
    features["import_ordinal_only_count"] = 0.0f;
    features["import_ordinal_only_ratio"] = 0.0f;
    features["avg_imports_per_dll"] = 0.0f;
    features["imports_density"] = 0.0f;
    features["imports_name_ratio"] = 0.0f;
    features["api_network_ratio"] = 0.0f;
    features["api_process_ratio"] = 0.0f;
    features["api_filesystem_ratio"] = 0.0f;
    features["api_registry_ratio"] = 0.0f;
  }

  if (bin.has_exports()) {
    const LIEF::PE::Export* ex = bin.get_export();
    if (!ex) {
      features["exports_count"] = 0.0f;
      features["export_name_avg_length"] = 0.0f;
      features["export_name_max_length"] = 0.0f;
      features["export_name_min_length"] = 0.0f;
      features["exports_density"] = 0.0f;
      features["exports_name_ratio"] = 0.0f;
    } else {
      features["exports_count"] = static_cast<float>(ex->entries().size());
    std::vector<std::size_t> name_lens;
    std::size_t name_count = 0;
      for (const auto& e : ex->entries()) {
      if (!e.name().empty()) {
        name_count++;
        name_lens.push_back(e.name().size());
      }
    }
    if (!name_lens.empty()) {
      double avg = std::accumulate(name_lens.begin(), name_lens.end(), 0.0) / static_cast<double>(name_lens.size());
      auto mm = std::minmax_element(name_lens.begin(), name_lens.end());
      features["export_name_avg_length"] = static_cast<float>(avg);
      features["export_name_max_length"] = static_cast<float>(*mm.second);
      features["export_name_min_length"] = static_cast<float>(*mm.first);
      features["exports_density"] = static_cast<float>(static_cast<double>(name_count) / static_cast<double>(file_size + 1));
        features["exports_name_ratio"] = static_cast<float>(static_cast<double>(name_count) / static_cast<double>(ex->entries().size() + 1));
    } else {
      features["export_name_avg_length"] = 0.0f;
      features["export_name_max_length"] = 0.0f;
      features["export_name_min_length"] = 0.0f;
      features["exports_density"] = 0.0f;
      features["exports_name_ratio"] = 0.0f;
    }
    }
  } else {
    features["exports_count"] = 0.0f;
    features["export_name_avg_length"] = 0.0f;
    features["export_name_max_length"] = 0.0f;
    features["export_name_min_length"] = 0.0f;
    features["exports_density"] = 0.0f;
    features["exports_name_ratio"] = 0.0f;
  }

  std::vector<std::string> section_names;
  std::vector<std::uint64_t> section_sizes;
  std::vector<std::uint64_t> section_vsizes;
  std::vector<double> section_entropies;

  std::uint64_t code_section_size = 0;
  std::uint64_t data_section_size = 0;
  std::uint64_t code_section_vsize = 0;
  std::uint64_t data_section_vsize = 0;
  std::size_t executable_sections_count = 0;
  std::size_t writable_sections_count = 0;
  std::size_t readable_sections_count = 0;
  std::size_t rwx_sections_count = 0;
  std::size_t non_standard_executable_sections_count = 0;
  std::size_t executable_writable_sections = 0;
  std::size_t alignment_mismatch_count = 0;

  std::unordered_set<std::string> common_exec = {".text", "text", ".code"};
  std::uint32_t file_align = bin.optional_header().file_alignment();
  std::uint32_t sect_align = bin.optional_header().section_alignment();

  for (const LIEF::PE::Section& section : bin.sections()) {
    std::string name = section.name();
    section_names.push_back(name);
    section_sizes.push_back(section.sizeof_raw_data());
    section_vsizes.push_back(section.virtual_size());

    const auto& content = section.content();
    if (content.size() >= SECTION_ENTROPY_MIN_SIZE) {
      std::vector<std::uint8_t> tmp(content.begin(), content.end());
      section_entropies.push_back(entropy_bytes(tmp));
    } else {
      section_entropies.push_back(0.0);
    }

    bool is_executable = (section.characteristics() & 0x20000000u) != 0;
    bool is_writable = (section.characteristics() & 0x80000000u) != 0;
    bool is_readable = (section.characteristics() & 0x40000000u) != 0;
    if (is_executable) executable_sections_count++;
    if (is_writable) writable_sections_count++;
    if (is_readable) readable_sections_count++;
    if (is_executable && is_writable) executable_writable_sections++;
    if (is_executable && is_writable && is_readable) rwx_sections_count++;

    if (is_executable && common_exec.find(lower_ascii(name)) == common_exec.end()) {
      non_standard_executable_sections_count++;
    }

    if (is_executable) {
      code_section_size += section.sizeof_raw_data();
      code_section_vsize += section.virtual_size();
    } else {
      data_section_size += section.sizeof_raw_data();
      data_section_vsize += section.virtual_size();
    }

    if (file_align != 0 && section.sizeof_raw_data() % file_align != 0) alignment_mismatch_count++;
    if (sect_align != 0 && section.virtual_size() % sect_align != 0) alignment_mismatch_count++;
  }

  features["section_names_count"] = static_cast<float>(section_names.size());
  features["section_total_size"] = static_cast<float>(std::accumulate(section_sizes.begin(), section_sizes.end(), 0ULL));
  features["section_total_vsize"] = static_cast<float>(std::accumulate(section_vsizes.begin(), section_vsizes.end(), 0ULL));

  if (!section_sizes.empty()) {
    double avg = std::accumulate(section_sizes.begin(), section_sizes.end(), 0.0) / static_cast<double>(section_sizes.size());
    auto mm = std::minmax_element(section_sizes.begin(), section_sizes.end());
    features["avg_section_size"] = static_cast<float>(avg);
    features["max_section_size"] = static_cast<float>(*mm.second);
    features["min_section_size"] = static_cast<float>(*mm.first);

    double var = 0.0;
    for (auto s : section_sizes) {
      double d = static_cast<double>(s) - avg;
      var += d * d;
    }
    double sd = std::sqrt(var / static_cast<double>(section_sizes.size()));
    features["section_size_std"] = static_cast<float>(sd);
    features["section_size_cv"] = static_cast<float>(avg != 0.0 ? (sd / avg) : 0.0);
  } else {
    features["avg_section_size"] = 0.0f;
    features["max_section_size"] = 0.0f;
    features["min_section_size"] = 0.0f;
    features["section_size_std"] = 0.0f;
    features["section_size_cv"] = 0.0f;
  }

  if (!section_vsizes.empty()) {
    double avg = std::accumulate(section_vsizes.begin(), section_vsizes.end(), 0.0) / static_cast<double>(section_vsizes.size());
    auto mm = std::minmax_element(section_vsizes.begin(), section_vsizes.end());
    features["avg_section_vsize"] = static_cast<float>(avg);
    features["max_section_vsize"] = static_cast<float>(*mm.second);
    features["min_section_vsize"] = static_cast<float>(*mm.first);
    double var = 0.0;
    for (auto s : section_vsizes) {
      double d = static_cast<double>(s) - avg;
      var += d * d;
    }
    double sd = std::sqrt(var / static_cast<double>(section_vsizes.size()));
    features["section_vsize_std"] = static_cast<float>(sd);
    features["section_vsize_cv"] = static_cast<float>(avg != 0.0 ? (sd / avg) : 0.0);
  } else {
    features["avg_section_vsize"] = 0.0f;
    features["max_section_vsize"] = 0.0f;
    features["min_section_vsize"] = 0.0f;
    features["section_vsize_std"] = 0.0f;
    features["section_vsize_cv"] = 0.0f;
  }

  if (!section_entropies.empty()) {
    double avg = std::accumulate(section_entropies.begin(), section_entropies.end(), 0.0) / static_cast<double>(section_entropies.size());
    auto mm = std::minmax_element(section_entropies.begin(), section_entropies.end());
    double var = 0.0;
    for (double e : section_entropies) {
      double d = e - avg;
      var += d * d;
    }
    double sd = std::sqrt(var / static_cast<double>(section_entropies.size()));
    features["avg_section_entropy"] = static_cast<float>(avg);
    features["max_section_entropy"] = static_cast<float>(*mm.second);
    features["min_section_entropy"] = static_cast<float>(*mm.first);
    features["section_entropy_std"] = static_cast<float>(sd);
    features["section_entropy_avg"] = features["avg_section_entropy"];
    features["section_entropy_min"] = features["min_section_entropy"];
    features["section_entropy_max"] = features["max_section_entropy"];
    features["packed_sections_ratio"] = static_cast<float>(
        static_cast<double>(std::count_if(section_entropies.begin(), section_entropies.end(),
                                          [](double e) { return e > ENTROPY_HIGH_THRESHOLD; })) /
        static_cast<double>(section_entropies.size()));
  } else {
    features["avg_section_entropy"] = 0.0f;
    features["max_section_entropy"] = 0.0f;
    features["min_section_entropy"] = 0.0f;
    features["section_entropy_std"] = 0.0f;
    features["section_entropy_avg"] = 0.0f;
    features["section_entropy_min"] = 0.0f;
    features["section_entropy_max"] = 0.0f;
    features["packed_sections_ratio"] = 0.0f;
  }

  features["code_section_ratio"] = static_cast<float>(static_cast<double>(code_section_size) / static_cast<double>(std::accumulate(section_sizes.begin(), section_sizes.end(), 0ULL) + 1));
  features["data_section_ratio"] = static_cast<float>(static_cast<double>(data_section_size) / static_cast<double>(std::accumulate(section_sizes.begin(), section_sizes.end(), 0ULL) + 1));
  features["code_vsize_ratio"] = static_cast<float>(static_cast<double>(code_section_vsize) / static_cast<double>(std::accumulate(section_vsizes.begin(), section_vsizes.end(), 0ULL) + 1));
  features["data_vsize_ratio"] = static_cast<float>(static_cast<double>(data_section_vsize) / static_cast<double>(std::accumulate(section_vsizes.begin(), section_vsizes.end(), 0ULL) + 1));

  features["executable_sections_count"] = static_cast<float>(executable_sections_count);
  features["writable_sections_count"] = static_cast<float>(writable_sections_count);
  features["readable_sections_count"] = static_cast<float>(readable_sections_count);
  features["rwx_sections_count"] = static_cast<float>(rwx_sections_count);
  features["non_standard_executable_sections_count"] = static_cast<float>(non_standard_executable_sections_count);
  features["executable_writable_sections"] = static_cast<float>(executable_writable_sections);

  double sec_denom = static_cast<double>(section_names.size() + 1);
  features["executable_sections_ratio"] = static_cast<float>(static_cast<double>(executable_sections_count) / sec_denom);
  features["writable_sections_ratio"] = static_cast<float>(static_cast<double>(writable_sections_count) / sec_denom);
  features["readable_sections_ratio"] = static_cast<float>(static_cast<double>(readable_sections_count) / sec_denom);
  features["rwx_sections_ratio"] = static_cast<float>(static_cast<double>(rwx_sections_count) / sec_denom);
  features["non_standard_executable_sections_ratio"] = static_cast<float>(static_cast<double>(non_standard_executable_sections_count) / sec_denom);
  features["executable_writable_ratio"] = static_cast<float>(static_cast<double>(executable_writable_sections) / sec_denom);
  features["executable_code_density"] = static_cast<float>(static_cast<double>(code_section_size) / static_cast<double>(file_size + 1));

  features["alignment_mismatch_count"] = static_cast<float>(alignment_mismatch_count);
  features["alignment_mismatch_ratio"] = static_cast<float>(static_cast<double>(alignment_mismatch_count) / static_cast<double>(section_names.size() + 1));

  std::vector<std::size_t> name_lens;
  for (const auto& n : section_names) if (!n.empty()) name_lens.push_back(n.size());
  if (!name_lens.empty()) {
    double avg = std::accumulate(name_lens.begin(), name_lens.end(), 0.0) / static_cast<double>(name_lens.size());
    auto mm = std::minmax_element(name_lens.begin(), name_lens.end());
    features["section_name_avg_length"] = static_cast<float>(avg);
    features["section_name_max_length"] = static_cast<float>(*mm.second);
    features["section_name_min_length"] = static_cast<float>(*mm.first);
  } else {
    features["section_name_avg_length"] = 0.0f;
    features["section_name_max_length"] = 0.0f;
    features["section_name_min_length"] = 0.0f;
  }

  std::vector<std::string> lower_names;
  lower_names.reserve(section_names.size());
  for (const auto& n : section_names) {
    lower_names.push_back(lower_ascii(n));
  }

  features["has_upx_section"] = std::any_of(lower_names.begin(), lower_names.end(), [](const std::string& n) { return n.find("upx") != std::string::npos; }) ? 1.0f : 0.0f;
  features["has_mpress_section"] = std::any_of(lower_names.begin(), lower_names.end(), [](const std::string& n) { return n.find("mpress") != std::string::npos; }) ? 1.0f : 0.0f;
  features["has_aspack_section"] = std::any_of(lower_names.begin(), lower_names.end(), [](const std::string& n) { return n.find("aspack") != std::string::npos; }) ? 1.0f : 0.0f;
  features["has_themida_section"] = std::any_of(lower_names.begin(), lower_names.end(), [](const std::string& n) { return n.find("themida") != std::string::npos; }) ? 1.0f : 0.0f;

  double total_chars = 0.0;
  double special_chars = 0.0;
  for (const auto& n : section_names) {
    total_chars += static_cast<double>(n.size());
    for (unsigned char c : n) {
      if (!std::isalnum(c) && c != '_' && c != '.') {
        special_chars += 1.0;
      }
    }
  }
  features["special_char_ratio"] = static_cast<float>(special_chars / (total_chars + 1.0));

  std::size_t packer_hits = 0;
  for (const char* kw : PACKER_SECTION_KEYWORDS) {
    if (!kw) continue;
    bool present = std::any_of(lower_names.begin(), lower_names.end(), [kw](const std::string& n) { return n.find(kw) != std::string::npos; });
    if (present) packer_hits++;
  }
  features["packer_keyword_hits_count"] = static_cast<float>(packer_hits);
  features["packer_keyword_hits_ratio"] = static_cast<float>(static_cast<double>(packer_hits) / static_cast<double>(section_names.size() + 1));

  std::unordered_set<std::string> uniq_sec(section_names.begin(), section_names.end());
  features["unique_sections_count"] = static_cast<float>(uniq_sec.size());
  features["unique_sections_ratio"] = static_cast<float>(static_cast<double>(uniq_sec.size()) / static_cast<double>(section_names.size() + 1));

  std::size_t long_sections_count = 0;
  std::size_t short_sections_count = 0;
  for (const auto& n : section_names) {
    if (n.size() > 6) long_sections_count++;
    if (n.size() < 3) short_sections_count++;
  }
  features["long_sections_count"] = static_cast<float>(long_sections_count);
  features["short_sections_count"] = static_cast<float>(short_sections_count);
  features["long_sections_ratio"] = static_cast<float>(static_cast<double>(long_sections_count) / static_cast<double>(section_names.size() + 1));
  features["short_sections_ratio"] = static_cast<float>(static_cast<double>(short_sections_count) / static_cast<double>(section_names.size() + 1));

  std::size_t max_end = 0;
  for (const LIEF::PE::Section& section : bin.sections()) {
    std::size_t end = static_cast<std::size_t>(section.pointerto_raw_data()) + static_cast<std::size_t>(section.sizeof_raw_data());
    if (end > max_end) max_end = end;
  }
  std::size_t trailing_data_size = 0;
  if (file_size > max_end) trailing_data_size = static_cast<std::size_t>(file_size - max_end);
  features["trailing_data_size"] = static_cast<float>(trailing_data_size);
  features["trailing_data_ratio"] = static_cast<float>(static_cast<double>(trailing_data_size) / static_cast<double>(file_size + 1));
  features["has_large_trailing_data"] = trailing_data_size > LARGE_OVERLAY_THRESHOLD ? 1.0f : 0.0f;

  std::vector<std::uint8_t> overlay;
  if (trailing_data_size >= OVERLAY_ENTROPY_MIN_SIZE) {
    read_file_bytes_seek(valid_path, max_end, trailing_data_size, overlay);
  }
  double overlay_entropy = entropy_bytes(overlay);
  features["overlay_entropy"] = static_cast<float>(overlay_entropy);
  features["overlay_high_entropy_flag"] = overlay_entropy > ENTROPY_HIGH_THRESHOLD ? 1.0f : 0.0f;
  features["has_high_entropy_overlay"] = features["overlay_high_entropy_flag"];

  features["resources_count"] = 0.0f;
  features["resource_types_count"] = 0.0f;
  features["has_resources"] = bin.has_resources() ? 1.0f : 0.0f;
  if (bin.has_resources()) {
    auto rm_res = bin.resources_manager();
    if (rm_res) {
      auto& rm = rm_res.value();
      auto types = rm.get_types();
      features["resource_types_count"] = static_cast<float>(types.size());
      features["resources_count"] = static_cast<float>(types.size());
    }
  }

  bool has_debug = bin.has_debug();
  features["has_debug_info"] = has_debug ? 1.0f : 0.0f;
  features["has_tls"] = bin.has_tls() ? 1.0f : 0.0f;
  features["has_relocs"] = bin.has_relocations() ? 1.0f : 0.0f;
  features["has_exceptions"] = 0.0f;

  std::size_t tls_cnt = 0;
  if (bin.has_tls()) {
    const auto* tls = bin.tls();
    tls_cnt = (tls && tls->addressof_callbacks() != 0) ? 1 : 0;
  }
  features["tls_callbacks_count"] = static_cast<float>(tls_cnt);

  std::size_t reloc_blocks = 0;
  std::size_t reloc_entries = 0;
  if (bin.has_relocations()) {
    for (const auto& b : bin.relocations()) {
      reloc_blocks++;
      reloc_entries += b.entries().size();
    }
  }
  features["relocation_blocks_count"] = static_cast<float>(reloc_blocks);
  features["relocation_entries_count"] = static_cast<float>(reloc_entries);
  features["reloc_blocks_count"] = static_cast<float>(reloc_blocks);
  features["reloc_entries_count"] = static_cast<float>(reloc_entries);

  std::size_t ptr_size = (bin.optional_header().magic() == LIEF::PE::PE_TYPE::PE32) ? 4 : 8;
  features["import_address_table_size"] = bin.has_imports() ? static_cast<float>(total_import_functions * ptr_size) : 0.0f;

  features["subsystem"] = static_cast<float>(bin.optional_header().subsystem());
  std::uint32_t dll_chars = bin.optional_header().dll_characteristics();
  features["dll_characteristics"] = static_cast<float>(dll_chars);
  features["has_nx_compat"] = (dll_chars & 0x0100u) ? 1.0f : 0.0f;
  features["has_aslr"] = (dll_chars & 0x0040u) ? 1.0f : 0.0f;
  features["has_seh"] = (dll_chars & 0x0400u) ? 0.0f : 1.0f;
  features["has_guard_cf"] = (dll_chars & 0x4000u) ? 1.0f : 0.0f;

  features["entry_point_ratio"] = static_cast<float>(static_cast<double>(bin.optional_header().addressof_entrypoint()) / static_cast<double>(file_size + 1));
  features["image_base"] = static_cast<float>(bin.optional_header().imagebase() > 0 ? 1.0 : 0.0);
  features["image_base_mod_64k"] = static_cast<float>(static_cast<double>(bin.optional_header().imagebase() % 65536u) / 65536.0);
  features["checksum"] = static_cast<float>(bin.optional_header().checksum());
  features["checksum_zero_flag"] = (bin.optional_header().checksum() == 0) ? 1.0f : 0.0f;
  features["pe_header_size"] = static_cast<float>(bin.optional_header().sizeof_headers());
  features["header_size_ratio"] = static_cast<float>(static_cast<double>(bin.optional_header().sizeof_headers()) / static_cast<double>(file_size + 1));

  std::string entry_section_name;
  auto entry_rva = bin.optional_header().addressof_entrypoint();
  for (const auto& s : bin.sections()) {
    if (entry_rva >= s.virtual_address() && entry_rva < s.virtual_address() + std::max<std::uint32_t>(s.virtual_size(), s.sizeof_raw_data())) {
      entry_section_name = lower_ascii(s.name());
      break;
    }
  }
  std::unordered_set<std::string> common_exec_names = {".text", "text", ".code"};
  features["entry_in_nonstandard_section"] =
      (!entry_section_name.empty() && common_exec_names.find(entry_section_name) == common_exec_names.end()) ? 1.0f : 0.0f;
  features["entry_in_nonstandard_section_flag"] = features["entry_in_nonstandard_section"];

  features["has_signature"] = bin.has_signatures() ? 1.0f : 0.0f;
  features["signature_size"] = 0.0f;
  features["signature_has_signing_time"] = 0.0f;
  if (bin.has_signatures()) {
    auto sigs = bin.signatures();
    auto it = sigs.begin();
    if (it != sigs.end()) {
      auto der = it->raw_der();
      features["signature_size"] = static_cast<float>(der.size());
      if (!der.empty()) {
        std::string blob(reinterpret_cast<const char*>(der.data()), der.size());
        bool has_st = (blob.find("signingTime") != std::string::npos) || (blob.find("1.2.840.113549.1.9.5") != std::string::npos);
        features["signature_has_signing_time"] = has_st ? 1.0f : 0.0f;
      }
    }
  }

  bool has_version = false;
  std::size_t company_name_len = 0;
  std::size_t product_name_len = 0;
  if (bin.has_resources()) {
    auto rm_res = bin.resources_manager();
    if (rm_res) {
      has_version = rm_res.value().has_version();
    }
  }
  features["version_info_present"] = has_version ? 1.0f : 0.0f;
  features["company_name_len"] = static_cast<float>(company_name_len);
  features["product_name_len"] = static_cast<float>(product_name_len);
  features["file_version_len"] = 0.0f;
  features["original_filename_len"] = 0.0f;

  std::size_t upx_section_count = 0;
  std::size_t compressed_section_count = 0;
  for (const auto& n : section_names) {
    std::string nl = lower_ascii(n);
    if (nl.find("upx") != std::string::npos) upx_section_count++;
    if (nl.find("comp") != std::string::npos || nl.find("zip") != std::string::npos || nl.find("lz") != std::string::npos) compressed_section_count++;
  }
  features["has_upx_section"] = upx_section_count > 0 ? 1.0f : 0.0f;
  features["upx_section_count"] = static_cast<float>(upx_section_count);
  features["has_compressed_section"] = compressed_section_count > 0 ? 1.0f : 0.0f;
  features["compressed_section_count"] = static_cast<float>(compressed_section_count);

  std::uint32_t timestamp = bin.header().time_date_stamp();
  features["timestamp"] = static_cast<float>(timestamp);
  std::uint32_t year = 0;
  if (timestamp > 0) {
    std::time_t tt = static_cast<std::time_t>(timestamp);
    std::tm tm{};
    gmtime_s(&tm, &tt);
    year = static_cast<std::uint32_t>(tm.tm_year + 1900);
  }
  features["timestamp_year"] = static_cast<float>(year);

  for (const char* sec : COMMON_SECTIONS) {
    std::string k = std::string("has_") + sec + "_section";
    bool present = false;
    for (const auto& n : section_names) {
      if (lower_ascii(n) == sec) {
        present = true;
        break;
      }
    }
    features[k] = present ? 1.0f : 0.0f;
  }

  return features;
}

static std::vector<std::string> build_feature_order() {
  static const std::vector<std::string> BASE = {
      "size","log_size","sections_count","symbols_count","imports_count","exports_count",
      "unique_imports","unique_dlls","unique_apis","section_names_count","section_total_size",
      "section_total_vsize","avg_section_size","avg_section_vsize","section_entropy_avg","section_entropy_min","section_entropy_max","section_entropy_std","packed_sections_ratio","subsystem","dll_characteristics",
      "code_section_ratio","data_section_ratio","code_vsize_ratio","data_vsize_ratio",
      "has_nx_compat","has_aslr","has_seh","has_guard_cf","has_resources","has_debug_info",
      "has_tls","has_relocs","has_exceptions","dll_name_avg_length","dll_name_max_length",
      "dll_name_min_length","section_name_avg_length","section_name_max_length","section_name_min_length",
      "export_name_avg_length","export_name_max_length","export_name_min_length","max_section_size",
      "min_section_size","long_sections_count","short_sections_count","section_size_std","section_size_cv",
      "executable_writable_sections","file_entropy_avg","file_entropy_min","file_entropy_max","file_entropy_range",
      "zero_byte_ratio","printable_byte_ratio","trailing_data_size","trailing_data_ratio","imported_system_dlls_count",
      "exports_density","has_large_trailing_data","pe_header_size","header_size_ratio","file_entropy_std",
      "file_entropy_q25","file_entropy_q75","file_entropy_median","high_entropy_ratio","low_entropy_ratio",
      "entropy_change_rate","entropy_change_std","executable_sections_count","writable_sections_count",
      "readable_sections_count","executable_sections_ratio","writable_sections_ratio","readable_sections_ratio",
      "executable_code_density","non_standard_executable_sections_count","rwx_sections_count","rwx_sections_ratio",
      "special_char_ratio","long_sections_ratio","short_sections_ratio","dll_imports_entropy","api_imports_entropy",
      "imported_system_dlls_ratio","resources_count","alignment_mismatch_count","alignment_mismatch_ratio","entry_point_ratio",
      "syscall_api_ratio","import_ordinal_only_count","import_ordinal_only_ratio","avg_imports_per_dll","exports_name_ratio","entry_in_nonstandard_section_flag","tls_callbacks_count","reloc_blocks_count","reloc_entries_count","checksum_zero_flag","api_network_ratio","api_process_ratio","api_filesystem_ratio","api_registry_ratio","overlay_entropy","overlay_high_entropy_flag","packer_keyword_hits_count","packer_keyword_hits_ratio"};

  std::vector<std::string> order = BASE;
  order.reserve(1500);
  for (const char* sec : COMMON_SECTIONS) {
    order.emplace_back(std::string("has_") + sec + "_section");
  }
  order.insert(order.end(),
               {"has_signature","signature_size","signature_has_signing_time","version_info_present","company_name_len","product_name_len","file_version_len","original_filename_len",
                "has_upx_section","has_mpress_section","has_aspack_section","has_themida_section","timestamp","timestamp_year"});
  return order;
}

std::vector<float> extract_combined_pe_features_from_path(
    const std::string& path,
    const std::optional<std::string>& allowed_root) {
  auto valid = validate_path(path, allowed_root);
  if (!valid) {
    return {};
  }

  std::unique_ptr<LIEF::PE::Binary> bin;
  try {
    bin = LIEF::PE::Parser::parse(*valid);
  } catch (...) {
    return {};
  }
  if (!bin) {
    return {};
  }

  std::unordered_map<std::string, float> all_features = extract_file_attributes(*valid, allowed_root);
  auto enh = extract_enhanced_pe_features(*valid, *bin);
  for (auto& kv : enh) all_features[kv.first] = kv.second;

  std::vector<float> combined;
  combined.assign(PE_FEATURE_VECTOR_DIM, 0.0f);

  std::vector<float> lw(LIGHTWEIGHT_FEATURE_DIM, 0.0f);
  {
    std::vector<float> tmp(LIGHTWEIGHT_FEATURE_DIM, 0.0f);
    if (bin->has_imports()) {
      for (const LIEF::PE::Import& imp : bin->imports()) {
        std::string dll = lower_ascii(imp.name());
        if (!dll.empty()) {
          std::uint8_t last = sha256_last_byte_utf8(dll);
          tmp[static_cast<std::size_t>(last % 128)] = 1.0f;
        }
        for (const auto& e : imp.entries()) {
          if (!e.name().empty()) {
            std::uint8_t last = sha256_last_byte_utf8(e.name());
            tmp[128 + static_cast<std::size_t>(last % 128)] = 1.0f;
          }
        }
      }
    }
    for (const auto& s : bin->sections()) {
      std::string n = s.name();
      if (!n.empty()) {
        std::uint8_t last = sha256_last_byte_utf8(n);
        tmp[224 + static_cast<std::size_t>(last % 32)] = 1.0f;
      }
    }
    l2_normalize(tmp);
    lw = std::move(tmp);
  }

  for (std::size_t i = 0; i < LIGHTWEIGHT_FEATURE_DIM; ++i) {
    combined[i] = lw[i] * LIGHTWEIGHT_FEATURE_SCALE;
  }

  auto order = build_feature_order();
  float log_size_norm = static_cast<float>(std::log(static_cast<double>(SIZE_NORM_MAX)));

  std::size_t base_offset = LIGHTWEIGHT_FEATURE_DIM;
  for (std::size_t i = 0; i < order.size() && base_offset + i < PE_FEATURE_VECTOR_DIM; ++i) {
    const std::string& key = order[i];
    float v = 0.0f;
    auto it = all_features.find(key);
    if (it != all_features.end()) v = it->second;

    if (key.find("size") != std::string::npos) {
      v = v / SIZE_NORM_MAX;
    } else if (key == "timestamp") {
      v = v / TIMESTAMP_MAX;
    } else if (key == "timestamp_year") {
      v = (v - TIMESTAMP_YEAR_BASE) / (TIMESTAMP_YEAR_MAX - TIMESTAMP_YEAR_BASE);
    } else if (key.rfind("has_", 0) == 0) {
      v = v > 0.0f ? 1.0f : 0.0f;
    } else if (key == "log_size") {
      v = log_size_norm > 0.0f ? (v / log_size_norm) : 0.0f;
    }
    if (!std::isfinite(v)) v = 0.0f;
    combined[base_offset + i] = v * PE_FEATURE_SCALE;
  }

  l2_normalize(combined);
  return combined;
}

std::optional<std::size_t> pe_feature_index(const std::string& name) {
  static const std::unordered_map<std::string, std::size_t> index_map = []() {
    std::unordered_map<std::string, std::size_t> m;
    auto order = build_feature_order();
    m.reserve(order.size());
    for (std::size_t i = 0; i < order.size(); ++i) {
      std::size_t idx = LIGHTWEIGHT_FEATURE_DIM + i;
      if (idx >= PE_FEATURE_VECTOR_DIM) break;
      m.emplace(order[i], idx);
    }
    return m;
  }();

  auto it = index_map.find(name);
  if (it == index_map.end()) return std::nullopt;
  return it->second;
}

}
