#include "kvd/api.h"

#if defined(_WIN32)
#include <windows.h>
#endif

#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <string>

static std::string get_arg_value(int argc, char** argv, const std::string& key) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (argv[i] == key) {
      return std::string(argv[i + 1]);
    }
  }
  return {};
}

static bool has_flag(int argc, char** argv, const std::string& flag) {
  for (int i = 1; i < argc; ++i) {
    if (argv[i] == flag) return true;
  }
  return false;
}

static int usage() {
  std::cerr
      << "Usage:\n"
      << "  kvd_scan_loader --target <file> [--dll <kvd.dll>] [--model <lightgbm_model.txt>]\n"
      << "                 [--model_normal <lightgbm_model_normal.txt>] [--model_packed <lightgbm_model_packed.txt>]\n"
      << "                 [--family <family_classifier.json>] [--allowed_root <dir>]\n"
      << "                 [--max_file_size <bytes>] [--threshold <0..1>]\n";
  return 2;
}

static bool parse_u32(const std::string& s, unsigned int& out) {
  if (s.empty()) return false;
  char* end = nullptr;
  unsigned long long v = std::strtoull(s.c_str(), &end, 10);
  if (!end || *end != '\0') return false;
  out = static_cast<unsigned int>(v);
  return true;
}

static bool parse_f32(const std::string& s, float& out) {
  if (s.empty()) return false;
  char* end = nullptr;
  float v = std::strtof(s.c_str(), &end);
  if (!end || *end != '\0') return false;
  out = v;
  return true;
}

int main(int argc, char** argv) {
  if (argc < 3 || has_flag(argc, argv, "--help")) {
    return usage();
  }

  std::string dll_path = get_arg_value(argc, argv, "--dll");
  if (dll_path.empty()) dll_path = "kvd.dll";

  std::string target = get_arg_value(argc, argv, "--target");
  if (target.empty()) {
    return usage();
  }

  std::string model_path = get_arg_value(argc, argv, "--model");
  std::string model_normal_path = get_arg_value(argc, argv, "--model_normal");
  std::string model_packed_path = get_arg_value(argc, argv, "--model_packed");
  std::string family_path = get_arg_value(argc, argv, "--family");
  std::string allowed_root = get_arg_value(argc, argv, "--allowed_root");
  std::string max_file_size_s = get_arg_value(argc, argv, "--max_file_size");
  std::string threshold_s = get_arg_value(argc, argv, "--threshold");

#if !defined(_WIN32)
  std::cerr << "This example is Windows-only (LoadLibrary/GetProcAddress).\n";
  return 1;
#else
  HMODULE mod = LoadLibraryA(dll_path.c_str());
  if (!mod) {
    std::cerr << "LoadLibrary failed: " << dll_path << "\n";
    return 1;
  }

  auto get = [&](const char* name) -> FARPROC {
    FARPROC p = GetProcAddress(mod, name);
    if (!p) {
      std::cerr << "GetProcAddress failed: " << name << "\n";
    }
    return p;
  };

  using kvd_create_fn = kvd_handle* (KVD_CALL*)(const kvd_config*);
  using kvd_destroy_fn = void (KVD_CALL*)(kvd_handle*);
  using kvd_scan_path_fn = int (KVD_CALL*)(kvd_handle*, const char*, char**, size_t*);
  using kvd_scan_bytes_fn = int (KVD_CALL*)(kvd_handle*, const unsigned char*, size_t, char**, size_t*);
  using kvd_free_fn = void (KVD_CALL*)(char*);

  auto kvd_create_p = reinterpret_cast<kvd_create_fn>(get("kvd_create"));
  auto kvd_destroy_p = reinterpret_cast<kvd_destroy_fn>(get("kvd_destroy"));
  auto kvd_scan_path_p = reinterpret_cast<kvd_scan_path_fn>(get("kvd_scan_path"));
  auto kvd_scan_bytes_p = reinterpret_cast<kvd_scan_bytes_fn>(get("kvd_scan_bytes"));
  auto kvd_free_p = reinterpret_cast<kvd_free_fn>(get("kvd_free"));

  if (!kvd_create_p || !kvd_destroy_p || !kvd_scan_path_p || !kvd_scan_bytes_p || !kvd_free_p) {
    return 1;
  }

  kvd_config cfg{};
  if (!model_path.empty()) cfg.model_path = model_path.c_str();
  if (!model_normal_path.empty()) cfg.model_normal_path = model_normal_path.c_str();
  if (!model_packed_path.empty()) cfg.model_packed_path = model_packed_path.c_str();
  if (!family_path.empty()) cfg.family_classifier_json_path = family_path.c_str();
  if (!allowed_root.empty()) cfg.allowed_scan_root = allowed_root.c_str();

  unsigned int max_file_size = 0;
  if (parse_u32(max_file_size_s, max_file_size)) cfg.max_file_size = max_file_size;

  float threshold = 0.0f;
  if (parse_f32(threshold_s, threshold)) cfg.prediction_threshold = threshold;

  kvd_handle* h = kvd_create_p(&cfg);
  if (!h) {
    std::cerr << "kvd_create failed\n";
    return 1;
  }

  char* out_json = nullptr;
  size_t out_len = 0;
  int rc = kvd_scan_path_p(h, target.c_str(), &out_json, &out_len);
  if (rc != 0) {
    std::cerr << "kvd_scan_path failed: " << rc << "\n";
    kvd_destroy_p(h);
    return 1;
  }

  if (out_json) {
    std::cout.write(out_json, static_cast<std::streamsize>(out_len));
    std::cout << "\n";
    kvd_free_p(out_json);
  }

  kvd_destroy_p(h);
  return 0;
#endif
}
