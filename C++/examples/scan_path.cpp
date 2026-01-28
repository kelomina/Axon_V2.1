#include "kvd/api.h"

#include <cstdlib>
#include <cstring>
#include <iostream>

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "Usage: kvd_scan <path>\n";
    return 2;
  }

  kvd_config cfg{};
  kvd_handle* h = kvd_create(&cfg);
  if (!h) {
    std::cerr << "kvd_create failed\n";
    return 1;
  }

  char* out_json = nullptr;
  size_t out_len = 0;
  int rc = kvd_scan_path(h, argv[1], &out_json, &out_len);
  if (rc != 0) {
    std::cerr << "kvd_scan_path failed: " << rc << "\n";
    kvd_destroy(h);
    return 1;
  }

  if (out_json) {
    std::cout.write(out_json, static_cast<std::streamsize>(out_len));
    std::cout << "\n";
    kvd_free(out_json);
  }

  kvd_destroy(h);
  return 0;
}
