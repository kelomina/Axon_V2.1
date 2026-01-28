#include "sha256.h"

#include <windows.h>
#include <bcrypt.h>

#include <array>
#include <vector>

#pragma comment(lib, "bcrypt.lib")

namespace kvd {

std::uint8_t sha256_last_byte_utf8(const std::string& s) {
  BCRYPT_ALG_HANDLE alg = nullptr;
  if (BCryptOpenAlgorithmProvider(&alg, BCRYPT_SHA256_ALGORITHM, nullptr, 0) != 0) {
    return 0;
  }

  DWORD hash_object_len = 0;
  DWORD data_len = 0;
  if (BCryptGetProperty(alg, BCRYPT_OBJECT_LENGTH, reinterpret_cast<PUCHAR>(&hash_object_len), sizeof(hash_object_len), &data_len, 0) != 0) {
    BCryptCloseAlgorithmProvider(alg, 0);
    return 0;
  }

  std::vector<std::uint8_t> hash_object(hash_object_len);
  BCRYPT_HASH_HANDLE hash = nullptr;
  if (BCryptCreateHash(alg, &hash, hash_object.data(), hash_object_len, nullptr, 0, 0) != 0) {
    BCryptCloseAlgorithmProvider(alg, 0);
    return 0;
  }

  if (!s.empty()) {
    if (BCryptHashData(hash, reinterpret_cast<PUCHAR>(const_cast<char*>(s.data())), static_cast<ULONG>(s.size()), 0) != 0) {
      BCryptDestroyHash(hash);
      BCryptCloseAlgorithmProvider(alg, 0);
      return 0;
    }
  }

  std::array<std::uint8_t, 32> digest{};
  digest.fill(0);
  if (BCryptFinishHash(hash, digest.data(), static_cast<ULONG>(digest.size()), 0) != 0) {
    BCryptDestroyHash(hash);
    BCryptCloseAlgorithmProvider(alg, 0);
    return 0;
  }

  BCryptDestroyHash(hash);
  BCryptCloseAlgorithmProvider(alg, 0);
  return digest.back();
}

}
