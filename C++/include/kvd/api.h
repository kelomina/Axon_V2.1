#pragma once

#include <stddef.h>

#if defined(_WIN32)
  #if defined(KVD_BUILD_DLL)
    #define KVD_API __declspec(dllexport)
  #else
    #define KVD_API __declspec(dllimport)
  #endif
#else
  #define KVD_API
#endif

#if defined(_WIN32)
  #define KVD_CALL __cdecl
#else
  #define KVD_CALL
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct kvd_handle kvd_handle;

typedef struct kvd_config {
  const char* model_path;
  const char* model_normal_path;
  const char* model_packed_path;
  const char* family_classifier_json_path;
  const char* allowed_scan_root;
  unsigned int max_file_size;
  float prediction_threshold;
} kvd_config;

KVD_API kvd_handle* KVD_CALL kvd_create(const kvd_config* config);
KVD_API void KVD_CALL kvd_destroy(kvd_handle* handle);

KVD_API int KVD_CALL kvd_scan_path(kvd_handle* handle, const char* path, char** out_json, size_t* out_len);
KVD_API int KVD_CALL kvd_scan_bytes(kvd_handle* handle, const unsigned char* bytes, size_t len, char** out_json, size_t* out_len);
KVD_API void KVD_CALL kvd_free(char* p);

#ifdef __cplusplus
}
#endif
