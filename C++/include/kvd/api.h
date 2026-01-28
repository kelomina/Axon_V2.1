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

typedef enum kvd_model_check_result {
  KVD_MODEL_OK = 0,
  KVD_MODEL_ERR_INVALID_ARGUMENT = -1,
  KVD_MODEL_ERR_MAIN_MISSING = -10,
  KVD_MODEL_ERR_MAIN_INVALID = -11,
  KVD_MODEL_ERR_ROUTE_INCOMPLETE = -12,
  KVD_MODEL_ERR_NORMAL_MISSING = -13,
  KVD_MODEL_ERR_NORMAL_INVALID = -14,
  KVD_MODEL_ERR_PACKED_MISSING = -15,
  KVD_MODEL_ERR_PACKED_INVALID = -16,
  KVD_MODEL_ERR_FAMILY_MISSING = -17,
  KVD_MODEL_ERR_FAMILY_INVALID = -18,
  KVD_MODEL_ERR_OOM = -100
} kvd_model_check_result;

KVD_API kvd_handle* KVD_CALL kvd_create(const kvd_config* config);
KVD_API void KVD_CALL kvd_destroy(kvd_handle* handle);

KVD_API int KVD_CALL kvd_scan_path(kvd_handle* handle, const char* path, char** out_json, size_t* out_len);
KVD_API int KVD_CALL kvd_scan_bytes(kvd_handle* handle, const unsigned char* bytes, size_t len, char** out_json, size_t* out_len);
KVD_API void KVD_CALL kvd_free(char* p);
KVD_API int KVD_CALL kvd_validate_models(const kvd_config* config, char** out_error, size_t* out_len);

#ifdef __cplusplus
}
#endif
