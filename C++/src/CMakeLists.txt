add_library(kvd SHARED)

target_sources(kvd
  PRIVATE
    api.cpp
    config.cpp
    features_byte_sequence.cpp
    features_file_attributes.cpp
    features_pe.cpp
    features_statistics.cpp
    lightgbm_infer.cpp
    malware_scanner.cpp
    family_classifier.cpp
    sha256.cpp
    util_filesystem.cpp
    kvd.def
)

target_include_directories(kvd
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

find_package(nlohmann_json CONFIG REQUIRED)
find_package(LIEF CONFIG REQUIRED)

find_path(LIGHTGBM_INCLUDE_DIR NAMES LightGBM/c_api.h)
find_library(LIGHTGBM_LIBRARY_RELEASE NAMES lib_lightgbm PATH_SUFFIXES lib)
find_library(LIGHTGBM_LIBRARY_DEBUG NAMES lib_lightgbm PATH_SUFFIXES debug/lib)

add_library(lightgbm::lightgbm UNKNOWN IMPORTED)
set_target_properties(lightgbm::lightgbm PROPERTIES
  IMPORTED_LOCATION_RELEASE "${LIGHTGBM_LIBRARY_RELEASE}"
  IMPORTED_LOCATION_DEBUG "${LIGHTGBM_LIBRARY_DEBUG}"
  INTERFACE_INCLUDE_DIRECTORIES "${LIGHTGBM_INCLUDE_DIR}"
)

target_link_libraries(kvd
  PUBLIC
    nlohmann_json::nlohmann_json
    LIEF::LIEF
    lightgbm::lightgbm
    bcrypt
)

target_compile_definitions(kvd PRIVATE KVD_BUILD_DLL=1)

if(WIN32)
  target_link_options(kvd PRIVATE "/DEF:${CMAKE_CURRENT_SOURCE_DIR}/kvd.def")
endif()
