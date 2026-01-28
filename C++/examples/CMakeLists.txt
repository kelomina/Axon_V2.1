add_executable(kvd_scan
  scan_path.cpp
)

target_link_libraries(kvd_scan
  PRIVATE
    kvd
)

add_custom_command(TARGET kvd_scan POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:kvd> $<TARGET_FILE_DIR:kvd_scan>
  COMMAND ${CMAKE_COMMAND} -E copy_if_different "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/bin/LIEF.dll" $<TARGET_FILE_DIR:kvd_scan>
  COMMAND ${CMAKE_COMMAND} -E copy_if_different "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/bin/lib_lightgbm.dll" $<TARGET_FILE_DIR:kvd_scan>
  COMMAND ${CMAKE_COMMAND} -E copy_if_different "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/bin/fmt.dll" $<TARGET_FILE_DIR:kvd_scan>
  COMMAND ${CMAKE_COMMAND} -E copy_if_different "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/bin/spdlog.dll" $<TARGET_FILE_DIR:kvd_scan>
)

add_executable(kvd_scan_loader
  kvd_scan_loader.cpp
)

target_include_directories(kvd_scan_loader
  PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

add_custom_command(TARGET kvd_scan_loader POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:kvd> $<TARGET_FILE_DIR:kvd_scan_loader>
  COMMAND ${CMAKE_COMMAND} -E copy_if_different "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/bin/LIEF.dll" $<TARGET_FILE_DIR:kvd_scan_loader>
  COMMAND ${CMAKE_COMMAND} -E copy_if_different "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/bin/lib_lightgbm.dll" $<TARGET_FILE_DIR:kvd_scan_loader>
  COMMAND ${CMAKE_COMMAND} -E copy_if_different "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/bin/fmt.dll" $<TARGET_FILE_DIR:kvd_scan_loader>
  COMMAND ${CMAKE_COMMAND} -E copy_if_different "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/bin/spdlog.dll" $<TARGET_FILE_DIR:kvd_scan_loader>
)
