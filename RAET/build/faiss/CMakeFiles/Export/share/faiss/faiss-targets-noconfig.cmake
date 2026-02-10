#----------------------------------------------------------------
# Generated CMake target import file.
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "faiss" for configuration ""
set_property(TARGET faiss APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(faiss PROPERTIES
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libfaiss.so"
  IMPORTED_SONAME_NOCONFIG "libfaiss.so"
  )

list(APPEND _IMPORT_CHECK_TARGETS faiss )
list(APPEND _IMPORT_CHECK_FILES_FOR_faiss "${_IMPORT_PREFIX}/lib/libfaiss.so" )

# Import target "faiss_avx2" for configuration ""
set_property(TARGET faiss_avx2 APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(faiss_avx2 PROPERTIES
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libfaiss_avx2.so"
  IMPORTED_SONAME_NOCONFIG "libfaiss_avx2.so"
  )

list(APPEND _IMPORT_CHECK_TARGETS faiss_avx2 )
list(APPEND _IMPORT_CHECK_FILES_FOR_faiss_avx2 "${_IMPORT_PREFIX}/lib/libfaiss_avx2.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
