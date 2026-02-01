# Find ONNX Runtime library
#
# Sets the following variables:
#   ONNXRUNTIME_FOUND        - True if ONNX Runtime was found
#   ONNXRUNTIME_INCLUDE_DIRS - Include directories for ONNX Runtime
#   ONNXRUNTIME_LIBRARIES    - Libraries to link
#   ONNXRUNTIME_VERSION      - Version of ONNX Runtime found
#
# Also creates the imported target ONNXRuntime::ONNXRuntime
#
# Hints:
#   ONNXRUNTIME_ROOT - Root directory of ONNX Runtime installation

# Allow users to specify ONNX Runtime location
set(ONNXRUNTIME_ROOT "" CACHE PATH "Root directory of ONNX Runtime installation")

# Search paths
set(_ONNXRUNTIME_SEARCH_PATHS
  ${ONNXRUNTIME_ROOT}
  $ENV{ONNXRUNTIME_ROOT}
  /usr/local
  /usr
  /opt/onnxruntime
)

# Find include directory
find_path(ONNXRUNTIME_INCLUDE_DIR
  NAMES onnxruntime_cxx_api.h
  PATHS ${_ONNXRUNTIME_SEARCH_PATHS}
  PATH_SUFFIXES
    include
    include/onnxruntime
    include/onnxruntime/core/session
)

# Find library
find_library(ONNXRUNTIME_LIBRARY
  NAMES onnxruntime
  PATHS ${_ONNXRUNTIME_SEARCH_PATHS}
  PATH_SUFFIXES
    lib
    lib64
    lib/x64
    lib/x86
)

# Try to get version from header
if(ONNXRUNTIME_INCLUDE_DIR)
  set(_version_file "${ONNXRUNTIME_INCLUDE_DIR}/onnxruntime_c_api.h")
  if(EXISTS "${_version_file}")
    file(STRINGS "${_version_file}" _version_line
      REGEX "^#define ORT_API_VERSION [0-9]+")
    if(_version_line)
      string(REGEX REPLACE "^#define ORT_API_VERSION ([0-9]+).*" "\\1" ONNXRUNTIME_VERSION "${_version_line}")
    endif()
  endif()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ONNXRuntime
  REQUIRED_VARS
    ONNXRUNTIME_LIBRARY
    ONNXRUNTIME_INCLUDE_DIR
  VERSION_VAR
    ONNXRUNTIME_VERSION
)

if(ONNXRUNTIME_FOUND)
  set(ONNXRUNTIME_LIBRARIES ${ONNXRUNTIME_LIBRARY})
  set(ONNXRUNTIME_INCLUDE_DIRS ${ONNXRUNTIME_INCLUDE_DIR})

  # Create imported target
  if(NOT TARGET ONNXRuntime::ONNXRuntime)
    add_library(ONNXRuntime::ONNXRuntime UNKNOWN IMPORTED)
    set_target_properties(ONNXRuntime::ONNXRuntime PROPERTIES
      IMPORTED_LOCATION "${ONNXRUNTIME_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIRS}"
    )
  endif()

  mark_as_advanced(ONNXRUNTIME_INCLUDE_DIR ONNXRUNTIME_LIBRARY)
endif()
