cmake_minimum_required(VERSION 3.10)
project(OpenCVExample VERSION 0.1.0)

include(CTest)
enable_testing()

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(main main.cpp)

target_link_libraries(main ${OpenCV_LIBS})

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

# Optional: check if OpenCV was built with CUDA
if ("${OpenCV_CUDA_VERSION}" MATCHES "^[0-9]")
    message(STATUS "OpenCV built with CUDA ${OpenCV_CUDA_VERSION}")
    target_compile_definitions(main PRIVATE HAVE_CUDA)
else()
    message(STATUS "OpenCV built without CUDA")
endif()
