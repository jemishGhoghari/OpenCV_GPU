cmake_minimum_required(VERSION 3.10)
project(OpenCVExample VERSION 0.1.0)

include(CTest)
enable_testing()

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(lesson_1 lesson_1.cpp)

target_link_libraries(lesson_1 ${OpenCV_LIBS})

# Copy the image file next to the executable after build
add_custom_command(TARGET lesson_1 POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_SOURCE_DIR}/lena.png
            $<TARGET_FILE_DIR:lesson_1>)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

# Optional: check if OpenCV was built with CUDA
if ("${OpenCV_CUDA_VERSION}" MATCHES "^[0-9]")
    message(STATUS "OpenCV built with CUDA ${OpenCV_CUDA_VERSION}")
    target_compile_definitions(lesson_1 PRIVATE HAVE_CUDA)
else()
    message(STATUS "OpenCV built without CUDA")
endif()
