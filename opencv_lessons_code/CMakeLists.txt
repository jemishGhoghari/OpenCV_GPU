cmake_minimum_required(VERSION 3.10)
project(OpenCVExample VERSION 0.1.0)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(lesson_1 lesson_1.cpp)
add_executable(lesson_2 lesson_2.cpp)
add_executable(calculate_histogram histogram_calculation.cpp)
add_executable(shi-tomasi shi-tomasi.cpp)

target_link_libraries(lesson_1 ${OpenCV_LIBS})
target_link_libraries(lesson_2 ${OpenCV_LIBS})
target_link_libraries(calculate_histogram ${OpenCV_LIBS})
target_link_libraries(shi-tomasi ${OpenCV_LIBS})

# Copy the image file next to the executable after build
add_custom_command(TARGET lesson_1 POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_SOURCE_DIR}/lena.png
            $<TARGET_FILE_DIR:lesson_1>)

# Optional: check if OpenCV was built with CUDA
if ("${OpenCV_CUDA_VERSION}" MATCHES "^[0-9]")
    message(STATUS "OpenCV built with CUDA ${OpenCV_CUDA_VERSION}")
    target_compile_definitions(lesson_1 PRIVATE HAVE_CUDA)
    target_compile_definitions(lesson_2 PRIVATE HAVE_CUDA)
    target_compile_definitions(calculate_histogram PRIVATE HAVE_CUDA)
    target_compile_definitions(shi-tomasi PRIVATE HAVE_CUDA)
else()
    message(STATUS "OpenCV built without CUDA")
endif()
