cmake_minimum_required(VERSION 3.16)

project(ImagickCLI LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc highgui)

add_executable(imagick
    src/main.cpp
    src/ImageLoader.cpp
    src/ImageOps.cpp
)

target_include_directories(imagick
    PRIVATE
        include
        ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(imagick
    PRIVATE
        ${OpenCV_LIBS}
)

if(MSVC)
    target_compile_options(imagick PRIVATE /W4 /permissive-)
else()
    target_compile_options(imagick PRIVATE -Wall -Wextra -pedantic)
endif()
