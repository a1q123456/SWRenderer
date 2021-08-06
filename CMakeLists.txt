cmake_minimum_required(VERSION 3.17)

project(swengine)

add_executable(swengine WIN32  "main.cc" "window.cc" "win32exception.cc" "swrenderer.cc" "canvas.cc")

target_compile_definitions(swengine PRIVATE NOMINMAX)
target_precompile_headers(swengine PRIVATE pch.h)
find_package(glm CONFIG REQUIRED)
target_link_libraries(swengine PRIVATE glm::glm)
find_package(Stb REQUIRED)
target_include_directories(swengine PRIVATE Stb_INCLUDE_DIR)

set_target_properties(swengine PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)
