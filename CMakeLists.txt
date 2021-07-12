cmake_minimum_required(VERSION 3.17)

project(swengine)

add_executable(swengine WIN32  "main.cc" "window.cc" "win32exception.cc" "swrenderer.cc" "canvas.cc")

target_compile_definitions(swengine PRIVATE NOMINMAX)
target_precompile_headers(swengine PRIVATE pch.h)


set_target_properties(swengine PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)
