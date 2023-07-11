cmake_minimum_required(VERSION 3.17)

project(swengine LANGUAGES CUDA CXX)

add_executable(swengine WIN32  
"main.cc" 
"stbi.cc"
"pixel_format.cc"
"swrenderer.cc"
"scene_controller.cc"
"raytracing_renderer.cc"
"raytracing_kernels.cu"
"headless_canvas.cc"
"shading/material/pbr_material.cu"
"shading/material/simple_vertex_program.cc"
"shading/material/phong_material.cc"
"shading/material/blinn_material.cc"
"shading/light/point_light.cc"
"image-processing/rescaling.cc"
)

if (WIN32)
    target_sources(swengine PRIVATE
        "win32_window.cc" 
        "win32exception.cc" 
        "win32_canvas.cc"
        "exception/throw.cc"
    )
endif()

target_compile_definitions(swengine PRIVATE NOMINMAX)
target_precompile_headers(swengine PRIVATE pch.h)

find_package(glm CONFIG REQUIRED)
target_link_libraries(swengine PRIVATE glm::glm)
find_package(Stb REQUIRED)
target_include_directories(swengine PRIVATE Stb_INCLUDE_DIR ${PROJECT_SOURCE_DIR} ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_compile_options(swengine PRIVATE $<$<AND:$<CXX_COMPILER_ID:MSVC>,$<COMPILE_LANGUAGE:CXX>>:/openmp>)
target_compile_options(swengine PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--diag-suppress 20012,3012 -std=c++20>)
target_compile_definitions(swengine PRIVATE $<BOOL:${WIN32}>:WIN32>)
set_target_properties(swengine PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CUDA_STANDARD 12
    CUDA_STANDARD_REQUIRED ON
)

