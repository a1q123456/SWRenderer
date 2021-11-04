cmake_minimum_required(VERSION 3.17)

project(swengine)

add_executable(swengine WIN32  
"main.cc" 
"stbi.cc"
"raytracing_renderer.cc"
"headless_window.cc"
"headless_canvas.cc"
"model/model_data.cc" 
"shading/material/simple_vertex_program.cc"
"shading/material/phong_material.cc"
"shading/material/blinn_material.cc"
"shading/light/point_light.cc"
)

if (WIN32)
    target_sources(swengine PRIVATE
        "win32_window.cc" 
        "win32exception.cc" 
        "win32_canvas.cc"
    )
endif()

target_compile_definitions(swengine PRIVATE NOMINMAX)
target_precompile_headers(swengine PRIVATE pch.h)
find_package(glm CONFIG REQUIRED)
target_link_libraries(swengine PRIVATE glm::glm)
find_package(Stb REQUIRED)
target_include_directories(swengine PRIVATE Stb_INCLUDE_DIR ${PROJECT_SOURCE_DIR})
target_compile_options(swengine PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/openmp /std:c++latest>)

if (NOT WIN32)
    set_target_properties(swengine PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
    )
endif()

target_compile_definitions(swengine PUBLIC SWR_HEADLESS_RENDERING)
