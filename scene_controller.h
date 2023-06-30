#pragma once
#include "platform_defines.h"
#include "swrenderer.h"
#include "raytracing_renderer.h"
#include "shading/material/simple_vertex_program.h"
#include "shading/material/blinn_material.h"
#include "native_window_handle.h"
#include "pixel_format.h"

class SceneController
{
    using RendererType = RayTracingRenderer;

    int width = 500;
    int height = 500;

    std::unique_ptr<std::uint8_t, void(*)(void*)> textureData;
    int textureW = 0;
    int textureH = 0;
    int textureChannels = 0;
    PointLight pointLight;
    AmbientLight ambientLight;
    float cameraDistance = 3.f;
    glm::vec3 cubeRotation = glm::vec3{-0.159999922, 0.0300000049, 0.0300000049};
    NativeWindowHandle hwnd = 0;
    bool mouseCaptured = false;
    int lastMouseX = -1;
    int lastMouseY = -1;
    int mouseX = -1;
    int mouseY = -1;

    glm::mat4 projectionMatrix;
    RendererType renderer;
    BlinnMaterial pixelProgram;
    SimpleVertexProgram vertexProgram;
    ModelData modelData;
    RendererType::ProgramContextType programCtx;

public:
    SceneController(CanvasType&& canvas);
    
    void CreateBuffer(EPixelFormat pixelFormat) { renderer.CreateBuffer(pixelFormat); }
    void SetHWND(NativeWindowHandle hwnd);

    void MouseDown();

    void MouseUp();

    void MouseWheel(int val);

    void MouseMove(int x, int y);

    void Render(float timeElapsed);

    auto& Canvas() noexcept
    {
        return renderer.Canvas();
    }
};
