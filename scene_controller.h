#pragma once
#include "icanvas.h"
#include "swrenderer.h"
#include "shading/material/simple_vertex_program.h"
#include "shading/material/blinn_material.h"
#include "native_window_handle.h"
#include "pixel_format.h"

template<CanvasDrawable TCanvas>
class SceneController
{
    int width = 500;
    int height = 500;

    std::unique_ptr<std::uint8_t, void(*)(void*)> textureData;
    int textureW = 0;
    int textureH = 0;
    int textureChannels = 0;
    PointLight pointLight;
    AmbientLight ambientLight;
    float cameraDistance = 3.f;
    glm::vec3 cubeRotation = glm::vec3{0, 0, 0};
    NativeWindowHandle hwnd = 0;
    bool mouseCaptured = false;
    int lastMouseX = -1;
    int lastMouseY = -1;

    glm::mat4 projectionMatrix;
    SWRenderer<TCanvas> renderer;
    BlinnMaterial pixelProgram;
    SimpleVertexProgram vertexProgram;
    ModelData modelData;
    ProgramContext programCtx;

public:
    template<CanvasDrawable T>
    SceneController(T&&);
    
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

#include "scene_controller.inl"