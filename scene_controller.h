#pragma once
#include "swrenderer.h"
#include "shading/material/simple_vertex_program.h"
#include "shading/material/blinn_material.h"

class SceneController
{
    int width = 500;
    int height = 500;

    std::shared_ptr<std::uint8_t> textureData;
    int textureW = 0;
    int textureH = 0;
    int textureChannels = 0;
    PointLight pointLight;
    AmbientLight ambientLight;
    float cameraDistance = 3.f;
    glm::vec3 cubeRotation = glm::vec3{0, 0, 0};
    HWND hwnd = 0;
    bool mouseCaptured = false;
    int lastMouseX = -1;
    int lastMouseY = -1;

    glm::mat4 projectionMatrix;
    SWRenderer renderer;
    BlinnMaterial pixelProgram;
    SimpleVertexProgram vertexProgram;
    ModelData modelData;
    ProgramContext programCtx;

public:
    SceneController(HDC hdc, int w, int h);
    
    void CreateBuffer(int pixelFormat) { renderer.CreateBuffer(pixelFormat); }
    void SwapBuffer() { renderer.SwapBuffer(); }
    HBITMAP GetBitmap() const { return renderer.GetBitmap(); }
    void SetHWND(HWND hwnd);

    void MouseDown();

    void MouseUp();

    void MouseWheel(int val);

    void MouseMove(int x, int y);

    void Render(float timeElapsed);
};
