#pragma once
#include "canvas.h"
#include "shading/light/light.h"
#include "shading/vertex_program.h"
#include "shading/pixel_program.h"
#include "model/model_data.h"
#include "shading/material/simple_vertex_program.h"
#include "shading/material/phong_material.h"
#include "shading/light/point_light.h"
#include "shading/light/ambient_light.h"
#include "shading/material/blinn_material.h"

class SWRenderer
{
    int width = 500;
    int height = 500;
    HDC hdc;
    HDC memDc;
    std::unique_ptr<Canvas> canvas[2];
    HBITMAP bitmaps[2];
    std::uint8_t *buffer[2];
    long bufferIndex = 0;
    float *zBuffer = nullptr;
    bool depthTestEnabled = true;
    bool depthWriteEnabled = true;
    bool backFaceCulling = true;

    std::shared_ptr<std::uint8_t[]> textureData;
    int textureW = 0;
    int textureH = 0;
    int textureChannels = 0;

    void ClearZBuffer();

    ModelData modelData;
    SimpleVertexProgram vertexProgram;
    BlinnMaterial pixelProgram;
    PointLight pointLight;
    AmbientLight ambientLight;

    glm::vec3 cubeRotation = glm::vec3{0, 0, 0};
    HWND hwnd = 0;
    bool mouseCaptured = false;
    int lastMouseX = -1;
    int lastMouseY = -1;

    std::list<float> stats;

    std::vector<void *> vsInput;
    std::vector<std::function<void(void *)>> vsInputDeleters;
    std::vector<void *> vsOutput;
    std::vector<std::function<void(void *)>> vsOutputDeleters;
    std::vector<std::function<void(void *)>> psInputDeleters;
    int vsOutputPosIdx = 0;
    int vsOutputUvIdx = 0;
    int vsOutputColorIdx = 0;
    int vsInputPosIdx = 0;
    int vsInputUvIdx = 0;
    int vsInputNormalIdx = 0;
    int vsInputColorIdx = 0;
    int psInputUvIdx = -1;
    int psInputColorIdx = -1;
    int psInputNormalIdx = -1;
    VertexAttributeTypes vsOutputPosType;
    VertexAttributeTypes vsOutputUvType;
    VertexAttributeTypes vsOutputColorType;

    std::vector<VertexDataDescriptor> vsInputDesc;
    std::vector<VertexDataDescriptor> vsOutputDesc;
    std::vector<VertexDataDescriptor> psInputDesc;

    uint32_t vertexAttributes;
    uint32_t inputVertexAttributes;
    VertexFunction vertexEntry;
    PixelFunction pixelEntry;

    bool vsOutputsUv = false;
    bool vsOutputsColor = false;
    std::map<int, int> vsPsIndexMap;

    void SetProgram();

public:
    void CreateBuffer(int pixelFormat);
    SWRenderer(HDC hdc, int w, int h);
    SWRenderer(const SWRenderer &) = delete;
    ~SWRenderer() noexcept;
    SWRenderer &operator=(const SWRenderer &) = delete;
    void SwapBuffer();
    void Render(float timeElapsed);
    HBITMAP GetBitmap() const;
    void SetHWND(HWND hwnd);
    void MouseDown();
    void MouseUp();
    void MouseMove(int x, int y);
};
