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

struct ProgramContext
{
    int vsOutputPosIdx = 0;
    int vsOutputUvIdx = 0;
    int vsOutputColorIdx = 0;
    VertexAttributeTypes vsOutputPosType;
    VertexAttributeTypes vsOutputUvType;
    VertexAttributeTypes vsOutputColorType;
    std::vector<VertexDataDescriptor> vsInputDesc;
    std::vector<VertexDataDescriptor> vsOutputDesc;
    std::vector<VertexDataDescriptor> psInputDesc;

    uint32_t inputVertexAttributes;
    VertexFunction vertexEntry;
    PixelFunction pixelEntry;

    VertexProgram *vertexProgram = nullptr;
    PixelProgram *pixelProgram = nullptr;

    bool vsOutputsUv = false;
    bool vsOutputsColor = false;
    std::map<int, int> psVsIndexMap;
};

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

    glm::mat4 viewTransform;
    glm::mat4 projectionMatrix;

    ModelData *modelData = nullptr;
    ProgramContext* programCtx = nullptr;

    std::list<float> stats;


public:
    static ProgramContext LinkProgram(VertexProgram &vp, PixelProgram &pp) noexcept;

    void CreateBuffer(int pixelFormat);
    SWRenderer(HDC hdc, int w, int h);
    SWRenderer(const SWRenderer &) = delete;
    SWRenderer &operator=(const SWRenderer &) = delete;
    void SwapBuffer();
    HBITMAP GetBitmap() const;

    std::uint8_t* GetColorBuffer() noexcept { return buffer[bufferIndex]; }
    void ClearZBuffer();
    void ClearColorBuffer(std::uint32_t color);
    void SetProgram(ProgramContext& programCtx);
    void SetMesh(ModelData &mesh);
    void SetViewMatrix(const glm::mat4 &view);
    void ProjectionMatrix(const glm::mat4 &proj);
    void Draw(float timeElapsed);
};
