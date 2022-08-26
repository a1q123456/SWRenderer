#pragma once
#include "icanvas.h"
#include "shading/light/light.h"
#include "shading/vertex_program.h"
#include "shading/pixel_program.h"
#include "model/model_data.h"
#include "shading/material/simple_vertex_program.h"
#include "shading/material/phong_material.h"
#include "shading/light/point_light.h"
#include "shading/light/ambient_light.h"
#include "shading/material/blinn_material.h"
#include "pixel_format.h"

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

template<CanvasDrawable TCanvas>
class SWRenderer
{
    static constexpr auto DEPTH_THRESHOLD = 0.01;
    TCanvas canvas;
    int width = 500;
    int height = 500;
    int multisampleLevel = 8;
    
    std::unique_ptr<float[]> colorBuffer = nullptr;
    std::unique_ptr<float[]> zBuffer = nullptr;
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

    void CreateBuffer(EPixelFormat pixelFormat);
    template<CanvasDrawable T>
    SWRenderer(T&& canvas);
    SWRenderer(const SWRenderer &) = delete;
    SWRenderer &operator=(const SWRenderer &) = delete;

    std::unique_ptr<float[]>& GetColorBuffer() noexcept { return colorBuffer; }
    void ClearZBuffer();
    void ClearColorBuffer(std::uint32_t color);
    void SetProgram(ProgramContext& programCtx);
    void SetMesh(ModelData &mesh);
    void SetViewMatrix(const glm::mat4 &view);
    void ProjectionMatrix(const glm::mat4 &proj);
    void Draw(float timeElapsed);
    
    auto& Canvas() noexcept
    {
        return canvas;
    }

private:
    std::vector<glm::vec3> GenerateSubsamples(glm::vec3 pt);
    void ClearBuffer(std::unique_ptr<float[]>& buffer, std::size_t nElement, float value);
};

#include "swrenderer.inl"
