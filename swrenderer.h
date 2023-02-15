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
    VertexAttributeTypes vsOutputPosType = VertexAttributeTypes::Float;
    VertexAttributeTypes vsOutputUvType = VertexAttributeTypes::Float;
    VertexAttributeTypes vsOutputColorType = VertexAttributeTypes::Float;
    std::vector<VertexDataDescriptor> vsInputDesc;
    std::vector<VertexDataDescriptor> vsOutputDesc;
    std::vector<VertexDataDescriptor> psInputDesc;

    uint32_t inputVertexAttributes;

    pro::proxy<VertexShaderFacade> vertexProgram;
    pro::proxy<PixelShaderFacade> pixelProgram;

    bool vsOutputsUv = false;
    bool vsOutputsColor = false;
    std::map<int, int> psVsIndexMap;
};

template<CanvasDrawable TCanvas>
class SWRenderer
{
    TCanvas canvas;
    int width = 500;
    int height = 500;
    int multisampleLevel = 0;
    
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
    static ProgramContext LinkProgram(
        pro::proxy<VertexShaderFacade> vp,
        pro::proxy<PixelShaderFacade> pp) noexcept;

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
    void SetMultiSampleLevel(int level) noexcept { multisampleLevel = level; }
    
    auto& Canvas() noexcept
    {
        return canvas;
    }

private:
    std::size_t GetNumberOfSubsamples() const noexcept;
    void GenerateSubsamples(glm::vec3 pt, std::vector<glm::vec3>& subsamples);
    void ClearBuffer(std::unique_ptr<float[]>& buffer, std::size_t nElement, float value);
};

#include "swrenderer.inl"
