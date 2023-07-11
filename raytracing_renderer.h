#pragma once
#include "platform_defines.h"
#include "model/model_data.h"
#include "pixel_format.h"
#include "shading/vertex_program.h"
#include "shading/pixel_program.h"
#include "cuda_utils.h"

class RayTracingProgramContext
{

};

class RayTracingRenderer
{
public:
    using ProgramContextType = RayTracingProgramContext;
    using ModelDataType = CudaModelData;

    RayTracingRenderer(CanvasType&& canvas);

    void CreateBuffer(EPixelFormat pixelFormat);
    void AddMesh(ModelDataType* mesh, ProgramContextType& programCtx);
    void ClearZBuffer();
    void ClearColorBuffer(std::uint32_t color);
    void Draw(float timeElapsed);
    CanvasType& Canvas();
    
    void SetMultiSampleLevel(int msaa);
    void ProjectionMatrix(glm::mat4x4 proj);

    static ProgramContextType LinkProgram(PixelProgram* pp) noexcept;

private:
    CudaPointer<std::uint8_t[]> colorBufferU8;
    CudaPointer<std::uint8_t[]> depthBufferU8;
    CanvasType canvas;
    int width = 0;
    int height = 0;
    glm::mat4 iviewMatrix;
    glm::mat4 iprojMatrix;
    ModelDataType* modelData = nullptr;
};
