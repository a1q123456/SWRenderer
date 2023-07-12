#pragma once
#include "model/model_data.h"
#include "renderers/raytracing_renderer.h"
#include "renderers/triangle.h"
#include "utils.h"

struct Model
{
    CudaModelData* modelData = nullptr;
    RayTracingProgramContext* programContext = nullptr;
};

struct Renderable
{
    glm::vec3 min;
    glm::vec3 max;

    cuda::std::span<Triangle> triangles;
    RayTracingProgramContext* programContext = nullptr;

    __host__ __device__ bool RayInterects(const Ray& ray) const noexcept
    {
        // TODO
        return true;
    }
};

