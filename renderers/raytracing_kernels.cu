#include "raytracing_kernels.h"

__device__ void float4_to_byte4(glm::vec4 src, std::uint8_t* dst)
{
    dst[0] = glm::clamp(src.b * 255.f, 0.f, 255.f);
    dst[1] = glm::clamp(src.g * 255.f, 0.f, 255.f);
    dst[2] = glm::clamp(src.r * 255.f, 0.f, 255.f);
    dst[3] = glm::clamp(src.a * 255.f, 0.f, 255.f);
}

__device__ Ray generateRay(glm::mat4x4 iproj, glm::mat4x4 iviewTransform, int w, int h)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    glm::vec4 nearPoint{ static_cast<float>(x) / static_cast<float>(w), static_cast<float>(y) / static_cast<float>(h), 0, 1 };
    glm::vec4 farPoint{ static_cast<float>(x) / static_cast<float>(w), static_cast<float>(y) / static_cast<float>(h), 1, 1 };

    nearPoint = iproj * iviewTransform * nearPoint;
    farPoint = iproj * iviewTransform * farPoint;
    
    Ray ray{glm::vec3{farPoint - nearPoint}, glm::vec3{nearPoint}};

    return ray;
}

__global__ void renderRay(glm::mat4x4 iproj, glm::mat4x4 iviewTransform, cuda::std::span<Renderable> renderables, int w,
                          int h, std::uint8_t* dst)
{
    Ray ray = generateRay(iproj, iviewTransform, w, h);

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    glm::vec3 rayColor;
    for (auto& renderable : renderables)
    {
        if (!renderable.RayInterects(ray))
        {
            continue;
        }

        for (auto& triangle : renderable.triangles)
        {
            glm::vec3 intersection;
            if (!triangle.Interect(ray, intersection))
            {
                continue;
            }
            // TODO: calculate UV
            // TODO: get diffuse color
            // TODO: accumulate the color of the ray
        }
    }

    float4_to_byte4(glm::vec4{1, 0, 0, 1}, dst + y * w * 4 + x * 4);
}

cudaError_t renderFrame(glm::mat4x4 iproj, glm::mat4x4 iviewTransform, cuda::std::span<Renderable> renderables, 
                        int w, int h, std::uint8_t* dst)
{
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(w / threadsPerBlock.x, h / threadsPerBlock.y);

    renderRay<<<numBlocks, threadsPerBlock>>>(iproj, iviewTransform, renderables, w, h, dst);
    return cudaGetLastError();
}

__global__ void transformModel(const Model& model, Renderable& renderable)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    auto p0 = model.modelData->GetVertexData<glm::vec3>(i * 3 + 0, VertexAttributes::Position);
    auto p1 = model.modelData->GetVertexData<glm::vec3>(i * 3 + 1, VertexAttributes::Position);
    auto p2 = model.modelData->GetVertexData<glm::vec3>(i * 3 + 2, VertexAttributes::Position);

    renderable.triangles[i].p0 = p0;
    renderable.triangles[i].p1 = p1;
    renderable.triangles[i].p2 = p2;

    for (int j = 0; j < 3; j++)
    {
        auto uv = model.modelData->GetVertexData<glm::vec3>(i * 3 + j, VertexAttributes::TextureCoordinate);
        for (int k = 0; k < 3; k++)
        {
            renderable.triangles[i].vsOutput[j].SetData(0, k, VertexAttributes::TextureCoordinate, uv[k]);
        }
    }

    for (int j = 0; j < 3; j++)
    {
        auto normal = model.modelData->GetVertexData<glm::vec3>(i * 3 + j, VertexAttributes::Normal);
        for (int k = 0; k < 3; k++)
        {
            renderable.triangles[i].vsOutput[j].SetData(0, k, VertexAttributes::Normal, normal[k]);
        }
    }
}

cudaError_t transformVertexes(cuda::std::span<Model> models, cuda::std::span<Renderable> renderables)
{
    size_t index = 0;
    for (auto& model : models)
    {
        int threadsPerBlock = 16;
        int numBlocks = model.modelData->GetNumberIndices() / 3 / threadsPerBlock;

        transformModel<<<numBlocks, threadsPerBlock>>>(model, renderables[index]);
        index++;
    }
    return cudaGetLastError();
}
