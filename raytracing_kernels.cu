#include "raytracing_kernels.h"

__device__ void float4_to_byte3(glm::vec4 src, std::uint8_t* dst)
{
    dst[0] = glm::clamp(src.r * 255.f, 0.f, 255.f);
    dst[1] = glm::clamp(src.g * 255.f, 0.f, 255.f);
    dst[2] = glm::clamp(src.b * 255.f, 0.f, 255.f);
}

__device__ Ray generateRay(glm::mat4x4 iproj, glm::mat4x4 iviewTransform, int w, int h)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    glm::vec4 near{ static_cast<float>(x) / static_cast<float>(w), static_cast<float>(y) / static_cast<float>(h), 0, 1 };
    glm::vec4 far{ static_cast<float>(x) / static_cast<float>(w), static_cast<float>(y) / static_cast<float>(h), 1, 1 };

    near = iproj * iviewTransform * near;
    far = iproj * iviewTransform * far;
    
    Ray ray
    {
        glm::vec3{far - near},
        glm::vec3{near}
    };

    return ray;
}

__global__ void renderRay(glm::mat4x4 iproj, glm::mat4x4 iviewTransform, int w, int h, std::uint8_t* dst)
{
    Ray ray = generateRay(iproj, iviewTransform, w, h);

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    dst[y * w * 4 + x * 4 + 0] = 0;
    dst[y * w * 4 + x * 4 + 1] = 0;
    dst[y * w * 4 + x * 4 + 2] = 255;
}

cudaError_t renderFrame(glm::mat4x4 iproj, glm::mat4x4 iviewTransform, int w, int h, std::uint8_t* dst)
{
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(w / threadsPerBlock.x, h / threadsPerBlock.y);

    renderRay<<<numBlocks, threadsPerBlock>>>(iproj, iviewTransform, w, h, dst);
    return cudaGetLastError();
}

