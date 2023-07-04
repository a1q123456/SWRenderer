#include "raytracing_kernels.h"

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

__global__ void renderRay(glm::mat4x4 iproj, glm::mat4x4 iviewTransform, int w, int h, float* dst)
{
    Ray ray = generateRay(iproj, iviewTransform, w, h);
}
