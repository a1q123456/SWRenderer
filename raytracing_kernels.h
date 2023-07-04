#pragma once
#include <glm/glm.hpp>
#include "utils.h"

#ifdef __INTELLISENSE__ 
#define __global__
#define __device__
struct
{
    int x, y;
} threadIdx, blockIdx, blockDim;
#endif

__device__ Ray generateRay(glm::mat4x4 iproj, glm::mat4x4 iviewTransform, int w, int h);

__global__ void renderRay(glm::mat4x4 iproj, glm::mat4x4 iviewTransform, int w, int h, float* dst);
