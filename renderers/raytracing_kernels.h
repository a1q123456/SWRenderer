#pragma once
#include <glm/glm.hpp>
#include <vector>
#include <array>
#include <memory>
#include <string_view>
#include <algorithm>
#include <iterator>
#include <stdexcept>
#include "cuda-support/cuda_utils.h"
#include "utils.h"
#include "model/model_data.h"


__device__ Ray generateRay(glm::mat4x4 iproj, glm::mat4x4 iviewTransform, int w, int h);

__global__ void renderRay(glm::mat4x4 iproj, glm::mat4x4 iviewTransform, int w, int h, std::uint8_t* dst);

cudaError_t renderFrame(glm::mat4x4 iproj, glm::mat4x4 iviewTransform, int w, int h, std::uint8_t* dst);
