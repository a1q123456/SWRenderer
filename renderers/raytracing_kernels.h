#pragma once
#include <glm/glm.hpp>
#include <vector>
#include <array>
#include <memory>
#include <string_view>
#include <algorithm>
#include <iterator>
#include <stdexcept>
#include <cuda/std/span>
#include "cuda-support/cuda_utils.h"
#include "utils.h"
#include "model/model_data.h"
#include "renderers/renderrable.h"

cudaError_t renderFrame(glm::mat4x4 iproj, glm::mat4x4 iviewTransform, cuda::std::span<Renderable> renderables, int w,
                        int h, std::uint8_t* dst);

cudaError_t transformVertexes(cuda::std::span<Model> models, cuda::std::span<Renderable> renderables);
