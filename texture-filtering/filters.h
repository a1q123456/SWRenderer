#pragma once
#include <glm/glm.hpp>
#include <vector>
#include <span>
#include <cstdint>

template<size_t Dim>
using TextureCoordinate = glm::vec<Dim, float, glm::qualifier::highp>;
using Texture1DCoordinate = TextureCoordinate<2>;
using Texture2DCoordinate = TextureCoordinate<3>;
using Texture3DCoordinate = TextureCoordinate<4>;

enum class ETextureFilteringMethods
{
    Nearest,
    Linear,
    Cubic,
};

#include "texture-filtering/linear_sampler_algorithm.h"
#include "texture-filtering/nearest_sampler_algorithm.h"
#include "texture-filtering/cubic_sampler_algorithm.h"
