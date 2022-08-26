#pragma once

template<size_t Dim>
using TextureCoordinate = glm::vec<Dim, float, glm::qualifier::highp>;

enum class TextureFilteringMethods
{
    Nearest,
    Linear,
    Cubic,
};

#include "texture-filtering/linear_sampler_algorithm.h"
#include "texture-filtering/nearest_sampler_algorithm.h"
#include "texture-filtering/cubic_sampler_algorithm.h"
