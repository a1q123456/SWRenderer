#pragma once
#include "pixel_format.h"

template<std::size_t N>
void rescaleImage(
    const std::span<std::uint8_t>& imageData,
    std::span<std::uint8_t>& rescaledData,
    const EPixelFormat& pixelFormat,
    const std::array<std::uint32_t, N>& fromSize, 
    const std::array<std::uint32_t, N>& toSize);

