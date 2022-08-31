#pragma once
#include "pixel_format.h"
#include "texture-filtering/filters.h"

void rescaleImage3D(
    ETextureFilteringMethods method,
    const std::span<std::uint8_t>& imageData,
    const EPixelFormat& pixelFormat,
    int srcW, int srcH, int srcD,
    int srcLineSize,
    int dstW, int dstH, int dstD,
    int dstLineSize, const std::span<std::uint8_t>& rescaledData) noexcept;

void rescaleImage2D(
    ETextureFilteringMethods method,
    const std::span<std::uint8_t>& imageData,
    const EPixelFormat& pixelFormat,
    int srcW, int srcH,
    int srcLineSize,
    int dstW, int dstH,
    int dstLineSize, const std::span<std::uint8_t>& rescaledData) noexcept;

void rescaleImage1D(
    ETextureFilteringMethods method,
    const std::span<std::uint8_t>& imageData,
    const EPixelFormat& pixelFormat,
    int srcW,
    int dstW,
    const std::span<std::uint8_t>& rescaledData) noexcept;
