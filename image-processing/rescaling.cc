#include "rescaling.h"
#include "texture-filtering/cubic_sampler_algorithm.h"

static std::pmr::unsynchronized_pool_resource memoryPool{};

template<std::size_t Dim>
void rescaleLineCubic(const std::span<std::uint8_t>& imageData,
                    const EPixelFormat& pixelFormat, 
                    const TextureBoundary<Dim>& srcBoundary,
                    std::size_t srcLineSize,
                    const TextureBoundary<Dim>& dstBoundary,
                    std::size_t dstLineSize,
                    const ResourceCoordinate<Dim>& increment,
                    const std::span<std::uint8_t>& rescaledData,
                    const ResourceCoordinate<Dim>& offset) noexcept
{
    auto desc = getPixelDesc(pixelFormat);
    auto res = Resource::Attach(imageData.data(), imageData.size());
    auto rv = ResourceView<Dim>{&res, 0, desc->dataType, desc->nbChannels, srcLineSize, srcBoundary};

    auto resDst = Resource::Attach(rescaledData.data(), rescaledData.size());
    auto rvDst = ResourceView<Dim>(&resDst, 0, desc->dataType, desc->nbChannels, dstLineSize, dstBoundary);
    int srcLen = glm::length(increment * srcBoundary);
    int dstLen = glm::length(increment * dstBoundary);

    auto gcd = std::gcd(srcLen, dstLen);
    auto kStep = static_cast<double>(srcLen / gcd) / static_cast<double>(dstLen / gcd);

    std::vector<std::array<float, 4>> kernels;
    kernels.resize(dstLen / gcd);
    for (int i = 0; i < kernels.size(); i++)
    {
        double t = (i) * kStep;
        t = t - static_cast<int>(t);

        kernels[i][0] = CubicSamplerAlgorithm::Kernel<float>(t + 1);
        kernels[i][1] = CubicSamplerAlgorithm::Kernel<float>(t);
        kernels[i][2] = CubicSamplerAlgorithm::Kernel<float>(t - 1);
        kernels[i][3] = CubicSamplerAlgorithm::Kernel<float>(t - 2);
    }

    for (int i = 0; i < dstLen; i++)
    {
        auto kernelIdx = i % kernels.size();
        auto kernelValues = kernels[kernelIdx];

        ResourceCoordinate<Dim> srcI0 = static_cast<ResourceCoordinate<Dim>::value_type>(i - 1 * (srcLen / gcd) / (dstLen / gcd)) * increment + offset;
        ResourceCoordinate<Dim> srcI1 = static_cast<ResourceCoordinate<Dim>::value_type>(i * (srcLen / gcd) / (dstLen / gcd)) * increment + offset;
        ResourceCoordinate<Dim> srcI2 = static_cast<ResourceCoordinate<Dim>::value_type>(i + 1 * (srcLen / gcd) / (dstLen / gcd)) * increment + offset;
        ResourceCoordinate<Dim> srcI3 = static_cast<ResourceCoordinate<Dim>::value_type>(i + 2 * (srcLen / gcd) / (dstLen / gcd)) * increment + offset;

        srcI0 = glm::clamp(srcI0, TextureBoundary<Dim>{0}, srcBoundary - increment);
        srcI1 = glm::clamp(srcI1, TextureBoundary<Dim>{0}, srcBoundary - increment);
        srcI2 = glm::clamp(srcI2, TextureBoundary<Dim>{0}, srcBoundary - increment);
        srcI3 = glm::clamp(srcI3, TextureBoundary<Dim>{0}, srcBoundary - increment);

        auto p0 = rv.template Get<float, 4, glm::defaultp>(srcI0);
        auto p1 = rv.template Get<float, 4, glm::defaultp>(srcI1);
        auto p2 = rv.template Get<float, 4, glm::defaultp>(srcI2);
        auto p3 = rv.template Get<float, 4, glm::defaultp>(srcI3);

        auto dstVal = kernelValues[0] * p0 + 
                        kernelValues[1] * p1 + 
                        kernelValues[2] * p2 + 
                        kernelValues[3] * p3;

        rvDst.Set(static_cast<TextureBoundary<Dim>::value_type>(i) * increment + offset, dstVal);
    }
}

void rescaleImage3D(ETextureFilteringMethods method, const std::span<std::uint8_t>& imageData,
                    const EPixelFormat& pixelFormat, int srcW, int srcH, int srcD, int srcLineSize, int dstW, int dstH,
                    int dstD, int dstLineSize, std::span<std::uint8_t>& rescaledData) noexcept
{

}

void rescaleImage2D(ETextureFilteringMethods method, const std::span<std::uint8_t>& imageData,
                    const EPixelFormat& pixelFormat, int srcW, int srcH, int srcLineSize, int dstW, int dstH,
                    int dstLineSize, const std::span<std::uint8_t>& rescaledData) noexcept
{
    switch (method)
    {
    case ETextureFilteringMethods::Cubic:
    {
        std::uint8_t* tempBuffer = nullptr;
        std::size_t tempBufferLen = 0;
        std::span<std::uint8_t> tempBufferSpan{};
        tempBufferLen = dstLineSize * srcH;
        tempBuffer = reinterpret_cast<std::uint8_t*>(memoryPool.allocate(tempBufferLen));
        tempBufferSpan = std::span{tempBuffer, tempBufferLen};
        for (int y = 0; y < srcH; y++)
        {
            rescaleLineCubic<1>(
                imageData.subspan(srcLineSize * y, srcLineSize), 
                pixelFormat, 
                Texture1DBoundary{static_cast<Texture1DBoundary::value_type>(srcW)},
                srcLineSize,
                Texture1DBoundary{static_cast<Texture1DBoundary::value_type>(dstW)}, 
                dstLineSize,
                ResourceCoordinate1D{1}, 
                tempBufferSpan.subspan(dstLineSize * y, dstLineSize),
                ResourceCoordinate1D{0});
        }
        // std::memset(rescaledData.data(), 0, rescaledData.size_bytes());
        // std::memcpy(rescaledData.data(), tempBufferSpan.data(), rescaledData.size_bytes());
        for (int x = 0; x < dstW; x++)
        {
            rescaleLineCubic<2>(
                tempBufferSpan, 
                pixelFormat, 
                Texture2DBoundary{static_cast<Texture2DBoundary::value_type>(dstW), static_cast<Texture2DBoundary::value_type>(srcH)}, 
                dstLineSize,
                Texture2DBoundary{static_cast<Texture2DBoundary::value_type>(dstW), static_cast<Texture2DBoundary::value_type>(dstH)},
                dstLineSize, 
                ResourceCoordinate2D{0, 1}, 
                rescaledData,
                ResourceCoordinate2D{x, 0});
        }
        memoryPool.deallocate(tempBuffer, tempBufferLen);
        break;
    }
    }
}

void rescaleImage1D(ETextureFilteringMethods method, const std::span<std::uint8_t>& imageData,
                    const EPixelFormat& pixelFormat, int srcW, int dstW,
                    const std::span<std::uint8_t>& rescaledData) noexcept
{
    switch (method)
    {
    case ETextureFilteringMethods::Cubic:
        return rescaleLineCubic<1>(
            imageData, 
            pixelFormat, 
            Texture1DBoundary{static_cast<Texture1DBoundary::value_type>(srcW)}, 
            imageData.size_bytes(),
            Texture1DBoundary{static_cast<Texture1DBoundary::value_type>(dstW)}, 
            rescaledData.size_bytes(),
            ResourceCoordinate1D{1}, 
            rescaledData,
            ResourceCoordinate1D{0});
    }
}