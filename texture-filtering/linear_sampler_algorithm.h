#pragma once
#include "texture-filtering/filters.h"
#include "resource.h"
#include <glm/gtx/compatibility.hpp>

class LinearSamplerAlgorithm
{
    template<std::size_t Dim>
    struct DimWrapper { };

    
    template<typename T, glm::length_t NChannels, glm::qualifier Q>
    static glm::vec<NChannels, T, Q> Lerp(
        const glm::vec<NChannels, T, Q>& x0,
        const glm::vec<NChannels, T, Q>& x1,
        float mix
    )
    {
        return glm::lerp(x0, x1, mix);
    }

    template<typename T, glm::length_t NChannels, glm::qualifier Q>
    static glm::vec<NChannels, T, Q> Lerp2D(
        const glm::vec<NChannels, T, Q>& x0y0,
        const glm::vec<NChannels, T, Q>& x1y0,
        const glm::vec<NChannels, T, Q>& x1y1,
        const glm::vec<NChannels, T, Q>& x0y1,
        float xMix,
        float yMix
    )
    {
        auto y0 = Lerp(x0y0, x1y0, xMix);
        auto y1 = Lerp(x0y1, x1y1, xMix);
        return Lerp(y0, y1, yMix);
    }

    template<typename T, glm::length_t NChannels, glm::qualifier Q>
    static glm::vec<NChannels, T, Q> Lerp3D(
        const glm::vec<NChannels, T, Q>& x0Y0Z0,
        const glm::vec<NChannels, T, Q>& x1Y0Z0,
        const glm::vec<NChannels, T, Q>& x1Y1Z0,
        const glm::vec<NChannels, T, Q>& x0Y1Z0,
        const glm::vec<NChannels, T, Q>& x0Y0Z1,
        const glm::vec<NChannels, T, Q>& x1Y0Z1,
        const glm::vec<NChannels, T, Q>& x1Y1Z1,
        const glm::vec<NChannels, T, Q>& x0Y1Z1,
        float xMix,
        float yMix,
        float zMix
    )
    {
        auto y0z0 = Lerp(x0Y0Z0, x1Y0Z0, xMix);
        auto y0z1 = Lerp(x0Y0Z1, x1Y0Z1, xMix);
        auto y1z1 = Lerp(x0Y1Z1, x1Y1Z1, xMix);
        auto y1z0 = Lerp(x0Y1Z0, x1Y1Z0, xMix);

        return Lerp2D(y0z0, y0z1, y1z1, y1z0, zMix, yMix);
    }

    template<typename T, glm::length_t NChannels, glm::qualifier Q>
    static glm::vec<NChannels, T, Q> InternalSample(
        const ResourceView<3>& resourceView,
        const glm::vec<3, T, Q>& location,
        DimWrapper<3>)
    {
        auto xMix = std::fmod(location.x, 1.0);
        auto yMix = std::fmod(location.y, 1.0);
        auto zMix = std::fmod(location.z, 1.0);

        return Lerp3D(
            resourceView.template Get<T, NChannels, Q>(
                ResourceCoordinate<3>{
                    std::floor(location.x),
                    std::floor(location.y),
                    std::floor(location.z)
                }
            ),
            resourceView.template Get<T, NChannels, Q>(
                ResourceCoordinate<3>{
                    std::ceil(location.x),
                    std::floor(location.y),
                    std::floor(location.z)
                }
            ),
            resourceView.template Get<T, NChannels, Q>(
                ResourceCoordinate<3>{
                    std::ceil(location.x),
                    std::ceil(location.y),
                    std::floor(location.z)
                }
            ),
            resourceView.template Get<T, NChannels, Q>(
                ResourceCoordinate<3>{
                    std::floor(location.x),
                    std::ceil(location.y),
                    std::floor(location.z)
                }
            ),
            resourceView.template Get<T, NChannels, Q>(
                ResourceCoordinate<3>{
                    std::floor(location.x),
                    std::floor(location.y),
                    std::ceil(location.z)
                }
            ),
            resourceView.template Get<T, NChannels, Q>(
                ResourceCoordinate<3>{
                    std::ceil(location.x),
                    std::floor(location.y),
                    std::ceil(location.z)
                }
            ),
            resourceView.template Get<T, NChannels, Q>(
                ResourceCoordinate<3>{
                    std::ceil(location.x),
                    std::ceil(location.y),
                    std::ceil(location.z)
                }
            ),
            resourceView.template Get<T, NChannels, Q>(
                ResourceCoordinate<3>{
                    std::floor(location.x),
                    std::ceil(location.y),
                    std::ceil(location.z)
                }
            ),
            xMix,
            yMix,
            zMix
        );
    }

    template<typename T, glm::length_t NChannels, glm::qualifier Q>
    static glm::vec<NChannels, T, Q> InternalSample(
        const ResourceView<2>& resourceView,
        const glm::vec<2, T, Q>& location,
        DimWrapper<2>)
    {
        auto xMix = std::fmod(location.x, 1.0);
        auto yMix = std::fmod(location.y, 1.0);

        return Lerp2D(
            resourceView.template Get<T, NChannels, Q>(
                ResourceCoordinate<2>{
                    std::floor(location.x),
                    std::floor(location.y)
                }
            ),
            resourceView.template Get<T, NChannels, Q>(
                ResourceCoordinate<2>{
                    std::ceil(location.x),
                    std::floor(location.y)
                }
            ),
            resourceView.template Get<T, NChannels, Q>(
                ResourceCoordinate<2>{
                    std::ceil(location.x),
                    std::ceil(location.y)
                }
            ),
            resourceView.template Get<T, NChannels, Q>(
                ResourceCoordinate<2>{
                    std::floor(location.x),
                    std::ceil(location.y)
                }
            ),
            xMix,
            yMix
        );
    }

   
public:
    static constexpr ETextureFilteringMethods ID = ETextureFilteringMethods::Linear; 

    template<typename T, glm::length_t NChannels, glm::qualifier Q, std::size_t Dim>
    static glm::vec<NChannels, T, Q> Sample(
        const TextureCoordinate<Dim + 1>& coord, 
        const std::vector<ResourceView<Dim>>& resourceViews)
    {
        TextureCoordinate<Dim> coordNoLevel{coord};
        double distance = coord[Dim];
        double level = 0;
        bool mipmap = resourceViews.size() > 1;

        if (mipmap)
        {
            level = glm::clamp(distance, 0.0, 1.0) * (resourceViews.size() - 1);
        }
        int mipmapLevel = level;
        double mipmapMix = std::fmod(level, 1.0);

        auto boundary = resourceViews[level].Boundary();

        glm::vec<Dim, T, Q> location = coordNoLevel;
        location *= boundary;

        auto ret = InternalSample<T, NChannels, Q>(resourceViews[mipmapLevel], location, DimWrapper<Dim>{});
        if (mipmap)
        {
            return Lerp(ret, 
                            InternalSample<T, NChannels, Q>(resourceViews[mipmapLevel + 1], location, DimWrapper<Dim>{}), 
                            mipmapMix);
        }
        else
        {
            return ret;
        }
    }
};
