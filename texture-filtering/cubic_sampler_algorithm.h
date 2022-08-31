#pragma once
#include "texture-filtering/filters.h"
#include "resource.h"

class CubicSamplerAlgorithm
{
   
public:
    enum class EInterpolationMethod
    {
        HermiteSpline,
        CubicConvolution
    };

    static constexpr EInterpolationMethod INTERPOLATION_METHOD = EInterpolationMethod::CubicConvolution;

    template<std::size_t Dim>
    struct DimWrapper { };

    template<typename T>
    using Interpolate2DParams = std::array<std::array<T, 4>, 4>;

    template<typename T>
    using Interpolate3DParams = std::array<Interpolate2DParams<T>, 4>;

    static constexpr double ALPHA = -0.5;
    static constexpr double C = 0.5;

    template<typename T>
    static T Kernel(T x)
    {
        if (x == 0)
        {
            return 1;
        }

        auto absX = x < 0 ? -x : x;
        auto xx = absX * absX;
        auto xxx = xx * absX;

        if (0 < absX && absX < 1)
        {
            return (ALPHA + 2) * xxx - (ALPHA + 3) * xx + 1;
        }
        else if (1 < absX && absX < 2)
        {
            return ALPHA * xxx - 5 * ALPHA * xx + 8 * ALPHA * absX - 4 * ALPHA;
        }
        else
        {
            return 0;
        }
    }

    // Cubic convolution interpolation by Keys
    template<typename TInterval, typename TVec>
    static TVec Interpolate(TInterval t, const std::array<TVec, 4>& p)
    {
        if constexpr (INTERPOLATION_METHOD == EInterpolationMethod::HermiteSpline)
        {
            auto m1 = (1 - static_cast<TVec::value_type>(C)) * (p[2] - p[0]) / static_cast<TVec::value_type>(2);
            auto m2 = (1 - static_cast<TVec::value_type>(C)) * (p[3] - p[1]) / static_cast<TVec::value_type>(2);
            auto ttt = t * t * t;
            auto tt = t * t;
            return static_cast<TVec::value_type>(2 * ttt - 3 * tt + 1) * p[1] +
                   static_cast<TVec::value_type>(-2 * ttt + 3 * tt) * p[2] +
                   static_cast<TVec::value_type>(ttt - 2 * tt + t) * m1 +
                   static_cast<TVec::value_type>(ttt - tt) * m2;
        }
        else if constexpr (INTERPOLATION_METHOD == EInterpolationMethod::CubicConvolution)
        {
            return p[0] * Kernel<typename TVec::value_type>(t + 1) +
                   p[1] * Kernel<typename TVec::value_type>(t + 0) +
                   p[2] * Kernel<typename TVec::value_type>(t - 1) +
                   p[3] * Kernel<typename TVec::value_type>(t - 2);


            // auto t0 = t + 1;
            // auto t1 = t + 0;
            // auto t2 = 1 - t;
            // auto t3 = 2 - t;
            // auto t02 = t0 * t0;
            // auto t03 = t02 * t0;
            // auto t12 = t1 * t1;
            // auto t13 = t12 * t1;
            // auto t22 = t2 * t2;
            // auto t23 = t22 * t2;
            // auto t32 = t3 * t3;
            // auto t33 = t32 * t3;

            // return p[0] * static_cast<typename TVec::value_type>(ALPHA * t03 - 5 * ALPHA * t02 + 8 * ALPHA * t0 - 4 * ALPHA) +
            //        p[1] * static_cast<typename TVec::value_type>((ALPHA + 2) * t13 - (ALPHA + 3) * t12 + 1) +
            //        p[2] * static_cast<typename TVec::value_type>((ALPHA + 2) * t23 - (ALPHA + 3) * t22 + 1) +
            //        p[3] * static_cast<typename TVec::value_type>(ALPHA * t33 - 5 * ALPHA * t32 + 8 * ALPHA * t3 - 4 * ALPHA);

            // typename TVec::value_type a = ALPHA;
            // return (((a * p[0] - 3 * a * p[1] + 3 * a * p[2] - a * p[3]) * t +
            // (p[0] + 5 * a * p[1] - 4 * a * p[2] + a * p[3])) * t +
            // (a * p[0] - a * p[2])) * t + p[1];
        }
    }

    template<typename T, glm::length_t NChannels, glm::qualifier Q>
    static glm::vec<NChannels, T, Q> Interpolate2D(
        const Interpolate2DParams<glm::vec<NChannels, T, Q>>& params,
        float xMix,
        float yMix
    )
    {
        auto y0 = Interpolate(xMix, std::array<glm::vec<NChannels, T, Q>, 4>{ params[0][0], params[1][0], params[2][0], params[3][0] });
        auto y1 = Interpolate(xMix, std::array<glm::vec<NChannels, T, Q>, 4>{ params[0][1], params[1][1], params[2][1], params[3][1] });
        auto y2 = Interpolate(xMix, std::array<glm::vec<NChannels, T, Q>, 4>{ params[0][2], params[1][2], params[2][2], params[3][2] });
        auto y3 = Interpolate(xMix, std::array<glm::vec<NChannels, T, Q>, 4>{ params[0][3], params[1][3], params[2][3], params[3][3] });

        return Interpolate(yMix, std::array<glm::vec<NChannels, T, Q>, 4>{ y0, y1, y2, y3 });
    }

    template<typename T, glm::length_t NChannels, glm::qualifier Q>
    static glm::vec<NChannels, T, Q> Interpolate3D(
        const Interpolate3DParams<glm::vec<NChannels, T, Q>>& params,
        float xMix,
        float yMix,
        float zMix
    )
    {
        auto y0z0 = Interpolate(xMix, params[0][0][0], params[1][0][0], params[2][0][0], params[3][0][0]);
        auto y0z1 = Interpolate(xMix, params[0][0][1], params[1][0][1], params[2][0][1], params[3][0][1]);
        auto y0z2 = Interpolate(xMix, params[0][0][2], params[1][0][2], params[2][0][2], params[3][0][2]);
        auto y0z3 = Interpolate(xMix, params[0][0][3], params[1][0][3], params[2][0][3], params[3][0][3]);
        auto y1z0 = Interpolate(xMix, params[0][1][0], params[1][1][0], params[2][1][0], params[3][1][0]);
        auto y1z1 = Interpolate(xMix, params[0][1][1], params[1][1][1], params[2][1][1], params[3][1][1]);
        auto y1z2 = Interpolate(xMix, params[0][1][2], params[1][1][2], params[2][1][2], params[3][1][2]);
        auto y1z3 = Interpolate(xMix, params[0][1][3], params[1][1][3], params[2][1][3], params[3][1][3]);
        auto y2z0 = Interpolate(xMix, params[0][2][0], params[1][2][0], params[2][2][0], params[3][2][0]);
        auto y2z1 = Interpolate(xMix, params[0][2][1], params[1][2][1], params[2][2][1], params[3][2][1]);
        auto y2z2 = Interpolate(xMix, params[0][2][2], params[1][2][2], params[2][2][2], params[3][2][2]);
        auto y2z3 = Interpolate(xMix, params[0][2][3], params[1][2][3], params[2][2][3], params[3][2][3]);
        auto y3z0 = Interpolate(xMix, params[0][3][0], params[1][3][0], params[2][3][0], params[3][3][0]);
        auto y3z1 = Interpolate(xMix, params[0][3][1], params[1][3][1], params[2][3][1], params[3][3][1]);
        auto y3z2 = Interpolate(xMix, params[0][3][2], params[1][3][2], params[2][3][2], params[3][3][2]);
        auto y3z3 = Interpolate(xMix, params[0][3][3], params[1][3][3], params[2][3][3], params[3][3][3]);

        return Interpolate2D({ 
            { y0z0, y0z1, y0z2, y0z3 },
            { y1z0, y1z1, y1z2, y1z3 },
            { y2z0, y2z1, y2z2, y2z3 },
            { y3z0, y3z1, y3z2, y3z3 },
        }, zMix, yMix);
    }

    template<typename T, glm::length_t NChannels, glm::qualifier Q>
    static glm::vec<NChannels, T, Q> InternalSample(
        const ResourceView<3>& resourceView,
        const glm::vec<3, T, Q>& location,
        DimWrapper<3>)
    {
        auto boundary = resourceView.Boundary();

        float xMix = location.x - static_cast<int>(location.x);
        float yMix = location.y - static_cast<int>(location.y);
        float zMix = location.z - static_cast<int>(location.z);

        int ix1 = location.x;
        int ix2 = std::min(ix1 + 1, boundary[0]);
        int ix3 = std::min(ix1 + 2, boundary[0]);
        int ix0 = std::max(0, ix1 - 1);

        int iy1 = location.y;
        int iy2 = std::min(iy1 + 1, boundary[1]);
        int iy3 = std::min(iy1 + 2, boundary[1]);
        int iy0 = std::max(0, iy1 - 1);

        int iz1 = location.z;
        int iz2 = std::min(iz1 + 1, boundary[2]);
        int iz3 = std::min(iz1 + 2, boundary[2]);
        int iz0 = std::max(0, iz1 - 1);

        return Interpolate3D({
            {
                {
                    resourceView.Get({ ix0, iy0, iz0 }), 
                    resourceView.Get({ ix0, iy0, iz1 }), 
                    resourceView.Get({ ix0, iy0, iz2 }), 
                    resourceView.Get({ ix0, iy0, iz3 })
                },
                {
                    resourceView.Get({ ix0, iy1, iz0 }), 
                    resourceView.Get({ ix0, iy1, iz1 }), 
                    resourceView.Get({ ix0, iy1, iz2 }), 
                    resourceView.Get({ ix0, iy1, iz3 })
                },
                {
                    resourceView.Get({ ix0, iy2, iz0 }), 
                    resourceView.Get({ ix0, iy2, iz1 }), 
                    resourceView.Get({ ix0, iy2, iz2 }), 
                    resourceView.Get({ ix0, iy2, iz3 })
                },
                {
                    resourceView.Get({ ix0, iy3, iz0 }), 
                    resourceView.Get({ ix0, iy3, iz1 }), 
                    resourceView.Get({ ix0, iy3, iz2 }), 
                    resourceView.Get({ ix0, iy3, iz3 })
                }
            },
            {
                {
                    resourceView.Get({ ix1, iy0, iz0 }), 
                    resourceView.Get({ ix1, iy0, iz1 }), 
                    resourceView.Get({ ix1, iy0, iz2 }), 
                    resourceView.Get({ ix1, iy0, iz3 })
                },
                {
                    resourceView.Get({ ix1, iy1, iz0 }), 
                    resourceView.Get({ ix1, iy1, iz1 }), 
                    resourceView.Get({ ix1, iy1, iz2 }), 
                    resourceView.Get({ ix1, iy1, iz3 })
                },
                {
                    resourceView.Get({ ix1, iy2, iz0 }), 
                    resourceView.Get({ ix1, iy2, iz1 }), 
                    resourceView.Get({ ix1, iy2, iz2 }), 
                    resourceView.Get({ ix1, iy2, iz3 })
                },
                {
                    resourceView.Get({ ix1, iy3, iz0 }), 
                    resourceView.Get({ ix1, iy3, iz1 }), 
                    resourceView.Get({ ix1, iy3, iz2 }), 
                    resourceView.Get({ ix1, iy3, iz3 })
                }
            },
            {
                {
                    resourceView.Get({ ix2, iy0, iz0 }), 
                    resourceView.Get({ ix2, iy0, iz1 }), 
                    resourceView.Get({ ix2, iy0, iz2 }), 
                    resourceView.Get({ ix2, iy0, iz3 })
                },
                {
                    resourceView.Get({ ix2, iy1, iz0 }), 
                    resourceView.Get({ ix2, iy1, iz1 }), 
                    resourceView.Get({ ix2, iy1, iz2 }), 
                    resourceView.Get({ ix2, iy1, iz3 })
                },
                {
                    resourceView.Get({ ix2, iy2, iz0 }), 
                    resourceView.Get({ ix2, iy2, iz1 }), 
                    resourceView.Get({ ix2, iy2, iz2 }), 
                    resourceView.Get({ ix2, iy2, iz3 })
                },
                {
                    resourceView.Get({ ix2, iy3, iz0 }), 
                    resourceView.Get({ ix2, iy3, iz1 }), 
                    resourceView.Get({ ix2, iy3, iz2 }), 
                    resourceView.Get({ ix2, iy3, iz3 })
                }
            },
            {
                {
                    resourceView.Get({ ix3, iy0, iz0 }), 
                    resourceView.Get({ ix3, iy0, iz1 }), 
                    resourceView.Get({ ix3, iy0, iz2 }), 
                    resourceView.Get({ ix3, iy0, iz3 })
                },
                {
                    resourceView.Get({ ix3, iy1, iz0 }), 
                    resourceView.Get({ ix3, iy1, iz1 }), 
                    resourceView.Get({ ix3, iy1, iz2 }), 
                    resourceView.Get({ ix3, iy1, iz3 })
                },
                {
                    resourceView.Get({ ix3, iy2, iz0 }), 
                    resourceView.Get({ ix3, iy2, iz1 }), 
                    resourceView.Get({ ix3, iy2, iz2 }), 
                    resourceView.Get({ ix3, iy2, iz3 })
                },
                {
                    resourceView.Get({ ix3, iy3, iz0 }), 
                    resourceView.Get({ ix3, iy3, iz1 }), 
                    resourceView.Get({ ix3, iy3, iz2 }), 
                    resourceView.Get({ ix3, iy3, iz3 })
                }
            }
        }, xMix, yMix, zMix);
    }

    template<typename T, glm::length_t NChannels, glm::qualifier Q>
    static glm::vec<NChannels, T, Q> InternalSample(
        const ResourceView<2>& resourceView,
        const glm::vec<2, T, Q>& location,
        DimWrapper<2>)
    {
        auto boundary = resourceView.Boundary();
        float xMix = location.x - static_cast<int>(location.x);
        float yMix = location.y - static_cast<int>(location.y);

        int ix1 = std::clamp<int>(location.x, 0, boundary.x - 1);
        int ix2 = std::clamp<int>(ix1 + 1, 0, boundary.x - 1);
        int ix3 = std::clamp<int>(ix1 + 2, 0, boundary.x - 1);
        int ix0 = std::clamp<int>(ix1 - 1, 0, boundary.x - 1);

        int iy1 = std::clamp<int>(location.y, 0, boundary.y - 1);
        int iy2 = std::clamp<int>(iy1 + 1, 0, boundary.y - 1);
        int iy3 = std::clamp<int>(iy1 + 2, 0, boundary.y - 1);
        int iy0 = std::clamp<int>(iy1 - 1, 0, boundary.y - 1);

        return Interpolate2D<T, NChannels, Q>(Interpolate2DParams<glm::vec<NChannels, T, Q>>{
            std::array<glm::vec<NChannels, T, Q>, 4>{
                resourceView.template Get<T, NChannels, Q>({ ix0, iy0 }), 
                resourceView.template Get<T, NChannels, Q>({ ix0, iy1 }), 
                resourceView.template Get<T, NChannels, Q>({ ix0, iy2 }), 
                resourceView.template Get<T, NChannels, Q>({ ix0, iy3 })
            },
            std::array<glm::vec<NChannels, T, Q>, 4>{
                resourceView.template Get<T, NChannels, Q>({ ix1, iy0 }), 
                resourceView.template Get<T, NChannels, Q>({ ix1, iy1 }), 
                resourceView.template Get<T, NChannels, Q>({ ix1, iy2 }), 
                resourceView.template Get<T, NChannels, Q>({ ix1, iy3 })
            },
            std::array<glm::vec<NChannels, T, Q>, 4>{
                resourceView.template Get<T, NChannels, Q>({ ix2, iy0 }), 
                resourceView.template Get<T, NChannels, Q>({ ix2, iy1 }), 
                resourceView.template Get<T, NChannels, Q>({ ix2, iy2 }), 
                resourceView.template Get<T, NChannels, Q>({ ix2, iy3 })
            },
            std::array<glm::vec<NChannels, T, Q>, 4>{
                resourceView.template Get<T, NChannels, Q>({ ix3, iy0 }), 
                resourceView.template Get<T, NChannels, Q>({ ix3, iy1 }), 
                resourceView.template Get<T, NChannels, Q>({ ix3, iy2 }), 
                resourceView.template Get<T, NChannels, Q>({ ix3, iy3 })
            }
        }, xMix, yMix);
    }

    static constexpr ETextureFilteringMethods ID = ETextureFilteringMethods::Cubic; 

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
        double mipmapMix = level - mipmapLevel;

        auto boundary = resourceViews[level].Boundary();

        glm::vec<Dim, T, Q> location = coordNoLevel * boundary;

        auto ret = InternalSample<T, NChannels, Q>(resourceViews[mipmapLevel], location, DimWrapper<Dim>{});
        if (mipmap)
        {
            glm::vec<Dim, T, Q> locationMipmap = coordNoLevel * resourceViews[mipmapLevel + 1].Boundary();
            return glm::lerp(ret, InternalSample<T, NChannels, Q>(resourceViews[mipmapLevel + 1], locationMipmap, DimWrapper<Dim>{}), static_cast<T>(mipmapMix));
        }
        else
        {
            return ret;
        }
    }
};
