#pragma once

struct ResourceDescriptor
{
    int width;
    int height;
};

template<typename T, std::size_t Dim, std::size_t N, typename TFloat, glm::qualifier Q>
concept Sampler = requires(T sampler)
{
    { sampler.AttachResource(std::declval<std::span<TFloat>>(), std::declval<ResourceDescriptor>()) };
    { sampler.AttachResource(std::declval<const std::vector<std::span<TFloat>>&>(), std::declval<ResourceDescriptor>()) };
    { sampler.Sample(std::declval<glm::vec<Dim, TFloat, Q>>()) } -> std::same_as<glm::vec<N, TFloat, Q>>;
};

template<typename T>
concept Sampler2D = Sampler<T, 2, 4, float, glm::defaultp>;

template<typename T>
concept Sampler3D = Sampler<T, 3, 4, float, glm::defaultp>;

template<typename T>
concept Sampler1D = Sampler<T, 1, 4, float, glm::defaultp>;
