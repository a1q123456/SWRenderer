#pragma once

template<typename T, size_t N>
constexpr size_t count_of(const T(& arr)[N])
{
    return N;
}

struct Ray
{
    glm::vec3 startPos;
    glm::vec3 direction;
};

constexpr std::uint32_t argb(std::uint8_t r, std::uint8_t g, std::uint8_t b, std::uint8_t a)
{
    return a << 24 | r << 16 | g << 8 | b;
}

