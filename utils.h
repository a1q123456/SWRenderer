#pragma once

template<typename T, size_t N>
constexpr size_t count_of(const T(& arr)[N])
{
    return N;
}
