#pragma once

template <typename T> class CudaAllocator
{
public:
    using value_type = T;
    using is_always_equal = std::true_type;

    CudaAllocator() = default;

    template <typename U> constexpr CudaAllocator(const CudaAllocator<U>&) noexcept
    {
    }

    [[nodiscard]] constexpr T* allocate(std::size_t n)
    {
        void* ret = nullptr;
        auto err = cudaMalloc(&ret, n * sizeof(T));
        if (err != cudaSuccess)
        {
            throw std::bad_alloc{};
        }
        return reinterpret_cast<T*>(ret);
    }

    constexpr void deallocate(T* p, std::size_t n)
    {
        cudaFree(p);
    }
};

template <typename T> bool operator==(const CudaAllocator<T>& a, const CudaAllocator<T>& b)
{
    return true;
}

template <typename T> bool operator!=(const CudaAllocator<T>& a, const CudaAllocator<T>& b)
{
    return false;
}
